"""
@author: Guo Shiguang
@software: PyCharm
@file: UIELightGenerator.py
@time: 2022/5/5 21:47
"""

import torch
import torch.nn.functional as F
from fairseq import search
from fairseq.utils import new_arange


def _skip(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [_skip(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: _skip(v, mask) for k, v in x.items()}

    raise NotImplementedError


def _skip_encoder_out(encoder, encoder_out, mask):
    if not mask.any():
        return encoder_out
    else:
        return encoder.reorder_encoder_out(encoder_out, mask.nonzero().squeeze())


def _fill(x, mask, y, padding_idx):
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))
    n_selected = mask.sum()
    assert n_selected == y.size(0)

    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, :y.size(1)] = y
        else:
            x[mask, :y.size(1), :] = y
    else:
        x[mask] = y
    return x


def _apply_ins_masks(
        in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx
):
    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = (
            new_arange(out_lengths, out_max_len)[None, :]
            < out_lengths[:, None]
    )

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
            .fill_(padding_idx)
            .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])

    return out_tokens, out_scores


def _apply_ins_words(
        in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx
):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            word_ins_masks, word_ins_scores[word_ins_masks]
        )
    else:
        out_scores = None

    return out_tokens, out_scores


def _apply_del_words(
        in_tokens, in_scores, word_del_pred, padding_idx, bos_idx, eos_idx
):
    # apply deletion to a tensor
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)

    reordering = (
        new_arange(in_tokens)
            .masked_fill_(word_del_pred, max_len)
            .sort(1)[1]
    )

    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(1, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)

    return out_tokens, out_scores


class UIELightGenerator(object):
    def __init__(
            self,
            tgt_dict,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.,
            unk_penalty=0.,
            retain_dropout=False,
            sampling=False,
            sampling_topk=-1,
            sampling_topp=-1.0,
            temperature=1.,
            diverse_beam_groups=-1,
            diverse_beam_strength=0.5,
            match_source_len=False,
            no_repeat_ngram_size=0,
            nar_max_length=-1
    ):
        self.max_ins_len = 10
        self.len_ratio = 2
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size

        self.eos_penalty = 0.0

        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.nar_max_length = nar_max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = search.BeamSearch(tgt_dict)

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        model = models[0]
        model.eval()
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)

        with torch.no_grad():
            encoder_out = model.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
            bsz = src_tokens.size(0)

            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(1)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * self.len_ratio + 2).clamp(min=10).long()
            plc_ins_len = torch.full_like(max_lens, self.max_ins_len)

            prev_tokens = self.make_emptyes(sample['target'], max_lens.max())
            scores = src_tokens.new(bsz, max_lens.max()).float().fill_(0)

            # for stage in ['event', 'argument']:
            for stage in ['event']:
                can_del_word = prev_tokens.ne(self.pad).sum(1) > 2
                if can_del_word.sum() != 0:  # we cannot delete, skip
                    word_del_out, word_del_attn = model.decoder.forward_word_del(
                        _skip(prev_tokens, can_del_word),
                        _skip_encoder_out(model.encoder, encoder_out, can_del_word),
                        stage=stage
                    )
                    word_del_score = F.log_softmax(word_del_out, 2)
                    word_del_pred = word_del_score.max(-1)[1].bool()

                    _tokens, _scores = _apply_del_words(
                        prev_tokens[can_del_word],
                        scores[can_del_word],
                        word_del_pred,
                        self.pad,
                        self.bos,
                        self.eos,
                    )
                    prev_tokens = _fill(prev_tokens, can_del_word, _tokens, self.pad)
                    scores = _fill(scores, can_del_word, _scores, 0)

                # insert placeholders
                can_ins_mask = prev_tokens.ne(self.pad).sum(1) < max_lens
                if can_ins_mask.sum() != 0:
                    mask_ins_out, _ = model.decoder.forward_mask_ins(
                        _skip(prev_tokens, can_ins_mask),
                        _skip_encoder_out(model.encoder, encoder_out, can_ins_mask),
                        stage=stage
                    )
                    mask_ins_score = F.log_softmax(mask_ins_out, 2)
                    if self.eos_penalty > 0.0:
                        mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - self.eos_penalty
                    mask_ins_pred = mask_ins_score.max(-1)[1]
                    mask_ins_pred = torch.min(
                        mask_ins_pred, plc_ins_len[can_ins_mask, None].expand_as(mask_ins_pred)
                    )

                    _tokens, _scores = _apply_ins_masks(
                        prev_tokens[can_ins_mask],
                        scores[can_ins_mask],
                        mask_ins_pred,
                        self.pad,
                        self.unk,
                        self.eos
                    )
                    prev_tokens = _fill(prev_tokens, can_ins_mask, _tokens, self.pad)
                    scores = _fill(scores, can_ins_mask, _scores, 0)

                # insert words
                can_ins_word = prev_tokens.eq(self.unk).sum(1) > 0
                if can_ins_word.sum() != 0:
                    word_ins_out, word_ins_attn = model.decoder.forward_word_ins(
                        _skip(prev_tokens, can_ins_word),
                        _skip_encoder_out(model.encoder, encoder_out, can_ins_word),
                        stage=stage
                    )
                    word_ins_score, word_ins_pred = F.log_softmax(word_ins_out, 2).max(-1)
                    _tokens, _scores = _apply_ins_words(
                        prev_tokens[can_ins_word],
                        scores[can_ins_word],
                        word_ins_pred,
                        word_ins_score,
                        self.unk,
                    )

                    prev_tokens = _fill(prev_tokens, can_ins_word, _tokens, self.pad)
                    scores = _fill(scores, can_ins_word, _scores, 0)

            cut_off = prev_tokens.ne(self.pad).sum(1).max()
            prev_tokens = prev_tokens[:, :cut_off]
            scores = scores[:, :cut_off]

            finalized = []
            for score_pred, word_pred in zip(scores, prev_tokens):
                finalized.append(
                    [{'tokens': word_pred[word_pred.ne(self.pad)], 'score': -1, 'alignment': None,
                      'positional_scores': score_pred}])
        return finalized

    def make_emptyes(self, tokens, max_len):
        bsz = tokens.size(0)
        bos = self.bos
        eos = self.eos
        pad = self.pad
        empty = [
            [bos, eos] for _ in range(bsz)
        ]
        empty_padded = [
            item + [pad for _ in range(max_len.max() - len(item))] for item in empty
        ]
        empty_tensor = torch.tensor(empty_padded, device=tokens.device)
        return empty_tensor
