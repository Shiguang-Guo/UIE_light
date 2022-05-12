"""
@author: Guo Shiguang
@software: PyCharm
@file: uie_light.py
@time: 2022/3/10 15:27
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerModel, TransformerEncoder
from fairseq.modules import (
    LayerNorm,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from bang_only.NgramTransformerDecoderARNARMixedLayer import NgramTransformerDecoderARNARMixedLayer
from bang_only.learned_positional_embedding import LearnedPositionalEmbedding
from bang_only.ngram_multihead_attention_AR_NAR_mixed import ngram_attention_bias

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512


def load_libnat():
    try:
        from fairseq import libnat
    except ImportError as e:
        import sys
        sys.stderr.write("ERROR: missing libnat. run `pip install --editable .`\n")
        raise e
    return libnat


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx):
    """
    比较in和out的差异
    @param in_tokens: [1,3,5]
    @param out_tokens: [1,2,3,4,5]
    @param padding_idx: 1
    @param unk_idx: ^
    @return:    masked_tgt_masks:out的每个位置与in的差别，[0,1,0,1,0]
                masked_tgt_tokens: 把out与in不同的位置用unk代替，[1,^,3,^,5]
                mask_ins_targets:in的每个间隔中需要插入的数量，[1,1]
    """
    libnat = load_libnat()

    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

    in_tokens_list = [
        [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
    ]
    out_tokens_list = [
        [t for t in s if t != padding_idx]
        for i, s in enumerate(out_tokens.tolist())
    ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )
    mask_inputs = [
        [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
    ]

    # generate labels
    masked_tgt_masks = []
    for mask_input in mask_inputs:
        mask_label = []
        for beam_size in mask_input[1:-1]:  # HACK 1:-1
            mask_label += [0] + [1 for _ in range(beam_size)]
        masked_tgt_masks.append(
            mask_label + [0 for _ in range(out_seq_len - len(mask_label))]
        )
    mask_ins_targets = [
        mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
        for mask_input in mask_inputs
    ]

    # transform to tensor
    masked_tgt_masks = torch.tensor(
        masked_tgt_masks, device=out_tokens.device
    ).bool()
    mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
    masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
    return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets


def _get_del_targets(in_tokens, out_tokens, padding_idx):
    libnat = load_libnat()

    out_seq_len = out_tokens.size(1)

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )
    word_del_targets = [b[-1] for b in full_labels]
    word_del_targets = [
        labels + [0 for _ in range(out_seq_len - len(labels))]
        for labels in word_del_targets
    ]

    # transform to tensor
    word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
    return word_del_targets


def _get_del_ins_targets(in_tokens, out_tokens, padding_idx):
    libnat = load_libnat()

    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )

    word_del_targets = [b[-1] for b in full_labels]
    word_del_targets = [
        labels + [0 for _ in range(out_seq_len - len(labels))]
        for labels in word_del_targets
    ]

    mask_inputs = [
        [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
    ]
    mask_ins_targets = [
        mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
        for mask_input in mask_inputs
    ]

    # transform to tensor
    mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
    word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
    return word_del_targets, mask_ins_targets


@register_model('uie_light')
class UIELightNAR(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.first_stage_nar = args.first_stage_nar
        self.second_stage_nar = args.second_stage_nar
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )
        parser.add_argument(
            "--early-exit",
            default="4,5,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--first-stage-nar",
            default=True,
            type=bool
        )
        parser.add_argument(
            "--second-stage-nar",
            default=True,
            type=bool
        )
        parser.add_argument(
            "--share-stage-layers",
            default=False,
            type=bool
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NgramTransformerDecoder(args, tgt_dict, embed_tokens)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    def forward(self, src_tokens=None, src_lengths=None, empty_tokens=None, event_only_tokens=None,
                include_arguments_tokens=None, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(empty_tokens, encoder_out=encoder_out, )
        return decoder_out


class NgramTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))
        self.num_buckets = args.num_buckets
        self.relative_max_distance = args.relative_max_distance

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_dim = embed_dim
        self.embed_tokens = embed_tokens
        self.embed_scale = None  # math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.embed_positions = LearnedPositionalEmbedding(
            args.max_target_positions + 2 + self.padding_idx, embed_dim, self.padding_idx,
        )

        self.ngram_input_embed = Embedding(1, input_embed_dim, None)

        self.layers = nn.ModuleList([])

        self.layers.extend([
            NgramTransformerDecoderARNARMixedLayer(
                1,
                args.decoder_embed_dim,
                args.decoder_ffn_embed_dim,
                args.decoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,

            )
            for _ in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.embed_dim ** -0.5)

        self.emb_layer_norm = LayerNorm(embed_dim)
        self.apply(init_bert_params)

    def forward(self,
                prev_output_tokens,
                encoder_out=None,
                incremental_state=None,
                **unused):
        x, extra = self.extract_features_NAR(prev_output_tokens, encoder_out, incremental_state, **unused)
        return self.output_layer(x), extra

    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = val_if_large.int()
        result = result + torch.where(is_small, n.int(), val_if_large)
        return result

    def cal_pretrain_relative_positions(self, real_positions):
        # [[1,2,3,4]]

        # main stream
        # [[
        #   [1, 2, 3, 4]
        # ]]
        main_stream_relative_positions = real_positions.unsqueeze(1)
        # [B,T,T/S]
        # [[
        #   [1, 2, 3, 4]
        #   [1, 2, 3, 4]
        #   [1, 2, 3, 4]
        #   [1, 2, 3, 4]
        # ]]
        main_stream_relative_positions = main_stream_relative_positions.repeat(1, real_positions.size(-1), 1)
        # [B,T,1]
        # [[
        #   [1]
        #   [2]
        #   [3]
        #   [4]
        # ]]
        real_positions_main = real_positions.unsqueeze(-1)
        # [[[0, 1, 2, 3]
        #   [-1, 0, 1, 2]
        #   [-2, -1, 0, 1]
        #   [-3, -2, -1, 0]]
        main_stream_relative_positions = main_stream_relative_positions - real_positions_main

        # predicting stream
        # input shift
        # [[0,1,2,3]]
        real_positions_shift_predicting_stream = real_positions - 1
        # [B,1, 2*T]
        # [[0,1,2,3,1,2,3,4]]
        predicting_stream_relative_positions = torch.cat((real_positions_shift_predicting_stream, real_positions),
                                                         dim=-1).unsqueeze(1)
        # [B,T, 2*T]
        # [[
        #   [0,1,2,3,1,2,3,4]
        #   [0,1,2,3,1,2,3,4]
        #   [0,1,2,3,1,2,3,4]
        #   [0,1,2,3,1,2,3,4]
        # ]]
        predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, real_positions.size(-1),
                                                                                           1)
        # [B,T, 1]
        real_positions_predicting_stream = real_positions.unsqueeze(-1)
        predicting_stream_relative_positions = predicting_stream_relative_positions - real_positions_predicting_stream
        i_buckets_main_stream = self._relative_positions_bucket(main_stream_relative_positions, bidirectional=False)
        i_bucket_relative_stream = self._relative_positions_bucket(predicting_stream_relative_positions,
                                                                   bidirectional=False)
        return i_buckets_main_stream, i_bucket_relative_stream

    def cal_finetune_relative_positions(self, real_positions):
        n_tokens = real_positions.size(-1)
        batch_size = real_positions.size(0)
        if not hasattr(self,
                       '_finetune_i_bucket_main_stream') or \
                self._finetune_i_bucket_main_stream is None \
                or self._finetune_i_bucket_main_stream.device != real_positions.device:
            fake_positions = torch.arange(1, self.max_target_positions + 1).repeat(1, 1)
            finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream = \
                self.cal_pretrain_relative_positions(fake_positions)
            self._finetune_i_bucket_main_stream = finetune_i_bucket_main_stream.to(real_positions.device)
            self._finetune_i_bucket_predicting_stream = finetune_i_bucket_predicting_stream.to(real_positions.device)
        finetune_i_bucket_main_stream = self._finetune_i_bucket_main_stream[:, :n_tokens, :n_tokens].repeat(batch_size,
                                                                                                            1, 1)
        finetune_i_bucket_predicting_stream = torch.cat([
            self._finetune_i_bucket_predicting_stream[:, :n_tokens, :n_tokens],
            self._finetune_i_bucket_predicting_stream[:, :n_tokens,
            self.max_target_positions:self.max_target_positions + n_tokens]
        ], 2).repeat(batch_size, 1, 1)
        return finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream

    def extract_features_AR(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        # embed positions
        # [bos, A, B, C, D, eos] with real positions [1,2,3,4,5,6](main stream), [2,3,4,5,6,7](predicting stream)
        # target [B,C,D] with prev [A,B,C] from [A,B,C,D] as pretraining span with real positions [2,3,4],
        # but target actually [3,4,5] for fine tune with another [bos].
        # thus [2,3,4] used for main stream shifted prev [A,B,C], [3,4,5] used for predicting [B,C,D]
        if 'positions' in unused:
            # pretrain procedure
            main_stream_pos_embed = self.embed_positions._forward(unused['positions'])
            real_positions = unused['positions']
            i_buckets_main_stream, i_bucket_relative_stream = \
                self.cal_pretrain_relative_positions(real_positions)
        else:
            # fine tune procedure
            main_stream_pos_embed, real_positions = self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions is not None else None
            if incremental_state is not None:
                i_buckets_main_stream, i_bucket_relative_stream = None, None
            else:
                i_buckets_main_stream, i_bucket_relative_stream = \
                    self.cal_finetune_relative_positions(real_positions)

        predicting_stream_pos_embed = self.embed_positions._forward(real_positions + 1)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if main_stream_pos_embed is not None:
                main_stream_pos_embed = main_stream_pos_embed[:, -1:]

        x = self.embed_tokens(prev_output_tokens)
        # embed tokens and positions
        if self.embed_scale is not None:
            x *= self.embed_scale

        if main_stream_pos_embed is not None:
            x += main_stream_pos_embed

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]
        if main_stream_pos_embed is None:
            print('positions should be used to predict ngrams')
            raise Exception()

        if self.embed_scale is not None:
            ngram_input_embed = self.embed_scale * self.ngram_input_embed.weight
        else:
            ngram_input_embed = self.ngram_input_embed.weight

        if incremental_state is not None:
            B = x.size(1)
            ngram_masks = [
                (ngram_input_embed[0] + predicting_stream_pos_embed).transpose(0, 1).repeat(1, B, 1)
                for ngram in range(self.ngram)]
        else:
            ngram_masks = [(ngram_input_embed[0] + predicting_stream_pos_embed).transpose(0, 1) for
                           ngram in range(self.ngram)]

        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None
        ngram_mask_matrix = self.buffered_future_mask_ngram(x) if incremental_state is None else None

        # TODO in train [(1+ngram)*T, B, C], in inference [T+ngram, B, C]
        x = torch.cat([x] + ngram_masks, 0)

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                ngram_mask_matrix=ngram_mask_matrix,
                i_buckets_main_stream=i_buckets_main_stream,
                i_bucket_relative_stream=i_bucket_relative_stream,
                real_positions=real_positions,
                flag_AR=True
            )
            inner_states.append(x)

        # TODO [(1+ngram)*T, B, C] -> [B, (1+ngram)*T, C]
        x_list = x.transpose(0, 1).chunk(1 + self.ngram, 1)
        if attn is not None:
            attn_list = attn.transpose(0, 1).chunk(1 + self.ngram, 1)
        else:
            attn_list = None

        return x_list, {'attn': attn_list}

    def extract_features_NAR(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        # embed positions
        # [bos, A, B, C, D, eos] with real positions [1,2,3,4,5,6](main stream), [2,3,4,5,6,7](predicting stream)
        # target [B,C,D] with prev [A,B,C] from [A,B,C,D] as pretraining span with real positions [2,3,4],
        # but target actually [3,4,5] for fine tune with another [bos].
        # thus [2,3,4] used for main stream shifted prev [A,B,C], [3,4,5] used for predicting [B,C,D]
        if 'positions' in unused:
            # pretrain procedure
            main_stream_pos_embed = self.embed_positions._forward(unused['positions'])
            real_positions = unused['positions']
            i_buckets_main_stream, i_bucket_relative_stream = \
                self.cal_pretrain_relative_positions(real_positions)
        else:
            # fine tune procedure
            main_stream_pos_embed, real_positions = self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions is not None else None
            if incremental_state is not None:
                i_buckets_main_stream, i_bucket_relative_stream = None, None
            else:
                i_buckets_main_stream, i_bucket_relative_stream = \
                    self.cal_finetune_relative_positions(real_positions)

        predicting_stream_pos_embed = self.embed_positions._forward(real_positions + 1)

        # embed tokens and positions
        if self.embed_scale is not None:
            x = (self.embed_scale * self.ngram_input_embed.weight[0] + predicting_stream_pos_embed).transpose(0, 1)
        else:
            x = (self.ngram_input_embed.weight[0] + predicting_stream_pos_embed).transpose(0, 1)

        attn = None
        inner_states = [x]

        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                ngram_mask_matrix=None,
                i_buckets_main_stream=i_buckets_main_stream,
                i_bucket_relative_stream=i_bucket_relative_stream,
                real_positions=real_positions,
                flag_AR=False
            )
            inner_states.append(x)

        # TODO [(1+ngram)*T, B, C] -> [B, (1+ngram)*T, C]
        if attn is not None:
            assert False
        else:
            attn_list = None

        return x.transpose(0, 1), {'attn': attn_list}

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out
        '''
        logits_list = net_output[0]
        if log_probs:
            return [utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in logits_list][0]
        else:
            return [utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in logits_list][0]
        '''
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(
            0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def buffered_future_mask_ngram(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_ngram_future_mask') or self._ngram_future_mask is None or self._ngram_future_mask.device != tensor.device:
            self._ngram_future_mask = ngram_attention_bias(self.max_target_positions, self.ngram).type(tensor.dtype).to(
                tensor.device)
        ngram_future_mask = torch.cat([self._ngram_future_mask[:, :dim, :dim],
                                       self._ngram_future_mask[:, :dim,
                                       self.max_target_positions: self.max_target_positions + dim]
                                       ], 2)
        return ngram_future_mask


@register_model_architecture('uie_light', 'uie_light_base')
def base_architecture(args):
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.load_sep = getattr(args, 'load_sep', False)

    args.first_stage_nar = getattr(args, '--first-stage-nar', True)
    args.second_stage_nar = getattr(args, '--second-stage-nar', True)
