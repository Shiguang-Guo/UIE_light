"""
@author: Guo Shiguang
@software: PyCharm
@file: uie_light.py
@time: 2022/3/10 15:27
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerEncoder

from UIE_light.uie_light_decoder import UIE_Light_Decoder

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
        decoder = UIE_Light_Decoder(args, tgt_dict, embed_tokens)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    def forward(self, src_tokens=None, src_lengths=None, empty_tokens=None, event_only_tokens=None,
                include_arguments_tokens=None, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        results = {}
        for stagename, prev_output_tokens, tgt_tokens, NAR_flag in [
            ("event", empty_tokens, event_only_tokens, self.first_stage_nar),
            ("arguments", event_only_tokens, include_arguments_tokens, self.second_stage_nar)]:
            # for stagename, prev_output_tokens, tgt_tokens, NAR_flag in [
            #     ("event", empty_tokens, event_only_tokens, self.first_stage_nar)]:
            masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
                prev_output_tokens, tgt_tokens, self.pad, self.unk
            )
            mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
            mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

            mask_ins_out, _ = self.decoder.forward_mask_ins(
                prev_output_tokens, encoder_out=encoder_out, nar_flag=NAR_flag, stage=stagename,
            )

            word_ins_out, _ = self.decoder.forward_word_ins(
                masked_tgt_tokens, encoder_out=encoder_out, NAR_flag=NAR_flag, stage=stagename,
            )

            # make online prediction
            if self.decoder.sampling_for_deletion:
                word_predictions = torch.multinomial(
                    F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1).view(
                    word_ins_out.size(0), -1)
            else:
                word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

            word_predictions.masked_scatter_(
                ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
            )

            # generate training labels for deletion
            word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
            word_del_out, _ = self.decoder.forward_word_del(
                word_predictions, encoder_out=encoder_out, NAR_flag=NAR_flag, stage=stagename, )
            word_del_masks = word_predictions.ne(self.pad)

            results[stagename] = {
                "mask_ins": {
                    "out": mask_ins_out, "tgt": mask_ins_targets,
                    "mask": mask_ins_masks, "ls": 0.01,
                },
                "word_ins": {
                    "out": word_ins_out, "tgt": tgt_tokens,
                    "mask": masked_tgt_masks, "ls": self.args.label_smoothing,
                    "nll_loss": True
                },
                "word_del": {
                    "out": word_del_out, "tgt": word_del_targets,
                    "mask": word_del_masks
                }
            }
        return results


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
