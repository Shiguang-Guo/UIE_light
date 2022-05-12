"""
@author: Guo Shiguang
@software: PyCharm
@file: uie_light.py
@time: 2022/3/10 15:27
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerModel
from fairseq.modules import (
    MultiheadAttention,
    LayerNorm,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from UIE_light.learned_positional_embedding import LearnedPositionalEmbedding
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


class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            export: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = None  # math.sqrt(embed_dim)
        self.embed_positions = LearnedPositionalEmbedding(
            args.max_source_positions + 1 + self.padding_idx, embed_dim, self.padding_idx,
        )

        self.layers = nn.ModuleList([])

        self.layers.extend([
            TransformerEncoderLayer(
                args.encoder_embed_dim,
                args.encoder_ffn_embed_dim,
                args.encoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,
            )
            for i in range(args.encoder_layers)
        ])

        self.emb_layer_norm = LayerNorm(embed_dim)

        self.apply(init_bert_params)

    def forward(self, src_tokens, src_lengths, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(src_tokens)
        # embed tokens and positions
        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            pos_emb, real_positions = self.embed_positions(src_tokens)
            x += pos_emb

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if encoder_padding_mask is not None:
            x *= 1 - encoder_padding_mask.unsqueeze(-1).type_as(x)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            # x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, real_positions=real_positions)
            x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, )

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


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
            ("argument", event_only_tokens, include_arguments_tokens, self.second_stage_nar)]:
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
