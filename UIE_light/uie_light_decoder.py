"""
@author: Guo Shiguang
@software: PyCharm
@file: uie_light_decoder.py
@time: 2022/4/28 0:13
"""
import math

import torch
from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import Embedding
from fairseq.modules import LayerNorm
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from torch import nn as nn
from torch.nn import functional as F

from UIE_light.NgramTransformerDecoderARNARMixedLayer import NgramTransformerDecoderARNARMixedLayer
from UIE_light.learned_positional_embedding import LearnedPositionalEmbedding
from UIE_light.ngram_multihead_attention_AR_NAR_mixed import ngram_attention_bias


class UIE_Light_Decoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)

        self.output_embed_dim = args.decoder_embed_dim
        self.embed_mask_ins = Embedding(128, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)
        self.early_exit = [int(i) for i in args.early_exit.split(',')]

        self.layers_msk = None
        self.layers_del = None

        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)

        self.register_buffer('version', torch.Tensor([3]))
        self.ngram = 1
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

        self.share_stage_layers = args.share_stage_layers

        if not self.share_stage_layers:
            self.second_stage_layers = nn.ModuleList([])
            self.second_stage_layers.extend([
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
        # main stream
        main_stream_relative_positions = real_positions.unsqueeze(1)
        # [B,T,T/S]
        main_stream_relative_positions = main_stream_relative_positions.repeat(1, real_positions.size(-1), 1)
        # [B,T,1]
        real_positions_main = real_positions.unsqueeze(-1)
        main_stream_relative_positions = main_stream_relative_positions - real_positions_main

        # predicting stream
        # input shift
        real_positions_shift_predicting_stream = real_positions - 1
        # [B,1, 2*T]
        predicting_stream_relative_positions = torch.cat((real_positions_shift_predicting_stream, real_positions),
                                                         dim=-1).unsqueeze(1)
        # [B,T, 2*T]
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

    def extract_features_NAR(
            self, prev_output_tokens, encoder_out=None, early_exit=None, layers=None, incremental_state=None,
            stage='event', **unused
    ):
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

        x = self.embed_tokens(prev_output_tokens)

        # embed tokens and positions
        if self.embed_scale is not None:
            x += (self.embed_scale * self.ngram_input_embed.weight[0] + predicting_stream_pos_embed)
        else:
            x += (self.ngram_input_embed.weight[0] + predicting_stream_pos_embed)

        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layers = self.layers
        if stage == 'argument' and not self.share_stage_layers:
            layers = self.second_stage_layers

        # decoder layers
        for layer in layers[:early_exit]:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
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

    def extract_features_AR(
            self, prev_output_tokens, encoder_out=None, early_exit=None, layers=None, incremental_state=None,
            stage='event', **unused
    ):
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

        layers = self.layers
        if stage == 'argument' and not self.share_stage_layers:
            layers = self.second_stage_layers

        # decoder layers
        for layer in layers[:early_exit]:
            x, attn = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
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

        return x_list[-1], {'attn': attn_list}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def forward_mask_ins(self, prev_output_tokens, encoder_out=None, NAR_Flag=True, stage='event', **unused):
        if NAR_Flag:
            features, extra = self.extract_features_NAR(prev_output_tokens, encoder_out=encoder_out,
                                                        early_exit=self.early_exit[1], layers=self.layers_msk,
                                                        stage=stage,
                                                        incremental_state=None, **unused)
        else:
            features, extra = self.extract_features_AR(prev_output_tokens, encoder_out=encoder_out,
                                                       early_exit=self.early_exit[1], layers=self.layers_msk,
                                                       stage=stage,
                                                       incremental_state=None, **unused)
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        mask_ins_out = F.linear(features_cat, self.embed_mask_ins.weight)
        return mask_ins_out, extra['attn']

    def forward_word_ins(self, prev_output_tokens, encoder_out=None, NAR_Flag=True, stage='event', **unused):
        if NAR_Flag:
            features, extra = self.extract_features_NAR(prev_output_tokens, encoder_out=encoder_out,
                                                        early_exit=self.early_exit[2], layers=self.layers, stage=stage,
                                                        incremental_state=None, **unused)
        else:
            features, extra = self.extract_features_AR(prev_output_tokens, encoder_out=encoder_out,
                                                       early_exit=self.early_exit[2], layers=self.layers, stage=stage,
                                                       incremental_state=None, **unused)
        return self.output_layer(features), extra['attn']

    def forward_word_del(self, prev_output_tokens, encoder_out=None, NAR_Flag=True, stage='event', **unused):
        if NAR_Flag:
            features, extra = self.extract_features_NAR(prev_output_tokens, encoder_out=encoder_out,
                                                        early_exit=self.early_exit[0], layers=self.layers_del,
                                                        stage=stage,
                                                        incremental_state=None, **unused)
        else:
            features, extra = self.extract_features_AR(prev_output_tokens, encoder_out=encoder_out,
                                                       early_exit=self.early_exit[0], layers=self.layers_del,
                                                       stage=stage,
                                                       incremental_state=None, **unused)

        return F.linear(features, self.embed_word_del.weight), extra['attn']
