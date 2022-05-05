"""
@author: Guo Shiguang
@software: PyCharm
@file: NgramTransformerDecoderARNARMixedLayer.py
@time: 2022/5/5 1:32
"""
from fairseq import utils
from fairseq.modules import (
    MultiheadAttention,
    LayerNorm,
)
from torch import nn as nn
from torch.nn import functional as F

from UIE_light.ngram_multihead_attention_AR_NAR_mixed import NgramMultiheadAttentionARNARMixed


class NgramTransformerDecoderARNARMixedLayer(nn.Module):
    def __init__(
            self,
            ngram=1,
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

    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.ngram_self_attn = NgramMultiheadAttentionARNARMixed(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            ngram=ngram
        )
        self.ngram = ngram
        assert ngram == 1, 'ngram only supports 1 now'

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.encoder_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            kdim=embedding_dim,
            vdim=embedding_dim,
            dropout=attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.need_attn = False

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_mask=None,
            incremental_state=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            ngram_mask_matrix=None,
            i_buckets_main_stream=None,
            i_bucket_relative_stream=None,
            real_positions=None,
            flag_AR=True
    ):
        # one main stream and ngram predicting streams
        residual = x

        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        x, attn = self.ngram_self_attn(
            query=x,
            key=x,
            value=x,
            incremental_state=incremental_state,
            need_weights=False,
            self_attn_mask=self_attn_mask,
            ngram_mask_matrix=ngram_mask_matrix,
            i_buckets_main_stream=i_buckets_main_stream,
            i_bucket_relative_stream=i_bucket_relative_stream,
            real_positions=real_positions,
            use_ar_attention=flag_AR
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        if prev_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=(not self.training and self.need_attn),
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn
