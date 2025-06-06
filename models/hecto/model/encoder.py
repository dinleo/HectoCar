"""
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on huggingface code base
 * https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert
"""

import math
from typing import Optional, Tuple, Dict, Any, Mapping

import torch
from torch import Tensor, device, dtype, nn
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN

from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
)
from transformers.modeling_utils import (
    PreTrainedModel,
)
from transformers.utils import logging
from models.hecto.model.base_model import BertConfigW

logger = logging.get_logger(__name__)


class HectoEncoder(PreTrainedModel):
    config_class = BertConfigW
    base_model_prefix = "HectoEncoder"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        query_tokens,
        ms_context_tokens,
        query_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        :param query_tokens: query tokens
        :param ms_context_tokens: List of Key-value features used for the attention computation.
        :param query_mask: Query mask (B, Q), 1 = valid, 0 = padding, Default None in which case all positions are attended.
        :param output_attentions: Whether or not to return attention probabilities. Defaults to the value in the model configuration.
        :param output_hidden_states: Whether or not to return all hidden states. Defaults to the value in the model configuration.
        :param return_dict: Whether or not to return outputs as a dictionary. Defaults to the value in the model configuration.
        :return: If `return_dict` is False, returns the final sequence output tensor. If `return_dict` is True, returns a `BaseModelOutputWithCrossAttentions` object containing the last hidden state, hidden states (if `output_hidden_states` is True), attentions (if `output_attentions` is True), and cross attentions (if applicable).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        batch_size, query_num, query_size = query_tokens.size()

        # ----------- Self-Attention Mask -----------
        if query_mask is None:
            query_mask = torch.ones((batch_size, query_num), device=query_tokens.device)
        query_mask = query_mask.float()
        # Expand to [B, 1, 1, Q] for broadcasting in attention
        self_mask = self.get_extended_attention_mask(query_mask, (batch_size, query_num), query_tokens.device)

        # ----------- Cross-Attention Mask -----------
        cross_mask = None

        encoder_outputs = self.encoder(
            query_tokens=query_tokens,
            ms_context_tokens=ms_context_tokens,
            self_mask=self_mask,
            cross_mask=cross_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = encoder_outputs[0]
        sequence_output = sequence_output.masked_fill(query_mask.unsqueeze(-1) == 0, 0)

        if not return_dict:
            return sequence_output

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: Tuple[int],
        device: device
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_sequery_length, to_sequery_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, sequery_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, sequery_length, sequery_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        query_tokens,
        ms_context_tokens,
        self_mask,
        cross_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        scale = len(ms_context_tokens)
        scale_interval = self.config.num_hidden_layers // scale
        assert (self.config.num_hidden_layers  % scale) == 0

        all_self_attentions = () if output_attentions else None
        all_hidden_states = (query_tokens,) if output_hidden_states else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            scale_idx = scale - (i // scale_interval) - 1
            layer_outputs = layer_module(
                query_tokens=query_tokens,
                context_tokens=ms_context_tokens[scale_idx],
                self_mask=self_mask,
                cross_mask=cross_mask,
                output_attentions=output_attentions,
            )

            query_tokens = layer_outputs[0]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (query_tokens,)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if not return_dict:
            return tuple(
                v
                for v in [
                    query_tokens,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=query_tokens,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.sequery_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        if (
            self.config.add_cross_attention
            and layer_num % self.config.cross_attention_freq == 0
        ):
            self.crossattention = BertAttention(
                config, is_cross_attention=self.config.add_cross_attention
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.activation = BertActivation(config)
        self.drn = BertDRN(config)

    def forward(
        self,
        query_tokens,
        context_tokens,
        self_mask,
        cross_mask,
        output_attentions=False
    ):
        self_attention_outputs = self.attention(
            query_tokens=query_tokens,
            context_tokens=None,
            mask=self_mask,
            output_attentions=output_attentions
        )
        query_tokens = self_attention_outputs[0]
        self_attn_map = self_attention_outputs[1]

        if self.has_cross_attention:
            cross_attention_outputs = self.crossattention(
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                mask=cross_mask,
                output_attentions=output_attentions,
            )
            query_tokens = cross_attention_outputs[0]
            cross_attn_map = cross_attention_outputs[1]
        else:
            cross_attn_map = None

        intermediate_output = self.activation(query_tokens)
        layer_output = self.drn(intermediate_output, query_tokens)

        outputs = (layer_output, self_attn_map, cross_attn_map)

        return outputs


class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.drn = BertSelfDRN(config)

    def forward(
        self,
        query_tokens,
        context_tokens,
        mask,
        output_attentions=False,
    ):
        self_outputs = self.self(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            mask=mask,
            output_attentions=output_attentions,
        )
        attention_output = self.drn(self_outputs[0], query_tokens)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.query_size % config.num_attention_heads != 0 and not hasattr(
                config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.query_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.query_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.query_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.context_size, self.all_head_size)
            self.value = nn.Linear(config.context_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.query_size, self.all_head_size)
            self.value = nn.Linear(config.query_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
                self.position_embedding_type == "relative_key"
                or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        """
        (Batch, Token, Dim) -> (Batch, Head, Token, Dim/Head==64)
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_tokens,
        context_tokens,
        mask,
        output_attentions=False,
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = context_tokens is not None

        # (Batch, Head, Query, Dim/Head)
        query_layer = self.transpose_for_scores(self.query(query_tokens))
        if is_cross_attention:
            # (Batch, Head, Key, Dim/Head)
            key_layer = self.transpose_for_scores(self.key(context_tokens))
            value_layer = self.transpose_for_scores(self.value(context_tokens))
        else:
            key_layer = self.transpose_for_scores(self.key(query_tokens))
            value_layer = self.transpose_for_scores(self.value(query_tokens))

        # past_key_value = (key_layer, value_layer)
        # Compute raw attention scores: (B, H, Q, D) x (B, H, K, D).T → (B, H, Q, K)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            token_length = query_tokens.size()[1]
            position_ids_l = torch.arange(
                token_length, dtype=torch.long, device=query_tokens.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                token_length, dtype=torch.long, device=query_tokens.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )
        # (Batch, Head, Query, Key)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)
        # Compute raw attention scores: (B, H, Q, K) x (B, H, K, D) → (B, H, Q, D)
        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer, None)
        )

        return outputs


class BertSelfDRN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.query_size, config.query_size)
        self.LayerNorm = nn.LayerNorm(config.query_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertDRN(nn.Module):
    """Dropout Residual Norm"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.query_size)
        self.LayerNorm = nn.LayerNorm(config.query_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.query_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.query_size, config.query_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# class Aq(BasePreTrainedModel):
#
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
#
#     def __init__(self, config):
#         super().__init__(config)
#
#         self.bert = BertModel(config, add_pooling_layer=False)
#         # self.cls = BertOnlyMLMHead(config)
#
#         self.init_weights()
#
#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder
#
#     def set_output_embeddings(self, new_embeddings):
#         self.cls.predictions.decoder = new_embeddings
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         head_mask=None,
#         query_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         labels=None,
#         past_key_values=None,
#         use_cache=True,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         return_logits=False,
#         is_decoder=True,
#         reduction="mean",
#     ):
#         r"""
#         encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, query_size)`, `optional`):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
#             the model is configured as a decoder.
#         encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#             the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
#             ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
#             ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
#         past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
#             Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
#             If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
#             (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
#             instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
#         use_cache (:obj:`bool`, `optional`):
#             If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
#             decoding (see :obj:`past_key_values`).
#         Returns:
#         Example::
#             >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
#             >>> import torch
#             >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#             >>> config = BertConfig.from_pretrained("bert-base-cased")
#             >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
#             >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#             >>> outputs = model(**inputs)
#             >>> prediction_logits = outputs.logits
#         """
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )
#         if labels is not None:
#             use_cache = False
#         if past_key_values is not None:
#             query_embeds = None
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             query_embeds=query_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             is_decoder=is_decoder,
#         )
#
#         sequence_output = outputs[0]
#         if query_embeds is not None:
#             sequence_output = outputs[0][:, query_embeds.shape[1] :, :]
#
#         prediction_scores = self.cls(sequence_output)
#
#         if return_logits:
#             return prediction_scores[:, :-1, :].contiguous()
#
#         lm_loss = None
#         if labels is not None:
#             # we are doing next-token prediction; shift prediction scores and input ids by one
#             shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
#             labels = labels[:, 1:].contiguous()
#             loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
#             lm_loss = loss_fct(
#                 shifted_prediction_scores.view(-1, self.config.vocab_size),
#                 labels.view(-1),
#             )
#             if reduction == "none":
#                 lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)
#
#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return ((lm_loss,) + output) if lm_loss is not None else output
#
#         return CausalLMOutputWithCrossAttentions(
#             loss=lm_loss,
#             logits=prediction_scores,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             cross_attentions=outputs.cross_attentions,
#         )
#
#     def prepare_inputs_for_generation(
#         self, input_ids, query_embeds, past=None, attention_mask=None, **model_kwargs
#     ):
#         # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
#         if attention_mask is None:
#             attention_mask = input_ids.new_ones(input_ids.shape)
#         query_mask = input_ids.new_ones(query_embeds.shape[:-1])
#         attention_mask = torch.cat([query_mask, attention_mask], dim=-1)
#
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             input_ids = input_ids[:, -1:]
#
#         return {
#             "input_ids": input_ids,
#             "query_embeds": query_embeds,
#             "attention_mask": attention_mask,
#             "past_key_values": past,
#             "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
#             "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
#             "is_decoder": True,
#         }
#
#     def _reorder_cache(self, past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             reordered_past += (
#                 tuple(
#                     past_state.index_select(0, beam_idx) for past_state in layer_past
#                 ),
#             )
#         return reordered_past

# class BertEmbeddings(nn.Module):
#     """Construct the embeddings from word and position embeddings."""
#
#     def __init__(self, config):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(
#             config.vocab_size, config.query_size, padding_idx=config.pad_token_id
#         )
#         self.position_embeddings = nn.Embedding(
#             config.max_position_embeddings, config.query_size
#         )
#
#         # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
#         # any TensorFlow checkpoint file
#         self.LayerNorm = nn.LayerNorm(config.query_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#         # position_ids (1, len position emb) is contiguous in memory and exported when serialized
#         self.register_buffer(
#             "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
#         )
#         self.position_embedding_type = getattr(
#             config, "position_embedding_type", "absolute"
#         )
#
#         self.config = config
#
#     def forward(
#         self,
#         input_ids=None,
#         position_ids=None,
#         query_embeds=None,
#         past_key_values_length=0,
#     ):
#         if input_ids is not None:
#             sequery_length = input_ids.size()[1]
#         else:
#             sequery_length = 0
#
#         if position_ids is None:
#             position_ids = self.position_ids[
#                 :, past_key_values_length : sequery_length + past_key_values_length
#             ].clone()
#
#         if input_ids is not None:
#             embeddings = self.word_embeddings(input_ids)
#             if self.position_embedding_type == "absolute":
#                 position_embeddings = self.position_embeddings(position_ids)
#                 embeddings = embeddings + position_embeddings
#
#             if query_embeds is not None:
#                 embeddings = torch.cat((query_embeds, embeddings), dim=1)
#         else:
#             embeddings = query_embeds
#
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings

# class BertPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.query_size, config.query_size)
#         if isinstance(config.hidden_act, str):
#             self.transform_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.transform_act_fn = config.hidden_act
#         self.LayerNorm = nn.LayerNorm(config.query_size, eps=config.layer_norm_eps)
#
#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states
#
#
# class BertLMPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.transform = BertPredictionHeadTransform(config)
#
#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(config.query_size, config.vocab_size, bias=False)
#
#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))
#
#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias
#
#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states
#
#
# class BertOnlyMLMHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.predictions = BertLMPredictionHead(config)
#
#     def forward(self, sequence_output):
#         prediction_scores = self.predictions(sequence_output)
#         return prediction_scores

#
#
# class BertForMaskedLM(BertPreTrainedModel):
#
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
#
#     def __init__(self, config):
#         super().__init__(config)
#
#         self.bert = BertModel(config, add_pooling_layer=False)
#         self.cls = BertOnlyMLMHead(config)
#
#         self.init_weights()
#
#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder
#
#     def set_output_embeddings(self, new_embeddings):
#         self.cls.predictions.decoder = new_embeddings
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         head_mask=None,
#         query_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         return_logits=False,
#         is_decoder=False,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
#             config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
#             (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
#         """
#
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             query_embeds=query_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             is_decoder=is_decoder,
#         )
#
#         if query_embeds is not None:
#             sequence_output = outputs[0][:, query_embeds.shape[1] :, :]
#         prediction_scores = self.cls(sequence_output)
#
#         if return_logits:
#             return prediction_scores
#
#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()  # -100 index = padding token
#             masked_lm_loss = loss_fct(
#                 prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
#             )
#
#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return (
#                 ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
#             )
#
#         return MaskedLMOutput(
#             loss=masked_lm_loss,
#             logits=prediction_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
