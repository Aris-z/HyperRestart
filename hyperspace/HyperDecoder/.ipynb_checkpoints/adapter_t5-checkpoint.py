# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified modeling_t5 code.
import copy
import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.t5.configuration_t5 import T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerFF,
    T5Stack,
    T5ForConditionalGeneration,
    __HEAD_MASK_WARNING_MSG,
)

from hyperspace.HyperDecoder.adapter_generators import ParameterGenerator
from hyperspace.HyperDecoder.adapter_layer import AdapterLayer, TaskSpecificAdapterLayer


class T5WithAdapterConfig(T5Config):
    def __init__(
        self,
        encoder_adapter_dim=64,
        decoder_adapter_dim=64,
        hypernetwork_bottleneck=128,
        encoder_adapter="task",
        decoder_adapter="task",
        tasks=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_adapter_dim = encoder_adapter_dim
        self.decoder_adapter_dim = decoder_adapter_dim
        self.hypernetwork_bottleneck = hypernetwork_bottleneck
        self.encoder_adapter = encoder_adapter
        self.decoder_adapter = decoder_adapter
        self.tasks = tasks


class T5LayerFFWithAdapter(T5LayerFF):
    def __init__(self, config, is_encoder=False):
        super().__init__(config)
        self.config = config
        if (is_encoder and config.encoder_adapter == "manual_specific") or (
            not is_encoder and config.decoder_adapter == "manual_specific"
        ):
            self.adapter_layer = TaskSpecificAdapterLayer(config, is_encoder=is_encoder)
        else:
            self.adapter_layer = AdapterLayer(config, is_encoder=is_encoder)

    def forward(self, hidden_states):
        normed_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(normed_states)
        adapter_input = (
            normed_states if self.config.adapter_norm_input else hidden_states
        )
        hidden_states = (
            hidden_states
            + self.dropout(forwarded_states)
            + self.adapter_layer(adapter_input)
        )
        return hidden_states


class T5BlockWithAdapter(T5Block):
    def __init__(self, config, has_relative_attention_bias=False, is_encoder=False):
        super().__init__(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer[-1] = T5LayerFFWithAdapter(config, is_encoder=is_encoder)


def mean_pooling(hidden_state, attention_mask):
    input_masked = hidden_state * attention_mask.unsqueeze(-1)
    return input_masked.sum(1) / attention_mask.sum(1).unsqueeze(-1)


class T5StackWithAdapter(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        blockClass = T5Block
        if (self.is_decoder and self.config.decoder_adapter != "none") or (
            (not self.is_decoder) and self.config.encoder_adapter != "none"
        ):
            blockClass = T5BlockWithAdapter
            kwargs = {"is_encoder": not self.is_decoder}
        else:
            kwargs = {}
        self.block = torch.nn.ModuleList(
            [
                blockClass(config, has_relative_attention_bias=bool(i == 0), **kwargs)
                for i in range(config.num_layers)
            ]
        )
        if (self.is_decoder and self.config.decoder_adapter == "generated") or (
            (not self.is_decoder) and self.config.encoder_adapter == "generated"
        ):
            self.param_gen = ParameterGenerator(
                config, config.hidden_size, is_encoder=not self.is_decoder
            )
            if self.config.process_encoder_output:
                self.mlp = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.ReLU(),
                    nn.Linear(config.d_model, config.d_model),
                )
            else:
                # no-op to make the forward function less of an if-maze
                self.mlp = lambda x: x
        elif (self.is_decoder and self.config.decoder_adapter == "task") or (
            (not self.is_decoder) and self.config.encoder_adapter == "task"
        ):
            self.param_gen = ParameterGenerator(
                config, config.hidden_size, is_encoder=not self.is_decoder
            )
            self.adapter_task_embedding = nn.Embedding(
                len(self.config.tasks), self.config.d_model
            )

    def forward(
        self,
        input_ids=None,
        encoder_hidden_states=None,
        tasks=None,
        **kwargs,
    ):
        # using input ids to determine whats going
        self.clear_adapters()
        if self.is_decoder and self.config.decoder_adapter == "generated":
            self.apply_params_to_adapters(
                encoder_hidden_states.size(0),
                self.param_gen(
                    self.mlp(
                        mean_pooling(
                            encoder_hidden_states, kwargs["encoder_attention_mask"]
                        )
                    )
                ),
            )
        elif (not self.is_decoder) and self.config.encoder_adapter == "generated":
            # for encoder generation, we first pass through the encoder, then set encoder adapters based on this.
            # currently using learnt adapters in the first pass, but potentially we could turn those off too?
            res = super().forward(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            self.apply_params_to_adapters(
                input_ids.size(0),
                self.param_gen(
                    self.mlp(
                        mean_pooling(res.last_hidden_state, kwargs["attention_mask"])
                    )
                ),
            )
        elif (self.is_decoder and self.config.decoder_adapter == "task") or (
            not self.is_decoder and self.config.encoder_adapter == "task"
        ):
            # at test time, we only test one task at a time.
            if not self.training:
                # simple sanity check
                if len(tasks) > 0:
                    assert(tasks[0] == tasks[1] and tasks[1] == tasks[-1])
                tasks = [tasks[0] for _ in range(input_ids.size(0))]
            indices = torch.tensor(
                [self.config.tasks.index(task) for task in tasks],
                device=input_ids.device,
                dtype=torch.long,
            )
            task_embed = self.adapter_task_embedding(indices)
            self.apply_params_to_adapters(input_ids.size(0), self.param_gen(task_embed))
        elif (self.is_decoder and self.config.decoder_adapter == "manual_specific") or (
            not self.is_decoder and self.config.encoder_adapter == "manual_specific"
        ):
            indices = torch.tensor(
                [self.config.tasks.index(task) for task in tasks],
                device=input_ids.device,
                dtype=torch.long,
            )
            self.apply_indices_to_adapters(indices)
        return super().forward(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, **kwargs
        )

    def clear_adapters(self):
        for block in self.block:
            for layer in block.layer:
                if isinstance(layer, T5LayerFFWithAdapter):
                    layer.adapter_layer.clear_adapter()

    def apply_params_to_adapters(self, batch_size, generated_params):
        for param, block in zip(generated_params, self.block):
            block.layer[-1].adapter_layer.apply_adapter_params(batch_size, *param)

    def apply_indices_to_adapters(self, indices):
        for block in self.block:
            block.layer[-1].adapter_layer.set_indices(indices)


class T5ForConditionalGenerationWithAdapter(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackWithAdapter(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5StackWithAdapter(decoder_config, self.shared)

        self.init_weights()

    # required to pass tasks through
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "tasks": kwargs["tasks"],
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        tasks=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tasks=tasks,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            tasks=tasks,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
