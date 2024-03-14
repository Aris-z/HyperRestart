import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datasets
import json
import os
import logging
from pathlib import Path
import dataclasses
from typing import List, Tuple, Dict, Union, Optional, Any
from transformers import T5Config, T5ForConditionalGeneration

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    GenerationMixin
)
from transformers.trainer_utils import EvaluationStrategy

from hyperspace.HyperDecoder.adapter_t5 import (
    T5WithAdapterConfig,
    T5ForConditionalGenerationWithAdapter,
)

from configs.hdConfig.training_args import (
    Seq2SeqTrainingArguments,
    ModelArguments,
    DataTrainingArguments,
    AdapterTrainingArguments,
)
from utils import (
    get_last_checkpoint_path,
    freeze_model,
    unfreeze_adapter_params_encoder,
    unfreeze_adapter_params_decoder,
    unfreeze_encoder,
    unfreeze_decoder,
    unfreeze_layer_norms,
    check_output_dir,
)

logger = logging.getLogger(__name__)

config_path = "/cpfs/29cd2992fe666f2a/user/wangzekun/zhangyihan/hsgdr/configs/hdConfig/hdconfig.json"

def Hyperdecoders(plm,
                 hp_config,
                 all_tasks=None,
                 freeze_plm=False,
                 tokenizer=None,
                 using_encoder_past_key_values: Optional[bool] = True,
                 using_decoder_past_key_values: Optional[bool] = True,
                 device=None,
                 seed=None):
        # See all possible arguments in src/transformers/training_args.py or by passing
        # the --help flag to this script. We now keep distinct sets of args, for a cleaner
        # separation of concerns.
        parser = HfArgumentParser(
            (
                ModelArguments,
                DataTrainingArguments,
                Seq2SeqTrainingArguments,
                AdapterTrainingArguments,
            )
        )
        
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=config_path)
        check_output_dir(training_args)

        model_class = T5ForConditionalGenerationWithAdapter
        config_class = T5WithAdapterConfig
        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = config_class.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        config.update(dataclasses.asdict(adapter_args))
        # mrqa is a single 'task' with many sub-tasks
        if "mrqa" in all_tasks or "mrqa_reg" in all_tasks:
            all_tasks += [
                "HotpotQA",
                "NaturalQuestionsShort",
                "NewsQA",
                "SearchQA",
                "SQuAD",
                "TriviaQA-web",
            ]
        config.update({"tasks": all_tasks})

        if model_args.not_load_t5_checkpoint:
            model = model_class(config=config)
        else:
            last_checkpoint_path = training_args.output_dir
            model_path = (
                model_args.model_name_or_path
                if (
                    (
                        training_args.optimize_from_scratch
                        and not training_args.optimize_from_scratch_with_loading_model
                    )
                    or not os.path.exists(
                        os.path.join(last_checkpoint_path, "pytorch_model.bin")
                    )
                )
                else last_checkpoint_path
            )
            logger.warning("model path loaded from : %s", model_path)
            model = model_class.from_pretrained(
                model_path,
                from_tf=".ckpt" in model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
            )

        # set num_beams for evaluation
        if data_args.eval_beams is None:
            data_args.eval_beams = model.config.num_beams

        # freezing the parameters.
        if model_args.freeze_model:
            freeze_model(model)
        if model_args.unfreeze_encoder_adapters:
            unfreeze_adapter_params_encoder(model)
        if model_args.unfreeze_decoder_adapters:
            unfreeze_adapter_params_decoder(model)
        if model_args.unfreeze_encoder:
            unfreeze_encoder(model)
        if model_args.unfreeze_decoder:
            unfreeze_decoder(model)
        if model_args.unfreeze_layer_norms:
            unfreeze_layer_norms(model)

        if training_args.print_num_parameters:
            #for name, param in model.named_parameters():
            #    if param.requires_grad:
            #        logger.info("Parameter name %s", name)
            total_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            logger.info("Total trainable parameters %s", total_trainable_params)
            logger.info("Total parameters %s", total_params)
        return model

#    def forward(self, input_ids, attention_mask, labels: torch.Tensor = None, tasks: List[str] = None):
#        # outputs = self.model(**batch_input, labels=labels)
#        # find out t5_trainer.py Line 622: training_step
#        inputs = {"input_ids":input_ids, "attention_mask": attention_mask}
#        output = self.model(**inputs, labels=labels, output_hidden_states=True)
#        return output
