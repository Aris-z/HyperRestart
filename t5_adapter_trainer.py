import haienv
haienv.set_env("hsgdr")
import collections
import math
import json
import tqdm
import numpy as np
import os
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel, logging
from transformers import Trainer
from transformers.models.fsmt.configuration_fsmt import FSMTConfig

from transformers.optimization import Adafactor, get_constant_schedule

from optimizers.scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim

from transformers.trainer_utils import TrainOutput, set_seed
from transformers.trainer_callback import TrainerState

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset

from data.adaSampler import MultiTaskBatchSampler
from data.LMdata import GLUEDataset
from data.preprocess import text_to_label, calculate_glue_score_macro_avg

from configs.config import Config
import warnings
import logging

from LLMs_B.adapter_configuration_t5 import T5ConfigAdapter
from LLMs_B.adapter_modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Tokenizer

logger = logging.getLogger(__name__)

with open("/Users/jiayigeng/hfai/hsgdr/configs/meta_adapter.json", "r") as f:
    config_dict = json.load(f) 
    
metaAda_config = Config(config_dict)   # MetaAdapterConfig  

def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)
        
        
def reset_config(model, config):
    """Reset model config."""
    model.config = config
    logger.info(f"config is reset for {model.__class__}")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

class T5HyperFormerTrainer(Trainer):
    def __init__(self, config=None, data_args=None, dataset_sizes=None, adapter_config=None,
                 multi_task_compute_metrics=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if config is None:
            assert isinstance(
                self.model, PreTrainedModel
            ), f"If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}"
            
            self.config = self._actual_model(self.model).config
        else:
            self.config = config
            
        self.data_args = data_args
        self.dataset_sizes = dataset_sizes
        self.adapter_config = adapter_config
        self.multi_task_compute_metrics = multi_task_compute_metrics
        
        if self.args.label_smoothing != 0 or (self.data_args is not None and self.data_args.ignore_pad_token_for_loss):
            assert self.config.pad_token_id is not None, (
            "Make sure that `config.pad_token_id` is correctly defined when ignoring `pad_token` for loss calculation "
            "or doing label smoothing."
        )

        if self.config.pad_token_id is None:
            self.config.pad_token_id = self.config.eos_token_id
            if self.config.pad_token_id is not None:
                logger.warn(
                    f"The `config.pad_token_id` was `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for padding.."
                )
            else:
                logger.warn(
                    "As neither `config.pad_token_id` nor `config.eos_token_id` are defined, no padding will be applied."
                )
        
        if self.args.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        else:
            # dynamically import label_smoothed_nll_loss
            self.loss_fn = label_smoothed_nll_loss
        
    def _get_train_sampler(self):
        if self.args.local_rank != -1:
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            num_replicas = 1
            rank = 0
        return MultiTaskBatchSampler(self.dataset_sizes, self.args.train_batch_size,
                                    self.args.temperature, rank=rank,
                                    num_replicas=num_replicas)

    def _compute_loss(self, model, inputs, labels):
        if self.args.label_smoothing == 0:
            if self.data_args is not None and self.data_args.ignore_pad_token_for_loss:
                # force training to ignore pad token
                logits = model(**inputs, use_cache=False)[0]
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                # compute usual loss via models
                loss, logits = model(**inputs, labels=labels, use_cache=False)[:2]
        else:
            # compute label smoothed loss
            logits = model(**inputs, use_cache=False)[0]
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, _ = self.loss_fn(lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id)
        return loss, logits
    
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        loss, _ = self._compute_loss(model, inputs, labels)
        return loss
    
    def __build_dataset(self, mode = "train", tokenizer= None, dir_to_data = None):
        dataset = GLUEDataset(mode=mode, tokenizer=tokenizer, encoder_max_len=self.max_seq_len, decoder_max_len=self.decoder_max_seq_len,seed=self.seed, dir_to_data=dir_to_data)
        
        assert len(dataset) > 0, "Dataset is empty"
        assert list(dataset.task_to_keys.keys()) == self.task_list, "Task list is not the same"
        return dataset
    
    def collate_fn(self, data, mode='train'):
        
        def pack_batch(batch):
            to_stack_input_ids = []
            to_stack_attention_mask = []
            # task_names = []
            for each in batch:
                to_stack_input_ids.append(
                    each['input_ids'] if type(each['input_ids']) is torch.Tensor else torch.tensor(each['input_ids']))
                to_stack_attention_mask.append(
                    each['attention_mask'] if type(each['attention_mask']) is torch.Tensor else torch.tensor(
                        each['attention_mask']))
            stacked_input_ids = torch.stack(to_stack_input_ids, dim=0)
            stacked_attention_mask = torch.stack(to_stack_attention_mask, dim=0)
            return {
                'input_ids': stacked_input_ids,
                'attention_mask': stacked_attention_mask
            }

        batch_input = [each['text'] for each in data]
        batch_target = [each['target'] for each in data]
        batch_task_name = [each['task_name'] for each in data]
        batch_textual_target = [each['textual_target'] for each in data]
        return pack_batch(batch_input), pack_batch(batch_target), batch_textual_target, batch_task_name
    
    def train(self, dir_to_data, output_dir,batch_size, training_steps,
              learning_rate, accumulation_steps, save_checkpoint_per_steps, sgdr_steps = 1,
              do_sgdr = True, do_eval=True, trial = None):
        # Use trial object to suggest hyperparameters
        if trial is not None:
            # assuming `trial` is an instance of optuna.trial.Trial
            learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            
        # Keeping track whether we can can len() on the dataset or not
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        n_gpu = torch.cuda.device_count()
        logger.info("Number of GPU: {}".format(n_gpu))
        logger.info("do_sgdr: {}".format(do_sgdr))
        logger.info("sgdr_steps: {}".format(sgdr_steps))
        logger.info("do_eval: {}".format(do_eval))
        
        logger.info("Load training dataset")
        train_dataset = self.__build_dataset(mode="train", tokenizer=self.tokenizer, dir_to_data=dir_to_data)
        train_dataloader = self.__build_dataloader(train_dataset, batch_size=batch_size, mode="train")
        num_batches = len(train_dataloader)
        # train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        
        logger.info("Load validation dataset")
        val_dataset = self.__build_dataset(mode="dev", tokenizer=self.tokenizer, dir_to_data=dir_to_data)
        val_dataloader = self.__build_dataloader(val_dataset, batch_size=batch_size, mode="dev")
        
        logger.info("Start training")
        self.model.to(self.device)
        self.model.train()
        if do_sgdr:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=sgdr_steps, T_mult=2, eta_min=1e-6)
        else:
            optimizer = Adafactor(self.model.parameters(), lr=learning_rate, scale_parameter=False, relative_step=False)
            scheduler = get_constant_schedule(optimizer)
            
        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
        
        global_step = 0
        for epoch in range(1, metaAda_config.num_epochs + 1):
            if global_step >= training_steps:
                break
            logger.info("Epoch: {}".format(epoch))
            for step, batch in tqdm(enumerate(train_dataloader), total = num_batches,
                                    desc="Epoch: {}".format(epoch)):
                if global_step >= training_steps:
                    break
                batch_input, batch_target, batch_textual_target, batch_task_name = batch
                batch_input['input_ids'] = batch_input['input_ids'].to(self.device)
                batch_input['attention_mask'] = batch_input['attention_mask'].to(self.device)
                batch_target['input_ids'] = batch_target['input_ids'].to(self.device)
                batch_target['attention_mask'] = batch_target['attention_mask'].to(self.device)
                
                labels = batch_target['input_ids']
                labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(**batch_input, labels=labels)
                # print(outputs)
                
                if n_gpu > 1:
                    loss = outputs.loss.mean()
                else:
                    loss = outputs.loss
                
                loss = loss / accumulation_steps
                loss_value = loss.item()
                
                if do_eval:
                    if global_step % save_checkpoint_per_steps == 0:
                        _, glue_score = self.eval(val_dataloader, output_dir, global_step, save_results=True, device=self.device)
                        logger.info("Glue score: {}".format(glue_score))
                        torch.save(self.model.state_dict(), os.path.join(output_dir, "checkpoint-step{}-loss:{}-glue:{}.pt".format(global_step, loss_value ,glue_score)))
                    # else:
                    #     continue
                        
                    elif global_step % (save_checkpoint_per_steps // 2)  == 0:
                        _, glue_score = self.eval(val_dataloader, output_dir, global_step, save_results=True, device=self.device)
                        logger.info("Glue score: {}".format(glue_score))
                        
                loss.backward()
                
                if (global_step + 1) % accumulation_steps == 0 or (global_step + 1) % num_batches == 0:
                
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                global_step += 1
                
        logger.info("Finish training")
        return TrainOutput(global_step, loss_value)

                
    def eval(self, data_loader, output_dir, global_step, save_results=False, device=None):
            
        self.model.eval()
        total = len(data_loader)
    
        with torch.no_grad():
            results = {
                each_task: {
                'predictions': [],
                'references': [],
                "textual_predictions": [],
                "textual_references": []
            } for each_task in self.task_list}
            
            for batch in tqdm(data_loader, total=total, desc="Evaluating"):
                batch_input, batch_target, batch_textual_target, batch_task_name = batch
                batch_input['input_ids'] = batch_input['input_ids'].to(self.device)
                batch_input['attention_mask'] = batch_input['attention_mask'].to(self.device)
                batch_target['input_ids'] = batch_target['input_ids'].to(self.device)
                batch_target['attention_mask'] = batch_target['attention_mask'].to(self.device)

                batch = {
                        'input_ids': batch_input['input_ids'],
                        'attention_mask': batch_input['attention_mask'],
                }
                    
                ground_truth = batch_textual_target
                
                output_sequences = self.model.generate(**batch)
                                        
                predictions = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                
                
                for i in range(len(predictions)):
                    
                    results[batch_task_name[i]]['predictions'].append(text_to_label(batch_task_name[i], predictions[i],mode = 'prediction', ground_truth = ground_truth[i]))
                    results[batch_task_name[i]]['references'].append(text_to_label(batch_task_name[i], ground_truth[i]))
                    results[batch_task_name[i]]['textual_predictions'].append(predictions[i])
                    results[batch_task_name[i]]['textual_references'].append(ground_truth[i])
                        
        scores_per_task = {}
        for each_task in results:
            scores_per_task[each_task] = self.glue_metrics[each_task].compute(predictions=results[each_task]['predictions'],
                                                                                references=results[each_task]['references'])
                
        one_score_per_task, glue_score = calculate_glue_score_macro_avg(scores_per_task)
            
        if save_results:
            with open(os.path.join(output_dir, "results_{}.json".format(global_step)), "w") as f:
                json.dump({
                               'glue_score': glue_score,
                               'scores_per_task': scores_per_task, 
                                 'one_score_per_task': one_score_per_task}, f, ensure_ascii=False, indent=4)
            with open(os.path.join(output_dir, "predictions_{}.json".format(global_step)), "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
        return one_score_per_task, glue_score
    
    def inference(self, text, task):
        self.model.eval()
        inputs = self.tokenizer.encode_plus(f"{task} {text}", return_tensors='pt', padding='max_length', truncation=True, max_length=self.config.max_length)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
            
        gen_kwargs = {
            "max_length": self.config.max_length,
            "num_beams": self.config.num_beams
        }
        gen_kwargs["task"] = task
        gen_kwargs["task_embedding"] = self.model.task_embedding_controller(task) if \
            (self.config.train_adapters and isinstance(self.config, metaAda_config)) else None
        
        with torch.no_grad():
            output_sequences = self.model.generate(input_ids=input_ids, 
                                                attention_mask=attention_mask,
                                                **gen_kwargs)

        if output_sequences.shape[-1] < gen_kwargs["max_length"]:
            output_sequences = self._pad_tensors_to_max_len(output_sequences, gen_kwargs["max_length"])

        return output_sequences

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id`"
                f" is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


def main(args):
    # TODO: add args, add logger
    # TODO: Get the adapter config and updates specific parameters
    # TODO: Setup the model and tokenizer
    # TODO: Define Trainer eval and save the results
    
    
    pass
if __name__ == '__main__':
    pass