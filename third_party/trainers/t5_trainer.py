# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""Implements a T5 trainer class doing training and evaluation.
modified from hyperformer and huggingface codebases.
"""

import collections
import math
import json

import numpy as np
import os
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import PreTrainedModel, logging
from transformers import Trainer
from transformers import FSMTConfig
from transformers.file_utils import is_torch_tpu_available, WEIGHTS_NAME
from transformers.integrations import hp_params
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.deepspeed import deepspeed_init
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import (
    TrainOutput,
    PredictionOutput,
    set_seed,
    denumpify_detensorize,
    EvalPrediction,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    nested_concat,
)

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from typing import Any, Dict, Optional, Tuple, Union, List
from torch.utils.data.dataset import Dataset

from utils import use_task_specific_params, reset_config
from data import MultiTaskBatchSampler
from collections import defaultdict

logger = logging.get_logger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    "cosine_annual_w_restarts": CosineAnnealingWarmRestarts,
}

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class T5Trainer(Trainer):
    def __init__(
        self,
        config=None,
        data_args=None,
        dataset_sizes=None,
        adapter_config=None,
        multi_task_compute_metrics=None,
        compute_gen_probs=False,
        answer_output_file="predicted_answers.json",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if config is None:
            assert isinstance(
                self.model, PreTrainedModel
            ), f"If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}"
            self.config = self._actual_model(self.model).config
        else:
            self.config = config

        self.adapter_config = adapter_config
        self.multi_task_compute_metrics = multi_task_compute_metrics
        self.dataset_sizes = dataset_sizes
        self.data_args = data_args
        self.compute_gen_probs = compute_gen_probs
        self.answer_output_file = answer_output_file
        self.vocab_size = (
            self.config.tgt_vocab_size
            if isinstance(self.config, FSMTConfig)
            else self.config.vocab_size
        )

        if self.args.label_smoothing != 0 or (
            self.data_args is not None and self.data_args.ignore_pad_token_for_loss
        ):
            assert (
                self.config.pad_token_id is not None
            ), "Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss calculation or doing label smoothing."

        if self.config.pad_token_id is None and self.config.eos_token_id is not None:
            logger.warn(
                f"The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for padding.."
            )

        if self.args.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(
                ignore_index=self.config.pad_token_id,
                reduction="None" if self.args.loss_scaling else "mean",
            )
        else:
            # dynamically import label_smoothed_nll_loss
            from third_party.utils import label_smoothed_nll_loss

            self.loss_fn = label_smoothed_nll_loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use
        something else, you can pass a tuple in the Trainer's init through
        :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.adafactor:
                self.optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                )

            else:
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    eps=self.args.adam_epsilon,
                )

        if self.lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler(num_training_steps)
        else:  # ignoring --lr_scheduler
            logger.warn(
                "scheduler is passed to `Seq2SeqTrainer`, `--lr_scheduler` arg is ignored."
            )

    def _get_lr_scheduler(self, num_training_steps):
        schedule_func = arg_to_scheduler[self.args.lr_scheduler]
        if self.args.lr_scheduler == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.args.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.args.warmup_steps
            )
        elif self.args.lr_scheduler == "cosine_annual_w_restarts":
            scheduler = schedule_func(
                self.optimizer, T_0=self.args.warmup_steps, T_mult=2, eta_min=1e-6
            )
        else:
            scheduler = schedule_func(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        return scheduler

    # def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
    #     if is_torch_tpu_available() and xm.xrt_world_size() > 1:
    #         num_replicas = xm.xrt_world_size()
    #         rank = xm.get_ordinal()
    #     elif self.args.local_rank != -1:
    #         num_replicas = torch.distributed.get_world_size()
    #         rank = torch.distributed.get_rank()
    #     else:
    #         num_replicas = 1
    #         rank = 0
    #     return MultiTaskBatchSampler(
    #         self.dataset_sizes,
    #         self.args.train_batch_size,
    #         self.args.temperature,
    #         rank=rank,
    #         num_replicas=num_replicas,
    #     )

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
            loss, _ = self.loss_fn(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.config.pad_token_id,
                reduce=False if self.args.loss_scaling else True,
            )
        return loss, logits

    # def get_train_dataloader(self) -> DataLoader:
    #     """
    #     Returns the training :class:`~torch.utils.data.DataLoader`.

    #     Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
    #     to distributed training if necessary) otherwise.

    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
    #     multitask_sampler = self._get_train_sampler()
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_sampler=multitask_sampler,
    #         collate_fn=self.data_collator,
    #     )

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        loss, _ = self._compute_loss(model, inputs, labels)
        return loss

    def evaluate(
        self, eval_datasets: Optional[Dict[str, Dataset]] = None, ignore_keys=None
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        results = {}
        if eval_datasets is None:
            eval_datasets = self.eval_dataset

        for eval_task, eval_dataset in eval_datasets.items():
            self.compute_metrics = self.multi_task_compute_metrics[eval_task]
            model_config = self.model.config.to_dict()

            use_task_specific_params(self.model, eval_task)

            if eval_dataset is not None and not isinstance(
                eval_dataset, collections.abc.Sized
            ):
                raise ValueError("eval_dataset must implement __len__")

            eval_dataloader = self.get_eval_dataloader(eval_dataset)

            output, gen_probs = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
            )
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

            if (
                eval_task == "squad" or "mrqa" in eval_task
            ):  # TODO: replace with list of 'squad eval tasks'
                answer_results = defaultdict(list)
                # we may have multiple answers for each q due to chunking (TODO: also report probs)
                for qid, prob, prediction in zip(
                    eval_dataset["id"], gen_probs, output.predictions
                ):
                    answer_results[qid].append(
                        (
                            self.tokenizer.decode(prediction, skip_special_tokens=True),
                            prob.tolist(),
                        )
                    )
                with open(
                    os.path.join(self.args.output_dir, str(self.state.global_step) + self.answer_output_file ), "w"
                ) as f:
                    json.dump(answer_results, f, indent=4)

            tasks_metric = {eval_task + "_" + k: v for k, v in output.metrics.items()}
            for key in sorted(tasks_metric.keys()):
                logger.info(f"  {key} = {tasks_metric[key]}")
            results.update(tasks_metric)
            reset_config(self.model, model_config)

        # Computes the average metrics across all the tasks without their corresponding losses.
        # For some cases, we want to only consider one metric subset, e.g. rouge2 instead of rouge1+2+L etc.
        # otherwise, rouge will be overrepresented in avg vs tasks with a single metric value (e.g. any glue task)
        metrics = [
            results[key]
            for key in results.keys()
            if "loss" not in key and key not in self.data_args.ignore_metric_keys
        ]
        results["eval_average_metrics"] = np.mean(metrics)
        logger.info(f'Average results ---> {str(results["eval_average_metrics"])} <---')
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, results
        )
        return results

    def train(
        self,
        model_path: Optional[str] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
    ):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)

            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = (
                len(train_dataloader) // self.args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = (
                    self.args.max_steps // num_update_steps_per_epoch
                    + int(self.args.max_steps % num_update_steps_per_epoch > 0)
                )
            else:
                max_steps = math.ceil(
                    self.args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        # Mixed precision training with apex (torch < 1.6)
        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, self.optimizer = amp.initialize(
                model, self.optimizer, opt_level=self.args.fp16_opt_level
            )

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
        # find_unused_parameters breaks checkpointing as per
        # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (
                    torch.distributed.get_world_size()
                    if self.args.local_rank != -1
                    else 1
                )
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        self._total_loss_scalar = 0.0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.args.per_device_train_batch_size,
        )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            total_train_batch_size,
        )
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", max_steps)

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(
            os.path.join(model_path, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(model_path, "trainer_state.json")
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.state.global_step % (
                num_update_steps_per_epoch
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info(
                "  Continuing training from global step %d", self.state.global_step
            )
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = (
            self.hp_name(trial) if self.hp_name is not None else None
        )
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(self.args.device)
        self._logging_loss_scalar = 0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flostraining_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and (
                isinstance(train_dataloader.sampler, DistributedSampler)
                or isinstance(train_dataloader.batch_sampler, MultiTaskBatchSampler)
                or isinstance(train_dataloader.batch_sampler, MultiTaskBatchSampler)
            ):
                if isinstance(train_dataloader.sampler, MultiTaskBatchSampler):
                    train_dataloader.sampler.set_epoch(epoch)
                else:
                    train_dataloader.batch_sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(
                    train_dataloader, [self.args.device]
                ).per_device_loader(self.args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                self.args, self.state, self.control
            )

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        self.args, self.state, self.control
                    )

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and _use_ddp_no_sync
                ):
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.args.max_grad_norm
                        )
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), self.args.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.args.max_grad_norm
                        )

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        self.args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, None)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(
                self.args, self.state, self.control
            )
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, None)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )

        if (
            self.args.load_best_model_at_end
            and self.state.best_model_checkpoint is not None
        ):
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(model, PreTrainedModel):
                self.model = model.from_pretrained(self.state.best_model_checkpoint)
                self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(
                    os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
                )
                self.model.load_state_dict(state_dict)

        if self._total_flos is not None:
            self.store_flos()
            self.log({"total_flos": self.state.total_flos})

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

        return TrainOutput(
            self.state.global_step, tr_loss.item() / self.state.global_step, None
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys=None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model.
                Most models expect the targets under the argument :obj:`labels`.
                Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self.model.config.max_length,
            "num_beams": self.model.config.num_beams,
        }
        gen_kwargs["tasks"] = inputs["tasks"]
        gen_probs = None
        if self.args.predict_with_generate and not self.args.prediction_loss_only:
            generated_output = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict_in_generate=True,
                output_scores=True,
                min_length=2,
                **gen_kwargs,
            )
            generated_tokens = generated_output.sequences
            # calculate sequence probabilities
            gen_sequences = generated_output.sequences[:, 1:]  # skip the bos token
            probs = torch.stack(generated_output.scores, dim=1)
            if self.compute_gen_probs:
                gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(
                    -1
                )
                gen_probs = gen_probs.masked_fill(
                    gen_sequences == self.tokenizer.pad_token_id, 0
                )
                # in case the batch is shorter than max length, the output should be padded
                # nll loss
                gen_probs = (
                    (
                        (gen_sequences != 0).int()
                        * torch.nn.CrossEntropyLoss(reduction="none")(
                            input=probs.permute(0, 2, 1), target=gen_sequences
                        )
                    )
                    .nan_to_num()
                    .mean(dim=1)
                )

            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_kwargs["max_length"]
                )

        labels = inputs.pop("labels")
        with torch.no_grad():
            # compute loss on predict data
            loss, logits = self._compute_loss(model, inputs, labels)

        loss = loss.mean().detach()
        if self.args.prediction_loss_only:
            return (loss, None, None, None)

        logits = generated_tokens if self.args.predict_with_generate else logits

        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, logits, labels, gen_probs)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        probs_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, SequentialDistributedSampler
            ):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )
            probs_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )
            labels_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]
            ).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels, gen_probs = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if logits is not None:
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            if gen_probs is not None:
                probs_host = (
                    gen_probs
                    if probs_host is None
                    else nested_concat(probs_host, gen_probs, padding_index=-100)
                )
            if labels is not None:
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                eval_losses_gatherer.add_arrays(
                    self._gather_and_numpify(losses_host, "eval_losses")
                )
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(
                        self._gather_and_numpify(preds_host, "eval_preds")
                    )
                    probs_gatherer.add_arrays(
                        self._gather_and_numpify(probs_host, "gen_probs")
                    )
                    labels_gatherer.add_arrays(
                        self._gather_and_numpify(labels_host, "eval_label_ids")
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, probs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(
            self._gather_and_numpify(losses_host, "eval_losses")
        )
        if not prediction_loss_only:
            preds_gatherer.add_arrays(
                self._gather_and_numpify(preds_host, "eval_preds")
            )
            probs_gatherer.add_arrays(self._gather_and_numpify(probs_host, "gen_probs"))
            labels_gatherer.add_arrays(
                self._gather_and_numpify(labels_host, "eval_label_ids")
            )

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        gen_probs = probs_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids)
            )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return (
            PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics),
            gen_probs,
        )

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.config.pad_token_id
            if self.config.pad_token_id is not None
            else self.config.eos_token_id
        )

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
