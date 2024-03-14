from configs.config import Config
import warnings
import logging
import json
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Optional


logger = logging.getLogger(__name__)

@dataclass
class MetaAdapterTrainingArguments(TrainingArguments):
    """
    Contains different training parameters such as dropout, optimizers parameters, ... .
    """
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    
    do_sgdr: bool = field(default=False, metadata={"help": "Whether to use SGDR."})
    
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    
    temperature: Optional[int] = field(default=1, metadata={"help": "Defines the temperature"
                                                                    "value for sampling across the multiple datasets."})
    train_adapters: Optional[bool] = field(default=False, metadata={"help":
                                                                        "Train an adapter instead of the full model."})
    do_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to run eval on the dev set."})

    