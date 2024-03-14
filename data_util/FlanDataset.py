import json
import os
import random
import numpy as np
import itertools
import contextlib
import torch
from functools import partial
from transformers import T5Tokenizer

import math
import datasets
from datasets import load_dataset
from dataclasses import dataclass
import logging
from torch.utils.data import IterableDataset, DataLoader, Dataset
import ijson

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    id: int = None
    data: dict = None
    targets: list = None
    task_name: str = None # Optional
    
DATASETS_TO_KEYS = { 
    'sni': 'natural-instructions',
    'flan': 'flan2021_submix_original',
    'p3': 'P3',
}

# DATASETS_TO_KEYS = {
#     'flan2022': 'FLAN_2022'
# }

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state) 
        
class SelectedIterableDataset(IterableDataset):
    def __init__(self, mode='train', tokenizer = None, data_dict = None, dir_to_data = None, if_sampled = True, 
                 encoder_max_len=512, decoder_max_len=32):
        super().__init__()
        self.data_dict = data_dict
        self.dir_to_data = dir_to_data
        self.if_sampled = if_sampled
        
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        
        self.dir_to_data = dir_to_data
        if self.dir_to_data:
            self.path_to_data = os.path.join(self.dir_to_data, '{}_features.json'.format(mode))
        else:
            self.path_to_data = None
        
        # load data
        if not (self.path_to_data and os.path.exists(self.path_to_data)):
            train_datasets, val_datasets, test_datasets = self.load_dataset('p3')
            
            train_datasets = train_datasets.map(
                    partial(self.tokenize_input_function, sentence_key='inputs'),
                    desc="Running tokenizer on train dataset"
                ) # label not translated, sliced features
            featured_data = list(train_datasets)
            json.dump(featured_data, open(os.path.join(self.dir_to_data, 'train_features.json'), 'w'))
            #
            val_datasets = val_datasets.map(
                    partial(self.tokenize_input_function, sentence_key='inputs'),
                    desc="Running tokenizer on val dataset"
                ) # label not translated, sliced features
            featured_data = list(val_datasets)
            json.dump(featured_data, open(os.path.join(self.dir_to_data, 'dev_features.json'), 'w'))
            #
            test_datasets = test_datasets.map(
                    partial(self.tokenize_input_function, sentence_key='inputs'),
                    desc="Running tokenizer on test dataset"
                )  # label not translated, sliced features
            featured_data = list(test_datasets)
            json.dump(featured_data, open(os.path.join(self.dir_to_data, 'test_features.json'), 'w'))
            #

            print('Dataset is created, plz run this code again.')
            exit()

        print("------load data num------")
        with open(self.path_to_data, 'r') as f:
            self.num_examples = len(json.load(f))
        f.close()
        
    def load_dataset(self, dataset_key, **kwargs):
        assert dataset_key in DATASETS_TO_KEYS.keys(), "Task key not found"
        prefix = f"./datasets/"
        path = DATASETS_TO_KEYS[dataset_key]
        # d_train, d_val, d_test = load_dataset(self.dir_to_data if self.dir_to_data else prefix+path)
        d = load_dataset(self.dir_to_data if self.dir_to_data else prefix+path)
        if len(d.keys()) == 1: # flan2021 dataset or else
            d = d["train"]
            d_train, d_val = d.train_test_split(test_size=0.002).values()
            d_val, d_test = d_val.train_test_split(test_size=0.5).values()
        elif len(d.keys()) == 2:
            d_train, d_val = d.values()
            d_train, d_test = d_train.train_test_split(test_size=0.1).values()
        elif len(d.keys()) == 3:
            d_train, d_val, d_test = d.values()
        else:
            raise ValueError("unknown case!")

        # train_datasets, val_datasets, test_datasets = self.build_dataset(d)
        
        return d_train, d_val, d_test
    
    
    def tokenize_input_function(self, example, sentence_key):

        sample = '{}: {}'.format(sentence_key, example[sentence_key])

        result = self.tokenizer(sample, padding='max_length', truncation=True, max_length=self.encoder_max_len,
                                return_tensors="pt")
        # result['input_ids'] = torch.tensor(result['input_ids'].squeeze())
        # result['attention_mask'] = torch.tensor(result['attention_mask'].squeeze())
        result['input_ids'] = result['input_ids'].squeeze().clone().detach()
        result['attention_mask'] = result['attention_mask'].squeeze().clone().detach()

        return result

    def tokenize_target_function(self, examples):
        mixed_feature = self.tokenizer(
            examples,
            padding='max_length', truncation=True, max_length=self.decoder_max_len, return_tensors="pt")
        sliced_feature = [{
            "input_ids": mixed_feature['input_ids'][i],
            "attention_mask": mixed_feature['attention_mask'][i],
        } for i in range(len(mixed_feature['input_ids']))]
        return sliced_feature

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        # read featured_data
        featured_data = ijson.items(open(self.path_to_data, 'r'), 'item')
        for each_feature in featured_data:
            each_feature['input_ids'] = torch.tensor(each_feature['input_ids'])
            each_feature['attention_mask'] = torch.tensor(each_feature['attention_mask'])
            targets = self.tokenize_target_function([each_feature['targets']])
        
            yield {
                "text": each_feature,
                "target": targets[0],
                "textual_target": each_feature['targets'],
            }


class SelectedDataset(Dataset):
    def __init__(self, mode='train', tokenizer = None, data_dict = None, dir_to_data = None, if_sampled = True, 
                 encoder_max_len=512, decoder_max_len=32,):
        super().__init__()
        self.data_dict = data_dict
        self.dir_to_data = dir_to_data
        self.if_sampled = if_sampled
        
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        
        self.dir_to_data = dir_to_data
        if self.dir_to_data:
            self.path_to_data = os.path.join(self.dir_to_data, '{}_features.json'.format(mode))
        else:
            self.path_to_data = None
        
        # load data
        if not (self.path_to_data and os.path.exists(self.path_to_data)):
            raise ValueError("Need featured data, please run code below.")

        # read featured_data
        if self.path_to_data and os.path.exists(self.path_to_data):
            logger.info('Reading featured data')
            featured_data = json.load(open(self.path_to_data, 'r'))
            for each_feature in featured_data:
                each_feature['input_ids'] = torch.tensor(each_feature['input_ids'])
                each_feature['attention_mask'] = torch.tensor(each_feature['attention_mask'])
            self.data = featured_data

        else:
            raise ValueError("Unknown case!")
            
        logger.info('Running tokenizer for target text')
        self.targets = self.tokenize_target_function(
                [each['targets'] for each in self.data]) # sliced features
        self.num_examples = self.data.num_rows if isinstance(self.data, datasets.arrow_dataset.Dataset) else len(self.data)
        # assert sum([self.fs_opt, self.zs_opt, self.fs_noopt, self.zs_noopt]) <= 1.0, "Sum of mixing ratio should be less than 1.0"
        # self.sampled_ratio = {'fs_opt': self.fs_opt, 'zs_opt': self.zs_opt, 'fs_noopt': self.fs_noopt, 'zs_noopt': self.zs_noopt}

    def tokenize_target_function(self, examples):
        mixed_feature = self.tokenizer(
            examples,
            padding='max_length', truncation=True, max_length=self.decoder_max_len, return_tensors="pt")
        sliced_feature = [{
            "input_ids": mixed_feature['input_ids'][i],
            "attention_mask": mixed_feature['attention_mask'][i],
        } for i in range(len(mixed_feature['input_ids']))]
        return sliced_feature

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        """
        remember : idx is based on the flattened mixture of all glue tasks
        """
        if idx < self.num_examples:
            return {
                "text": self.data[idx],
                "target": self.targets[idx],
                "textual_target": self.data[idx]['targets'],
            }
        else:
            raise IndexError("Index out of range")


if __name__ == "__main__":
    
    ### Sampled Test ###
    tokenizer = T5Tokenizer.from_pretrained('/apdcephfs_cq10/share_1567347/arisyhzhang/Restart_HyperInstrucT/plm_models/flan-t5-base')
    dir_to_data = './datasets/P3'
    # load_dataset(dir_to_data)
    dataset = SelectedDataset(mode='train', tokenizer=tokenizer, dir_to_data=dir_to_data)
    print(len(dataset))
    