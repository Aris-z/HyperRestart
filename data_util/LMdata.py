# import haienv
# haienv.set_env("hsgdr")

from functools import partial
import os.path
import time

import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from datasets import load_dataset
from functools import partial
import datasets
print(os.getcwd())
from .preprocess import label_to_text
import json
from datasets import Features, Value
import logging

logger = logging.getLogger(__name__)

class GLUEDataset(Dataset):
    
    def __init__(self, mode='train', tokenizer: T5Tokenizer = None, encoder_max_len=512, decoder_max_len=32,
                 dir_to_data=None, seed=3407):
        """
        :param mode: train, dev, test
        """
        # self.task_to_keys = {
        #     "cola": ("sentence", None),  # [unacceptable, acceptable]
        #     "mnli": ("premise", "hypothesis"),  # [entailment, neutral, contradiction]
        #     "mrpc": ("sentence1", "sentence2"),  # [not equivalent, equivalent]
        #     "qnli": ("question", "sentence"),  # [entailment, not entailment]
        #     "qqp": ("question1", "question2"),  # [not duplicate, duplicate]
        #     "rte": ("sentence1", "sentence2"),  # ['entailment', 'not entailment']
        #     "sst2": ("sentence", None),  # ['negative', 'positive']
        #     "stsb": ("sentence1", "sentence2"),  # float to string
        #     # "wnli": ("sentence1", "sentence2"),  # [not entailment, entailment]
        # }
        
        self.task_to_keys = {
            "cola": ("sentence", None,"label"),  # [unacceptable, acceptable]
            "mnli": ("premise", "hypothesis","label"),  # [entailment, neutral, contradiction]
            "mrpc": ("sentence1", "sentence2","label"),  # [not equivalent, equivalent]
            "qnli": ("question", "sentence","label"),  # [entailment, not entailment]
            "qqp": ("question1", "question2","label"),  # [not duplicate, duplicate]
            "rte": ("sentence1", "sentence2","label"),  # ['entailment', 'not entailment']
            "sst2": ("sentence", None,"label"),  # ['negative', 'positive']
            "stsb": ("sentence1", "sentence2","label"),  # float to string
        #    "wnli": ("sentence1", "sentence2","label"),  # [not entailment, entailment]
        }
            

        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        self.dir_to_data = dir_to_data
        if self.dir_to_data:
            self.path_to_data = os.path.join(self.dir_to_data, 'glue_{}_features.json'.format(mode))
        else:
            self.path_to_data = None

        if not (self.path_to_data and os.path.exists(self.path_to_data)):
            if mode == 'train':
                self.data = {each_task: load_dataset('glue', each_task, cache_dir='Restart_HyperInstrucT/datasets/glue')['train'] for each_task in
                             self.task_to_keys.keys()}
            #     self.data1 = {each_task: load_dataset('csv', data_files=tasks_paths[each_task], delimiter='\t', skip_index=True,
            #                           cache_dir='/Users/jiayigeng/HyperSpace/datasets/glue')["train"]
            #   for each_task in self.task_to_keys.keys()}

                print(self.data)
                
            elif mode == 'dev':
                self.data = {each_task: load_dataset('glue', each_task, cache_dir='Restart_HyperInstrucT/hsgdr/datasets/glue')[
                    "validation_matched" if each_task == "mnli" else "validation"] for each_task in
                             self.task_to_keys.keys()}
                # self.data = {each_task: load_dataset('text', data_files=tasks_paths[each_task], cache_dir='/Users/jiayigeng/HyperSpace/datasets/glue')[
                #     "validation_matched" if each_task == "mnli" else "validation"] for each_task in self.task_to_keys.keys()}
            else:
                assert mode == 'test'
                self.data = {
                    each_task: load_dataset('glue', each_task, cache_dir='Restart_HyperInstrucT/datasets/glue')["test_matched" if each_task == "mnli" else "test"]
                    for each_task in self.task_to_keys.keys()}
                # self.data = {
                #     each_task: load_dataset('csv',data_files=tasks_paths[each_task], cache_dir='/Users/jiayigeng/HyperSpace/datasets/glue')[
                #         "test_matched" if each_task == "mnli" else "test"]for each_task in self.task_to_keys.keys()}

        if self.path_to_data and not os.path.exists(self.path_to_data):
            self.data = {
                each_task: self.data[each_task].map(
                    partial(self.tokenize_input_function, sentence1_key=self.task_to_keys[each_task][0],
                            sentence2_key=self.task_to_keys[each_task][1], task=each_task),
                    desc="Running tokenizer on dataset {}".format(each_task)
                ) for each_task in self.task_to_keys.keys()
            }  # label not translated, sliced features
            featured_data = {
                each_task: list(self.data[each_task])
                for each_task in self.task_to_keys.keys()
            }
            json.dump(featured_data, open(self.path_to_data, 'w'))

        elif self.path_to_data and os.path.exists(self.path_to_data):
            featured_data = json.load(open(self.path_to_data, 'r'))
            self.data = {}
            for each_task in self.task_to_keys.keys():
                list_features = featured_data[each_task]
                for each_feature in list_features:
                    each_feature['input_ids'] = torch.tensor(each_feature['input_ids'])
                    each_feature['attention_mask'] = torch.tensor(each_feature['attention_mask'])
                self.data[each_task] = list_features

        elif not self.path_to_data:
            self.data = {
                each_task: self.data[each_task].map(
                    partial(self.tokenize_input_function, sentence1_key=self.task_to_keys[each_task][0],
                            sentence2_key=self.task_to_keys[each_task][1], task=each_task),
                    desc="Running tokenizer on dataset {}".format(each_task)
                ) for each_task in self.task_to_keys.keys()
            }  # label not translated, sliced features
        else:
            raise ValueError("Unknown case!")

        logger.info('Running tokenizer for target text')
        self.targets = {
            each_task: self.tokenize_target_function(
                [label_to_text(each_task, each['label']) for each in self.data[each_task]])
            for each_task in self.task_to_keys.keys()
        }  # sliced features

        self.num_examples_per_task = {
            each_task: self.data[each_task].num_rows if isinstance(self.data[each_task],
                                                                   datasets.arrow_dataset.Dataset) else len(
                self.data[each_task])
            for each_task in self.task_to_keys.keys()
        }

    def tokenize_input_function(self, example, sentence1_key, sentence2_key, task):

        if sentence2_key is None:
            sample = '{} {}: {}'.format(task, sentence1_key, example[sentence1_key])
        else:
            sample = '{} {}: {} {}: {}'.format(task, sentence1_key, example[sentence1_key], sentence2_key, example[sentence2_key])

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
        return sum([self.num_examples_per_task[each_task] for each_task in self.task_to_keys.keys()])

    def __getitem__(self, idx):
        """
        remember : idx is based on the flattened mixture of all glue tasks
        """
        for each_task in self.task_to_keys.keys():
            if idx < self.num_examples_per_task[each_task]:
                return {
                    "text": self.data[each_task][idx],
                    "target": self.targets[each_task][idx],
                    "textual_target": label_to_text(each_task, self.data[each_task][idx]['label']),
                    "task_name": each_task
                }
            else:
                idx -= self.num_examples_per_task[each_task]
        raise IndexError("Index out of range")


class SuperGLUEDataset(Dataset):
    
    def __init__(self, mode='train', tokenizer: T5Tokenizer = None, encoder_max_len=512, decoder_max_len=32,
                 dir_to_data=None, seed=3407):
        """
        :param mode: train, dev, test
        """
        self.task_to_keys = {
            "boolq": ("question", "passage", "label"),  # [false, true]
            "cb": ("premise", "hypothesis", "label"),  # [entailment, contradiction, neutral]
            "copa": ("premise", "choice1", "choice2", "label"),  # [cause, effect]
            "multirc": ("paragraph", "question", "answer", "label"),  # [incorrect, correct]
            "record": ("passage", "query", "entities", "label"),  # [not entailment, entailment]
            "rte": ("premise", "hypothesis", "label"),  # [not entailment, entailment]
            "wic": ("sentence1", "sentence2", "word", "label"),  # [false, true]
            "wsc": ("text", "span1_text","span2_text", "label")  # [false, true]
        }
        
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        self.dir_to_data = dir_to_data
        if self.dir_to_data:
            self.path_to_data = os.path.join(self.dir_to_data, 'superglue_{}_features.json'.format(mode))
        else:
            self.path_to_data = None
            
            
        if not (self.path_to_data and os.path.exists(self.path_to_data)):
            if mode == 'train':
                self.data = {each_task: load_dataset('super_glue', each_task, cache_dir='./datasets/super_glue')['train'] for each_task in
                            self.task_to_keys.keys()}
                    
            elif mode == 'dev':
                self.data = {each_task: load_dataset('super_glue', each_task, cache_dir='./datasets/super_glue')[
                    "validation"] for each_task in self.task_to_keys.keys()}
                    
            else:
                assert mode == 'test'
                self.data = {
                    each_task: load_dataset('super_glue', each_task, cache_dir='./datasets/super_glue')["test"]
                    for each_task in self.task_to_keys.keys()}
                
        if self.path_to_data and not os.path.exists(self.path_to_data):
            self.data = {
                each_task: self.data[each_task].map(
                    partial(self.tokenize_input_function, sentence1_key=self.task_to_keys[each_task][0],
                            sentence2_key=self.task_to_keys[each_task][1], task=each_task),
                    desc="Running tokenizer on dataset {}".format(each_task)
                ) for each_task in self.task_to_keys.keys()
            }  # label not translated, sliced features
            featured_data = {
                each_task: list(self.data[each_task])
                for each_task in self.task_to_keys.keys()
            }
            json.dump(featured_data, open(self.path_to_data, 'w'))

        elif self.path_to_data and os.path.exists(self.path_to_data):
            featured_data = json.load(open(self.path_to_data, 'r'))
            self.data = {}
            for each_task in self.task_to_keys.keys():
                list_features = featured_data[each_task]
                for each_feature in list_features:
                    each_feature['input_ids'] = torch.tensor(each_feature['input_ids'])
                    each_feature['attention_mask'] = torch.tensor(each_feature['attention_mask'])
                self.data[each_task] = list_features

        elif not self.path_to_data:
            self.data = {
                each_task: self.data[each_task].map(
                    partial(self.tokenize_input_function, sentence1_key=self.task_to_keys[each_task][0],
                            sentence2_key=self.task_to_keys[each_task][1], task=each_task),
                    desc="Running tokenizer on dataset {}".format(each_task)
                ) for each_task in self.task_to_keys.keys()
            }  # label not translated, sliced features
        else:
            raise ValueError("Unknown case!")

        logger.info('Running tokenizer for target text')
        self.targets = {
            each_task: self.tokenize_target_function(
                [label_to_text(each_task, each['label']) for each in self.data[each_task]])
            for each_task in self.task_to_keys.keys()
        }  # sliced features

        self.num_examples_per_task = {
            each_task: self.data[each_task].num_rows if isinstance(self.data[each_task],
                                                                   datasets.arrow_dataset.Dataset) else len(
                self.data[each_task])
            for each_task in self.task_to_keys.keys()
        }
    def tokenize_input_function(self, example, sentence1_key, sentence2_key, task):
    
        if sentence2_key is None:
            sample = '{} {}: {}'.format(task, sentence1_key, example[sentence1_key])
        else:
            sample = '{} {}: {} {}: {}'.format(task, sentence1_key, example[sentence1_key], sentence2_key, example[sentence2_key])

        result = self.tokenizer(sample, padding='max_length', truncation=True, max_length=self.encoder_max_len,
                                return_tensors="pt")
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
        return sum([self.num_examples_per_task[each_task] for each_task in self.task_to_keys.keys()])
    
    def __getitem__(self, idx):
        """
        remember : idx is based on the flattened mixture of all superglue tasks
        """
        for each_task in self.task_to_keys.keys():
            if idx < self.num_examples_per_task[each_task]:
                return {
                    "text": self.data[each_task][idx],
                    "target": self.targets[each_task][idx],
                    "textual_target": label_to_text(each_task, self.data[each_task][idx]['label']),
                    "task_name": each_task
                }
            else:
                idx -= self.num_examples_per_task[each_task]
        raise IndexError("Index out of range")

        
        
if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    dataset = GLUEDataset(mode = 'train', tokenizer = tokenizer, dir_to_data = 'datasets/glue')
   
    # dataset = SuperGLUEDataset(mode='train', tokenizer=tokenizer, dir_to_data='/Users/jiayigeng/HyperSpace/data')
    print(len(dataset))
    
    first_element = dataset[0]
    print(first_element)

    print(first_element['text'])
    print(first_element['target'])
    print(first_element['textual_target'])
    print(first_element['task_name'])
