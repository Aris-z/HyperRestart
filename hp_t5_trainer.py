from types import SimpleNamespace
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from data_util.LMdata import GLUEDataset
from optimizers.scheduler import CosineAnnealingWarmRestarts
import json, os, re
from collections import namedtuple
from transformers.optimization import Adafactor, get_constant_schedule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from hyperspace.HyperDecoder.hyperdecoder import Hyperdecoders
from hyperspace.HyperDecoder.adapter_t5 import T5ForConditionalGenerationWithAdapter
from data_util.preprocess import text_to_label, calculate_score_macro_avg
import matplotlib.pyplot as plt
from data_util.FlanDataset import SelectedDataset, SelectedIterableDataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import argparse, time
from tqdm import tqdm
from evaluate import load
import wandb
import time
import logging
logging.basicConfig(format='%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO)

logger = logging.getLogger(__name__)

from eval_metric.glue_metric import Glue, RougeMetric


class T5Trainer():
    def __init__(self, plm_path, config_path, init_restore_dir=None, seed=2000, model = "t5"):
        #self.task_list = ["sst2","stsb", "wnli"] # for debug use
        # self.task_list = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
        
        hp_config = json.load(open(config_path, "r"), object_hook=config_object_hook)
        
        if "t5" not in plm_path:
            raise ValueError("plm_path must be a T5 model")
        self.plm_path = plm_path
        self.hp_config = hp_config
        self.init_restore_dir = init_restore_dir
        self.seed = seed
        self.max_seq_len = hp_config.encoder_max_len
        self.batch_size = hp_config.batch_size
        self.decoder_max_seq_len = hp_config.decoder_max_len
        self.model = model
        self.plm = T5ForConditionalGeneration.from_pretrained(plm_path)
        self.tokenizer = T5Tokenizer.from_pretrained(plm_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model == 't5':
            self.model = self.plm
            # temp_pth = ""
            # self.model.load_state_dict(torch.load(temp_pth))
   
        elif self.model == 'hyperdecoder':
            
            self.model = Hyperdecoders(self.plm,
                                     hp_config=hp_config,
                                     tokenizer=self.tokenizer,
                                     seed=self.seed)
            
        else:
            raise ValueError("Only available t5 baseline/hyperprompt")
      
        if self.init_restore_dir:
            self.model.load_state_dict(torch.load(self.init_restore_dir))
        
        self.generation_arguments = {
            "max_length": self.decoder_max_seq_len,
            "num_beams": 5,
            "length_penalty": 1.0,
            "max_new_tokens": 10,
            "repetition_penalty": 1.0,
            "early_stopping": True,
            "use_cache": True,
            "do_sample": False,
            "top_k": 0,
            "top_p": 0.9,
            "bad_words_ids": [[self.tokenizer.pad_token_id]]
        }
    
    def __build_dataset(self, mode = "train", tokenizer: T5Tokenizer = None, dir_to_data = None):
        if "glue" in dir_to_data:
            dataset = GLUEDataset(mode=mode, tokenizer=tokenizer, encoder_max_len=self.max_seq_len, decoder_max_len=self.decoder_max_seq_len,seed=self.seed, dir_to_data=dir_to_data)
        else:
            if mode == "train":
                dataset = SelectedIterableDataset(mode=mode, tokenizer=tokenizer, encoder_max_len=self.max_seq_len, decoder_max_len=self.decoder_max_seq_len, dir_to_data=dir_to_data)
            elif mode == "dev":
                dataset = SelectedDataset(mode=mode, tokenizer=tokenizer, encoder_max_len=self.max_seq_len, decoder_max_len=self.decoder_max_seq_len, dir_to_data=dir_to_data)
        
        # assert len(dataset) > 0, "Dataset is empty"
        # assert list(dataset.task_to_keys.keys()) == self.task_list, "Task list is not the same"
        return dataset
    
    def collate_fn(self, data, mode='train'):
        
        def pack_batch(batch):
            to_stack_input_ids = []
            to_stack_attention_mask = []
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
        batch_textual_target = [each['textual_target'] for each in data]

        return pack_batch(batch_input), pack_batch(batch_target), batch_textual_target
    
    def __build_dataloader(self, dataset, batch_size, mode="train"):
        if mode == "train":
            return DataLoader(dataset, num_workers=0, collate_fn=self.collate_fn, shuffle=False)
        else:
            return DataLoader(dataset, num_workers=4, batch_size=batch_size, collate_fn=self.collate_fn, batch_sampler=None, shuffle=False)
        
    def train(self,dir_to_data, output_dir, accumulation_steps, batch_size, training_steps,
              learning_rate, save_checkpoint_per_steps, do_sgdr=True, sgdr_steps = 1, do_eval=True, eta_min=1e-6):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        n_gpu = torch.cuda.device_count()
        logger.info("Number of GPU: {}".format(n_gpu))
        logger.info("do_sgdr: {}".format(do_sgdr))
        logger.info("sgdr_steps: {}".format(sgdr_steps))

        logger.info("Load training dataset")
        train_dataset = self.__build_dataset(mode="train", tokenizer=self.tokenizer, dir_to_data=dir_to_data)
        train_dataloader = self.__build_dataloader(train_dataset, batch_size=batch_size, mode="train")
        num_batches = len(train_dataloader)

        logger.info("Load validation dataset")
        val_dataset = self.__build_dataset(mode="dev", tokenizer=self.tokenizer, dir_to_data=dir_to_data)
        val_dataloader = self.__build_dataloader(val_dataset, batch_size=batch_size, mode="dev")

        logger.info("loading glue evaluators")
        # self.glue_metrics = {each_task: load('glue', each_task, ) for each_task in self.task_list}
        self.metrics = RougeMetric()

        logger.info("Start training")
        self.model.to(self.device)
        self.model.train()

        if do_sgdr:
            # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=sgdr_steps, T_mult=2, eta_min=eta_min)
        else:
            optimizer = Adafactor(self.model.parameters(), lr=learning_rate, scale_parameter=False, relative_step=False)
            scheduler = get_constant_schedule(optimizer)
        global_step = 0
        loss_list = []
        for epoch in range(1, self.hp_config.num_epochs + 1):
            if global_step >= training_steps:
                break
            logger.info("Epoch: {}".format(epoch))
            for step, batch in tqdm(enumerate(train_dataloader), total = num_batches, desc="Epoch: "):
                if global_step % 300000 == 0:
                    torch.cuda.empty_cache()
                if global_step >= training_steps:
                    break
                batch_input, batch_target, batch_textual_target = batch
                batch_input['input_ids'] = batch_input['input_ids'].to(self.device)
                batch_input['attention_mask'] = batch_input['attention_mask'].to(self.device)
                batch_target['input_ids'] = batch_target['input_ids'].to(self.device)
                batch_target['attention_mask'] = batch_target['attention_mask'].to(self.device)
                
                labels = batch_target['input_ids']
                labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(**batch_input, labels=labels)
                # outputs = self.model(**batch_input, labels=labels, tasks=batch_task_name)
                # print(outputs)
                
                if n_gpu > 1:
                    loss = outputs.loss.mean()
                else:
                    loss = outputs.loss
                
                # loss = loss / accumulation_steps
                loss_value = loss.item()
                
                loss_list.append(loss_value)
                
                if do_eval:
                    if global_step % save_checkpoint_per_steps == 0:
                        logger.info("Start eval.")
                        rouge_score = self.eval(val_dataloader, output_dir, global_step, save_results=True)
                        logger.info("Rouge score: {}".format(rouge_score))
                        torch.save(self.model.state_dict(), os.path.join(output_dir, "checkpoint-step{}-loss:{}-rouge:{}.pt".format(global_step, loss_value ,rouge_score)))
                    # else:
                    #     continue
                        
                    elif global_step % (save_checkpoint_per_steps // 2)  == 0:
                        logger.info("Start eval.")
                        rouge_score = self.eval(val_dataloader, output_dir, global_step, save_results=True)
                        logger.info("Rouge score: {}".format(rouge_score))
                        wandb.log({"RougeL Fmeature": rouge_score})
                
                optimizer.zero_grad()        
                loss.backward()
                optimizer.step()
                
                if (global_step + 1) % accumulation_steps == 0:
                    loss_log = sum(loss_list)/len(loss_list)
                    loss_list = []
                    wandb.log({"loss":loss_log,
                        "learning_rate":optimizer.param_groups[0]['lr']
                        })
                    scheduler.step()
                global_step += 1
        # if not do_sgdr:
        #     np.save("no_warm_loss_list.npy", loss_list)
        # else:
        #     np.save("warm_loss_list.npy", loss_list)
        # draw_loss(loss_list, global_step, do_sgdr)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "final-step{}.pt".format(global_step)))
        logger.info('=========training done!=========')

                
    def eval(self, data_loader, output_dir, global_step, save_results=False):
            
        self.model.eval()
        total = len(data_loader)
    
        with torch.no_grad():
            results = {
                'predictions': [],
                'references': [],
                "textual_predictions": [],
                "textual_references": []
            }
            
            
            for step, batch in tqdm(enumerate(data_loader), total = total, desc="Eval"):
                batch_input, batch_target, batch_textual_target = batch
                batch_input['input_ids'] = batch_input['input_ids'].to(self.device)
                batch_input['attention_mask'] = batch_input['attention_mask'].to(self.device)
                batch_target['input_ids'] = batch_target['input_ids'].to(self.device)
                batch_target['attention_mask'] = batch_target['attention_mask'].to(self.device)


                # batch = {
                #         'input_ids': batch_input['input_ids'],
                #         'attention_mask': batch_input['attention_mask'],
                #         'tasks': batch_task_name,
                # }
                batch = {
                       'input_ids': batch_input['input_ids'],
                       'attention_mask': batch_input['attention_mask'],
                }
                    
                ground_truth = batch_textual_target
                
                output_sequences = self.model.generate(**batch)
                                        
                predictions = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                
                for i in range(len(predictions)):
                    
                    results['predictions'].append(predictions[i])
                    results['references'].append(ground_truth[i])
                    results['textual_predictions'].append(predictions[i])
                    results['textual_references'].append(ground_truth[i])
                        
        scores = self.metrics.compute(predictions=results['predictions'], references=results['references'])  
        score = calculate_score_macro_avg(scores)
            
        if save_results:
            with open(os.path.join(output_dir, "results_{}.json".format(global_step)), "w") as f:
                json.dump({
                                'rouge_score': score}, f, ensure_ascii=False, indent=4)
            with open(os.path.join(output_dir, "predictions_{}.json".format(global_step)), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
        return score
        
        
    def test(self, dir_to_data, output_dir, batch_size, tag='test'):
        logger.info('loading tesk data...')
        test_dataset = self.__build_dataset(mode='test', tokenizer=self.tokenizer, dir_to_data=dir_to_data)
        test_dataloader = self.__build_dataloader(test_dataset, batch_size=batch_size, mode='test')

        logger.info('loading glue evaluators...')
        self.metrics = RougeMetric()

        self.model.cuda()

        self.eval(test_dataloader, output_dir, global_step=tag, save_results=True)
        logger.info('=========test done!=========')
        
    def inference(self, text):
        self.model.eval()

        with torch.no_grad():
            
            feature = self.tokenizer(text, return_tensors='pt')
            input_ids = feature
            attention_mask = feature.attention_mask
            this_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            _, predictions = self.model.generate(**this_input, **self.generation_arguments)
        return predictions


def config_object_hook(config_dict):
     return SimpleNamespace(**config_dict)

def main(args):
    os.environ["WANDB_MODE"] = "offline"
    mode = args.mode
    plm_path = args.plm_path
    data_dir = args.data_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    eta_min = args.eta_min
    accumulation_steps = args.accumulation_steps
    training_steps = args.training_steps
    save_checkpoint_per_steps = args.save_checkpoint_per_steps
    do_eval = args.do_eval
    config_path = args.config_path
    seed = args.seed
    # device = args.device
    exp_tag = args.exp_tag
    init_restore_dir = args.init_restore_dir
    do_sgdr = args.do_sgdr
    sgdr_steps = args.sgdr_steps
    model = args.model
    sgdr = "sgdr" if do_sgdr else "no"
    # wandb log
    now = time.strftime("%m-%d",time.localtime(time.time()))
    wandb.init(
        project="hyper_restart",
        name = f"{model}_{eta_min}_{now}",
        notes = "P3",
        config=args
    )
    # wandb.init(
    #     project="hyper_restart",
    #     resume=True,
    #     config=args
    # )
    
    # hp_config = json.load(open(config_path, "r"), object_hook=config_object_hook)
    # timestamp = time.strftime('%H-%M-%Y-%m-%d')
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    if exp_tag:
        stamp = '{}-{}'.format(exp_tag, timestamp)
    else:
        stamp = timestamp
    output_dir = os.path.join(output_dir, stamp)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    trainer = T5Trainer(plm_path, config_path, init_restore_dir, seed, model)
    if mode == 'train':
        print(eta_min)
        trainer.train(data_dir, output_dir, accumulation_steps, batch_size, training_steps, learning_rate,
                    save_checkpoint_per_steps, do_sgdr=do_sgdr, sgdr_steps=sgdr_steps, do_eval=do_eval, eta_min=eta_min)
        
    else:
        raise NotImplementedError("Not implemented yet")

        
# def draw_loss(loss_list, step, warmup):
#     fig = plt.figure(figsize=(12,4))
#     plt.cla()
#     x1 = range(1, step+1)
#     y1 = loss_list
#     plt.title('Train loss', fontsize=20)
#     plt.plot(x1[1:step+1:int((step+1)/2000)], y1[1:step+1:int((step+1)/2000)])
#     plt.xlabel('steps', fontsize=20)
#     plt.ylabel('Train loss', fontsize=20)
#     plt.grid()
#     if warmup:
#         plt.savefig("./lossfig/Train_loss_t5_warmup.png", dpi=800)
#     else:
#         plt.savefig("./lossfig/Train_loss_t5_no_warmup.png", dpi=800)
#     plt.show()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="T5 training script")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--plm_path", type=str, default="./plm_models/flan-t5-base", help="pretrained language model path: t5-base or flan-t5-base")
    parser.add_argument("--data_dir", type=str, default="./datasets/P3", help="data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=10000, help="accumulation steps")
    parser.add_argument("--training_steps", type=int, default=310000, help="training steps")
    parser.add_argument("--save_checkpoint_per_steps", type=int, default=10000, help="save checkpoint per steps")
    parser.add_argument("--do_eval", type=bool, default=True, help="do evaluation")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--exp_tag", type=str, default=None, help="experiment tag")
    parser.add_argument("--init_restore_dir", type=str, default=None, help="initial restore directory")
    parser.add_argument("--sgdr_steps", type=int, default=1, help="sgdr steps")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="eta_min")
    #
    parser.add_argument("--do_sgdr", action="store_true", help="Run or not.")
    parser.add_argument("--model", type=str, default="t5", help="t5, hyperdecoder")
    parser.add_argument("--output_dir", type=str, default="./checkpoint/", help="output directory")
    parser.add_argument("--config_path", type=str, default="./configs/T5config.json", help="config path")
    
    args = parser.parse_args()
    main(args)
