## Restart HyperRestart
Official repository for the paper: Enhancing Language Model Hypernetworks with Restart: A Study on Optimization
### Experiment Setup (Section: 4)
#### Setup
For the dataset, we use <https://huggingface.co/datasets/bigscience/P3#source-data>. All the tasks we use can be found in `./P3_tasks.txt`. We put these datas in `./datasets`
We use flan-t5 model as backbone of Hyperdecoders and the baseline in this section. This model can be downloaded from `https://huggingface.co/google/flan-t5-base`. It is recommended to put these models in `./plm_models/`.

#### Run
For experiment in section 4, the commands we use are in the file `run.sh`. You can adjust `--eta_min` in our commands to replicate our experiments for different eta_min values.

### Experiment Setup (Section: 5, 7)
We refer to the code of Hyperdecoders <https://github.com/allenai/hyperdecoders>. The configs we use in section 5 and 7 are in `./hyperdecoder_configs`.
#### Setup
Before you run this code, you need to add restart scheduler in the code. Specifically, you need to add the following code in `hyperdecoders/hyperdecoder/third_party/trainers/t5_trainer.py`:
Line 97: `"cosine_annual_w_restarts": CosineAnnealingWarmRestarts,`
Line 220: in `def _get_lr_scheduler` 
```python
        elif self.args.lr_scheduler == "cosine_annual_w_restarts":
            scheduler = schedule_func(
                self.optimizer, T_0=1, T_mult=2, eta_min=1e-6
            )
```

### Experiment Setup (Section: 6)
We refer to the code of Hyperformers <https://github.com/rabeehk/hyperformer>. The configs we use in section 6 are in `./hyperformer_configs`
#### Setup
Similar to Hyperdecoders above, you need to add the same code in `hyperformer/hyperformer/third_party/trainers/t5_trainer.py`:
Line 71: `"cosine_annual_w_restarts": CosineAnnealingWarmRestarts,`
Line 160: in `def _get_lr_scheduler` 
```python
        elif self.args.lr_scheduler == "consine_annealing_restart":
            scheduler = schedule_func(
                self.optimizer, T_0=1, T_mult=2, eta_min=1e-6
            )
```
