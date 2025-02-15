import os
import sys
from typing import List
from datasets import load_dataset,Dataset
import loguru
from transformers import set_seed
from dataclasses import dataclass, field
import logging
import json
import transformers
from rl.config import GRPOConfig, GRPOScriptArguments
from rl.prompt import SYSTEM_PROMPT
from rl.reward import accuracy_reward, format_reward, get_cosine_scaled_reward, get_repetition_penalty_reward, len_reward, reasoning_steps_reward
from rl.utils.callbacks import get_callbacks
from rl.utils.wandb_logging import init_wandb_training
from transformers.trainer_utils import get_last_checkpoint
import pandas as pd
import datasets
import torch

from trl import GRPOTrainer,ModelConfig, TrlParser,get_peft_config
logger = logging.getLogger(__name__)

class TrainerPipline():
    def __init__(self,script_args, training_args, model_args):
        self.desc = "trainer pipline"
        self.script_args, self.training_args, self.model_args = script_args, training_args, model_args
        
    def get_reward_func(self) -> List:
        '''
        get reward fun
        '''
        # Get reward functions
        REWARD_FUNCS_REGISTRY = {
            "accuracy": accuracy_reward,
            "format": format_reward,
            "reasoning_steps": reasoning_steps_reward,
            "cosine": get_cosine_scaled_reward(
                min_value_wrong=script_args.cosine_min_value_wrong,
                max_value_wrong=script_args.cosine_max_value_wrong,
                min_value_correct=script_args.cosine_min_value_correct,
                max_value_correct=script_args.cosine_max_value_correct,
                max_len=script_args.cosine_max_len,
            ),
            "repetition_penalty": get_repetition_penalty_reward(
                ngram_size=script_args.repetition_n_grams,
                max_penalty=script_args.repetition_max_penalty,
            ),
            "length": len_reward,
        }
        reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
        return reward_funcs

    def local_data_to_datasets(self):
        _, file_extension = os.path.splitext(script_args.local_dataset_path)
        file_extension = file_extension.lower()
        data_list =[]
        if file_extension == ".json" or file_extension == ".jsonl":
            with open(script_args.local_dataset_path,"r",encoding="utf-8") as file:
                data_list = [data for data in json.loads(file.read())]
        elif file_extension == ".parquet":
            data_df = pd.read_parquet(script_args.local_dataset_path)
            data_list = [data.to_dict() for index,data in data_df.iterrows()]
        df = pd.DataFrame(data_list)
        dataset = Dataset.from_pandas(df)
        return dataset

    
    def load_or_prepare_datasets(self)->Dataset:
        '''
        本地加载和HF加载 local_path=script_args.local_dataset_path,
                                                                  script_args=script_args.random_sample
        '''
        try:
            if os.path.exists(script_args.local_dataset_path):
                dataset = self.local_data_to_datasets()
            else:
                logger.info(f"from huggingface datastes download {script_args.dataset_name}")
                dataset = load_dataset(script_args.dataset_name)
            # split the dataset into train and test
                # select a random subset of 50k samples
            if script_args.random_sample:
                dataset = dataset.shuffle(seed=42).select(range(script_args.random_sample))
            train_test_split = dataset.train_test_split(test_size=0.1)
            train_dataset = train_test_split["train"]
            test_dataset = train_test_split["test"]
            ##进行预处理
            train_dataset= self._prepare_dataset(dataset=train_dataset)
            test_dataset = self._prepare_dataset(dataset=test_dataset)
            return train_dataset,test_dataset
        except RuntimeError as e:
            logger.info(f"load datsets error:{e}")
 
     # Format into conversation不同复现策略这里有点区别 都是数学题目
    def make_conversation(self,example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }           

    def _prepare_dataset(self,dataset:Dataset):
        dataset = dataset.map(self.make_conversation)
        return dataset 

    def _set_logging_info(self):
        ###############
        # Setup logging
        ###############
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process a small summary
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f" distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Model parameters {self.model_args}")
        logger.info(f"Script parameters {self.script_args}")
        logger.info(f"Training parameters {self.training_args}")     
        
            
    def _set_args_paresr(self):
        parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
        script_args, training_args, model_args = parser.parse_args_and_config()
        return script_args, training_args, model_args
    
    def train(self):
        # Set seed for reproducibility
        set_seed(self.training_args.seed)
        self._set_logging_info()

        # Check for last checkpoint
        last_checkpoint = None
        if os.path.isdir(self.training_args.output_dir):
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
        if last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

        if "wandb" in self.training_args.report_to:
            init_wandb_training(self.training_args)
        ##
        train_datset,test_dataset = self.load_or_prepare_datasets()

        logger.info("*** Initializing model kwargs ***")
        torch_dtype = (
            self.model_args.torch_dtype if self.model_args.torch_dtype in ["auto", None] else getattr(torch, self.model_args.torch_dtype)
        )
        model_kwargs = dict(
            revision=self.model_args.model_revision,
            trust_remote_code=self.model_args.trust_remote_code,
            attn_implementation=self.model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if self.training_args.gradient_checkpointing else True,
        )
        self.training_args.model_init_kwargs = model_kwargs
        
        #############################
        # Initialize the GRPO trainer
        #############################
        trainer = GRPOTrainer(
            model=self.model_args.model_name_or_path,
            reward_funcs=self.get_reward_func(),
            args=self.training_args,
            train_dataset=train_datset,
            eval_dataset=test_dataset if self.training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(self.model_args),
            callbacks=get_callbacks(self.training_args, self.model_args),
        )
        ###############
        # Training loop
        ###############
        logger.info("*** Train ***")
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_datset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        ##################################
        # Save model and create model card
        ##################################
        logger.info("*** Save model ***")
        trainer.save_model(self.training_args.output_dir)
        logger.info(f"Model saved to {self.training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "dataset_name": self.script_args.dataset_name,
            "tags": ["open-r1"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(self.training_args.output_dir)

        ##########
        # Evaluate
        ##########
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(test_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)                 
    
    def train_run(self):
        self.train()

if __name__ == "__main__":
    loguru.logger.info("train reasoning starting...")
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()  
    pipline =TrainerPipline(script_args, training_args, model_args)
    pipline.train_run()
        

    