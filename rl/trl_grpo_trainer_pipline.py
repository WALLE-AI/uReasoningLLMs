import os
import sys
from typing import List
import loguru
from transformers import set_seed
import logging
import transformers
from rl.backend.trl_vlm_vllm_grpo import Qwen2VLGRPOVLLMTrainer
from rl.config import GRPOConfig, GRPOScriptArguments
from rl.reward import (
    accuracy_reward,
    code_reward,
    format_reward,
    format_reward_func,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    int_reward_func,
    len_reward,
    reasoning_steps_reward,
    reflection_reward_ratio,
    simple_length_reward,
    soft_format_reward_func,
    strict_format_reward_func,
    tag_count_reward,
    xmlcount_reward_func,
)
from rl.utils.callbacks import get_callbacks
from rl.utils.data_utils import dataset_format_alignment, load_or_prepare_datasets, local_data_to_datasets
from rl.utils.model_utils import get_tokenizer
from rl.utils.wandb_logging import init_wandb_training
from transformers.trainer_utils import get_last_checkpoint
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
            "len_reward": len_reward,
            "soft_format_reward_func":soft_format_reward_func,
            "strict_format_reward_func":strict_format_reward_func,
            "int_reward_func":int_reward_func,
            "xmlcount_reward_func":xmlcount_reward_func,
            "format_reward_func":format_reward_func,
            "reflection_reward_ratio":reflection_reward_ratio,
            "simple_length_reward":simple_length_reward,
            "code_reward":code_reward,
            "get_code_format_reward":get_code_format_reward,
            "tag_count_reward":tag_count_reward
            
        }
        reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
        return reward_funcs

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
        
        ## Load datasets
        train_dataset,test_dataset = load_or_prepare_datasets(script_args=self.script_args)
        loguru.logger.info(f"dataset size:{len(train_dataset)}")
        loguru.logger.info(f"dataset example:{train_dataset[0]}")
        
        # Load tokenizer
        ################
        tokenizer = get_tokenizer(self.model_args, self.training_args)
        tokenizer.pad_token = tokenizer.eos_token

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
        match self.training_args.model_type:
            case "llm":
                trainer = GRPOTrainer(
                    model=self.model_args.model_name_or_path,
                    reward_funcs=self.get_reward_func(),
                    args=self.training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset if self.training_args.eval_strategy != "no" else None,
                    peft_config=get_peft_config(self.model_args),
                    callbacks=get_callbacks(self.training_args, self.model_args),
                    processing_class=tokenizer,
                )
            case "vlm":
                ##多模态训练
                trainer = Qwen2VLGRPOVLLMTrainer(
                    model=self.model_args.model_name_or_path,
                    reward_funcs=self.get_reward_vlm_func(),
                    args=self.training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset if self.training_args.eval_strategy != "no" else None,
                    peft_config=get_peft_config(model_args),
                    attn_implementation=self.model_args.attn_implementation,
                    max_pixels=self.script_args.max_pixels,
                    min_pixels=self.script_args.min_pixels,
                )
        # trainer = GRPOTrainer(
        #     model=self.model_args.model_name_or_path,
        #     reward_funcs=self.get_reward_func(),
        #     args=self.training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=test_dataset if self.training_args.eval_strategy != "no" else None,
        #     peft_config=get_peft_config(self.model_args),
        #     callbacks=get_callbacks(self.training_args, self.model_args),
        #     processing_class=tokenizer,
        # )
        ###############
        # Training loop
        ###############
        logger.info("*** Train ***")
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
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
        

    
