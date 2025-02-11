from data.download_data import extract_hash_answer
from rl.reward import correctness_reward_func, int_reward_func, soft_format_reward_func, strict_format_reward_func, xmlcount_reward_func
from unsloth import FastLanguageModel, PatchFastRL
from vllm import SamplingParams

from unsloth import is_bfloat16_supported
import torch
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
import loguru
max_seq_length = 512 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

model_path = "meta-llama/meta-Llama-3.1-8B-Instruct"

def load_model_or_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )
    return model,tokenizer
# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    import pandas as pd
    data_df = pd.read_parquet("data/gsm8k/train-00000-of-00001.parquet")
    data_list = [data.to_dict() for index,data in data_df.iterrows()]
    datasets_list = []
    for data in data_list:
        data_dict = {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': data['question']}
            ],
            'answer': extract_hash_answer(data['answer'])
            
        }
        datasets_list.append(data_dict)
    df = pd.DataFrame(datasets_list)
    dataset = Dataset.from_pandas(df)
    # data = data_df.map(lambda x: { # type: ignore
    #     'prompt': [
    #         {'role': 'system', 'content': SYSTEM_PROMPT},
    #         {'role': 'user', 'content': x['question']}
    #     ],
    #     'answer': extract_hash_answer(x['answer'])
    # }) # type: ignore
    return dataset # type: ignore
##补充一下模型训练部分的研究内容，大概写几个点就行。股份公司课题。后面还有具体的技术路线、研究内容展开及国内外发展现状调研的东西要写。但你得把核心的这部分研究内容先给我。技术路线部分，我明天再想办法弄。
dataset = get_gsm8k_questions()

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)
def train():
    model,tokenizer = load_model_or_tokenizer()

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()
    model.save_lora("grpo_saved_lora")
    
def inference():
    model,tokenizer = load_model_or_tokenizer()
    text = tokenizer.apply_chat_template([
        {"role" : "user", "content" : "Calculate pi."},
    ], tokenize = False, add_generation_prompt = True)
    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text
    print(output)
    
if __name__ == "__main__":
    loguru.logger.info("reasoning starting...")
    train()

    