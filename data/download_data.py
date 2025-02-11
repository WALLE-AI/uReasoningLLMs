import os
from datasets import load_dataset, Dataset
import loguru

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k():
    data = load_dataset('openai/gsm8k', 'main')["train"]
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': "SYSTEM_PROMPT"},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    loguru.logger.info(f"data:{data[0]}")
    
def get_countdown():
    dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset = load_dataset(dataset_id, split="train")
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42).select(range(50000))
