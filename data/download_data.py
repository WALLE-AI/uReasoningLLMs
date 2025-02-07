import os
from datasets import load_dataset, Dataset
import loguru
os.environ["HF_DATASETS_CACHE"]="data/cache"

def get_gsm8k():
    data = load_dataset('openai/gsm8k', 'main')["train"]
    loguru.logger.info(f"data:{data[0]}")


