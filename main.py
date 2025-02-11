import loguru

from dotenv import load_dotenv
import os
from data.download_data import get_countdown, get_gsm8k
load_dotenv()
os.environ["HF_DATASETS_CACHE"]="data/hf_cache"

if __name__ == "__main__":
    loguru.logger.info("reasoning starting...")
    get_gsm8k()