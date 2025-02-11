import loguru

from dotenv import load_dotenv
import os
from data.download_data import get_countdown, get_gsm8k
from tests.test_reward_fn import test_format_or_equation_func
load_dotenv()
os.environ["HF_DATASETS_CACHE"]="data/hf_cache"

if __name__ == "__main__":
    loguru.logger.info("reasoning starting...")
    test_format_or_equation_func()