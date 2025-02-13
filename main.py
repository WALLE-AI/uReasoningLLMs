import loguru

from dotenv import load_dotenv
import os
from data.download_data import get_countdown, get_gsm8k
from rl.models.llama3_8b_GRPO import train
from tests.test_reward_fn import test_format_or_equation_func
from tests.test_trainer_pipline import test_tranier_pipline
load_dotenv()
os.environ["HF_DATASETS_CACHE"]="data/hf_cache"

if __name__ == "__main__":
    loguru.logger.info("reasoning starting...")
    test_tranier_pipline()