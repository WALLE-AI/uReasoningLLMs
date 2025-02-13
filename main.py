import loguru

from dotenv import load_dotenv
import os
from tests.test_trainer_pipline import test_tranier_pipline, test_tranier_pipline_cmd
load_dotenv()
os.environ["HF_DATASETS_CACHE"]="data/hf_cache"

if __name__ == "__main__":
    loguru.logger.info("reasoning starting...")
    test_tranier_pipline_cmd()