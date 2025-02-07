import loguru

from dotenv import load_dotenv

from data.download_data import get_gsm8k
load_dotenv()

if __name__ == "__main__":
    loguru.logger.info("reasoning starting...")
    get_gsm8k()