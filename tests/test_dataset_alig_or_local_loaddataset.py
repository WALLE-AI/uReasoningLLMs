


import loguru
from rl.config import GRPOScriptArguments
from rl.utils.data_utils import load_or_prepare_datasets


def test_local_load_dataset():
    script = GRPOScriptArguments(dataset_name="hello world")
    script.local_dataset_path = ["Bespoke-Stratos-17k","ReasonFlux_SFT_15k__medical_r1","OpenThoughts-114k"]
    script.local_dataset_path = ["X-R1-7500"]
    script.local_dataset_path = ["OpenR1-Math-220k"]
    script.random_sample=10
    train_dataset,test_dataset=load_or_prepare_datasets(script_args=script)
    loguru.logger.info(f"all datasets index {train_dataset[0]}")
    