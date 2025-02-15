import json
import os
from datasets import load_dataset,Dataset
import loguru
import pandas as pd


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def test_dataset():
    import pandas as pd
    data_df = pd.read_parquet("data/xr1/test-00000-of-00001.parquet")
    data_list = [data.to_dict() for index,data in data_df.iterrows()]
    df = pd.DataFrame(data_list)
    dataset = Dataset.from_pandas(df)
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    loguru.logger.info(f"datasets index {dataset[0]}")


def local_data_to_datasets():
    test_data_path="data/xr1/test-00000-of-00001.parquet"
    test_data_path ="data/medical/medical_o1_sft_Chinese.json"
    _, file_extension = os.path.splitext(test_data_path)
    file_extension = file_extension.lower()
    data_list =[]
    if file_extension == ".json" or file_extension == ".jsonl":
        with open(test_data_path,"r",encoding="utf-8") as file:
            data_list = [data for data in json.loads(file.read())]
    elif file_extension == ".parquet":
        data_df = pd.read_parquet(test_data_path)
        data_list = [data.to_dict() for index,data in data_df.iterrows()]
    df = pd.DataFrame(data_list)
    dataset = Dataset.from_pandas(df)
    return dataset

def local_medical_datasets_prepare():
    data_list=[]
    test_data_path ="data/medical/medical_o1_sft_Chinese.json"
    with open(test_data_path,"r",encoding="utf-8") as file:
        for data in json.loads(file.read()):
            data["solution"] = data["Response"]
            data["problem"] = data["Question"]

if __name__ =="__main__":
    loguru.logger.info("test datastes")
    test_dataset()