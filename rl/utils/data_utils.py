import json
import os
from datasets import load_dataset,Dataset,concatenate_datasets
import loguru
import pandas as pd

from rl.grpo import SYSTEM_PROMPT

dataset_root = os.getenv("DATASETS_ROOT")


LOCAL_DATASETS_REGISTRY={
    "Bespoke-Stratos-17k":dataset_root+"Bespoke-Stratos-17k/data",
    "OpenR1-Math-220k": dataset_root+"OpenR1-Math-220k/data",
    "ReasonFlux_SFT_15k__medical_r1":dataset_root+"ReasonFlux_SFT_15k_medical_r1",
    "OpenThoughts-114k":dataset_root +"OpenThoughts-114k/data",
    "Countdown-Tasks-3to4":dataset_root +"Countdown-Tasks-3to4",
    "X-R1-7500":dataset_root +"X-R1-7500",
    "medical-o1-reasoning-SFT": dataset_root+"medical-o1-reasoning-SFT"
    
}

def load_local_dataset(dataset_path:str):
    data_list=[]
    _, file_extension = os.path.splitext(dataset_path)
    file_extension = file_extension.lower()
    if file_extension == ".json" or file_extension == ".jsonl":
        with open(dataset_path,"r",encoding="utf-8") as file:
            data_list = [data for data in json.loads(file.read())]
    elif file_extension == ".parquet":
        data_df = pd.read_parquet(dataset_path)
        data_list = [data.to_dict() for index,data in data_df.iterrows()]
    df = pd.DataFrame(data_list)
    dataset = Dataset.from_pandas(df)
    return dataset

def local_data_to_datasets(script_args):
    all_dataset_dict = {}
    for dataset_name in script_args.local_dataset_path:
        dataset_path_dir = LOCAL_DATASETS_REGISTRY[dataset_name]
        try:
            # 遍历文件夹中的所有文件
            datasets_list=[]
            for filename in os.listdir(dataset_path_dir):
                file_path = os.path.join(dataset_path_dir, filename)
                if os.path.isfile(file_path):
                    datasets_list.append(load_local_dataset(file_path))
            dataset = concatenate_datasets(datasets_list)
            ##每个一个数据集进行随机抽样
            if script_args.random_sample:
                dataset = dataset.shuffle(seed=42).select(range(script_args.random_sample))
            all_dataset_dict[dataset_name]=dataset
        except Exception as e:
            loguru.logger.info(f"preproces file dir error: {str(e)}")
    return all_dataset_dict


def dataset_format_alignment(all_dataset_dict:dict):
    '''
        {
        "messages": [
            {
                "role": "system",
                "content": "作为一名助理，你需要通过系统的长期思考过程对问题进行深入探讨，然后提供最终的精确和准确的解决方案。这就需要进行全面的分析、总结、探索、重新评
估、反思、回溯和迭代循环，以形成深思熟虑的思维过程。请将答题结构分为两大部分： 思考和解决方案。在 “思考 ”部分，请使用指定格式详细说明您的推理过程： <|begin_of_thought|> {thought with steps separated with ‘\n\n’} <|end_of_thought|> 每个步骤都应包括详细的考虑因素，如分析问题、总结相关发现、集思广益提出新想法、验证当前步骤的准确性、改进>任何错误以及重温之前的步骤。在 “解决方案 ”部分，根据 “思考 ”部分的各种尝试、探索和反思，系统地提出你认为正确的最终解决方案。解决方案应保持逻辑严密、准确、简洁的表达风格
，并详细说明得出结论所需的必要步骤，格式如下： <|begin_of_solution|> {最终格式化的、精确的、清晰的解决方案}。<|end_of_solution|> 现在，请根据上述指导原则尝试解决下面的>问题："
            },
            {
                "role": "user",
                "content": "Prove: When a, b, c, d > 0, (a+b+c+d)/4 ≥ ⁴√(abcd). What method should be chosen?"
            },
            {
                "role": "assistant",
                "content": "<|begin_of_thought|>\n1. The problem requires proving that the arithmetic mean of four numbers is greater than or equal to their geometric mean.\n2. All variables are positive numbers, satisfying the preconditions for application.\n3. This is a standard form of the n-variable AM-GM inequality, requiring no additional transformation.\n4. Need to confirm the generalized form of the basic inequality when n=4.\n5. This is a typical multi-variable inequality proof scenario.\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\nThe appropriate method for this scenario is the application of the basic inequality for three or n variables. The generalized form of the arithmetic mean ≥ geometric mean inequality can be directly applied to the four-variable case. The equality holds if and only if a=b=c=d, perfectly matching the current proof requirement.<|end_of_solution|>"
            }
        ]
    }
    SFT所有的数据格式全部对齐成message格式 采用dataset map策略来完成
    
    GRPO format 采用 reward fun均采用此数据格式设计
    {
        "prompt":[xxxx],
        "solution":str
    }
    '''
    def preproce_conversation(example):
        data_dict ={
            "messages":[
                {"role":data["from"],"content":data["value"]} for data in example['conversations']               
            ]
        }
        data_dict['messages'].append({"role":"system","content":example['system']})
        return data_dict
    
    def countdown_tasks_3to4_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        { 
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }]
        return {"prompt":r1_prefix, "target": target} 
    
     #Format into conversation不同复现策略这里有点区别 都是数学题目
    def make_conversation(example):
        ##可能在不同的情况下 system prompt存在差异
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }
    
    all_datasets_alig_list=[]
    for key,value in all_dataset_dict.items():
        match key:
            case "Bespoke-Stratos-17k":
                local_dataset = value.map(preproce_conversation)
                dataset = local_dataset.remove_columns(["system","conversations"])
                all_datasets_alig_list.append(dataset)
            case "OpenThoughts-114k":
                local_dataset = value.map(preproce_conversation)
                dataset = local_dataset.remove_columns(["system","conversations"])
                all_datasets_alig_list.append(dataset)
            case "ReasonFlux_SFT_15k__medical_r1":
                all_datasets_alig_list.append(value)
            case "OpenR1-Math-220k":
                ##需要对齐格式
                dataset = value.map(make_conversation)
                all_datasets_alig_list.append(dataset)
            ##https://www.philschmid.de/mini-deepseek-r1 针对于 https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4 数据集性message 构建
            ##reward fun有差异    
            case "Countdown-Tasks-3to4":
                dataset = value.map(lambda x: countdown_tasks_3to4_prompt(x["nums"], x["target"]))
                all_datasets_alig_list.append(dataset)
            case "X-R1-7500":
                dataset = value.map(make_conversation)
                all_datasets_alig_list.append(dataset)
    all_datasets= concatenate_datasets(all_datasets_alig_list)
    return all_datasets

def load_or_prepare_datasets(script_args)->Dataset:
    '''
    本地加载和HF加载 local_path=script_args.local_dataset_path,
    script_args=script_args.random_sample
    '''
    try:
        dataset_dict = local_data_to_datasets(script_args=script_args)
        dataset = dataset_format_alignment(dataset_dict)
        # split the dataset into train and test
            # select a random subset of 50k samples
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        return train_dataset,test_dataset
    except RuntimeError as e:
        loguru.logger.info(f"load datsets error:{e}")
        

            
                
    
    
    
    
    
    
