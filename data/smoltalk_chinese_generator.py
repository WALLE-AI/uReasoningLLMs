import os
import loguru
from rl.utils.data_utils import load_local_dataset,local_data_to_datasets
from datasets import concatenate_datasets

'''
math 
{'conversations': [{...}, {...}], 'source': 'Math23K', 'answer': '20', 'text_len': 285}
分别抽取每一个类别的1W数据 要求能够筛查出该类别具体案例 OpenThoughts-114k ChatDoctor-HealthCareMagic-100k 
math部分 比重要求20%  ReasonFlux_SFT_15k OpenR1-Math-220k
在加入construct、medical,law，谜语、脑经急转弯等数据 得到 20个类别的多轮对话指令数据集
1、全部都带think合成
2、只有math reasoning medical 谜语 脑筋急转弯带think数据 其他只是蒸馏答案 medical:ChatDoctor-HealthCareMagic-100k 
'''

def local_data_to_datasets(dataset_path_dir,random_sample:int=1):
    all_dataset_dict = {}
    try:
        # 遍历文件夹中的所有文件
        datasets_list=[]
        for filename in os.listdir(dataset_path_dir):
            file_path = os.path.join(dataset_path_dir, filename)
            if os.path.isfile(file_path):
                dataset =load_local_dataset(file_path)
                ##每个一个数据集进行随机抽样
                if random_sample:
                    dataset = dataset.shuffle(seed=42).select(range(random_sample))
                if "classify" in dataset[0]:
                    datasets_list.append({dataset[0]["classify"]:dataset})
                else:
                    loguru.logger.info(f"no classify dataset index {dataset[0]}")
        dataset = concatenate_datasets(datasets_list)
        loguru.logger.info(f"all dataset {len(dataset)}")
    except Exception as e:
        loguru.logger.info(f"preproces file dir error: {str(e)}")

def show_datasets():
    data_path_dir = "/home/dataset1/gaojing/datasets/posttrain/chinese_smolTalk_dataset/smoltalk-chinese/data"
    dataset = local_data_to_datasets(dataset_path_dir=data_path_dir)
    loguru.logger.info(f"dataset index {dataset[0]}")