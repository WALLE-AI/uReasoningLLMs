##合并多个数据集
'''
task1: 基于Qwen2.5-7B-base模型,利用公开的指令数据集 使用R1合成增加的指令与推理数据集，分别参考如下数据集
QwQ-LongCoT-Verified-130K
OpenR1-Math-220k
Bespoke-Stratos-17k
Magpie-Reasoning-V1-150K
数据任务的类别
信息搜索：提供准确简明的信息，涵盖广泛的主题，帮助用户寻找特定事实、概念解释或详细信息。
推理：专注于逻辑思考和解决复杂问题，帮助用户组织复杂思想、分析情况，并得出结论。
规划：协助用户制定有效的计划和策略，整理思想，设定目标，并为任务或活动创造可行的解决方案。
编辑：通过提供语法、风格、清晰度和整体结构的建议来改善书面内容，帮助用户优化写作。
编码：协助用户编写、审查和调试各种编程语言的代码，提供清晰的解释和最佳实践。
数学：解答广泛的数学学科问题，从基础概念到高级主题，提供清晰简明的解释和解决方案。
角色扮演：参与各种角色扮演场景，根据用户请求采取不同角色，以创造身临其境的互动用户体验。
数据分析：帮助用户理解和从数据集中提取有用信息，提供数据趋势的洞察，并进行分析任务。
创意写作：支持创意写作任务，协助用户撰写引人入胜的故事、诗歌、文章和其他创意文本。
寻求建议：提供深思熟虑的建议和指导，帮助用户应对个人、专业或生活挑战。
头脑风暴：生成创意并促进创造性思维，协助用户探索可能性并提出创新概念。
格式约束：严格按照用户指定的格式响应，遵守所有格式要求。
重写：根据用户要求重写文本，使其更简洁、集中或改变语气，类似于编辑。
摘要：根据用户指示总结文本，满足特定的摘要要求。
安全：识别非法内容并合理拒绝响应，或在检测到非法指令时提供适当的建议。
翻译：根据用户请求在英语和中文之间进行翻译，满足特定的翻译要求。
文档：根据参考文本回答用户问题，努力使用参考材料中的信息，而不引入外部知识。
医学诊断：medical_o1_sft_Chinese.json
建筑/建造/造价：看看如何合成
'''

'''
对于

'''
import json
import os
import pandas as pd


common_system_prompt=(
     "作为一名助理，你需要通过系统的长期思考过程对问题进行深入探讨，然后提供最终的精确和准确的解决方案。这就需要进行全面的分析、总结、探索、重新评估、反思、回溯和迭代循环，以形成深思熟虑的思维过程。请将答题结构分为两大部分： 思考和解决方案。在 “思考 ”部分，请使用指定格式详细说明您的推理过程： <|begin_of_thought|> {thought with steps separated with ‘\n\n’} <|end_of_thought|> 每个步骤都应包括详细的考虑因素，如分析问题、总结相关发现、集思广益提出新想法、验证当前步骤的准确性、改进任何错误以及重温之前的步骤。在 “解决方案 ”部分，根据 “思考 ”部分的各种尝试、探索和反思，系统地提出你认为正确的最终解决方案。解决方案应保持逻辑严密、准确、简洁的表达风格，并详细说明得出结论所需的必要步骤，格式如下： <|begin_of_solution|> {最终格式化的、精确的、清晰的解决方案}。<|end_of_solution|> 现在，请根据上述指导原则尝试解决下面的问题："
)

datasets_root="D:/WALLE-AI/uReasoningLLMs/data/"

datasets_task = [
    {
        "task":"medical_zh",
        "path":datasets_root +"/medical_o1_sft_Chinese.json"
     },
     {
        "task":"medical_en",
        "path":"/home/dataset1/gaojing/datasets/medical/medical-o1-reasoning-SFT/medical_o1_sft.json"
     },
     {
         "task":"smoltalk-chinese",
         "path":"/home/dataset1/gaojing/datasets/posttrain/smoltalk/data/all/"
     },
     {
         "task":"QwQ-LongCoT-Verified-130K",
        "path":"/home/dataset1/gaojing/datasets/posttrain/math/QwQ-LongCoT-Verified-130K/verified/"
    },
    {
         "task":"Magpie-Reasoning-V1-150K",
        "path":"/home/dataset1/gaojing/datasets/posttrain/math/Magpie-Reasoning-V1-150K/data/"

     },
    {
        "task":"Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B",
        "path":"/home/dataset1/gaojing/datasets/posttrain/math/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B/data/"
     },
     {
        "task":"Bespoke-Stratos-17k",
        "path":"/home/dataset1/gaojing/datasets/posttrain/math/Bespoke-Stratos-17k/data/" 
    },
    {
        "task":"construct",
        "path":""
    }
]

def preprocess_medical_data():
    ## <|begin_of_thought|> {thought with steps separated with ‘\n\n’} <|end_of_thought|> 
    medical_data_path = datasets_root +"medical/medical_o1_sft_Chinese.json"
    data_list=[]
    with open(medical_data_path,"r",encoding="utf-8") as file:
        for data in json.loads(file.read()):
            data_dict = {
                "messages":[
                    {
                        "role":"system",
                        "content":common_system_prompt
                    },
                    {
                        "role":"user",
                        "content":data["Question"]
                    },
                    {
                        "role":"assistant",
                        "content":"<|begin_of_thought|>" +data['Complex_CoT'] +"<|end_of_thought|>\n\n" +  "<|begin_of_solution|>"+data["Response"] +"<|end_of_solution|>"
                    }
                ]
            }
            data_list.append(data_dict)
    return data_list
    # save_data_path=datasets_root+"medical_o1_sft_Chinese_r1.json"
    # with open(save_data_path,"w",encoding="utf-8") as file:
    #     file.write(json.dumps(data_list,ensure_ascii=False,indent=4))

def reason_flux_data():
    medical_data_path = datasets_root +"ReasonFlux_SFT_15k/ReasonFlux_SFT_15k.json"
    data_list=[]
    with open(medical_data_path,"r",encoding="utf-8") as file:
        for data in json.loads(file.read()):
            think = data["output"].split("</think>")
            answer=think[0].split("<think>")
            data_dict = {
                "messages":[
                    {
                        "role":"system",
                        "content":common_system_prompt
                    },
                    {
                        "role":"user",
                        "content":data["input"]
                    },
                    {
                        "role":"assistant",
                        "content":"<|begin_of_thought|>"+answer[-1]+ "<|end_of_thought|>\n\n"+ "<|begin_of_solution|>"+think[-1]+"<|end_of_solution|>"
                    }
                ]
            }
            data_list.append(data_dict)
    data_list = data_list + preprocess_medical_data()
    save_data_path=datasets_root+"ReasonFlux_SFT_15k/ReasonFlux_SFT_15k__medical_r1.json"
    with open(save_data_path,"w",encoding="utf-8") as file:
        file.write(json.dumps(data_list,ensure_ascii=False,indent=4))


def merge_parquet_files(folder_path, output_file=None):
    """
    合并指定文件夹中的所有 .parquet 文件
    
    参数:
        folder_path (str): 包含 .parquet 文件的文件夹路径
        output_file (str, optional): 如果提供，将合并结果保存到该文件路径
    
    返回:
        pandas.DataFrame: 合并后的数据框架
    
    异常:
        ValueError: 如果文件夹不存在或没有有效的 .parquet 文件
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        raise ValueError(f"文件夹路径 {folder_path} 不存在")

    # 获取所有 .parquet 文件
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    
    # 如果没有找到 .parquet 文件
    if not parquet_files:
        raise ValueError(f"在文件夹 {folder_path} 中没有找到 .parquet 文件")

    # 初始化一个空列表来存储 DataFrame
    dfs = []
    
    # 读取每个 .parquet 文件
    for file in parquet_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_parquet(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
    
    # 如果没有成功读取任何文件
    if not dfs:
        raise ValueError("没有成功读取任何有效的 .parquet 文件")

    # 合并所有 DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 如果提供了输出路径，则保存结果
    if output_file:
        merged_df.to_parquet(output_file)
        print(f"合并后的文件已保存到: {output_file}")
    
    return merged_df


def read_dataset(file_path):
    """
    根据文件扩展名读取数据集，并将其转换为 Pandas DataFrame。
    
    参数:
        file_path (str): 文件路径。
    
    返回:
        pandas.DataFrame: 读取的数据集。
    
    异常:
        ValueError: 如果文件格式不受支持。
        FileNotFoundError: 如果文件不存在。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.parquet':
        return pd.read_parquet(file_path)
    elif ext == '.jsonl':
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def merge_datasets(file_paths, output_file=None, join_method='outer'):
    """
    读取并合并多个数据集，支持不同格式和列结构。
    
    参数:
        file_paths (list): 包含文件路径的列表。
        output_file (str, optional): 如果提供，将合并结果保存到该文件路径。
        join_method (str): 合并时的列处理方式，可选 'outer'（保留所有列）或 'inner'（只保留共有列）。
    
    返回:
        pandas.DataFrame: 合并后的数据集。
    
    异常:
        ValueError: 如果没有成功读取任何数据集。
    """
    datasets = []
    for file_path in file_paths:
        try:
            df = read_dataset(file_path)
            datasets.append(df)
            print(f"成功读取文件: {file_path}，列数: {len(df.columns)}")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
    
    if not datasets:
        raise ValueError("没有成功读取任何数据集")
    
    # 检查列名是否一致并打印信息
    all_columns = [set(df.columns) for df in datasets]
    common_columns = set.intersection(*all_columns)
    if len(common_columns) < len(datasets[0].columns):
        print("警告: 数据集列不一致。")
        print(f"共有列: {common_columns}")
        print(f"所有列: {set.union(*all_columns)}")
    
    # 合并数据集
    try:
        merged_df = pd.concat(datasets, ignore_index=True, join=join_method)
        print(f"合并完成，使用 {join_method} 方式，列数: {len(merged_df.columns)}")
    except Exception as e:
        print(f"合并失败: {e}")
        return None
    
    # 如果提供了输出文件路径，则保存结果
    if output_file:
        ext = os.path.splitext(output_file)[1].lower()
        if ext == '.parquet':
            merged_df.to_parquet(output_file)
        elif ext == '.jsonl':
            merged_df.to_json(output_file, orient='records', lines=True)
        else:
            raise ValueError(f"不支持的输出文件格式: {ext}")
        print(f"合并后的文件已保存到: {output_file}")
    
    return merged_df
            
            
import json

def extract_qa_from_md():
    file_path = "xxxx/uReasoningLLMs/data/brain-teasers_zh/三千个脑筋急转弯.md"
    """
    从 .md 文件中提取脑筋急转弯的问题和答案，并返回包含字典的列表。
    
    参数:
        file_path (str): .md 文件的路径
    
    返回:
        list: 包含字典的列表，每个字典格式为 {"question": "xxxx", "answer": "xxx"}
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 存储问题和答案的列表
    qa_list = []
    
    # 处理每一行
    for line in lines:
        # 只处理以 <br> 开头的行
        if line.startswith('<br>'):
            # 去掉 <br> 并移除首尾空白
            line = line[4:].strip()
            # 找到 — 的位置
            dash_index = line.find('—')
            if dash_index != -1:
                # — 后面的是剩余内容
                rest = line[dash_index + 1:].strip()
                # 找到 答案： 的位置
                answer_index = rest.find('答案：')
                if answer_index != -1:
                    # 提取问题和答案
                    question = rest[:answer_index].strip()
                    answer = rest[answer_index + 3:].strip()
                    # 添加到列表
                    qa_list.append({"question": question, "answer": answer})
    save_data = "D:/LLM/project/dsr1/uReasoningLLMs/data/brain-teasers_zh/brain-teaser_zh_3649.jsonl"               
    with open(save_data,"w",encoding="utf-8") as file:
        file.write(json.dumps(qa_list,ensure_ascii=False,indent=4))