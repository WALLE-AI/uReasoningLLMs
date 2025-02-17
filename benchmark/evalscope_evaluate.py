##https://modelscope.cn/datasets/modelscope/R1-Distill-Math-Test/summary
'''
evalscope eval `
 --model DeepSeek-R1-Distill-Qwen-32B `
 --api-url http://xxxxxv1/chat/completions `
 --api-key EMPTY `
 --eval-type service `
 --datasets gsm8k `
 --limit 10

'''


from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType
import loguru

task_cfg = TaskConfig(
    model='DeepSeek-R1-Distill-Qwen-32B',   # 模型名称 
    api_url='http://xxxx/v1/chat/completions',  # 推理服务地址
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,   # 评测类型，SERVICE表示评测推理服务
    datasets=[
        'data_collection',  # 数据集名称(固定为data_collection表示使用混合数据集)
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': 'modelscope/R1-Distill-Math-Test'  # 数据集ID 或 数据集本地路径
        }
    },
    eval_batch_size=2,       # 发送请求的并发数
    generation_config={       # 模型推理配置
        'max_tokens': 8196,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1,                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
)

if __name__ =="__main__":
    loguru.logger.info(f"benchmark starting......")
    run_task(task_cfg=task_cfg)