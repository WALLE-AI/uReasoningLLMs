##https://community.modelscope.cn/67aeae0dc8869b4726b4a965.html
##evalscope

import subprocess

##powershell
'''
# 定义 evalscope 命令和参数
evalscope perf `
--parallel 64 `
--url http://xxxxx/chat/completions `
--model DeepSeek-R1-Distill-Qwen-32B `
--log-every-n-query 5 `
--connect-timeout 6000 `
--read-timeout 6000 `
--api openai `
--prompt "写一个科幻小说，不少于2000字，请开始你的表演" `
--wandb-api-key 'xxxxx' `
--name 'DeepSeek-R1-Distill-Qwen-32B-inference-performance' `
-n 500

'''


def run_gpt_performance_test(
    parallel=10,
    url="http://127.0.0.1:8801/v1/chat/completions",
    model="DeepSeek-R1-Distill-Qwen-1.5B",
    log_every_n_query=5,
    connect_timeout=6000,
    read_timeout=6000,
    api="openai",
    prompt="写一个科幻小说，不少于2000字，请开始你的表演",
    num_requests=100,
    log_file=None
):
    """
    运行使用 evalscope perf 测试给定 API 的性能

    参数:
        parallel : int
            并行请求的数量
        url : str
            API 的 URL
        model : str
            使用的模型
        log_every_n_query : int
            每隔多少次请求记录一次日志
        connect_timeout : int
            连接超时时间（毫秒）
        read_timeout : int
            读取超时时间（毫秒）
        api : str
            使用的 API 类型
        prompt : str
            请求的 prompt 内容
        num_requests : int
            总请求次数
        log_file : str (可选)
            记录命令输出的文件路径
    """
    command = [
        "evalscope", "perf",
        f"--parallel={parallel}",
        f"--url={url}",
        f"--model={model}",
        f"--log-every-n-query={log_every_n_query}",
        f"--connect-timeout={connect_timeout}",
        f"--read-timeout={read_timeout}",
        f"--api={api}",
        f"--prompt={prompt}",
        f"-n={num_requests}"
    ]

    try:
        # 运行命令
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=dict(BEARER_TOKEN="your-token")  # 如果需要环境变量
        )

        # 打印或保存日志
        output = result.stdout + result.stderr
        print(output)

        if log_file:
            with open(log_file, "w") as f:
                f.write(output)

    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        print(f"输出: {e.output}")

# 示例使用
if __name__ == "__main__":
    run_gpt_performance_test(
        parallel=10,
        url="http://36.103.239.202:9005/starvlm/v1/chat/completions",
        model="DeepSeek-R1-Distill-Qwen-32B",
        num_requests=100,
        log_file="perf_test.log"
    )