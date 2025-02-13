from rl.trl_trainer_pipline import TrainerPipline


def test_tranier_pipline():
    pipline =TrainerPipline()
    pipline.train_run()
    
    
def test_tranier_pipline_cmd():
    import subprocess

    # 定义命令
    command = [
        "python", "rl/trl_trainer_pipline.py",
        "--config", "receipes/Qwen2.5-3B-Instruct/grpo/grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml"
    ]
    # 执行命令
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("命令执行成功!")
        print("输出:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("命令执行失败!")
        print("错误信息:\n", e.stderr)