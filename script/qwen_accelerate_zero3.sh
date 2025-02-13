# One 1 node of 8 x A100s 40G
accelerate launch --num_processes 7 \
    --config_file receipes/accelerate_configs/deepspeed_zero3_cpu_offload.yaml \
    rl/trl_trainer_pipline.py \
    --config receipes/Qwen2.5-3B-Instruct/grpo/grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml \