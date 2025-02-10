# uReasoningLLMs
* Deepseek-r1复现科普与资源汇总,目前复现主要针对于R1蒸馏模型（领域模型或者自有SFT模型）和R1-Zero的复现（观测"顿悟"情况）
## Introduction
* [图解Deepseek-r1](https://zhuanlan.zhihu.com/p/22070707889)
* [可视化指导推理模型](https://zhuanlan.zhihu.com/p/22193737362)
* [深度理解推理模型](https://zhuanlan.zhihu.com/p/22660720550)
* Deepseek开源砸了多少人"饭碗"
* Deepseek技术创新真的是"国运创新"？
## Prompt
* [Deepseek官方Prompt](https://api-docs.deepseek.com/zh-cn/prompt-library)
* [DeepSeek从入门到精通(20250204)](docs/DeepSeek从入门到精通(20250204).pdf)
## Datasets
### Datasets Builds
* 在Deepseek的技术报告冷启动Long-Cot数据、60W推理数据和20W非推理数据在整个训练过程至关重要，非常有必要在数据集合成与评测上面进行复现研讨，一般业务场景对SFT数据中增强推理能力，比如领域业务场景中合成推理数据集，提高领域模型自身推理能力
* [rStar-Math](https://github.com/microsoft/rStar)
* [distilabel_r1_distill](https://github.com/huggingface/open-r1/blob/main/src/open_r1/generate.py)
* [Math-spepherd][https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/math_shepherd]
* [magpie](https://magpie-align.github.io/)
* [Camel_ai_cot_agent](https://docs.camel-ai.org/cookbooks/data_generation/self_improving_cot_generation.html)
### Opensource Datasets
* [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) 使用Deeepseek-R1对[Sky-T1_data_17k](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k)进行蒸馏 创建Bespoke-Stratos-17k——一个包含问题、推理轨迹和答案的推理数据集。
* [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR)
* [Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd)
* [QwQ-LongCoT-500K-Cleaned](https://huggingface.co/datasets/qingy2024/QwQ-LongCoT-500K-Cleaned)
* [trl-lib/tldr](https://huggingface.co/datasets/trl-lib/tldr)
## Distillation
* [Distill Math Reasoning Data from DeepSeek R1 with CAME](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)
* [LLMs-Distillation-Quantification](https://github.com/Aegis1863/LLMs-Distillation-Quantification/tree/main)
## RL
### 一些RL知识
* [什么是GRPO Trainer](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)
### PRM(Process Reward Models)
* [Process Reinforcement Through Implicit Rewards](https://github.com/PRIME-RL/PRIME)
### ORM(Outcome Reward Models)
### Progress of Replication Project
* [Llama3.1_(8B)-GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=XjjUb0hqE6nm)
* [Logic-RL](https://github.com/Unakar/Logic-RL)
* [open-r1](https://github.com/huggingface/open-r1)
* [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
* [oat-zero](https://oatllm.notion.site/oat-zero)
* [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
* [s1: Simple test-time scaling](https://github.com/simplescaling/s1)
* [RAGEN](https://github.com/ZihanWang314/RAGEN)
* [R1-V](https://github.com/Deep-Agent/R1-V)
### Progress of Replication Domain Project
* [diagnosis_zero](https://github.com/wizardlancet/diagnosis_zero) 在医疗的疾病诊断任务上拿Qwen2.5-1.5B/3B/7B用R1-Zero那种rule based reward尝试复现了一下
## AIInfra
* 这里主要针对于低成本复现R1的策略,本项目会针对于上述开源项目进行复现，并且具体给出对应的硬件成本估算
* [Sky-T1: Train your own O1 preview model within $450](https://novasky-ai.github.io/posts/sky-t1/)
## Papers
* [《Distilling the Knowledge in a Neural Network》](https://arxiv.org/abs/1503.02531)
* [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs viaReinforcement Learning](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
* [OpenAI o3-mini](https://openai.com/index/openai-o3-mini/)
* [Gemini 2.0 Flash Thinking](https://deepmind.google/technologies/gemini/flash-thinking/)
* [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
* [Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
* [Scaling Scaling Laws with Board Games](https://arxiv.org/abs/2104.03113)
* [Speculations on Test-Time Scaling](https://srush.github.io/awesome-o1/o1-tutorial.pdf)
## Reference
* [DeepSeek R1推理相关项目源码分析](https://mp.weixin.qq.com/s/dvy_4uJ5og9IS6J8mPKIQQ)
