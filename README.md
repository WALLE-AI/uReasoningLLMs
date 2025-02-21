# uReasoningLLMs
* Deepseek-r1复现科普与资源汇总,目前复现主要针对于R1蒸馏模型（领域模型或者自有SFT模型）和R1-Zero的复现（观测"顿悟"情况）
## Introduction
* [图解Deepseek-r1](https://zhuanlan.zhihu.com/p/22070707889)
* [可视化指导推理模型](https://zhuanlan.zhihu.com/p/22193737362)
* [深度理解推理模型](https://zhuanlan.zhihu.com/p/22660720550)
* [A Visual Guide to Mixture of Experts (MoE)](https://zhuanlan.zhihu.com/p/23074047123)
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
* [Math-spepherd](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/math_shepherd)
* [magpie](https://magpie-align.github.io/)
* [Math-Verify](https://github.com/huggingface/Math-Verify)
* [Camel_ai_cot_agent](https://docs.camel-ai.org/cookbooks/data_generation/self_improving_cot_generation.html)
### Opensource Datasets
* [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) 使用Deeepseek-R1对[Sky-T1_data_17k](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k)进行蒸馏 创建Bespoke-Stratos-17k——一个包含问题、推理轨迹和答案的推理数据集。
* [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
* [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR)
* [Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd)
* [QwQ-LongCoT-500K-Cleaned](https://huggingface.co/datasets/qingy2024/QwQ-LongCoT-500K-Cleaned)
* [trl-lib/tldr](https://huggingface.co/datasets/trl-lib/tldr)
* [prm800k](https://huggingface.co/datasets/HuggingFaceH4/prm800k-trl-dedup)
* [ReasonFlux_SFT_15k](https://huggingface.co/datasets/Gen-Verse/ReasonFlux_SFT_15k)
* [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/main/extended)
* [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
* [X-R1-7500](https://huggingface.co/datasets/xiaodongguaAIGC/X-R1-7500)
* [R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)
* [dolphin-r1](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1)
* [r1_mix_1024.jsonl](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/r1_mix_1024.jsonl)
* [Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)
### Medical R1 Datastes
* [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
## Multi-Model Datasets
* [open-r1-video-4k](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)
## Distillation
* [Distill Math Reasoning Data from DeepSeek R1 with CAME](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)
* [LLMs-Distillation-Quantification](https://github.com/Aegis1863/LLMs-Distillation-Quantification/tree/main)
* [open-thoughts](https://github.com/open-thoughts/open-thoughts)通过R1来合成推理数据集来训练最先进的小型推理模型，该模型在数学和代码推理基准上超越DeepSeek-R1-Distill-Qwen-32B和DeepSeek-R1-Distill-Qwen-7B 。
* [s1: Simple test-time scaling](https://github.com/simplescaling/s1)
* [LIMO](https://github.com/GAIR-NLP/LIMO)
## RL
### 一些RL知识
* [什么是GRPO Trainer](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)
* [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
* [【手撕LLM-GRPO】你只管给Reward, 剩下的交给RL（附代码）](https://zhuanlan.zhihu.com/p/20812786520)
* [PPO & GRPO 可视化介绍](https://mp.weixin.qq.com/s/HE5wUIzg5c2u2yqEVVB9fw)
* [从理论到代码剖析DeepSeek-R1：从PPO到Reinforce++，再对比GRPO](https://mp.weixin.qq.com/s/7NiKjgGSujrDvndxqyWk-Q)
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
* [RAGEN](https://github.com/ZihanWang314/RAGEN)
* [Mini-R1: Reproduce Deepseek R1 „aha moment“ a RL tutorial](https://www.philschmid.de/mini-deepseek-r1)
* [unlock-deepseek](https://github.com/datawhalechina/unlock-deepseek)
* [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
* [mini-r1-zero](https://github.com/ahxt/mini-r1-zero)
* [ReasonFlux](https://github.com/Gen-Verse/ReasonFlux)之开源了SFT阶段训练代码和数据集，RL部分没有开源
* [grpo_demo](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)
* [X-R1](https://github.com/dhcode-cpp/X-R1)
* [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
* [lmm-r1](https://github.com/TideDra/lmm-r1)
### Multi-Models R1
* [Open-R1-Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video)
* [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)
* [VLM-R1](https://github.com/om-ai-lab/VLM-R1) 在 open-r1-multimodal修改参数
* [R1-V](https://github.com/Deep-Agent/R1-V)
### Progress of Replication Domain Project
* [diagnosis_zero](https://github.com/wizardlancet/diagnosis_zero) 在医疗的疾病诊断任务上拿Qwen2.5-1.5B/3B/7B用R1-Zero那种rule based reward尝试复现了一下
* [PatientSeek](https://huggingface.co/whyhow-ai/PatientSeek)使用unsloth 单卡Int4上面基于DeepSeek-R1-Distill-Llama-8B微调作为的非常浅，但是业务场景思路很好
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
* [Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models](https://github.com/YangLing0818/buffer-of-thought-llm/tree/main)
* [Process Reinforcement through Implicit Rewards](https://curvy-check-498.notion.site/Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f#1604fcb9c42180bbabc6f67a2866d2c3)
* [Distillation Scaling Laws]()
## Reference
* [DeepSeek R1推理相关项目源码分析](https://mp.weixin.qq.com/s/dvy_4uJ5og9IS6J8mPKIQQ)
* [DeepSeek R1 论文解读&关键技术点梳理](https://mp.weixin.qq.com/s/wckZqmgSmocnIgUPcg5QcQ)
* [综述 DeepSeek R1、LIMO、S1 等 6 篇文章的关键结论](https://mp.weixin.qq.com/s/04HEd5CUWETck6Ug-pOYjg)
* [开源22万条DeepSeek R1的高质量数据！你也能复现DeepSeek了](https://mp.weixin.qq.com/s/K4msDYxwYNhNsTRNK0sVwA)
* [DeepSeek R1 论文解读&关键技术点梳理](https://mp.weixin.qq.com/s/wckZqmgSmocnIgUPcg5QcQ)
* [A vision researcher’s guide to some RL stuff: PPO & GRPO](https://yugeten.github.io/posts/2025/01/ppogrpo/)