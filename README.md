# uReasoningLLMs
* Deepseek-r1复现科普与资源汇总,目前复现主要针对于R1蒸馏模型（领域模型或者自有SFT模型）和R1-Zero的复现（观测"顿悟"情况）
## 科普
* 图解Deepseek-r1
* 可视化指导推理模型
* 深度理解推理模型
* Deepseek开源砸了多少人"饭碗"
* Deepseek技术创新真的是"国运创新"？
## 数据集构建策略
* 在Deepseek的技术报告冷启动Long-Cot数据、60W推理数据和20W非推理数据在整个训练过程至关重要，非常有必要在数据集合成与评测上面进行复现研讨，一般业务场景对SFT数据中增强推理能力，比如领域业务场景中合成推理数据集，提高领域模型自身推理能力
* [rStar-Math](https://github.com/microsoft/rStar)
## 蒸馏策略
* [Distill Math Reasoning Data from DeepSeek R1 with CAME](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)
* [LLMs-Distillation-Quantification](https://github.com/Aegis1863/LLMs-Distillation-Quantification/tree/main)
## RL策略
* [Llama3.1_(8B)-GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=XjjUb0hqE6nm)
* [Logic-RL](https://github.com/Unakar/Logic-RL)
* [open-r1](https://github.com/huggingface/open-r1)
* [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
* [oat-zero](https://oatllm.notion.site/oat-zero)
* [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
* [s1: Simple test-time scaling](https://github.com/simplescaling/s1)
## AIInfra
* 这里主要针对于低成本复现R1的策略
## 关键论文
* 《Distilling the Knowledge in a Neural Network》(https://arxiv.org/abs/1503.02531)
* DeepSeek-R1: Incentivizing Reasoning Capability in LLMs viaReinforcement Learning(https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
* [OpenAI o3-mini](https://openai.com/index/openai-o3-mini/)
* [Gemini 2.0 Flash Thinking](https://deepmind.google/technologies/gemini/flash-thinking/)
* [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
* [Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
* [Scaling Scaling Laws with Board Games](https://arxiv.org/abs/2104.03113)
* [Speculations on Test-Time Scaling](https://srush.github.io/awesome-o1/o1-tutorial.pdf)
