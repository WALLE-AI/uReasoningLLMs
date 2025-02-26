SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
##参考VLM-R1 得到思考过程，不是可以使用QVQ来合成，目前使用multimodal-open-r1-8k-verified 合成COT还是太短了
PROMPT_FORMAT = """I will provide you with an image, an original question, and its answer related to the image. Your task is to rewrite the question in such a way that answering it requires step-by-step Chain-of-Thought (CoT) reasoning with numerical or mathematical expressions where applicable. The reasoning process can include expressions like "let me think," "oh, I see," or other natural language thought expressions.

Please make sure your question is to ask for a certain answer with a certain value, do not ask for open-ended answer, and the answer is correct and easy to verify via simple protocol, like "2" or "A".

Please strictly do not include "Answer:" in the question part to avoid confusion and leakage.

Input Format:
Original Question: {original_question}
Original Answer: {original_answer}

Output Format:
Question: [rewrite the question if necessary]
Answer: [answer with reasoning steps, including calculations where applicable]
<think>step-by-step reasoning process</think>
<answer>easy to verify answer</answer>
"""

IMAGE_SYSTEM_PROMPT=(
    "You are an expert to analyze the image and provide useful information for users."
)
