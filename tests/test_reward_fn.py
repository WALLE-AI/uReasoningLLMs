from rl.reward import equation_reward_func, format_reward_func

correct_sample_1 = """We need to find an equation using the numbers 19, 36, 55, and 7
exactly once, with basic arithmetic operations, that equals 65. One possible
combination is 55 + 36 - 19 + 7... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""
 
correct_sample_2 = """ ... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""
 
wrong_format = """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""
 
wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:                      
95 + 88 = 183                                                                                                              
Now, let's subtract 104 from 183 to get 79:
183 - 104 = 79
<think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""
 
wrong_result = """ ... </think>
<answer> 55 + 36 - 7 - 18 </answer>"""
 
def test_format_or_equation_func():
    test_rewards = format_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result], target=["65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 5)
    assert test_rewards == [1.0, 1.0, 0.0, 0.0, 1.0], "Reward function is not working"
    test_rewards = equation_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result], target=["65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 5)
    assert test_rewards == [1.0, 1.0, 0.0, 0.0, 0.0], "Reward function is not working"