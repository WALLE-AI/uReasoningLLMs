medical_template = {
"Hypothesis testing simulated data": '''我看到一个模型在以下问题性能很好：

Hypothesis Testing (假设检验) 问题1：如果一个患者表现出某种症状，模型能否准确判断这种症状是否与某种疾病相关？ 问题2：在给定一组临床数据的情况下，模型能否验证某种假设（例如，某种药物是否有效）？ 问题3：模型能否根据患者的病史和症状，判断某种疾病的假设是否成立？
Medical Summarization (医疗总结) 问题1：模型能否从一份详细的病历中提取出关键信息并生成简洁的总结？ 问题2：在处理一份复杂的医疗报告时，模型能否准确识别出主要诊断和治疗建议？ 问题3：模型能否将多份病历整合成一份清晰的总结，同时保留所有重要信息？
Treatment Planning (治疗计划) 问题1：根据患者的病情和病史，模型能否推荐一个合适的治疗方案？ 问题2：在面对一种罕见疾病时，模型能否提供创新的治疗建议？ 问题3：模型能否根据患者的反应和进展，动态调整治疗计划？
你能帮我生成一个，关于：

Hypothesis Testing (假设检验) 问题1：如果一个患者表现出某种症状，模型能否准确判断这种症状是否与某种疾病相关？
具体的患者症状，以及标准的正确答案么，让我来检验这个医学模型

=====================================================

以下是针对假设检验问题1设计的测试案例，包含具体症状描述、标准答案以及验证要点。案例设计基于常见临床场景，同时覆盖典型和边缘情况''',
"Data from medical summaries simulations":'''我看到一个模型在以下问题性能很好：

Hypothesis Testing (假设检验) 问题1：如果一个患者表现出某种症状，模型能否准确判断这种症状是否与某种疾病相关？ 问题2：在给定一组临床数据的情况下，模型能否验证某种假设（例如，某种药物是否有效）？ 问题3：模型能否根据患者的病史和症状，判断某种疾病的假设是否成立？
Medical Summarization (医疗总结) 问题1：模型能否从一份详细的病历中提取出关键信息并生成简洁的总结？ 问题2：在处理一份复杂的医疗报告时，模型能否准确识别出主要诊断和治疗建议？ 问题3：模型能否将多份病历整合成一份清晰的总结，同时保留所有重要信息？
Treatment Planning (治疗计划) 问题1：根据患者的病情和病史，模型能否推荐一个合适的治疗方案？ 问题2：在面对一种罕见疾病时，模型能否提供创新的治疗建议？ 问题3：模型能否根据患者的反应和进展，动态调整治疗计划？
你能帮我生成一个，关于： 2. Medical Summarization (医疗总结) 问题1：模型能否从一份详细的病历中提取出关键信息并生成简洁的总结？

具体的患者病历，以及比较的正确医疗总结答案么，供我来检验这个医学模型

=====================================================

以下是一个模拟的详细患者病历示例以及对应的正确医疗总结参考答案，可用于检验医疗总结模型的准确性:''',
"Data from treatment plan simulations":'''我看到一个模型在以下问题性能很好：

Hypothesis Testing (假设检验) 问题1：如果一个患者表现出某种症状，模型能否准确判断这种症状是否与某种疾病相关？ 问题2：在给定一组临床数据的情况下，模型能否验证某种假设（例如，某种药物是否有效）？ 问题3：模型能否根据患者的病史和症状，判断某种疾病的假设是否成立？
Medical Summarization (医疗总结) 问题1：模型能否从一份详细的病历中提取出关键信息并生成简洁的总结？ 问题2：在处理一份复杂的医疗报告时，模型能否准确识别出主要诊断和治疗建议？ 问题3：模型能否将多份病历整合成一份清晰的总结，同时保留所有重要信息？
Treatment Planning (治疗计划) 问题1：根据患者的病情和病史，模型能否推荐一个合适的治疗方案？ 问题2：在面对一种罕见疾病时，模型能否提供创新的治疗建议？ 问题3：模型能否根据患者的反应和进展，动态调整治疗计划？
你能帮我生成一个，关于：

Treatment Planning (治疗计划) 问题1：根据患者的病情和病史，模型能否推荐一个合适的治疗方案？
具体的患者病情与病史，以及标准的、合适的治疗方案作为黄金答案，让我来检验这个医学模型

=====================================================

老师的模拟案例：

以下是一个可用于测试医学模型在治疗计划方面能力的完整病例示例，包含患者信息、黄金标准答案及评分维度建议：'''

}

##https://github.com/lemonhall/eval_PatientSeek-Q4_K_M/blob/main/%E6%B2%BB%E7%96%97%E8%AE%A1%E5%88%92%E6%A8%A1%E6%8B%9F%E7%9A%84%E6%95%B0%E6%8D%AE.md