from vllm import LLM, SamplingParams

import torch 
# from modelscope import snapshot_download 
# osmodel_dir = snapshot_download('ZhipuAI/glm-4-9b-chat', cache_dir='/root/autodl-tmp', revision='master')

prompts = [
    '你是一个手语老师',
    '今天，你要给法语老师上课'
]
sample_params = SamplingParams(temperature=0.8, top_p=0.85)
model_name = '/root/autodl-tmp/ZhipuAI/glm-4-9b-chat'
# llm = LLM(model=model_name, trust_remote_code=True)
llm = LLM(model=model_name, max_model_len=2048,trust_remote_code=True)
outputs = llm.generate(prompts, sample_params)
for output in outputs:
    print(f"{output.prompt}, generate: {output.outputs[0].text}")