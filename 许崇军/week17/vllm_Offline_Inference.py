#下⼀步，创建 check-vllm.py ⽂件并在其中编写如下代码：
from vllm import LLM, SamplingParams
prompts = [
 "中国十二地支中代表马的是 ",
 "国内的科技巨头有",
 "历史上的金融危机发生在",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen3-0.6B-Base", tensor_parallel_size=1)

responses = llm.generate(prompts, sampling_params)
for response in responses:
  print(response.outputs[0].text)