# 使用量化方式加载文本生成大模型并进行推理
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


# 加载4-bit量化模型
local_model_path ="/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B-Base"
model_4bit = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True  # 允许加载远程代码
)


tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
trust_remote_code=True
)
# 使用4-bit量化模型进行文本生成
prompt ="人工智能的未来发展方向是"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model_4bit.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True))

