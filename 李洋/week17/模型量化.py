from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配
    torch_dtype=torch.float16
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 测试推理
inputs = tokenizer("你好，最近怎么样？", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))