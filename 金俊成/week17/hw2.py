import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map='cuda:0',
    quantization_config=quantization_config,
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

input_data = tokenizer('天为什么是蓝的', return_tensors='pt').to('cuda:0')

predict = model_4bit.generate(**input_data, max_new_tokens=100)

indices = predict[0].tolist()
tokenizer.decode(indices)