import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map='cuda',
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

input_data = tokenizer('Once upon a year', return_tensors='pt').to('cuda')

predict = model_4bit.generate(**input_data, max_new_tokens=100)

indices = predict[0].tolist()
print(tokenizer.decode(indices))