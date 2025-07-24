import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_4bit=AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B',
    device_map='cuda:0',
    quantization_config=quantization_config
)

tokenizer_llm = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
input_data = tokenizer_llm('讲一个笑话:',return_tensors ='pt').to('cuda:0')
predict = model_4bit.generate(**input_data,max_new_tokens=100)
indices = predict[0].tolist()
tokenizer_llm.decode(indices)