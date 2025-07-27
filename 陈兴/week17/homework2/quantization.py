import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 在Mac上实现"量化"的说明 ---
# 1. 问题：MPS后端在处理某些模型时存在兼容性问题，导致矩阵乘法维度错误
# 2. 解决方案：强制使用CPU来确保稳定运行
#    - 使用半精度(float16)来减少内存占用
#    - 在CPU上运行虽然稍慢，但更稳定

# 1. 强制使用CPU以避免MPS兼容性问题
device = torch.device("cpu")
print("使用CPU设备以确保稳定运行。")
torch_dtype = torch.float16  # 使用半精度减少内存占用

# 2. 直接加载半精度模型（模拟量化效果）
print("正在加载模型 'Qwen/Qwen3-0.6B' 到半精度...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch_dtype,  # 直接加载为半精度，减少内存占用
    trust_remote_code=True
).to(device)

print(f"模型加载完成！模型数据类型: {model.dtype}")
print(f"模型内存占用: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 3. 准备输入数据
input_data = tokenizer('介绍一下AI的历史：', return_tensors='pt').to(device)

# 4. 使用半精度模型生成文本
print("正在使用半精度模型生成文本...")
with torch.no_grad():  # 推理时不需要梯度
    predict = model.generate(**input_data, max_new_tokens=100)

# 5. 解码并打印结果
indices = predict[0].tolist()
result = tokenizer.decode(indices, skip_special_tokens=True)

print("\n--- 半精度模型生成结果 ---")
print(result)
