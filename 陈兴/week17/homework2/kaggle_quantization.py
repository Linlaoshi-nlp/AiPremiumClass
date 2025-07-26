import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

# --- Kaggle GPU P100 量化脚本 ---
# https://www.kaggle.com/code/mitrecx/notebookee39512fa3
# 这个脚本专门为 Kaggle 的 NVIDIA GPU P100 环境设计
# 使用 bitsandbytes 库进行真正的 4位量化

def setup_quantization():
    """设置4位量化配置"""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 4位量化
        bnb_4bit_compute_dtype=torch.float16, # 计算时使用float16
        bnb_4bit_quant_type="nf4",           # 使用nf4量化类型
        bnb_4bit_use_double_quant=True,      # 使用双重量化进一步压缩
    )
    return quantization_config

def load_quantized_model(model_name="Qwen/Qwen3-0.6B"):
    """加载量化模型"""
    print(f"正在加载模型: {model_name}")
    
    # 设置量化配置
    quantization_config = setup_quantization()
    
    # 加载量化模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # 自动处理设备映射
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    print("模型加载完成！")
    return model

def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    """使用量化模型生成文本"""
    print(f"输入提示: {prompt}")
    
    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成文本
    print("正在生成文本...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    """主函数"""
    print("=== Kaggle GPU P100 量化模型演示 ===")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA GPU，此脚本需要在NVIDIA GPU环境中运行")
        return
    
    print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # 1. 加载量化模型
        model = load_quantized_model()
        
        # 2. 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 3. 显示模型信息
        print(f"\n模型信息:")
        print(f"- 模型设备: {next(model.parameters()).device}")
        print(f"- 模型数据类型: {next(model.parameters()).dtype}")
        
        # 4. 生成多个示例
        prompts = [
            "介绍一下AI的历史：",
            "讲一个关于机器学习的笑话：",
            "解释什么是深度学习：",
            "写一首关于人工智能的诗："
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- 示例 {i} ---")
            result = generate_text(model, tokenizer, prompt)
            print(f"生成结果:\n{result}")
            print("-" * 50)
        
        # 5. 清理内存
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("\n内存已清理完成！")
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("请确保在Kaggle环境中运行，并且已安装必要的依赖库")

if __name__ == "__main__":
    main() 