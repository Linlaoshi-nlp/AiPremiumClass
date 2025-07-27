import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import re


tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files="corpus.txt", vocab_size=50000)
vocab_size = tokenizer.get_vocab_size()

def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    return tokenizer.decode(ids)

def apply_repetition_penalty(logits, penalty=1.2, prev_tokens=None):
    if prev_tokens is not None:
        for token in set(prev_tokens):
            logits[token] /= penalty
    return logits

def postprocess(text):
    # 清理标点空格
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    # 首字母大写
    sentences = text.split('. ')
    sentences = [s.capitalize() for s in sentences]
    return '. '.join(sentences)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, dropout)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.GELU(), 
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, head_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    def __init__(self, n_embd, head_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_embd, bias=False)
        self.query = nn.Linear(n_embd, head_embd, bias=False)
        self.value = nn.Linear(n_embd, head_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x):
        B, T, C = input_x.shape
        k = self.key(input_x)
        q = self.query(input_x)
        v = self.value(input_x)
        

        with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                is_causal=True,
                dropout_p=0.1 if self.training else 0
            )
        return attn_output

class NanoGPT(nn.Module):
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        
        self.temperature = 0.7
        self.top_k = 50

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        prev_tokens = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            logits = apply_repetition_penalty(logits, penalty=1.2, prev_tokens=prev_tokens)

            probs = F.softmax(logits / self.temperature, dim=-1)
            top_probs, top_indices = probs.topk(self.top_k)
            idx_next = torch.multinomial(top_probs, num_samples=1)
            idx_next = top_indices.gather(1, idx_next)
            
            prev_tokens.append(idx_next.item())
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def load_dataset():
    # HuggingFace数据集
    dataset = load_dataset("bookcorpus", split="train")
    return dataset

if __name__ == '__main__':
    # 配置参数
    block_size = 256
    batch_size = 64
    max_iter = 10000
    learn_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 768
    n_head = 12
    n_layer = 12
    dropout = 0.1
    eval_interval = 500
    eval_iters = 100
    
    # 加载数据集
    dataset = load_dataset()
    data = torch.tensor([encode(text) for text in dataset], dtype=torch.long)
    
    # 拆分数据集
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]
    
    # 初始化模型
    model = NanoGPT(block_size, vocab_size, n_embd, n_head, n_layer, dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=1e-2)
    
    # 训练循环
    for iter in range(max_iter):
 
        if iter % 1000 == 0:
            start_text = "人工智能的未来"
            input_ids = torch.tensor([encode(start_text)], device=device)
            generated = model.generate(input_ids, max_new_tokens=100)
            generated_text = decode(generated[0].tolist())
            print(postprocess(generated_text))

      from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# === 量化配置 ===
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# === 模型加载 ===
model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen-1.8B",  # 或 "mistralai/Mistral-7B-v0.1"
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

def generate_text(prompt, max_length=100, temperature=0.8, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # 使用KV缓存加速
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True  # 启用KV缓存
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return postprocess(generated_text)  # 应用后处理

prompt = "人工智能的未来"
print(generate_text(prompt))

from vllm import LLM, SamplingParams
import requests
import concurrent.futures
import time

# === 启动vLLM服务 ===
def start_vllm_server(model_name="qwen/Qwen-1.8B", dtype="half", port=8000):
    import subprocess
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--dtype", dtype,
        "--port", str(port),
        "--api-key", "YOUR_SECURE_API_KEY"
    ]
    return subprocess.Popen(command)

# === 压力测试函数 ===
def stress_test(api_url, prompts, workers=10):
    def send_request(prompt):
        start = time.time()
        response = requests.post(
            f"{api_url}/completions",
            json={
                "model": "qwen",
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7
            },
            headers={"Authorization": "Bearer YOUR_SECURE_API_KEY"}
        )
        latency = (time.time() - start) * 1000
        return {
            "text": response.json()["choices"][0]["text"],
            "latency": latency
        }
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(send_request, prompt) for prompt in prompts]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    return results

# === 性能监控 ===
def monitor_performance(results):
    latencies = [r["latency"] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    throughput = len(results) / (sum(latencies) / 1000)
    
    print(f"平均延迟: {avg_latency:.2f}ms")
    print(f"吞吐量: {throughput:.2f} 请求/秒")
    print(f"P95延迟: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")

if __name__ == "__main__":
    # 启动服务器
    server_process = start_vllm_server()
    time.sleep(120)  # 等待服务器启动
    
    # 准备测试数据
    prompts = ["人工智能的未来", "机器学习的发展", "深度学习应用"] * 20
    
    # 运行压力测试
    results = stress_test("http://localhost:8000/v1", prompts, workers=20)
    
    # 分析性能
    monitor_performance(results)
    
    # 关闭服务器
    server_process.terminate()
