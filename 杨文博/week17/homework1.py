import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

class GPTTrainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.setup_data()
        self.init_model()
        
    def setup_data(self):
        """准备训练数据"""
        if not os.path.exists(self.config['data_file']):
            self.download_and_process_data()
            
        with open(self.config['data_file'], 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 简单的字符级tokenizer
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        # 分割训练和验证集
        n = int(0.9*len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
    
    def download_and_process_data(self):
        """示例数据下载和处理"""
        print("下载示例唐诗数据...")
        # 实际应用中替换为真实数据下载逻辑
        example_poems = [
            "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
            "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"
        ]
        with open(self.config['data_file'], 'w', encoding='utf-8') as f:
            f.write("\n".join(example_poems))
    
    def init_model(self):
        """初始化一个简单的GPT模型"""
        from torch import nn
        class SimpleGPT(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.tok_emb = nn.Embedding(config['vocab_size'], config['n_embd'])
                self.pos_emb = nn.Embedding(config['block_size'], config['n_embd'])
                self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])
                self.ln_f = nn.LayerNorm(config['n_embd'])
                self.head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
                
            def forward(self, idx, targets=None):
                B, T = idx.shape
                tok_emb = self.tok_emb(idx)
                pos_emb = self.pos_emb(torch.arange(T, device=self.device))
                x = tok_emb + pos_emb
                x = self.blocks(x)
                x = self.ln_f(x)
                logits = self.head(x)
                
                if targets is None:
                    loss = None
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                return logits, loss
            
        class Block(nn.Module):
            """Transformer block"""
            def __init__(self, config):
                super().__init__()
                self.ln1 = nn.LayerNorm(config['n_embd'])
                self.ln2 = nn.LayerNorm(config['n_embd'])
                self.attn = nn.MultiheadAttention(config['n_embd'], config['n_head'])
                self.mlp = nn.Sequential(
                    nn.Linear(config['n_embd'], 4 * config['n_embd']),
                    nn.GELU(),
                    nn.Linear(4 * config['n_embd'], config['n_embd']),
                    nn.Dropout(config['dropout']),
                )
                
            def forward(self, x):
                x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
                x = x + self.mlp(self.ln2(x))
                return x
        
        self.model = SimpleGPT(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
    
    def train(self):
        """训练循环"""
        self.model.train()
        for epoch in range(self.config['max_epochs']):
            # 获取一小批数据
            ix = torch.randint(len(self.train_data) - self.config['block_size'], (self.config['batch_size'],))
            x = torch.stack([self.train_data[i:i+self.config['block_size']] for i in ix])
            y = torch.stack([self.train_data[i+1:i+self.config['block_size']+1] for i in ix])
            x, y = x.to(self.device), y.to(self.device)
            
            # 前向传播
            logits, loss = self.model(x, y)
            
            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            if epoch % self.config['log_interval'] == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    def generate(self, max_new_tokens=100):
        """生成文本"""
        self.model.eval()
        start = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.model(start[:, -self.config['block_size']:])
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                start = torch.cat((start, next_token), dim=1)
        
        return ''.join([self.itos[i.item()] for i in start[0]])

class QuantizedModelInference:
    """量化模型推理"""
    def __init__(self, model_name="Qwen/Qwen-1_8B-Chat"):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
    
    def load_model(self):
        """加载量化模型"""
        from transformers import BitsAndBytesConfig
        
        # 4-bit量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"加载量化模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    
    def generate(self, prompt, max_length=100):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class VLLMInference:
    """vLLM高性能推理"""
    def __init__(self, model_name="Qwen/Qwen-1_8B-Chat"):
        self.model_name = model_name
        self.llm = None
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=200,
            stop_token_ids=[151329, 151336]  # Qwen的停止token
        )
    
    def load_model(self):
        """加载vLLM模型"""
        print(f"使用vLLM加载模型: {self.model_name}")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,  # 单GPU
            gpu_memory_utilization=0.8,
            max_num_seqs=32,
            max_model_len=1024
        )
    
    def generate(self, prompts):
        """生成文本"""
        if not self.llm:
            self.load_model()
        
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

def main():
    parser = argparse.ArgumentParser(description="大模型训练与推理实践")
    subparsers = parser.add_subparsers(dest='command')
    
    # nanoGPT训练命令
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data', type=str, default='poems.txt', help='训练数据文件')
    train_parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    
    # 量化推理命令
    quant_parser = subparsers.add_parser('quant')
    quant_parser.add_argument('--model', type=str, default="Qwen/Qwen-1_8B-Chat", help='HuggingFace模型名称')
    quant_parser.add_argument('--prompt', type=str, required=True, help='生成提示词')
    
    # vLLM推理命令
    vllm_parser = subparsers.add_parser('vllm')
    vllm_parser.add_argument('--model', type=str, default="Qwen/Qwen-1_8B-Chat", help='HuggingFace模型名称')
    vllm_parser.add_argument('--prompts', nargs='+', required=True, help='生成提示词列表')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # 训练配置
        config = {
            'data_file': args.data,
            'vocab_size': 0,  # 将在setup_data中设置
            'n_embd': 128,
            'n_head': 4,
            'n_layer': 4,
            'block_size': 64,
            'batch_size': 32,
            'dropout': 0.0,
            'learning_rate': 6e-4,
            'max_epochs': args.epochs,
            'log_interval': 100
        }
        
        trainer = GPTTrainer(config)
        print("开始训练...")
        trainer.train()
        
        print("\n生成示例:")
        print(trainer.generate(max_new_tokens=50))
    
    elif args.command == 'quant':
        print("\n量化模型推理:")
        infer = QuantizedModelInference(args.model)
        result = infer.generate(args.prompt)
        print("生成结果:", result)
    
    elif args.command == 'vllm':
        print("\nvLLM高性能推理:")
        infer = VLLMInference(args.model)
        results = infer.generate(args.prompts)
        for prompt, result in zip(args.prompts, results):
            print(f"提示: {prompt}\n生成: {result}\n")

if __name__ == "__main__":
    main()
