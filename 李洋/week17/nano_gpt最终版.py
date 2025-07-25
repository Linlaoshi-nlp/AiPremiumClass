import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import time

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split, train_data, valid_data, batch_size, block_size):
    data = train_data if split == 'train' else valid_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in indices])
    return x.to(device), y.to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(torch.tril(torch.ones(T, T, device=device))[None, None] == 0, float('-inf'))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=device))== 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差连接
        x = x + self.ffn(self.ln2(x))  # 残差连接
        return x


class ImprovedModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, dropout=0.1, block_size=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # 计算损失
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),label_smoothing=0.1)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 如果序列长度超过模型最大处理长度，截断
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 前向传播
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :] / temperature
            # 可选的top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            # 添加采样的索引到序列
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def train_model(model, train_data, valid_data, config):
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_iter'])

    # 训练循环
    best_val_loss = float('inf')
    model = model.to(device)

    for epoch in tqdm(range(config['max_iter']), desc="Training"):
        # 获取一批训练数据
        xb, yb = get_batch('train', train_data, valid_data,
                           config['batch_size'], config['block_size'])

        # 前向传播
        logits, loss = model(xb, yb)

        # 反向传播和优化
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪
        if config['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        scheduler.step()

        # 评估
        if epoch % config['eval_interval'] == 0 or epoch == config['max_iter'] - 1:
            losses = estimate_loss(model, train_data, valid_data, config)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")

            # 保存最佳模型
            if losses['valid'] < best_val_loss:
                best_val_loss = losses['valid']
                torch.save(model.state_dict(), "./best_model/pytorch_model.bin")
                torch.save({model.token_embed.weight.data,
                            model.pos_embed.weight.data},'./best_model/pytorch_embedding.bin')
                print(f"Saved best model at epoch {epoch} with val loss: {best_val_loss:.4f}")

    return model


@torch.no_grad()
def estimate_loss(model, train_data, valid_data, config):
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(config['eval_num'])
        for k in range(config['eval_num']):
            x, y = get_batch(split, train_data, valid_data,
                             config['batch_size'], config['block_size'])
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    os.makedirs('./best_model', exist_ok=True)
    # 训练配置
    config = {
        'block_size': 128,
        'batch_size': 64,
        'max_iter': 3000,
        'learning_rate': 3e-4,
        'weight_decay': 3e-2,
        'dropout': 0.3,
        'eval_interval': 500,
        'eval_num': 100,
        'grad_clip': 1.0,
        'save_path_model': 'best_model.pth',
        'save_path_embedding': 'best_embedding.pth',
    }

    # 读取数据
    try:
        with open('../网游之命轮之主_命给你行不行.txt', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("数据集文件未找到，请确保文件路径正确。")
        return

    # 构建词汇表
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")

    # 创建编码器/解码器
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 分割数据
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.8 * len(data))
    train_data = data[:n]
    valid_data = data[n:]
    print(f"训练数据大小: {len(train_data)}, 验证数据大小: {len(valid_data)}")

    

    # 初始化模型
    model = ImprovedModel(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        dropout=config['dropout'],
        block_size=config['block_size']
    )
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 训练模型
    start_time = time.time()
    model = train_model(model, train_data, valid_data, config)
    end_time = time.time()
    print(f"训练完成，耗时: {(end_time - start_time) / 60:.2f} 分钟")

    # 加载最佳模型
    model.load_state_dict(torch.load(config['save_path_model']))
    model.eval()

    # 生成文本
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    print("\n生成的文本:")
    print(decode(generated[0].tolist()))


if __name__ == "__main__":
    main()