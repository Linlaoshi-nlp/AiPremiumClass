import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def get_batch(mode):
    data = train_data if mode =='train' else val_data
    #动态获取索引
    ix = torch.randint(len(data) - block_size,(batch_size,))
    # stack 把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推,这个是将多个二维的矩阵平面按照序列的形式堆积，得到3维矩阵
    x = torch.stack([data[i:i+block_size] for i in ix]) #这里是一个batch维度的多个list无法用shape，所以要用stack拼接变为shape=(4,8)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x,y


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, dropout)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # 残差连接
        x = x + self.ffwd(self.ln2(x)) # 残差连接
        return x

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, head_embd, dropout):
        super().__init__()    #nn.ModuleList 把这一串子模块注册到网络里：
        self.heads = nn.ModuleList([Head(n_embd, head_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
             
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_embd, bias=False)
        self.query = nn.Linear(n_embd, head_embd, bias=False)
        self.value = nn.Linear(n_embd, head_embd, bias=False)
        ## register_buffer不可训练的缓存，但随模型保存/移动，注册到内存，详细区别如下：
        # | 名称                | 属于参数？ | 参与反向传播？ | `state_dict` 保存？ | 典型用途 
        # | `nn.Parameter`      | ✅        | ✅           | ✅                | 权重、偏置 
        # | `register_buffer`   | ❌        | ❌           | ✅                | 均值、方差、mask、固定的位置编码等

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x):
        # B, T, C = input_x.shape
        k = self.key(input_x)   # (B, T, head_size)
        q = self.query(input_x) # (B, T, head_size)
        v = self.value(input_x) # (B, T, 16)

        # wei = q @ k.transpose(-2,-1) * C ** -0.5

        # T = wei.shape[-1]
        # tril = torch.tril(torch.ones(T, T, device=wei.device))
        # wei = wei.masked_fill(tril == 0, float('-inf'))
        # wei = wei.softmax(dim=-1)
        # wei = self.dropout(wei)
        
        # out = wei @ v
        # return out

        ##################### flash attention 2 #########################

        # FlashAttention2 只支持bfloat16或float16类型的张量
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16) # 或调用 v.bfloat16()

        with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return attn_output


class BingramLanguageModel(nn.Module):
    
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        # 每个token直接输出的logits值作为下一个token的映射
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx和target都是维度为 (B,T) 的整型tensor
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device), ) # (T,C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx指当前语料集(B,T)中的索引
        for _ in range(max_new_tokens):
            # 限定索引列的取值范围
            idx_cond = idx[:, -block_size:]
            # 推理
            logits, loss = self(idx_cond)
            # 只提取最后一个时间步的结果
            logits = logits[:, -1, :]  # (B,C)
            # 通过softmax转换为概率值
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # 把采样的索引追加在当前解码序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

        
    
    
if __name__ == '__main__':
    
    # 1、定义超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block_size = 8
    batch_size = 32
    max_iter = 5000
    learn_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 32
    eval_interval = 500
    eval_iters = 200
    head_size = 8
    num_layers = 4
    dropout = 0.1
    
    # 2、读取数据
    with open('input.txt','rb') as f:
        text = f.read().decode('utf-8')
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # 3、构建字典、编码器、解码器(都是函数)
    #这里构建的是字典，然后下面编码器可以直接过来搜索
    stoi = {s:i for i,s in enumerate(chars)}
    itos = {i:s for i,s in enumerate(chars)}
    
    #如果不用lambda就得将这俩函数单独定义，用lambda可以直接传 参数
    encoder = lambda s: [stoi[c] for c in s]
    decoder = lambda l: ''.join([itos[i] for i in l])
    
    # 4、将文本转换为数据，定义训练和验证集
    data = torch.tensor(encoder(text),dtype=torch.long)
    
    train_data = data[:int(0.9*len(data))]
    val_data = data[int(0.9*len(data)):]
    
    # 5、定义模型，准备模型训练
    model = BingramLanguageModel(block_size, vocab_size, n_embd, head_size, num_layers, dropout)
    # model.to(device)
    # 6、定义优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=learn_rate)
    
    # 7、定义模型评估，每500轮进行一次评估，每次分别使用训练和验证做200次的评估，然后打印用的损失用均值
    @torch.no_grad() #这个装饰器是为了不计算梯度，等价于with torch.no_grad():
    def estimate_loss():
        out={}
        model.eval()#指定模型为验证模式
        for mode in ['train','val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x,y = get_batch(mode)
                logits,loss = model(x,y)
                losses[k] = loss.item()
            out[mode] = losses.mean()
        model.train()
        return out
    # 8、模型训练 1000轮
    for iter in range(1000):
        if iter % eval_interval == 0:
            loss = estimate_loss()
            print(f"step {iter}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")
            
        xb,yb = get_batch("train")
        
        logits,loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    #模型推理
    idx = torch.zeros((1,1),dtype = torch.long)
    print(decoder(model.generate(idx,max_new_tokens=100)[0].tolist()))