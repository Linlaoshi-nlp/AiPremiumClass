import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json

def read_data(path1,flag):
    """
    读取训练数据返回数据集合
    """

    enc_data = []
    with open(path1,'r',encoding='utf-8') as f:
        # 读取记录行
        lines = f.read().split('\n')

        for line in lines:
            if line == '':
                continue
            
            # 数据清洗
            enc = line.replace(',','').replace('.','').replace('!','').replace('?','')
           
            # 保存
            if(flag==1):
                enc_tks = ['BOS'] + list(enc) + ['EOS']
            else :
                enc_tks = enc.split()
             # 分词
            enc_data.append(enc_tks)
    return enc_data

def get_proc(enc_voc, dec_voc):

    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        """
        批次数据处理并返回
        """
        enc_ids, dec_ids, labels = [],[],[]
        for enc,dec in data:
            # token -> token index
            enc_idx = [enc_voc[tk] for tk in enc]
            dec_idx = [dec_voc[tk] for tk in dec]

            # encoder_input
            enc_ids.append(torch.tensor(enc_idx))
            # decoder_input
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            # label
            labels.append(torch.tensor(dec_idx[1:]))

        
        # 数据转换张量 [batch, max_token_len]
        # 用批次中最长token序列构建张量
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        targets = pad_sequence(labels, batch_first=True)

        # 返回数据都是模型训练和推理的需要
        return enc_input, dec_input, targets

    # 返回回调函数
    return batch_proc    

class Vocabulary:

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        # 字典构建（字符为token、词汇为token）
        no_repeat_tokens = set()
        for cmt in documents:
            no_repeat_tokens.update(list(cmt))  # token list
        # set转换为list，第0个位置添加统一特殊token
        tokens = ['PAD','UNK'] + list(no_repeat_tokens)

        vocab = { tk:i for i, tk in enumerate(tokens)}

        return cls(vocab)

def save_fenci(path1,path2):
    
    enc_data = read_data(path1,0)
    dec_data = read_data(path2,1)
    
    print('enc length', len(enc_data))
    print('dec length', len(dec_data))

    enc_vocab = Vocabulary.from_documents(enc_data)
    dec_vocab = Vocabulary.from_documents(dec_data)

    print('编码器词汇数量', len(enc_vocab.vocab))
    print('解码器词汇数量', len(dec_vocab.vocab))
    
    # 编码+解码（训练样本）
    dataset = list(zip(enc_data, dec_data))
    # Dataloader

    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=get_proc(enc_vocab.vocab, dec_vocab.vocab)   # callback
    )



    # 数据整体json数据集（json）
    with open('week8/encoder.json', 'w', encoding='utf-8') as f:
        json.dump(enc_data, f)  
    
    with open('week8/decoder.json', 'w', encoding='utf-8') as f:
        json.dump(dec_data, f)  

    import pickle
    with open('week8/vocab.bin','wb') as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab),f)


