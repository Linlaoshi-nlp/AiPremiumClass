import os
import fasttext
import jieba
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def local_save(filename="data/Fasttext/rich_corpus.txt"):
    rich_corpus = [
        "自然语言处理是人工智能领域的重要方向，它研究人与计算机之间如何进行语言交流。",
        "机器学习是一种利用算法从数据中学习模式并进行预测的技术。",
        "深度学习是机器学习的一个分支，使用神经网络模拟人脑处理信息的方式。",
        "在文本挖掘中，词向量能够将词语转换为具有语义的稠密向量表示。",
        "FastText 由 Facebook AI 研究院开发，它能生成更细粒度的词嵌入表示。",
        "使用词向量可以进行情感分析、文本分类和问答系统等任务。",
        "TF-IDF 是一种经典的文本表示方法，但不能捕捉词语之间的语义关系。",
        "Word2Vec 和 FastText 的主要区别在于是否考虑子词结构。",
        "在自然语言处理中，处理同义词和歧义是提升系统准确性的关键。",
        "构建中文语料时需要考虑分词、停用词过滤和简繁体统一。",
        "语料质量直接影响词向量模型的表现，高质量语料能提升模型泛化能力。",
        "词向量的可视化可以帮助我们更好地理解词汇之间的语义分布。",
    ]
    # 保存为训练文件
    with open(filename, "w", encoding="utf-8") as f:
        for line in rich_corpus:
            f.write(line + "\n")


def splitTexts(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    tokenized_lines = []
    for line in lines:
        # 去除行尾的换行符
        line = line.strip()
        # 使用 jieba 进行分词
        tokenized_line = " ".join(jieba.lcut(line))
        tokenized_lines.append(tokenized_line)

    with open(output_file_path, "w", encoding="utf-8") as file:
        for line in tokenized_lines:
            file.write(line + "\n")


def train_unsupervised(filename):
    # 训练模型（Skip-gram 模式）
    model = fasttext.train_unsupervised(
        input=filename,
        dim=300,
        epoch=30,
        lr=0.1,
        ws=5,
        minCount=1,
        model="skipgram",  # 或 "cbow"
    )
    # 查询指定词汇的词向量（长度为 dim）
    vec = model.get_word_vector("旅行")
    print(f"旅行 的词向量前5维：{vec[:5]}")

    # # 保存模型（保存为 .bin 格式）
    # model.save_model("dist/Fasttext/fasttext_model.bin")

    # # 加载模型
    # model = fasttext.load_model("dist/Fasttext/fasttext_model.bin")

    # 查询最相似的词
    similar = model.get_nearest_neighbors("旅行", k=5)
    print("\n与 '旅行' 最相近的词：")
    for sim, word in similar:
        print(f"{word}：相似度 {sim:.4f}")

    print("\n model.words lengths", len(model.words))

    # 词向量可视化
    writer = SummaryWriter(log_dir="runs/fasttext")
    words_data = model.words
    embeddings = []
    for word in words_data:
        embeddings.append(model.get_word_vector(word))

    writer.add_embedding(torch.tensor(np.array(embeddings)), metadata=words_data)
    writer.close()

    print("\n📊 已保存至 TensorBoard。运行：tensorboard --logdir=runs/fasttext")


# 训练词向量 - https://zhuanlan.zhihu.com/p/575814154
if __name__ == "__main__":
    # filename = "data/Fasttext/rich_corpus.txt"
    # local_save(filename)

    filename = "data/Fasttext/the_wandering_earth.txt"
    # 要检查的文件路径
    dist_path = "dist/Fasttext/the_wandering_earth.txt"

    # 判断文件或目录是否存在
    if os.path.exists(dist_path):
        print(f"{dist_path} 存在。")
    else:
        print(f"{dist_path} 不存在。")
        splitTexts(filename, dist_path)

    train_unsupervised(dist_path)
