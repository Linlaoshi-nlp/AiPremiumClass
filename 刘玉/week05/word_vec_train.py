import jieba
from gensim.models import Word2Vec
from torch.utils.tensorboard import SummaryWriter
import numpy as np

hot_words = [
    "词向量",
    "人工智能",
    "自然语言处理",
    "推荐系统",
    "搜索引擎",
    "情感分析",
    "命名实体识别",
    "词袋模型",
    "语义搜索",
]
# ✅ 添加高频复合词到词典，避免被切分
for word in hot_words:
    jieba.add_word(word)

# ✅ 丰富语料（18行）
documents = [
    "人工智能正在改变世界，尤其是在医疗、金融和交通等领域。",
    "自然语言处理是人工智能的重要分支，处理语言理解与生成问题。",
    "词向量可以捕捉词与词之间的语义关系。",
    "深度学习模型在语音识别和图像分类中表现卓越。",
    "情感分析可以判断一段文本是正面还是负面。",
    "推荐系统常常使用用户历史行为和内容相似度来做出推荐。",
    "搜索引擎使用词向量提升查询与文档的匹配质量。",
    "知识图谱能够帮助机器更好地理解实体和关系。",
    "大模型如GPT在多种NLP任务中都取得了领先效果。",
    "机器翻译是自然语言处理中的一个核心应用场景。",
    "TF-IDF 是一种衡量关键词重要性的经典方法。",
    "Word2Vec 和 FastText 都可以训练出高质量的词向量。",
    "文本分类是判断文本所属类别的任务，常用于新闻、商品等场景。",
    "命名实体识别用于识别文本中的人名、地名、组织名等实体。",
    "词袋模型简单高效，但不能保留词序和语义信息。",
    "BERT模型能够更好地捕捉上下文语义，实现双向编码。",
    "用户搜索行为中常含有多义词，词向量可以帮助消歧义。",
    "语义搜索比关键词搜索更智能，能理解查询的真实意图。",
]

# ✅ 中文分词
tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]

# ✅ 训练 Word2Vec 模型
model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=20,
)

# ✅ 查询词相似度
print("\n🔍 与 '词向量' 最相近的词：")
if "词向量" in model.wv:
    for word, score in model.wv.most_similar("词向量", topn=5):
        print(f"{word} : {score:.4f}")
else:
    print("词 '词向量' 不在词表中")

print("\n📏 相似度（人工智能 vs 自然语言处理）：")
if "人工智能" in model.wv and "自然语言处理" in model.wv:
    sim = model.wv.similarity("人工智能", "自然语言处理")
    print(f"相似度分数：{sim:.4f}")
else:
    print("词不在词表中")

# ✅ 保存词向量到 TensorBoard Embedding Projector
writer = SummaryWriter(log_dir="runs/word2vec")
words = list(model.wv.index_to_key)[:100]
vectors = [model.wv[word] for word in words]
writer.add_embedding(mat=np.array(vectors), metadata=words, tag="word2vec-chinese")
writer.close()

print("\n📊 已保存至 TensorBoard。运行：tensorboard --logdir=runs/word2vec")
