from collections import defaultdict
import math
import jieba
import warnings
import numpy as np
import sys
import os

# 将 utils 目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.handle_file_comment import handle_file
from sklearn.metrics.pairwise import cosine_similarity

# 忽略警告
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.feature_extraction.text"
)


class BookProfile:
    def __init__(self, book_name, author_id, star, time, likenum, text):
        self.book_name = book_name
        self.author_id = author_id
        self.star = star
        self.time = time
        self.likenum = likenum
        self.comments = [" ".join(jieba.lcut(text))]  # 分词处理

    def append(self, text):
        self.comments.append(" ".join(jieba.lcut(text)))  # 分词处理


def contract_comment():
    comments = handle_file()
    current_book = None
    book_profiles = []
    # 处理评论
    for comment in comments:
        if current_book is None or current_book.book_name != comment.book_name:
            current_book = BookProfile(
                comment.book_name,
                comment.author_id,
                comment.star,
                comment.time,
                comment.likenum,
                comment.body,
            )
            book_profiles.append(current_book)
        else:
            current_book.append(comment.body)

    print(f"Total number of books: {len(book_profiles)}")
    print(list(map(lambda x: x.book_name, book_profiles))[:8])
    return book_profiles


# 计算TF-IDF
def calculate_tfidf(doc_words, stop_words=None):
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    return vectorizer.fit_transform(doc_words)  # 稀疏矩阵


# 计算余弦相似度
def calculate_tfidf_similarity(tfidf_matrix, doc_index=0):
    # 通常接受两个二维数组作为输入，用于计算多个向量之间的相似度矩阵
    similarities = cosine_similarity(
        tfidf_matrix[doc_index], tfidf_matrix
    )  # 计算余弦相似度，返回一个稀疏矩阵，表示查询与所有文档之间的相似度
    print("TF-IDF Similarities:", similarities.shape)
    return similarities[0]  # 返回查询与所有文档之间的相似度得分


def run_calculate_tfidf(doc_words, stop_words, doc_index):
    tfidf_matrix = calculate_tfidf(doc_words, stop_words)
    similarities = calculate_tfidf_similarity(tfidf_matrix, doc_index)
    # 获取排序后的索引
    sorted_indices = np.argsort(-similarities)  # 降序排列
    print(similarities.shape, sorted_indices.shape)
    print("TF - IDF 余弦相似度:")
    # 打印前10个相似度最高的文档
    for i in sorted_indices[1:11]:
        print(f"文档-{book_profiles[i].book_name} 的相似度: {similarities[i]:.4f}")


class BM25:
    def __init__(self, doc_words, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_words = doc_words
        self.doc_total = len(doc_words)
        # 计算语料库中每个文档的长度（以单词数量计）
        self.doc_lengths = [len(doc.split()) for doc in doc_words]
        # 计算语料库中所有文档的平均长度
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        # 用于存储每个文档中各个词的词频(TF)
        self.doc_freqs = []
        # 用于存储每个词在语料库中的逆文档频率(IDF)
        self.idf_dicts = {}

        self._initialize()

    def _initialize(self):
        """
        计算每个词的逆文档频率(IDF)，BM25 IDF（用于 BM25 排序模型）
        IDF = log(1 + (N - df + 0.5) / (df + 0.5))
        N 是文档总数，df 是包含该词的文档数量
        这里使用了拉普拉斯平滑来避免除以零的情况，即加 0.5 是为了平滑，避免极值或除零错误
        计算每个词的逆文档频率(IDF)，使用拉普拉斯平滑
        """
        repeat_words = []
        word_df_dicts = defaultdict(int)
        epoch = 1
        for words in self.doc_words:
            frequencies = defaultdict(int)
            for word in words.split():
                frequencies[word] += 1
                repeat_words.append(word)
                # 对于每个词，计算包含该词的文档数量（文档频率 df）
                word_df_dicts[word] = epoch
            epoch += 1
            # 将每个文档的词频字典添加到 self.doc_freqs 列表中
            self.doc_freqs.append(frequencies)

        uniq_words = set(repeat_words)

        for word in uniq_words:
            df_value = word_df_dicts[word]
            # IDF = log(1 + (N - df + 0.5) / (df + 0.5))
            self.idf_dicts[word] = math.log(
                1 + (self.doc_total - df_value + 0.5) / (df_value + 0.5)
            )

    def get_score(self, target_index, index):
        """
        计算每个词的 BM25 分数
        score = IDF * TF * (k1 + 1) / (TF + k1 * (1 - b + b * doc_length / avgdl))
        其中 IDF 是逆文档频率，TF 是词频，k1 和 b 是调节参数
        """
        query = self.doc_words[target_index]
        score = 0

        tf_doc_freq = self.doc_freqs[index]
        doc_length = self.doc_lengths[index]
        for word in query.split():
            if word in tf_doc_freq:
                score += (self.idf_dicts[word] * tf_doc_freq[word] * (self.k1 + 1)) / (
                    tf_doc_freq[word]
                    + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                )
        return score

    def get_scores(self, target_index):
        return [self.get_score(target_index, i) for i in range(len(self.doc_words))]


if __name__ == "__main__":
    stop_words = [
        line.strip()
        for line in open("data/stopwords.txt", encoding="utf-8").readlines()
    ]
    book_profiles = contract_comment()

    doc_words = [" ".join(book.comments) for book in book_profiles]

    book_name = "钢铁是怎样炼成的"
    # book_name = input("Enter the book name to calculate TF-IDF: ")

    book_indexs = list(map(lambda x: x.book_name, book_profiles))
    if book_name not in book_indexs:
        print(f"Book '{book_name}' not found.")
    else:
        target_index = book_indexs.index(book_name)
        print(f"Book '{book_name}' found at index {target_index}")
        run_calculate_tfidf(doc_words, stop_words, target_index)

        # bm25 = BM25(doc_words, k1=1.5, b=0.5)
        # scores = bm25.get_scores(target_index)
        # # 获取排序后的索引
        # sorted_indices = np.argsort(-np.array(scores))  # 降序排列
        # # 使用 BM25 算法计算查询与文档之间的相似度得分。得分越高，说明查询与文档越相似
        # for i in sorted_indices[1:11]:
        #     print(f"Document {book_profiles[i].book_name} score: {scores[i]:.4f}")
