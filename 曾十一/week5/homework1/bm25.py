##   1. 实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF及BM25两种算法实现）

import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import bm25_code

def load_data(filename):
    # 图书评论信息集合
    book_comments = {}  # {书名: “评论1词 + 评论2词 + ...”}

    with open(filename,'r') as f:
        reader = csv.DictReader(f,delimiter='\t')  # 识别格式文本中标题列
        for item in reader:
            book = item['book']
            comment = item['body']
            comment_words = jieba.lcut(comment)

            if book == '': continue  # 跳过空书名
            
            # 图书评论集合收集
            book_comments[book] = book_comments.get(book,[])
            book_comments[book].extend(comment_words)
    return book_comments

if __name__ == '__main__':

    # 加载停用词列表
    stop_words = [line.strip() for line in open("/mnt/data_1/zfy/4/week5/homework/stopwords.txt", "r", encoding="utf-8")]
    
    # 加载图书评论信息
    book_comments = load_data("/mnt/data_1/zfy/4/week5/homework/douban_comments_fixed.txt")
    print(len(book_comments))

    # 提取书名和评论文本
    book_names = []
    book_comms = []
    for book, comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)
    
    # 构建TF-IDF特征矩阵
    ##vectorizer = TfidfVectorizer(stop_words=stop_words)
    ##tfidf_matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])

    #BM25
    bm25_matrix = bm25_code.bm25(book_comms)
    

    # 计算图书之间的余弦相似度
    similarity_matrix = cosine_similarity(bm25_matrix)

    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input("请输入图书名称：")
    book_idx = book_names.index(book_name)  # 获取图书索引

    # 获取与输入图书最相似的图书
    recommend_book_index = np.argsort(-similarity_matrix[book_idx])[1:11]
    # 输出推荐的图书
    for idx in recommend_book_index:
        print(f"《{book_names[idx]}》 \t 相似度：{similarity_matrix[book_idx][idx]:.4f}")
    print()





