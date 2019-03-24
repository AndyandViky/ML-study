# coding: utf-8

# use gensim 自然语言处理python常用的的包之一，其中的 word2vector 模块是为了将单词全部映射成向量进行解析（例如计算相似度）word2vector 能够较好地分别出有一定的关联的词语 例如（宾馆， 酒店）

'''
核心概念和简单例子
从宏观来看，gensim提供了一个发现文档语义结构的工具，通过检查词出现的频率。
gensim读取一段语料，输出一个向量，表示文档中的一个词。词向量可以用来训练各种分类器模型。
这三个模型是理解gensim的核心概念，所以接下来依次介绍。同时，会以一个简单例子贯穿讲述。
'''

from gensim import corpora, models

raw_corpus = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

stoplist = set('for a of the and to in'.split(' ')) # 去除词语干扰项
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in raw_corpus] # 将所有的句子分词

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1 # 计算单词出现频次

precessed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(precessed_corpus)

bow_corpus = [dictionary.doc2bow(text) for text in precessed_corpus]

tfidf = models.TfidfModel(bow_corpus)
string = "system minors"
string_bow = dictionary.doc2bow(string.lower().split())
string_tfidf = tfidf[string_bow]
print(string_bow)
print(string_tfidf)