# -*- coding: utf-8 -*-

import graphlab

knn_model = graphlab.load_model("knn_model")

datas = graphlab.SFrame.read_csv('./article.csv')
word_count = graphlab.text_analytics.count_words(datas['content'])
datas['word_count'] = word_count
tfidf = graphlab.text_analytics.tf_idf(datas['word_count'])
datas['tfidf'] = tfidf # 将计算出来的tfidf赋给语料库

result = knn_model.query(datas[datas['id'] == 21234], k=10)

print result