# -*- coding: utf-8 -*-

import graphlab

datas = graphlab.SFrame.read_csv('./article.csv')
datas['word_count'] = graphlab.text_analytics.count_words(datas['content'])
tfidf = graphlab.text_analytics.tf_idf(datas['word_count'])
datas['tfidf'] = tfidf # 将计算出来的tfidf赋给语料库
knn_model = graphlab.nearest_neighbors.create(datas, features=['tfidf'], label='id')

knn_model.save('knn_model')

# result = knn_model.query(datas[datas['name'] == 'Barack Obama'], k=10)
# print(result)