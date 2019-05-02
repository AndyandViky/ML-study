# -*- coding: utf-8 -*-

import graphlab

datas = graphlab.SFrame('./datasets/people_wiki.gl')
datas['word_count'] = graphlab.text_analytics.count_words(datas['text'])
tfidf = graphlab.text_analytics.tf_idf(datas['word_count'])
datas['tfidf'] = tfidf # 将计算出来的tfidf赋给语料库
knn_model = graphlab.nearest_neighbors.create(datas, features=['tfidf'], label='name')

result = knn_model.query(datas[datas['name'] == 'Barack Obama'])
print(result)