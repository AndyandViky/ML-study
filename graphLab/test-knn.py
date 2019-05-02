# -*- coding: utf-8 -*-

import graphlab

knn_model = graphlab.load_model("knn_model")

datas = graphlab.SFrame('./datasets/people_wiki.gl')
datas['word_count'] = graphlab.text_analytics.count_words(datas['text'])
tfidf = graphlab.text_analytics.tf_idf(datas['word_count'])
datas['tfidf'] = tfidf # 将计算出来的tfidf赋给语料库

result = knn_model.query(datas[datas['name'] == 'Barack Obama'], k=10)
print(result)