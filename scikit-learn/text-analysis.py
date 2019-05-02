# coding: utf-8

# sklearn 对文本做分析
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count_vector = CountVectorizer()

X_train_counts = count_vector.fit_transform(twenty_train.data) # 拟合归一化数据

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) # 拟合归一化数据

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

clf = SGDClassifier().fit(X_train_tfidf, twenty_train.target)

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
x_twenty_test = tfidf_transformer.transform(count_vector.transform(twenty_test.data))
score = clf.predict(x_twenty_test)

mean = np.mean(score == twenty_test.target)
print(mean)

# ================================== 使用 pipline 简单方便
# from sklearn.pipeline import Pipeline
#
# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', MultinomialNB()),
# ])
#
# model = text_clf.fit(twenty_train.data, twenty_train.target)
#
# result = model.predict(twenty_test.data)
# mean = np.mean(result == twenty_test.target)



