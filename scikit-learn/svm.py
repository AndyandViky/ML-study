# -*- coding: utf-8 -*-#
from sklearn.svm import SVC # Using SVC is to classify and SVR is to regress

import matplotlib.pyplot as plt

import numpy as np

X = np.array([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[3,1],[4,1],[5,1],

         [5,2],[6,1],[6,2],[6,3],[6,4],[3,3],[3,4],[3,5],[4,3],[4,4],[4,5]])

Y = np.array([1]*14+[-1]*6)

T = np.array([[0.5, 0.5], [1.5, 1.5], [3.5, 3.5], [4, 5.5]])

# X为训练样本，Y为训练样本标签(1和-1)，T为测试样本

svc = SVC(kernel='poly', degree=2, gamma=1, coef0=0)

svc.fit(X, Y)

pre = svc.predict(T)

print(pre)     #输出预测结果

print(svc.n_support_)   #输出正类和负类支持向量总个数

print(svc.support_)    #输出正类和负类支持向量索引

print(svc.support_vectors_) #输出正类和负类支持向量