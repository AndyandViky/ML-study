# -*- coding: utf-8 -*-
# https://blog.csdn.net/cxmscb/article/details/56277984  博客


# SVC params discrib
# sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
#
# tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)
#
# 参数：
#
# l  C：C-SVC的惩罚参数C?默认值是1.0
#
# C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
#
# l  kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
#
#   　　0 – 线性：u'v
#
#  　　 1 – 多项式：(gamma*u'*v + coef0)^degree
#
#   　　2 – RBF函数：exp(-gamma|u-v|^2)
#
#   　　3 –sigmoid：tanh(gamma*u'*v + coef0)
#
# l  degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
#
# l  gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
#
# l  coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
#
# l  probability ：是否采用概率估计？.默认为False
#
# l  shrinking ：是否采用shrinking heuristic方法，默认为true
#
# l  tol ：停止训练的误差值大小，默认为1e-3
#
# l  cache_size ：核函数cache缓存大小，默认为200
#
# l  class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
#
# l  verbose ：允许冗余输出？
#
# l  max_iter ：最大迭代次数。-1为无限制。
#
# l  decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
#
# l  random_state ：数据洗牌时的种子值，int值
#
# 主要调节的参数有：C、kernel、degree、gamma、coef0。


# example
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
np.random.seed(8) # 保证随机的唯一性

# 线性可分：
# array = np.random.randn(20,2)
# X = np.r_[array-[3,3],array+[3,3]]
# y = [0]*20+[1]*20
#
# # 建立svm模型
# clf = svm.SVC(kernel='linear')
# clf.fit(X,y)
#
# x1_min, x1_max = X[:,0].min(), X[:,0].max(),
# x2_min, x2_max = X[:,1].min(), X[:,1].max(),
# xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
# # 得到向量w  : w_0x_1+w_1x_2+b=0
# w = clf.coef_[0]
# f = w[0]*xx1 + w[1]*xx2 + clf.intercept_[0]+1  # 加1后才可绘制 -1 的等高线 [-1,0,1] + 1 = [0,1,2]
# plt.contour(xx1, xx2, f, [0,1,2], colors = 'r') # 绘制分隔超平面、H1、H2
# plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired)
# plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],color='k') # 绘制支持向量点
# plt.show()


# 非线性可分：
from sklearn import datasets
from sklearn.model_selection import train_test_split
# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.)  # 分割训练集和测试集

from sklearn.preprocessing import StandardScaler # 标准化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.model_selection import GridSearchCV
# 交叉验证，调整参数

param_grid = {'C':[1e1,1e2,1e3, 5e3,1e4,5e4],
              'gamma':[0.0001,0.0008,0.0005,0.008,0.005,]}
clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=10)
clf = clf.fit(X_train_std, y_train)

score = clf.score(X_test_std,y_test)
y_pred = clf.predict(X_test_std)
print(y_pred)
print(y_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# recall表示召回率 = #(True positive) / (#(True positive)+ #(False negative))，表示样本中的正例有多少被预测正确。

# precision表示精确率 = #(True positive) / (#(True positive)+ #(False negative))，表示预测为正的样本中有多少是真正的正样本。

# f1-score（F1指标）表示召回率和精确率两个指标的调和平均数，召回率和精确率越接近,F1指标越高。F1 = 2 / （1/recall + 1/precision）。召回率和精确率差距过大的学习模型，往往没有足够的实用价值。
print(classification_report(y_test,y_pred,target_names=iris.target_names))

print(confusion_matrix(y_test,y_pred,labels=range(iris.target_names.shape[0])))
# 纵坐标表示预测的是谁，横坐标表示标准的是谁。对角线的值越大，预测能力越好。