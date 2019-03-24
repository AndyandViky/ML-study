# _*_ coding: utf-8 _*_
# 交叉验证  分别使用训练集和测试集去训练， 之后获取验证的结果进行比较

import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split # using this function splits dataset to get tranData and testData

iris = datasets.load_iris()  # 安德森鸢尾花卉数据集 Anderson’s Iris data set
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, train_size=0.4, random_state=0)

# ====================================================== 标准化数据集
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X_train)
# X_train_std = scaler.fit_transform(X_train)
# X_test_std = scaler.transform(X_test)
#
# clf = svm.SVC(kernel='linear', C=1)
# clf.fit(X_train_std, Y_train) # train model
# score = clf.score(X_test_std, Y_test)

# or do this
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(StandardScaler(), svm.SVC(C=1))
# scores = cross_val_score(clf, iris.data, iris.target, cv=5)
# ======================================================

# ====================================================== 交叉验证
# from sklearn import metrics # 默认情况下，每次CV迭代计算score是估计量的得分方法。可以通过使用评分参数来改变其评分参数：
# from sklearn.model_selection import cross_val_score
#
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro') # 获取交叉验证的得分

# or do this
# from sklearn.model_selection import ShuffleSplit
# n_samples = iris.data.shape[0]
# cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
# scores = cross_val_score(clf, iris.data, iris.target, cv=cv, scoring='f1_macro')
# ======================================================

# ====================================================== 交叉验证2 多度量评估
# from sklearn.metrics import recall_score # 默认情况下，每次CV迭代计算score是估计量的得分方法。可以通过使用评分参数来改变其评分参数：
# from sklearn.model_selection import cross_validate
#
# clf = svm.SVC(kernel='linear', C=1)
# scoring = ['precision_macro', 'recall_macro']
# # or scoring = {'prec_macro': 'precision_macro', 'rec_micro': make_scorer(recall_score, average='macro')}
# scores = cross_validate(clf, iris.data, iris.target, cv=5, scoring=scoring, return_train_score=False) # 获取交叉验证的得分
# ======================================================

# ====================================================== 交叉验证3 预测得分
# from sklearn.model_selection import cross_val_predict
# from sklearn import metrics
# clf = svm.SVC(kernel='linear', C=1)
# predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
# score = metrics.accuracy_score(iris.target, predicted)
# ======================================================

# ====================================================== 交叉验证4 不同条件使用不同的方法
# http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
'''
    1。 假设一些数据是独立的并且是相同分布的（i.i.d.）假设所有的样本来源于相同的生成过程，并且假设生成过程没有记忆过去生成的样本。
    而i.i.d.数据是机器学习理论中的一个常见假设，它在实践中很少成立。如果知道样本是使用时间相关过程生成的，则使用时间序列感知交叉验证方案更安全。
    类似地，如果我们知道生成过程具有组结构（从不同主体收集的样本，实验测量设备）更安全地使用分组交叉验证。（一般不用）
    
    2。 KFold将样本组中的所有样本（称为折叠）划分为相同大小（如果可能）。使用折叠学习预测函数，并且省略折叠用于测试。
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> X = ["a", "b", "c", "d"]
    >>> kf = KFold(n_splits=2)
    >>> for train, test in kf.split(X):
    ...     print("%s %s" % (train, test))
    [2 3] [0 1]
    [0 1] [2 3]
    
    3。 Repeated K-Fold 重复K次折叠n次。当需要运行KFold n次时，可以使用它，在每次重复中产生不同的分割。
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> random_state = 12883823
    >>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
    >>> for train, test in rkf.split(X):
    ...     print("%s %s" % (train, test))
    ...
    [2 3] [0 1]
    [0 1] [2 3]
    [0 2] [1 3]
    [1 3] [0 2]
    
    4。  Leave One Out（但是在交叉验证的k值不是很大的情况下此方法的计算量更高） 是一个简单的交叉验证。每个学习集合都是通过除了一个样本以外的所有样本创建的，因此，对于样本，我们有不同的训练集和不同的测试集。
    这种交叉验证程序不会浪费太多数据，因为只有一个样本从训练集中删除：
    >>> from sklearn.model_selection import LeaveOneOut
    >>> X = [1, 2, 3, 4]
    >>> loo = LeaveOneOut()
    >>> for train, test in loo.split(X):
    ...     print("%s %s" % (train, test))
    [1 2 3] [0]
    [0 2 3] [1]
    [0 1 3] [2]
    [0 1 2] [3]
    
    5。 Leave P Out 与 Leave One Out相似， 去除p个样本
    >>> from sklearn.model_selection import LeavePOut

    >>> X = np.ones(4)
    >>> lpo = LeavePOut(p=2)
    >>> for train, test in lpo.split(X):
    ...     print("%s %s" % (train, test))
    [2 3] [0 1]
    [1 3] [0 2]
    [1 2] [0 3]
    [0 3] [1 2]
    [0 2] [1 3]
    [0 1] [2 3]
    
    6。 （和k-fold一样好用）随机排列交叉验证a.k.a. ShuffleSplit 将样品随机打乱， 再切割。通过显设置random_state伪随机数发生器，可以控制结果再现性的随机性。
    >>> from sklearn.model_selection import ShuffleSplit
    >>> X = np.arange(5)
    >>> ss = ShuffleSplit(n_splits=3, test_size=0.25,
    ...     random_state=0)
    >>> for train_index, test_index in ss.split(X):
    ...     print("%s %s" % (train_index, test_index))
    ...
    [1 3 4] [2 0]
    [1 4 3] [0 2]
    [4 0 2] [1 3]
    
    7. Cross-validation iterators for grouped data 对于特定领域的分组数据的交叉验证 。。。
    
    在这种情况下，我们想知道一个特定组的训练模型是否能很好地适用于看不见的组。为了衡量这一点，
    我们需要确保验证对象中的所有样本来自在配对训练折叠中完全没有表现出来的组。
    
        1。 Group k-fold
        2。 Leave One Group Out
        3。 Leave P Groups Out
        4。 Group Shuffle Split
        
    8。 Cross validation of time series data  交叉验证时间序列数据
    
    时间序列数据的特点是时间上接近的观测值（自相关）。然而，经典的交叉验证技术，如KFold和ShuffleSplit假设样本是独立的且分布相同的，
    并且会导致训练和测试实例之间不合理的相关性（产生广义误差估计值很差）与时间序列数据有关。因此，评估我们关于“未来”观测的时间序列数据的模型非常重要，
    至少像用于训练模型的数据那样。
    >>> from sklearn.model_selection import TimeSeriesSplit

    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit(n_splits=3)
    >>> print(tscv)  
    TimeSeriesSplit(max_train_size=None, n_splits=3)
    >>> for train, test in tscv.split(X):
    ...     print("%s %s" % (train, test))
    [0 1 2] [3]
    [0 1 2 3] [4]
    [0 1 2 3 4] [5]
    

'''


# ======================================================
pass