# coding: utf-8

'''
L0 L1 L2 范数
L0 为原向量中为0的个数

L1 指向量中各个元素绝对值之和  L1范数是L0范数的最优凸近似，而且它比L0范数要容易优化求解

# L1范数和L0范数可以实现稀疏，L1因具有比L0更好的优化求解特性而被广泛应用。

L2 为原向量的膜， 也是广泛使用的一个范数

'''

'''
让我们的参数稀疏有什么好处呢？
https://blog.csdn.net/zouxy09/article/details/24971995
1）特征选择(Feature Selection)：

       大家对稀疏规则化趋之若鹜的一个关键原因在于它能实现特征的自动选择。
       一般来说，xi的大部分元素（也就是特征）都是和最终的输出yi没有关系或者不提供任何信息的，
       在最小化目标函数的时候考虑xi这些额外的特征，虽然可以获得更小的训练误差，但在预测新的样本时，
       这些没用的信息反而会被考虑，从而干扰了对正确yi的预测。稀疏规则化算子的引入就是为了完成特征自动选择的光荣使命，
       它会学习地去掉这些没有信息的特征，也就是把这些特征对应的权重置为0。

2）可解释性(Interpretability)：

       另一个青睐于稀疏的理由是，模型更容易解释。例如患某种病的概率是y，
       然后我们收集到的数据x是1000维的，也就是我们需要寻找这1000种因素到底是怎么影响患上这种病的概率的。
       假设我们这个是个回归模型：y=w1*x1+w2*x2+…+w1000*x1000+b（当然了，为了让y限定在[0,1]的范围，
       一般还得加个Logistic函数）。通过学习，如果最后学习到的w*就只有很少的非零元素，例如只有5个非零的wi，
       那么我们就有理由相信，这些对应的特征在患病分析上面提供的信息是巨大的，决策性的。也就是说，患不患这种病只和这5个因素有关，
       那医生就好分析多了。但如果1000个wi都非0，医生面对这1000种因素，累觉不爱。

'''

'''
Lasso是一个估计稀疏系数的线性模型。
它使用坐标下降算法
它在某些情况下很有用，因为它倾向于选择较少参数值的解决方案，
从而有效地减少给定解决方案所依赖的变量数量。

作用： 约束回归系数， 避免造成过拟合
      可以用来做特征的选择器

lasso回归可以适应的情况是： 样本量比较小，但是指标非常多，即小N大P问题。
适用于高维统计，传统的方法无法应对这样的数据。并且lasso可以进行特征选择

# 时间复杂度 O(np^2)
# alpha参数控制估计系数的稀疏程度。

=================选择最优正则化参数====================
# select alpha by cross-validation  
    LassoCV（在高维度的数据集中使用） and LassoLarsCV（在极少量的数据和特征时效果好）
    LassoLarsIC 
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

# #############################################################################
# Generate some sparse data to play with
np.random.seed(42)

n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal(size=n_samples)

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

# #############################################################################
# Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

# #############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()