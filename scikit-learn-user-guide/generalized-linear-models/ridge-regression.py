# coding: utf-8

# 岭回归

# 时间复杂度为 O(np^2)

'''

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# ############################################################# ################
# Compute paths

n_alphas = 200

# 此参数是一个阀值， 小于这个值的参数需要重视，大于的直接舍弃（或者相反）。为了避免过拟合的情况出现
alphas = np.logspace(-10, -2, n_alphas) # 创建等比数列, 此参数为邻回归函数的超参数

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# 使用交叉验证选取最好的一条模型
# ridge = linear_model.RidgeCV(alphas=alphas)
# ridge.fit(X, y)

# #############################################################################
# Display results

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()