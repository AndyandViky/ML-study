# coding: utf-8

# 普通最小二乘法

# 最小化残差平方和

# 时间复杂度为 O(np^2)

'''
然而，普通最小二乘法的系数估计依赖于模型项的独立性。
当条件相关并且设计矩阵的列具有近似的线性相关性时，设计矩阵变得接近于奇异，结果，最小二乘估计对观测响应中的随机误差高度敏感，产生大的方差。
例如，当没有实验设计收集数据时，就会出现这种多重共线性的情况。

预测精度：这里要处理好这样一个问题，即样本的数量和特征的数量
    n>=p 时，最小二乘回归会有较小的方差
    n约等于p 时，容易产生过拟合
    n 时，最小二乘回归得不到有意义的结果
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


