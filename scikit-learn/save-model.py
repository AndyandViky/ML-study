# _*_ coding: utf-8 _*_

# how to save model by pickle

# example

from sklearn.svm import SVC

import numpy as np

X = np.array([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[3,1],[4,1],[5,1],

         [5,2],[6,1],[6,2],[6,3],[6,4],[3,3],[3,4],[3,5],[4,3],[4,4],[4,5]])

Y = np.array([1]*14+[-1]*6)

T = np.array([[0.5, 0.5], [1.5, 2], [3.5, 3.5], [4, 5.5]])

model = SVC(kernel='poly', degree=2, gamma=1, coef0=0, probability=True)

model.fit(X, Y)

result = model.predict(T)

# save to String
# import pickle
# s = pickle.dumps(model)

# save to File
# from sklearn.externals import joblib
# # joblib.dump(model, 'test.pkl')
#
# model = joblib.load('test.pkl')
#
# result = model.predict(T)
