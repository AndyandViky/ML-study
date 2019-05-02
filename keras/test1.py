from keras import Convolution2D, Dense, models, MaxPooling2D, Flatten
import numpy as np

model = models.Sequential()
# Convolution2D 第一个参数代表25个filter，第二和第三代表filter形状为3*3
# input_shape表示输入图片的形状28*28，第三个1表示为黑白图片，若为彩色图片为3
model.add(Convolution2D(25, 3, 3, input_shape=(28, 28, 1)))

# 其中参数代表选取的矩阵大小
model.add(MaxPooling2D((2, 2)))

# filter的数目一般越来越多
model.add(Convolution2D(50, 3, 3))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

# 全连接
model.add(Dense(output_dim=100, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))

