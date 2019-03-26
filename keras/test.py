import numpy as np
import gzip
import struct
import keras as ks
import logging
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.utils import np_utils


def read_data(label_url,image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II",flbl.read(8))
        label = np.fromstring(flbl.read(),dtype=np.int8)
    with gzip.open(image_url,'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16))
        image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols)
    return (label, image)


(train_lbl, train_img) = read_data('../MNIST_data/train-labels-idx1-ubyte.gz','../MNIST_data/train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data('../MNIST_data/t10k-labels-idx1-ubyte.gz','../MNIST_data/t10k-images-idx3-ubyte.gz')


def to4d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255


train_img = to4d(train_img)
val_img = to4d(val_img)
#important
train_LBL = np_utils.to_categorical(train_lbl)
val_LBL = np_utils.to_categorical(val_lbl)

model = ks.models.Sequential()
model.add(Dense(input_dim=784, units=500, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(x=train_img,y=train_LBL,batch_size=100,nb_epoch=10,verbose=1,validation_data=(val_img,val_LBL))