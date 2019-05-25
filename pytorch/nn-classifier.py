# encoding: utf-8
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: nn-classifier.py
@time: 2019/5/25 上午8:21
@desc: 使用pytorch创建一个简单的分类器
'''

'''
1 Load and normalizing the CIFAR10 training and test datasets using torchvision
2 Define a Convolutional Neural Network
3 Define a loss function
4 Train the network on the training data
5 Test the network on the test data
'''

import torch
import torch.utils as utils
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root = '../datasets', train = True,
                                        download = True, transform = transform)
trainloader = utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                       download=True, transform=transform)
testloader = utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# -----------------img-show-----------------
# show image
import matplotlib.pyplot as plt
import numpy as np

def img_show(img):
    img = img / 2 + 0.5
    nimg = img.numpy()
    plt.imshow(np.transpose(nimg, (1, 2, 0)))
    plt.imshow(nimg)
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

img_show(torchvision.utils.make_grid(images)) # make_grid的作用是将若干幅图像拼成一幅图像
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------

# -----------------定义网络结构-----------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

























