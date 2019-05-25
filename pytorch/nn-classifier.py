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
import torch.nn.functional as F

# -----------------数据加载----------------
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
# ---------------数据加载结束-----------------


# -----------------img-show-----------------
# show image
import matplotlib.pyplot as plt
import numpy as np
#
def img_show(img):
    img = img / 2 + 0.5
    nimg = img.numpy()
    plt.imshow(np.transpose(nimg, (1, 2, 0)))
    plt.imshow(nimg)
    plt.show()

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# img_show(torchvision.utils.make_grid(images)) # make_grid的作用是将若干幅图像拼成一幅图像
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------


# -----------------定义网络结构-----------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积池化两次
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten
        x = x.view(-1, 16 * 5 * 5)

        # 全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# ---------------定义网络结构结束-----------------


# -----------------定义Loss函数和optimizer函数-----------------
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# ---------------定义Loss函数结束-----------------


# -----------------开始train-----------------
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step() # 更新

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
# -----------------结束train-----------------


# -----------------开始test-----------------
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print('Accuracy of the network on the 10000 test images: %d %%\n' % (
    100 * correct / total))
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
# -----------------结束test-----------------





