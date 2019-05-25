# encoding: utf-8
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: neural-network.py
@time: 2019/5/24 下午7:13
@desc: 使用pytorch创建一个简单的神经网络
'''

import torch
import torch.nn as nn # 神经网络包
import torch.nn.functional as F
import torch.optim as optim # 激活函数包

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        new_features = 1
        for s in size:
            new_features *= s
        return new_features

if __name__ == '__main__':
    # test
    net = Net()
    # print(net)

    # params = list(net.parameters())
    # print(len(params))
    # print(params[0].size()) # conv1's .weight

    input = torch.randn(1, 1, 32, 32) # 从正态分布中随机抽取
    # print(input)
    # out = net(input)
    # print(out)

    # net.zero_grad() # 使用随机梯度将所有参数和反向支持的梯度缓冲区归零
    # out.backward(torch.randn(1, 10))

    '''
    torch.nn仅支持 mini-batch， 因此为输入只有一个 sample 时使用 input.unsqueeze(0) 添加假的 sample
    '''

    # -----------------Loss-----------------
    # Loss
    # output = net(input)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()
    # loss = criterion(output, target)
    # print(loss)

    # net.zero_grad()
    # print('conv1.bias.grad before backward')
    # print(net.conv1.bias.grad)
    #
    # loss.backward()
    #
    # print('conv1.bias.grad after backward')
    # print(net.conv1.bias.grad)
    # ----------------------------------


    # Update the weights
    # ----简单python代码实现weight更新----
    # params = net.parameters()
    # learning_rate = 0.01
    # for f in params:
    #     f.data.sub_(f.grad.data * learning_rate)
    # ----------------------------------

    # ----use optim 包----
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step() # Does the update
    # ----------------------------------















