# encoding: utf-8
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: adversarial-example.py
@time: 2019/5/26 上午11:00
@desc: adversarial-example，生成对抗样本。在现实生活中，往往改动一小部分数据将会对model的结果产生巨大的影响。
那么我们需要将这种微小的变化考虑进model中，增强model的泛化能力。
我们可以自主生成对抗样本，让model基于对抗样本和训练样本学习。
https://undefinedf.github.io/2018/11/02/%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BFGSM/
'''

"""
白盒与黑盒
白盒的意思为：在已经获取机器学习模型内部的所有信息和参数上进行攻击
盒黑的意思为：在神经网络结构为黑箱时，仅通过模型的输入和输出，逆推生成对抗样本。

误分类 和 目标误分类
误分类的意思是：不关心输出的分类是否存在，只要不与原分类相同即可
目标误分类的意思是：规定检测出来的类别需要与给定的一致
"""

"""
Fast Gradient Sign Attack（FGSM）
FGSM是一种简单的对抗样本生成算法，该算法的思想直观来看就是在输入的基础上沿损失函数的梯度方向加入了一定的噪声，使目标模型产生了误判

公式如下：
perturbed_image=image+epsilon∗sign(data_grad)=x+ϵ∗sign(∇xJ(θ,x,y))
"""

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "../datasets/lenet_mnist_model.pth"
use_cuda=True

# ---------------定义网络结构---------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../datasets', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True
)
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

model = Net().to(device)

model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

# -------------定义网络结构结束---------------


# -------------定义对抗图片生成函数---------------
def fgsm_attack(image, epsilon, data_grad):
    """
    获取扰动图片
    :param image: 原始图片
    :param epsilon: 扰动量
    :param data_grad: 损失梯度
    :return:
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
# ---------------------------------------------


# -------------定义测试函数---------------------
def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        # loss
        loss = F.nll_loss(output, target) # 用于多分类的负对数似然损失函数（Negative Log Likelihood) loss(x,label)=−xlabel

        model.zero_grad()

        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
# ---------------------------------------------


# ---------------------开始测试-----------------
accuracies = []
examples = []
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
# ---------------------------------------------

















