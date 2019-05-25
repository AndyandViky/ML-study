# encoding: utf-8
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: utils.py
@time: 2019/5/25 下午7:01
@desc: 存放一些基础工具函数
'''

' a utils module '

from skimage import transform
import numpy as np
import torch

# ---------------图片相关-----------------
# 以下函数在 torchvision.transforms 下有提供
class Rescale(object):
    """
    图片缩放，将图片缩放到指定大小
    output_size： 可接收 int， 和tuple两个类型
    int: 表示将较短的边赋值为output_size, 另一边等比缩放
    tuple：规定长度和宽度
    """

    def __init__(self, output_size, has_landmarks=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.has_landmarks = has_landmarks

    def __call__(self, sample):
        if self.has_landmarks:
            image, landmarks = sample['image'], sample['landmarks']
        else:
            image, landmarks = sample['image'], 0

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """
    图片随机裁剪
    output_size： 可接收 int， 和tuple两个类型
    int: 方形裁剪
    tuple：规定长度和宽度裁剪
    """

    def __init__(self, output_size, has_landmarks=False):
        assert isinstance(output_size, (int, tuple))
        self.has_landmarks = has_landmarks
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if self.has_landmarks:
            image, landmarks = sample['image'], sample['landmarks']
        else:
            image, landmarks = sample['image'], [0, 0]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """
    将ndarrays转换为tensor
    """
    def __int__(self, has_landmarks=False):
        self.has_landmarks = has_landmarks

    def __call__(self, sample):
        if self.has_landmarks:
            image, landmarks = sample['image'], sample['landmarks']
        else:
            image, landmarks = sample['image'], []

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {
            'image': torch.from_numpy(image),
            'landmarks': torch.from_numpy(landmarks)
        }
# ---------------------------------------