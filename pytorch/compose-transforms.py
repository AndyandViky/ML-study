# encoding: utf-8
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: compose-transforms.py
@time: 2019/5/25 下午8:31
@desc: 使用compose转换图片格式，使用utils中的工具类
'''

import torchvision.transforms as transforms
from utils import Rescale, RandomCrop, ToTensor
from dataclass import MdataSample

data = MdataSample("", "",
                   transform=transforms.Compose([
                       Rescale(256),
                       RandomCrop(224),
                       ToTensor()
                   ]))