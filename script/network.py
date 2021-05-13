#!/usr/bin/env python
# encoding: utf-8
"""
@author: HeWenYong
@contact: 1060667497@qq.com
@software: Pycharm
@file: network.py
@time: 2021/5/13 10:06
"""
import numpy as np
from util import sigmoid
from util import softmax
from util import cross_entry_error

class TwoLayerNet:
    # 模型初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        # 第一层的权重
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 第一层的偏置
        self.params['b1'] = np.zeros(hidden_size)
        # 请添加网络第二层的权重和偏置的初始化代码
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 模型前向传播过程
    def forward(self, x):
        # 请从参数字典获取网络参数
        w1, b1 = self.params['W1'], self.params['b1']
        w2, b2 = self.params['W2'], self.params['b2']

        # 实现第一层的运算
        z1 = np.dot(x, w1) + b1
        h1 = sigmoid(z1)
        # 请实现第二层的运算
        z2 = np.dot(h1, w2) + b2
        return softmax(z2)

    # 定义损失函数
    def loss(self, x, t):  # x:输入数据, t:监督数据
        y = self.forward(x)
        # 请补充计算并返回损失函数值的代码
        return cross_entry_error(y, t)

    # 计算预测的准确率
    def accuracy(self, x, t):  # 假定输入的数据x和标签t都是mini-batch形式的
        # 请补充实现模型前向输出的代码
        y = self.forward(x)
        # 请补充提取模型预测类别和标签真实类别的代码
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        # 请补充计算并返回模型类别预测准确率的代码
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 求梯度
    def gradient(self, x, t):
        # 请从参数字典获取网络参数
        w1, b1 = self.params['W1'], self.params['b1']
        w2, b2 = self.params['W2'], self.params['b2']
        # 保存梯度结果
        grads = {}

        # forward
        a1 = np.dot(x, w1) + b1
        h1 = sigmoid(a1)
        a2 = np.dot(h1, w2) + b2
        output = softmax(a2)

        # backward
        dy = (output - t) / x.shape[0]
        grads['W2'] = np.dot(h1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        """
            grads['b2'] = np.sum(dy, axis=0),为什么求和？
                - 首先输出为多少维度，那么b就是多少维的向量，和样本数量无关
            因为正向传播过程中，偏置b向量会分别加到每一个样本数据上，因此只需把这些值加起来就好
            也就是说：第一个样本产生由于b产生误差 dy1
                    第二个样本产生由于b产生误差 dy2
                    ...
                    b产生的总误差为: dy1 + dy2 + ...      
        """
        da1 = np.dot(dy, w2.T)
        ha1 = sigmoid(a1)
        dz1 = (1.0 - ha1) * ha1 * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
