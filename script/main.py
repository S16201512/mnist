#!/usr/bin/env python
# encoding: utf-8
"""
@author: HeWenYong
@contact: 1060667497@qq.com
@software: Pycharm
@file: main.py
@time: 2021/5/13 10:00
"""

import numpy as np
from util import load_mnist
from network import TwoLayerNet
import matplotlib.pyplot as plt
# 数据集路径
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}


# 读取数据
(x_train, t_train), (x_test, t_test) = load_mnist(key_file, url_base,
                                                  normalize=True, flatten=True, one_hot_label=True)

# 参数设置
input_size = 784
hidden_size = 50
output_size = 10
network = TwoLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)  # 网络结构初始化

# 获取训练数据规模
train_size = x_train.shape[0]
# 定义训练批次大小
batch_size = 100
# 计算一个epoch所需的训练迭代次数（一个epoch定义为所有训练数据都遍历过一次所需的迭代次数）
iter_per_epoch = max(train_size / batch_size, 1)
# 定义训练循环迭代次数
epoch_num = 100
# 定义学习率
learning_rate = 0.1

# 创建记录模型训练损失值的列表
train_loss_list = []
# 创建记录模型在训练数据集上预测精度的列表
train_acc_list = []
# 创建记录模型在测试数据集上预测精度的列表
test_acc_list = []


# 开始训练
for i in range(int(epoch_num)):
    for j in range(int(iter_per_epoch)):
        # 在每次训练迭代内部选择一个批次的数据
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 请补充计算梯度的代码
        grads = network.gradient(x_batch, t_batch)
        # 请补充更新模型参数的代码
        for key in network.params.keys():
            network.params[key] -= learning_rate * grads[key]

        # 请补充向train_loss_list列表添加本轮迭代的模型损失值的代码
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if j % iter_per_epoch == 0:
            # 完成了一个epoch，即所有训练数据都遍历完一遍
            # 请补充向train_acc_list列表添加当前模型对于训练集预测精度的代码
            train_acc = network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            # 请补充向test_acc_list列表添加当前模型对于测试集预测精度的代码
            test_acc = network.accuracy(x_test, t_test)
            test_acc_list.append(test_acc)

            # 输出一个epoch完成后模型分别在训练集和测试集上的预测精度以及损失值
            print("iteration:{} ,train acc:{}, test acc:{} ,loss:{}|".format(i, train_acc, test_acc, loss))

print(train_acc_list)
print(test_acc_list)
x_dim = np.array(np.arange(0, len(train_loss_list)))
plt.plot(x_dim, train_loss_list, label='train line')
# plt.plot(x_dim, test_acc_list, color='red', linewidth=1.0, linestyle='--', label='test line')
# plt.legend(handles=[train_line, test_line], labels=['train', 'test'], loc='best')
plt.xlabel('iteration')
plt.ylabel('train Loss')

# 设置坐标轴刻度
my_x_ticks = np.arange(0, len(train_loss_list), 20)
my_y_ticks = np.arange(0, 2, 0.1)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()



