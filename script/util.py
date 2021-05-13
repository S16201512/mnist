#!/usr/bin/env python
# encoding: utf-8
"""
@author: HeWenYong
@contact: 1060667497@qq.com
@software: Pycharm
@file: util.py
@time: 2021/5/13 10:06
"""
import os
import urllib.request
import gzip
import numpy as np
import pickle

"""*******************************获得数据集***************************"""
def load_mnist(file_names, url_base, normalize=True, flatten=True, one_hot_label=False):
    data_path = os.path.dirname(os.path.dirname(__file__))  # 当前文件父路径
    # 1.数据集下载
    for file_name in file_names.values():
        if not os.path.exists(os.path.join(data_path, file_name)):
            _download(data_path, file_name, url_base)
    print("load_mnist finish!")

    # 2.数据集保存为pkl文件
    save_file = data_path + "/mnist.pkl"
    if not os.path.exists(save_file):
        init_mnist(data_path, file_names, save_file)

    # 3.读取数据集
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 4. 类型转换
    dataset['train_img'] = dataset['train_img'].astype(np.float32)  # 类型转换，方便numpy计算
    dataset['test_img'] = dataset['test_img'].astype(np.float32)  # 类型转换，方便numpy计算

    # 5.判断是否进行归一化
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] /= 255.0
        print("normalize finish！")

    # 6.标签one-hot编码
    if one_hot_label:
        for key in ('train_label', 'test_label'):
            dataset[key] = _change_one_hot_label(dataset[key])
        print("one_hot_label finish！")

    # 7.是否把图片展开
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)  # batch, one_channel, height, width
            print("flatten finish!!!")
    print("load_mnist finish!!!")

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


"""**************************下载数据集*****************************"""
def _download(data_path, file_name, url_base):
    """
        dataset_dir: 数据集保存路径
        file_name：下载的文件名
        url_base：数据请求地址
    """
    file_path = data_path + "/" + file_name
    if os.path.exists(file_path):
        return
    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)


def load_img(data_path, file_name):
    """读取图片数据"""
    file_path = os.path.join(data_path, file_name)

    print("Img loading...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    print("Img loading finish!")
    return data


def load_label(data_path, file_name):
    """读取图片标签"""
    file_path = os.path.join(data_path, file_name)

    print("Label loading...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Label loading finish!")
    return labels


def init_mnist(data_path, file_names, save_file):
    """数据初始化"""
    dataset = {}
    dataset['train_img'] = load_img(data_path, file_names['train_img'])
    dataset['train_label'] = load_label(data_path, file_names['train_label'])
    dataset['test_img'] = load_img(data_path, file_names['test_img'])
    dataset['test_label'] = load_label(data_path, file_names['test_label'])

    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)


def _change_one_hot_label(obj):
    """one-hot编码"""
    T = np.zeros((obj.size, 10))
    for idx, row in enumerate(T):
        row[obj[idx]] = 1

    return T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# def softmax(x):
#     x = x - np.max(x)  # 防止溢出
#     exp_x = np.exp(x)
#     return exp_x / np.sum(exp_x)


def cross_entry_error(y, t):
    """ 计算交叉熵 """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.shape)
    # 标签是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
