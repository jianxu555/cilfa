import torchvision
from torchvision import datasets, transforms
import gzip
import shutil

import numpy as np
import struct

from PIL import Image
import os
import  torch
# #
# data_file = '/devdata/xujian_data/MNIST/raw/train-images-idx3-ubyte'
# # It's 47040016B, but we should set to 47040000B
# data_file_size = 47040016
# data_file_size = str(data_file_size - 16) + 'B'
# data_buf = open(data_file, 'rb').read()
# magic, numImages, numRows, numColumns = struct.unpack_from(
#     '>IIII', data_buf, 0)
# datas = struct.unpack_from(
#     '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
# datas = np.array(datas).astype(np.uint8).reshape(
#     numImages, 1, numRows, numColumns)
# label_file = '/devdata/xujian_data/MNIST/raw/train-labels-idx1-ubyte'
# # It's 60008B, but we should set to 60000B
# label_file_size = 60008
# label_file_size = str(label_file_size - 8) + 'B'
#
# label_buf = open(label_file, 'rb').read()
#
# magic, numLabels = struct.unpack_from('>II', label_buf, 0)
# labels = struct.unpack_from(
#     '>' + label_file_size, label_buf, struct.calcsize('>II'))
# labels = np.array(labels).astype(np.int64)
#
# datas_root = 'mnist_train'
# if not os.path.exists(datas_root):
#     os.mkdir(datas_root)
#
# for i in range(10):
#     file_name = datas_root + os.sep + str(i)
#     if not os.path.exists(file_name):
#         os.mkdir(file_name)
#
# for ii in range(numLabels):
#     img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
#     label = labels[ii]
#     file_name = datas_root + os.sep + str(label) + os.sep + \
#                 'mnist_train_' + str(ii) + '.png'

# 下载MNIST数据集
# mnist_trainset = datasets.MNIST(root='/devdata/xujian_data/', train=True, download=True, transform=transforms.ToTensor())
# mnist_testset = datasets.MNIST(root='/devdata/xujian_data/', train=False, download=True, transform=transforms.ToTensor())
#
# 下载MNIST-M数据集
# mnist_m_trainset = datasets.MNIST(root='/devdata/xujian_data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))]))
# mnist_m_testset = datasets.MNIST(root='/devdata/xujian_data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))]))
#

#
# # 下载USPS数据集
# usps_trainset = datasets.USPS(root='/devdata/xujian_data/', train=True, download=True, transform=transforms.ToTensor())
# usps_testset = datasets.USPS(root='/devdata/xujian_data/', train=False, download=True, transform=transforms.ToTensor())
#
# # 下载SYN数据集
# syn_trainset = datasets.SYN(root='/devdata/xujian_data/', download=True, transform=transforms.ToTensor())
# syn_testset = datasets.SYN(root='/devdata/xujian_data/', download=True, transform=transforms.ToTensor())




# # 下载SVHN数据集


import torchvision
import torchvision.transforms as transforms

# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 下载和加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='/devdata/xujian_data/CIFAR10/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='/devdata/xujian_data/CIFAR10/', train=False, transform=transform, download=True)


# 创建数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 现在，你可以使用`train_loader`和`test_loader`来访问CIFAR-10数据集的训练和测试数据。