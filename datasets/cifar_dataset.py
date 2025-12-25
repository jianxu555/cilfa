import torch
import os
import numpy as np
from torch.utils.data import Dataset, TensorDataset
import pickle
def load_cifar10_c_level1(dataroot):
    path = f'data/cifar10_c_level1.pkl'
    if not os.path.exists(path):
        print("genenrating cifar10_c_level1")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[0:10000]
                index = index + 1
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level1")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)
    return dataset
def load_cifar10_c_level2(dataroot):
    path = f'data/cifar10_c_level2.pkl'
    if not os.path.exists(path):
        print("genenrating cifar10_c_level2")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[10000:20000]
                index = index + 1
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level2")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)
    return dataset
def load_cifar10_c_level3(dataroot):
    path = f'data/cifar10_c_level3.pkl'
    if not os.path.exists(path):
        print("generating cifar10_c_level3")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[20000:30000]
                index = index + 1
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level3")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)
    return dataset
def load_cifar10_c_level4(dataroot):
    path = f'data/cifar10_c_level4.pkl'
    if not os.path.exists(path):
        print("genenrating cifar10_c_level4")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[30000:40000]
                index = index + 1
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level4")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)
    return dataset
def load_cifar10_c_level5(dataroot):
    path = f'data/cifar10_c_level5.pkl'
    if not os.path.exists(path):
        print("genenrating cifar10_c_level5")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[40000:50000]
                index = index + 1
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level5")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)
    return dataset
def load_cifar10_c(dataroot):
    y = np.load(os.path.join(dataroot, 'labels.npy'))
    print("y.shape:",y.shape)
    y_single = y[0:10000]
    x1 = torch.zeros((190000,3,32,32))
    x2 = torch.zeros((190000,3,32,32))
    x3 = torch.zeros((190000,3,32,32))
    x4 = torch.zeros((190000,3,32,32))
    x5 = torch.zeros((190000,3,32,32))
    for j in range(19):
        if j == 0:
            y_total = y_single
        else:
            y_total = np.hstack((y_total,y_single))
    print("y_total.shape:",y_total.shape)
    index = 0
    for filename in os.listdir(dataroot):
        if filename=='labels.npy':
            continue
        else:
            x = np.load(os.path.join(dataroot,filename))
            x = x.transpose(0,3,1,2)
            x = torch.tensor(x)
            x = x.float()/255.
            print(x.shape)
            x1[index*10000:(index+1)*10000] = x[0:10000]
            x2[index*10000:(index+1)*10000] = x[10000:20000]
            x3[index*10000:(index+1)*10000] = x[20000:30000]
            x4[index*10000:(index+1)*10000] = x[30000:40000]
            x5[index*10000:(index+1)*10000] = x[40000:50000]
            index = index + 1
    # x1, x2, x3, x4, x5, y_total = torch.tensor(x1), torch.tensor(x2), torch.tensor(x3),\
                                    # torch.tensor(x4),torch.tensor(x5),torch.tensor(y_total)
    y_total = torch.tensor(y_total)
    dataset1 = TensorDataset(x1, y_total)
    dataset2 = TensorDataset(x2, y_total)
    dataset3 = TensorDataset(x3, y_total)
    dataset4 = TensorDataset(x4, y_total)
    dataset5 = TensorDataset(x5, y_total)
    return dataset1,dataset2,dataset3,dataset4,dataset5

def load_cifar10_c_class(dataroot,CORRUPTIONS):
    y = np.load(os.path.join(dataroot, 'labels.npy'))
    y_single = y[0:10000]
    y_single = torch.tensor(y_single)
    print("y.shape:",y.shape)
    x = np.load(os.path.join(dataroot,CORRUPTIONS+'.npy'))
    print("loading data of",os.path.join(dataroot,CORRUPTIONS+'.npy'))
    x = x.transpose(0,3,1,2)
    x = torch.tensor(x)
    x = x.float()/255.
    dataset = []
    for i in range(5):
        x_single = x[i*10000:(i+1)*10000]
        dataset.append(TensorDataset(x_single, y_single))
    return dataset

if __name__ == '__main__':
    dataroot = '/devdata/xujian_data/CIFAR-10-C/'
    ds1, ds2,ds3,ds4,ds5 = load_cifar10_c(dataroot)