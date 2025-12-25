
import os
import numpy as np
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from RandomAug import RandAugment

import random
class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x
def imshow(img_root):
    img = Image.open(img_root)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    # print(img)
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))

    plt.show()



def dataset_folders():

    domains = ['/devdata/xujian_data/MNIST/raw/ori/train/',
               '/devdata/xujian_data/MNIST_M/train/']

    source_index = 0
    source = domains[source_index]
    target = []
    for i in range(len(domains)):
        if i != source_index:
            target.append(domains[i])

    print('source domain:', source)
    print('target domain:', target)

    train_folders = [os.path.join(source, label) \
                         for label in os.listdir(os.path.join(source)) \
                         if os.path.isdir(os.path.join(source, label)) \
                         ]
    test_folders = []
    for d in target:
        t = [os.path.join( d, label) \
                        for label in os.listdir(d) \
                        if os.path.isdir(os.path.join(d, label)) \
                        ]
        test_folders.append(t)

    return train_folders, test_folders, source, target

class CategoryUtils(object):

    def __init__(self, train_folders):
        self.category = {}
        for class_index in range(len(train_folders)):
            self.category[train_folders[class_index].split('/')[-1]] = class_index

class TrainDataTask(object):

    def __init__(self, train_folders, cate, bs, ite):
        self.train_folders = train_folders

        self.n_train_roots = []
        self.n_train_labels = []
        sample_1cls = train_folders[ite%6]
        sample_1cls_samples = random.sample(os.listdir(sample_1cls), bs*2)
        self.a_train_roots = [os.path.join(sample_1cls, x) for x in sample_1cls_samples[0:bs]]
        self.p_train_roots = [os.path.join(sample_1cls, x) for x in sample_1cls_samples[bs:bs*2]]
        self.a_train_labels = [cate[self.get_class(x)] for x in self.a_train_roots]
        self.p_train_labels = [cate[self.get_class(x)] for x in self.p_train_roots]
        j = 0
        for i in range(bs):
            sample_other_cls = train_folders[j%len(train_folders)]
            if(sample_other_cls == sample_1cls):
                j = j + 1
                sample_other_cls = train_folders[j%len(train_folders)]

            temp1 = [os.path.join(sample_other_cls, x) for x in os.listdir(sample_other_cls)]
            n_sample = random.sample(temp1, 1)
            self.n_train_roots.append(n_sample[0])
            j = j + 1

        # random.shuffle(self.n_train_roots)
        self.n_train_labels = [cate[self.get_class(x)] for x in self.n_train_roots]

    def get_class(self, sample):
        return sample.split('/')[-2]


class TestDataTask(object):

    def __init__(self, test_folders, cate):

        self.test_folders = test_folders
        self.test_roots = []  ####  [ [xxx, xxx, ...], [yyy, yyy, ...], [zzz, zzz, ...]   ]
        self.test_labels = []

        for test_f_i in range(len(test_folders)):
            tr = []
            for t in range(len(test_folders[test_f_i])):
                tr += [os.path.join(test_folders[test_f_i][t], x) for x in os.listdir(test_folders[test_f_i][t])]
            self.test_roots += [tr]
            self.test_labels += [[cate[self.get_class(x)] for x in tr]]

        for i in range(len(self.test_roots)):
            # print('self.test_roots:', i, self.test_roots[i])
            # print('self.test_labels', i, self.test_labels[i])
            print('self.test_roots[i] len :', i,  len(self.test_roots[i]))
            print('self.test_labels[i] len', i ,len(self.test_labels[i]))

    def get_class(self, sample):
        return sample.split('/')[-2]

class TrainDataset(Dataset):
    def __init__(self, task, flag):
        if(flag == 'a'):
            self.image_roots = task.a_train_roots
            self.labels = task.a_train_labels
        elif(flag == 'p'):
            self.image_roots = task.p_train_roots
            self.labels = task.p_train_labels
        elif(flag == 'n'):
            self.image_roots = task.n_train_roots
            self.labels = task.n_train_labels
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.transform = transforms.Compose([
            transforms.Resize([32,32]),
            # transforms.Grayscale(num_output_channels=3),  # 将图像转换为3通道的灰度图
            # transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),  # 随机颜色调整
            # transforms.RandomGrayscale(p=0.1),  # 以 0.1 的概率进行图像灰度化
            transforms.ToTensor(),
            ])
    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image_root, image, label

    def __len__(self):
        return len(self.image_roots)

class TestDataset(Dataset):
    def __init__(self, task, transform, test_id=-1):
        self.image_roots = task.test_roots[test_id]
        self.labels = task.test_labels[test_id]
        self.transform = transform

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image_root, image, label

    def __len__(self):
        return len(self.image_roots)


def get_test_data_loader(task, batch_size=64, shuffle = False, t_id=-1):
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    dataset = TestDataset(task, transform=transforms.Compose([
        transforms.Resize([32,32]),
            transforms.ToTensor(),]), test_id=t_id)
    loader = DataLoader(dataset,  batch_size=batch_size, shuffle=shuffle )
    return loader

def get_train_data_loader(task, flag, bs):
    dataset = TrainDataset(task, flag)
    loader = DataLoader(dataset,  batch_size=bs)
    return loader

if __name__ == '__main__':

    train_folders, test_folders, source, target = dataset_folders()
    print(train_folders)
    print(test_folders)
    print(source)
    print(target)

    category_utils = CategoryUtils(train_folders)
    print(category_utils.category)
    #
    train_dt = TrainDataTask(train_folders, category_utils.category, 32, 0)
    print('1:', train_dt.a_train_roots)
    print('2:',train_dt.p_train_roots)
    print('3:',train_dt.n_train_roots)
    print('4:',train_dt.a_train_labels)
    print('5:',train_dt.p_train_labels)
    print('6:',train_dt.n_train_labels)
    #
    test_dt = TestDataTask(test_folders, category_utils.category)
    test_dl = get_test_data_loader(test_dt, batch_size=32, shuffle=False, t_id=0)

    # for test_data in (test_dl):
    #     t_img_root, t_imgs, t_targets = test_data
    #     print(t_targets)
    # #
    a_train_dl = get_train_data_loader(train_dt, flag='a', bs=32)
    p_train_dl = get_train_data_loader(train_dt, flag='p', bs=32)
    n_train_dl = get_train_data_loader(train_dt, flag='a', bs=32)
    # for train_data in (a_train_dl):
    #     img_root, imgs, targets = train_data
    #     print(img_root)
    #     print(imgs.shape)
    #     print(targets)
    # print(train_folders)
    # print(test_folders)
    # dt = DataTask(train_folders, test_folders)
    #
    # dl = get_data_loader(dt, 'train', batch_size=32, shuffle=True)
    #
    # for eph in range(2):
    #     for data in tqdm(dl):
    #         img_root, imgs, targets = data
