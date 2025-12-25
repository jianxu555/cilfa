
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
from Augment import RandAugment_all
from tools import autoaugment, causalaugment_v3, randaugment

import random
import time
FACTOR_NUM = 16

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


def dataset_folders(s_id):
    # dateset_dir = '/data/xujian_data/datasets/pacs/images1/'
    # target_dateset_dir = '/data/xujian_data/datasets/pacs/images/'

    #### used for ala
    # dateset_dir = '/data/xujian_data/datasets/pacs/images1/'
    # target_dateset_dir = '/data/xujian_data/datasets/pacs/images/'
    #
    # dateset_dir = '/data/xujian_data/datasets/VLCS_2/'
    # target_dateset_dir = '/data/xujian_data/datasets/VLCS_1/'


    # dateset_dir = '/data/xujian_data/datasets/T/terra_incognita_1/'
    # target_dateset_dir = '/data/xujian_data/datasets/T/terra_incognita'

    # multi source domains
    # dateset_dir = '/data/xujian_data/datasets/T/terra_incognita_2/'
    # target_dateset_dir = '/data/xujian_data/datasets/T/terra_incognita'

    # dateset_dir = '/devdata/xujian_data/'
    #
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    # domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
    # domains = ['location_38', 'location_43', 'location_46', 'location_100']
    # domains = ['location384346', 'location_100']
    # domains = ['MNIST/raw/ori/train/', 'MNIST_M/testing/']

    source_index = s_id
    source = domains[source_index]
    target = []
    for i in range(len(domains)):
        if i != source_index:
            target.append(domains[i])

    print('source domain:', source)
    print('target domain:', target)

    train_folders = [os.path.join(dateset_dir, source, label) \
                         for label in os.listdir(os.path.join(dateset_dir, source)) \
                         if os.path.isdir(os.path.join(dateset_dir, source, label)) \
                         ]

    test_folders = []
    for d in target:
        t = [os.path.join(target_dateset_dir, d, label) \
                        for label in os.listdir(os.path.join(target_dateset_dir, d)) \
                        if os.path.isdir(os.path.join(target_dateset_dir, d, label)) \
                        ]
        test_folders.append(t)

    return train_folders, test_folders, source, target

class CategoryUtils(object):

    def __init__(self, train_folders):
        self.category = {}
        for class_index in range(len(train_folders)):
            self.category[train_folders[class_index].split('/')[-1]] = class_index


class MyDataset1(Dataset):
    def __init__(self, task, cate):
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.task = task
        self.cate = cate
        self.a_image_roots = task.train_roots
        self.a_labels = task.train_labels

        self.tensor_transform = transforms.Compose([
            transforms.ToTensor()])
        ra = RandAugment_all(m=5, factor_num=FACTOR_NUM, randm=True)
        aa = autoaugment()
        CA = causalaugment_v3.MultiCounterfactualAugment(FACTOR_NUM, 5)
        self.aug_transform = transforms.Compose([
            ra,
            transforms.ToTensor()])
        self.aug_transform1 = transforms.Compose([
            CA,
            transforms.ToTensor()])
        # ------------- 构建 label -> indices 映射（正样本采样需要） -----
        self.label_to_indices = {}
        for idx, y in enumerate(self.a_labels):
            if y not in self.label_to_indices:
                self.label_to_indices[y] = []
            self.label_to_indices[y].append(idx)
    def __getitem__(self, idx):
        image_root = self.a_image_roots[idx]
        image = Image.open(image_root).convert('RGB')
        image = image.resize((64, 64))
        label = self.a_labels[idx]
        a_img = self.tensor_transform(image)

        style_img = self.aug_transform(image)
        style_imgv1 = self.aug_transform1(image)
        # ---------------- 找一个正样本（相同 label） ----------------
        same_class_idxs = self.label_to_indices[label]

        # 从同标签下选一个不同的样本
        if len(same_class_idxs) == 1:
            # 只有一个样本，只能用它自己
            pos_idx = same_class_idxs[0]
        else:
            # 随机挑一个不同的
            while True:
                pos_idx = random.choice(same_class_idxs)
                if pos_idx != idx:
                    break
        semi_root = self.a_image_roots[pos_idx]
        semi_img = Image.open(semi_root).convert("RGB")
        semi_img = semi_img.resize((64,64))
        semi_tensor = self.tensor_transform(semi_img)
        # return image_root, a_img, label, semi_root, semi_tensor, style_img
        return {
            "image_root": image_root,
            "a_img": a_img,
            "label": label,
            "semi_root": semi_root,
            "semi_label": self.cate[self.task.get_class(semi_root)],
            "semi_tensor": semi_tensor,
            "style_img": style_img,
            'style_imgv1': style_imgv1,
                }

    def __len__(self):
        return len(self.a_image_roots)

class DataTask1(object):

    def __init__(self, train_folders, cate, bs):
        self.train_folders = train_folders
        self.train_roots = []
        for tf in self.train_folders:
            for file in os.listdir(tf):
                self.train_roots.append(os.path.join(tf, file))
        self.train_labels = [cate[self.get_class(x)] for x in self.train_roots]

    def get_class(self, sample):
        return sample.split('/')[-2]

class TrainDataTask(object):

    def __init__(self, train_folders, cate, bs):
        self.train_folders = train_folders
        self.a_train_roots = []
        self.p_style_train_roots = []
        self.p_semi_train_roots = []

        c_i = 0
        import random

        for i in range(bs):
            c = random.choice(self.train_folders)
            samples = random.sample(os.listdir(c), 1)
            self.a_train_roots += [os.path.join(c, x) for x in samples]
            self.p_style_train_roots += [os.path.join(c, x) for x in samples]
            samples = random.sample(os.listdir(c), 1)
            self.p_semi_train_roots += [os.path.join(c, x) for x in samples]

        self.a_train_roots, self.p_style_train_roots, self.p_semi_train_roots = \
            zip(*random.sample(list(zip(self.a_train_roots, self.p_style_train_roots, self.p_semi_train_roots)),
                               len(self.a_train_roots)))
        # for c in self.train_folders:
        #     # if(c_i != 2):
        #     #     c_i += 1
        #     #     continue
        #     samples = random.sample(os.listdir(c), bs)
        #     self.a_train_roots += [os.path.join(c, x) for x in samples]
        #     self.p_style_train_roots += [os.path.join(c, x) for x in samples]
        #     samples = random.sample(os.listdir(c), bs)
        #     self.p_semi_train_roots += [os.path.join(c, x) for x in samples]
        #     # break

        # seed = time.time()  # 你可以选择不同的种子
        # random.seed(seed)
        # random.shuffle(self.a_train_roots)
        # random.seed(seed)
        # random.shuffle(self.p_style_train_roots)
        # random.seed(seed)
        # random.shuffle(self.p_semi_train_roots)

        self.a_train_labels = [cate[self.get_class(x)] for x in self.a_train_roots]
        self.p_style_train_labels = [cate[self.get_class(x)] for x in self.p_style_train_roots]
        self.p_semi_train_labels = [cate[self.get_class(x)] for x in self.p_semi_train_roots]

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
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if(flag == 'a'):
            self.image_roots = task.a_train_roots
            self.labels = task.a_train_labels
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        elif(flag == 'p_style'):
            self.image_roots = task.p_style_train_roots
            self.labels = task.p_style_train_labels
            ra = RandAugment_all( m=5,factor_num=FACTOR_NUM, randm=True)
            self.transform = transforms.Compose([
                ra,
                transforms.ToTensor()])
        elif(flag == 'p_semi'):
            self.image_roots = task.p_semi_train_roots
            self.labels = task.p_semi_train_labels
            self.transform = transforms.Compose([
                transforms.ToTensor()])


    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root).convert('RGB')

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
        image = Image.open(image_root).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image_root, image, label

    def __len__(self):
        return len(self.image_roots)


def get_test_data_loader(task, batch_size=64, shuffle = False, t_id=-1):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = TestDataset(task, transform=transforms.Compose([transforms.ToTensor()]), test_id=t_id)
    loader = DataLoader(dataset,  batch_size=batch_size, shuffle=shuffle, num_workers=16 )
    return loader

def get_train_data_loader(task, flag, bs):

    # if(flag == 'a' or flag == 'p'):
    #     data_transform = transforms.Compose([
    #         transforms.CenterCrop(224),  # 随机裁剪和调整大小到 224x224
    #         # transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转
    #         # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),  # 随机颜色调整
    #         # transforms.RandomGrayscale(p=0.1),  # 以 0.1 的概率进行图像灰度化
    #     ])
    # else:
    #     data_transform = transforms.Compose([
    #         transforms.CenterCrop(224) # 随机裁剪和调整大小到 224x224
    #     ])

    dataset = TrainDataset(task, flag)
    loader = DataLoader(dataset,  batch_size=bs, num_workers=8)
    return loader

# if __name__ == '__main__':
#
#     train_folders, test_folders, source, target = dataset_folders()
#     print(train_folders)
#     print(test_folders)
#     print(source)
#     print(target)
#
#     category_utils = CategoryUtils(train_folders)
#     print(category_utils.category)
#
#     train_dt = TrainDataTask(train_folders, category_utils.category, 32)
#     print('1:', train_dt.a_train_roots)
#     print('2:',train_dt.p_train_roots)
#     print('3:',train_dt.n_train_roots)
#     print('4:',train_dt.a_train_labels)
#     print('5:',train_dt.p_train_labels)
#     print('6:',train_dt.n_train_labels)
#
#     test_dt = TestDataTask(test_folders, category_utils.category)
#     test_dl = get_test_data_loader(test_dt, batch_size=32, shuffle=False, t_id=0)
#     # for test_data in (test_dl):
#     #     t_img_root, t_imgs, t_targets = test_data
#     #     print(t_targets)
#
#     a_train_dl = get_train_data_loader(train_dt, flag='a', bs=32)
#     p_train_dl = get_train_data_loader(train_dt, flag='p', bs=32)
#     n_train_dl = get_train_data_loader(train_dt, flag='a', bs=32)
#     for train_data in (a_train_dl):
#         img_root, imgs, targets = train_data
#         print(img_root)
#         print(imgs.shape)
#         print(targets)
#     # print(train_folders)
#     # print(test_folders)
#     # dt = DataTask(train_folders, test_folders)
#     #
#     # dl = get_data_loader(dt, 'train', batch_size=32, shuffle=True)
#     #
#     # for eph in range(2):
#     #     for data in tqdm(dl):
#     #         img_root, imgs, targets = data
