import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

def channel_matrix_gen(z_score_normalized_data):
    z_score_normalized_data_trans = z_score_normalized_data.transpose(0, 1)
    neiji = torch.mm(z_score_normalized_data_trans, z_score_normalized_data)
    l2_norm = torch.norm(z_score_normalized_data, dim=0).unsqueeze(1)
    l2_norm_transpose = l2_norm.transpose(0, 1)
    l2_matrix = torch.mm(l2_norm, l2_norm_transpose)
    channel_cos_similar_matrix = neiji / (l2_matrix + 1e-8)
    return channel_cos_similar_matrix

import math
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def imshow(imgs, mode='a'):
    # print(img)
    npimg = imgs.cpu().numpy()
    for i in range(npimg.shape[0]):
        plt.figure( )
        plt.imshow(np.transpose(npimg[i],(1,2,0)))
        plt.axis('off')
        # plt.show()
        plt.savefig(os.path.join('/home/xujian/single_dg/savefig/', str(mode)+str(i)+ 'jpg'))
        plt.close()
        plt.clf()
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_distance, neg_distance):
        # 计算损失，确保锚定样本与正样本之间的距离小于锚定样本与负样本之间的距离
        loss = torch.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()


def euclidean_dist_sq(x1, x2, eps=1e-8):
    x1_norm = F.normalize(x1, p=2, dim=-1, eps=eps)  # 归一化后 ||x1_norm||₂ = 1
    x2_norm = F.normalize(x2, p=2, dim=-1, eps=eps)  # 同理

    dist_sq = torch.sum((x1_norm - x2_norm) ** 2, dim=-1)

    return dist_sq
def kl_divergence(p_logits, q_logits):
    p_log_soft = F.log_softmax(p_logits, dim=-1)
    q_log_soft = F.log_softmax(q_logits, dim=-1)
    p_soft = p_log_soft.exp()

    return torch.mean(torch.sum(p_soft * (p_log_soft - q_log_soft), dim=-1))

class MultiLossUncertainty(nn.Module):
    def __init__(self, num_losses=3):
        super().__init__()
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_losses))

    def forward(self, L_list):
        assert len(L_list) == len(self.log_sigma_sq), "Loss number mismatch"
        total_loss = 0.0
        weights = []
        for i, L in enumerate(L_list):
            sigma_sq = torch.exp(self.log_sigma_sq[i])  # σ^2 = exp(log σ^2) >0
            weight = 1.0 / (2.0 * sigma_sq)
            total_loss += weight * L + 0.5 * self.log_sigma_sq[i]  # 对应 paper 中 log σ
            weights.append(weight.item())
        return total_loss, weights