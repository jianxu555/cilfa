
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

import PACS_dataset_9p1_gau as pacs
from models.aug_model_93p1_gau import Generator
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from tools.utils import MultiLossUncertainty, channel_matrix_gen,weights_init,imshow,TripletLoss
from tools.utils import euclidean_dist_sq, kl_divergence

import argparse
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            *list(model.children())[:-2],  #
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)


    def forward(self, x):
        x = self.conv(x)
        x1 = self.avgpool(x)
        out = x1.view(x1.size(0), -1)
        return x, out


class Classfier(nn.Module):
    def __init__(self, dim, class_num):
        super(Classfier, self).__init__()
        self.class_num = class_num

        self.m = nn.Linear(dim * 2, dim)
        self.fc = nn.Linear(dim, self.class_num)  # num_classes为分类类别数

    def forward(self, x, prevent=False):
        if prevent == True:
            x = self.m(x)
        x = self.fc(x)
        return x




def get_args():
    parser = argparse.ArgumentParser(description='Causal DG Training')
    # training settings
    parser.add_argument('--MAX_ITE', type=int)
    parser.add_argument('--LEARNING_RATE', type=float, default=5e-4)
    parser.add_argument('--LEARNING_RATE_1', type=float, default=1e-3)
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int)
    parser.add_argument('--TEST_BATCH_SIZE', type=int,)
    parser.add_argument('--lr_scheduler', type=str, default='Step')

    # augmentation / generation
    parser.add_argument('--GEN_TIMES', type=int)
    parser.add_argument('--GEN_HID', type=int)

    # loss / hyperparameters
    parser.add_argument('--TRIPLE_MARGIN', type=float)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--K', type=int)
    parser.add_argument('--lamb', type=float)

    parser.add_argument('--a1', type=float)
    parser.add_argument('--a2', type=float)
    parser.add_argument('--a3', type=float)

    # model dimensions
    parser.add_argument('--C_DIM', type=int)
    parser.add_argument('--causal_dim', type=int)
    parser.add_argument('--noncausal_dim', type=int)

    # dataset
    parser.add_argument('--DATA_SET', type=str)
    parser.add_argument('--CLASS_NUM', type=int)

    # gpu
    parser.add_argument('--GPU', type=int)

    return parser.parse_args()

def train(args):
    source_indexs = [2]

    for source_index in source_indexs:

        model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        f_extractor = FeatureExtractor(model)
        clsfier = Classfier(dim=args.causal_dim, class_num=args.CLASS_NUM)
        generator = Generator(f_d=args.causal_dim, z_d=args.C_DIM, gau_num=args.K, hd=args.GEN_HID)

        multi_loss = MultiLossUncertainty(num_losses=3)
        loss_weight_optimizer = torch.optim.Adam([multi_loss.log_sigma_sq], lr=1e-3)

        clsfier.apply(weights_init)
        generator.apply(weights_init)
        f_extractor = f_extractor.cuda(args.GPU)
        clsfier = clsfier.cuda(args.GPU)
        generator = generator.cuda(args.GPU)

        f_optimizer = optim.Adam(f_extractor.parameters(),  lr=args.LEARNING_RATE)
        clsfier_optimizer = optim.Adam(clsfier.parameters(),  lr=args.LEARNING_RATE)
        gen_optimizer = optim.Adam(generator.parameters(),  lr=1e-5)

        if args.lr_scheduler == 'cosine':
            f_scheduler = CosineAnnealingLR(f_optimizer, args.MAX_ITE)
            clsfier_scheduler = CosineAnnealingLR(clsfier_optimizer, args.MAX_ITE)
            gen_scheduler = CosineAnnealingLR(gen_optimizer,   args.MAX_ITE)

        elif args.lr_scheduler == 'Step':
            f_scheduler = StepLR(f_optimizer, step_size=100,gamma=0.5)
            clsfier_scheduler = StepLR(clsfier_optimizer, step_size=100,gamma=0.5)
            gen_scheduler = StepLR(gen_optimizer, step_size=100, gamma=0.5)


        train_folders, test_folders, source_dm, target_dm = pacs.dataset_folders(source_index)
        category_utils = pacs.CategoryUtils(train_folders)
        test_dt = pacs.TestDataTask(test_folders, category_utils.category)
        MAX_ACC = []
        for i in range(len(target_dm)):
            MAX_ACC.append(0)
        bin_ent = nn.BCEWithLogitsLoss().cuda(args.GPU)
        cross_ent = nn.CrossEntropyLoss().cuda(args.GPU)

        train_dt = pacs.DataTask1(train_folders, category_utils.category, args.TRAIN_BATCH_SIZE)
        dataset = pacs.MyDataset1(train_dt, category_utils.category)
        dataloader = DataLoader(dataset,  batch_size=32, shuffle=True, num_workers=8)


        for p in f_extractor.parameters():
            p.requires_grad = False
        for p in clsfier.parameters():
            p.requires_grad = False
        generator.train()

        g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

        WARMUP_STEPS = 10  #
        TEMP = 0.07

        for step in range(WARMUP_STEPS):

            batch = next(iter(dataloader))
            imgs = batch["a_img"].cuda(args.GPU)
            label = batch["label"].cuda(args.GPU)

            with torch.no_grad():
                _, avg_feat = f_extractor(imgs)

            a_causal = avg_feat[:, :args.causal_dim]
            a_noncausal = avg_feat[:, args.causal_dim:args.noncausal_dim]

            # 生成增强特征
            gen_ag, gen_ap = generator(a_causal, a_causal, a_causal, times=1)

            pos_sim = F.cosine_similarity(a_causal, gen_ag).mean()

            sim_matrix = torch.mm(gen_ag, a_causal.t()) / TEMP
            sim_matrix = sim_matrix - torch.eye(sim_matrix.size(0)).cuda(args.GPU) * 1e9
            neg_loss = torch.logsumexp(sim_matrix, dim=1).mean()

            L_reg = F.mse_loss(gen_ag, a_causal)

            L_contrast = -pos_sim + neg_loss

            loss = L_contrast + 0.1 * L_reg

            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()


        # Unfreeze all
        for p in f_extractor.parameters():
            p.requires_grad = True
        for p in clsfier.parameters():
            p.requires_grad = True
        generator.train()

        STAGE1_EPOCHS = 5
        STAGE2_EPOCHS = 5


        for epo in range(1000):

            f_extractor.train()
            clsfier.train()
            generator.train()

            if epo >= STAGE1_EPOCHS + STAGE2_EPOCHS:
                a1, a2, a3 = 1, 1, 1  # Stage3
            elif epo >= STAGE1_EPOCHS:
                a1, a2, a3 = 0.0, 1, 0.0  # Stage2
            else:
                a1, a2, a3 = 0.0, 0.0, 0.0  # Stage1

            for batch in dataloader:

                a_imgs = batch['a_img'].cuda(args.GPU)
                style_imgs = batch['style_img'].cuda(args.GPU)
                semi_imgs = batch['semi_tensor'].cuda(args.GPU)
                label = batch['label'].cuda(args.GPU)
                semi_label = batch['semi_label'].cuda(args.GPU)

                a_conv, a_avg = f_extractor(a_imgs)
                s_conv, s_avg = f_extractor(style_imgs)
                semi_conv, semi_avg = f_extractor(semi_imgs)

                a_yes, a_no = a_avg[:, :args.causal_dim], a_avg[:, args.causal_dim:args.noncausal_dim]
                s_yes, s_no = s_avg[:, :args.causal_dim], s_avg[:, args.causal_dim:args.noncausal_dim]
                semi_yes, semi_no = semi_avg[:, :args.causal_dim], semi_avg[:, args.causal_dim:args.noncausal_dim]

                yes_feat = F.normalize(torch.cat([a_yes, s_yes, semi_yes], dim=0), dim=1)
                no_feat = F.normalize(torch.cat([a_no, s_no, semi_no], dim=0), dim=1)

                pred_yes = clsfier(yes_feat)
                pred_no = clsfier(no_feat)

                num_classes = pred_no.shape[-1]
                uniform_target = torch.ones_like(pred_no) / num_classes

                L_cls_init = (
                        cross_ent(pred_yes, label.repeat(3))
                        + 100 * F.kl_div(F.log_softmax(pred_no, dim=-1),
                                         uniform_target,
                                         reduction="batchmean")
                )

                if epo >= STAGE1_EPOCHS:
                    gen_yes_ag, gen_yes_ap = generator(a_yes, s_yes, semi_yes, times=args.GEN_TIMES)
                    gen_no_ag, gen_no_ap = generator(a_no, s_no, semi_no, times=args.GEN_TIMES)

                    gen_yes_feat = F.normalize(torch.cat([gen_yes_ag, gen_yes_ap], dim=0), dim=1)
                    gen_no_feat = F.normalize(torch.cat([gen_no_ag, gen_no_ap], dim=0), dim=1)

                    pred_gen_no = clsfier(gen_no_feat)
                    uniform_target = torch.ones_like(pred_gen_no) / num_classes
                    # 分类损失
                    L_cls_aug = (
                            cross_ent(clsfier(gen_yes_feat), label.repeat(args.GEN_TIMES * 2))
                            + 100 * F.kl_div(F.log_softmax(pred_gen_no, dim=-1),
                                             uniform_target,
                                             reduction="batchmean")
                    )

                    dist_yes_ag = euclidean_dist_sq(a_yes.repeat(args.GEN_TIMES, 1), gen_yes_ag)
                    dist_no_ag = euclidean_dist_sq(a_no.repeat(args.GEN_TIMES, 1), gen_no_ag)
                    dist_yes_ap = euclidean_dist_sq(a_yes.repeat(args.GEN_TIMES, 1), gen_yes_ap)
                    dist_no_ap = euclidean_dist_sq(a_no.repeat(args.GEN_TIMES, 1), gen_no_ap)

                    L_aug = (
                            torch.mean(dist_yes_ag)
                            + torch.mean(F.relu(dist_yes_ag - dist_no_ag - args.delta))
                            + torch.mean(dist_yes_ap)
                            + torch.mean(F.relu(dist_yes_ap - dist_no_ap - args.delta))
                    )
                else:
                    L_cls_aug = torch.tensor(0.0).cuda(args.GPU)
                    L_aug = torch.tensor(0.0).cuda(args.GPU)

                if epo >= STAGE1_EPOCHS + STAGE2_EPOCHS:

                    N_c = yes_feat.shape[0]
                    N_b = no_feat.shape[0]

                    yes_exp = yes_feat.unsqueeze(1).repeat(1, N_b, 1)
                    no_exp = no_feat.unsqueeze(0).repeat(N_c, 1, 1)

                    combined_feat = torch.cat([yes_exp, no_exp], dim=-1).reshape(-1, yes_feat.shape[1] + no_feat.shape[1])
                    all_labels = label.repeat(3).unsqueeze(1).repeat(1, N_b).reshape(-1)

                    pred_intervened = clsfier(combined_feat, prevent=True)
                    pred_causal = clsfier(yes_feat)
                    pred_causal_exp = pred_causal.unsqueeze(1).repeat(1, N_b, 1).reshape(-1, pred_causal.shape[1])

                    L_int = cross_ent(pred_intervened, all_labels) + kl_divergence(pred_intervened, pred_causal_exp)

                    L_ind = (yes_feat * no_feat).sum(dim=1).mean()

                else:
                    L_int = torch.tensor(0.0).cuda(args.GPU)
                    L_ind = torch.tensor(0.0).cuda(args.GPU)

                loss = L_cls_init + L_cls_aug + a1 * L_ind + a2 * L_aug + a3 * L_int

                f_optimizer.zero_grad()
                clsfier_optimizer.zero_grad()
                gen_optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(f_extractor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(clsfier.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)

                f_optimizer.step()
                clsfier_optimizer.step()
                gen_optimizer.step()





def main():
    args = get_args()
    train(args)

if __name__ == '__main__':
    main()