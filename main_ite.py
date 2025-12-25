
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
        MAX_AVG_ACC = 0.0

        model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        f_extractor = FeatureExtractor(model)
        clsfier = Classfier(dim=args.causal_dim, class_num=args.CLASS_NUM)
        generator = Generator(f_d=args.causal_dim, z_d=args.C_DIM, gau_num=args.K, hd=args.GEN_HID)
        # generator = Generator1(f_d=causal_dim, z_d=C_DIM, hd=GEN_HID)

        multi_loss = MultiLossUncertainty(num_losses=3)
        loss_weight_optimizer = torch.optim.Adam([multi_loss.log_sigma_sq], lr=1e-3)

        # f_extractor.apply(weights_init)
        clsfier.apply(weights_init)
        generator.apply(weights_init)
        # causal_clsfier.apply(weights_init)

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
        print('category:', category_utils.category)
        test_dt = pacs.TestDataTask(test_folders, category_utils.category)
        MAX_ACC = []
        for i in range(len(target_dm)):
            MAX_ACC.append(0)
        bin_ent = nn.BCEWithLogitsLoss().cuda(args.GPU)
        cross_ent = nn.CrossEntropyLoss().cuda(args.GPU)

        train_dt = pacs.DataTask1(train_folders, category_utils.category, args.TRAIN_BATCH_SIZE)
        dataset = pacs.MyDataset1(train_dt, category_utils.category)
        dataloader = DataLoader(dataset,  batch_size=32, shuffle=True, num_workers=8)
        for ite in range(args.MAX_ITE):
            f_extractor.train()
            clsfier.train()
            generator.train()

            train_dt = pacs.TrainDataTask(train_folders, category_utils.category, args.TRAIN_BATCH_SIZE)
            a_train_dl = pacs.get_train_data_loader(train_dt, flag='a', bs=args.TRAIN_BATCH_SIZE*args.CLASS_NUM)
            p_style_train_dl = pacs.get_train_data_loader(train_dt, flag='p_style', bs=args.TRAIN_BATCH_SIZE*args.CLASS_NUM)
            p_semi_train_dl = pacs.get_train_data_loader(train_dt, flag='p_semi', bs=args.TRAIN_BATCH_SIZE*args.CLASS_NUM)

            a_roots, a_imgs, a_labels = next(iter(a_train_dl))
            p_style_roots, p_style_imgs, p_style_labels = next(iter(p_style_train_dl))
            p_semi_roots, p_semi_imgs, p_semi_labels = next(iter(p_semi_train_dl))

            # imshow(p_style_imgs, mode='ptyle')
            # imshow(p_semi_imgs, mode='psemi')
            # imshow(a_imgs, mode='a')
            # imshow(p_imgs, 'p')

            # aug_a_imgs , aug_p_imgs = foureis(a_roots, p_roots)

            # a_imgs = torch.cat((a_imgs, aug_a_imgs), 0)
            # p_imgs = torch.cat((p_imgs, aug_p_imgs), 0)

            a_imgs = a_imgs.cuda(args.GPU)
            a_labels = a_labels.cuda(args.GPU)
            p_style_imgs = p_style_imgs.cuda(args.GPU)
            p_style_labels = p_style_labels.cuda(args.GPU)
            p_semi_imgs = p_semi_imgs.cuda(args.GPU)
            p_semi_labels = p_semi_labels.cuda(args.GPU)

            macs_total = 0

            a_conv_out, a_avg_out = f_extractor(a_imgs) # [bs, 512, 8, 8]  [bs, 2048],
            # macs, params = profile(f_extractor, inputs=(a_imgs,))
            # macs_total += macs
            p_style_conv_out, p_style_avg_out = f_extractor(p_style_imgs) # [bs, 512, 8, 8]  [bs, 2048],
            # macs, params = profile(f_extractor, inputs=(p_style_imgs))
            # macs_total += macs

            p_semi_conv_out, p_semi_avg_out = f_extractor(p_semi_imgs) # [bs, 512, 8, 8]  [bs, 2048],
            # macs, params = profile(f_extractor, inputs=(p_semi_imgs))
            # macs_total += macs


            a_yes_conv_out = a_conv_out[:, 0:args.causal_dim]
            a_no_conv_out = a_conv_out[:, args.causal_dim:args.noncausal_dim]
            p_style_yes_conv_out = p_style_conv_out[:, 0:args.causal_dim]
            p_style_no_conv_out = p_style_conv_out[:, args.causal_dim:args.noncausal_dim]
            p_semi_yes_conv_out = p_semi_conv_out[:, 0:args.causal_dim]
            p_semi_no_conv_out = p_semi_conv_out[:, args.causal_dim:args.noncausal_dim]

            a_yes_avg_out = a_avg_out[:, 0:args.causal_dim]
            a_no_avg_out = a_avg_out[:, args.causal_dim:args.noncausal_dim]
            p_style_yes_avg_out = p_style_avg_out[:, 0:args.causal_dim]
            p_style_no_avg_out = p_style_avg_out[:, args.causal_dim:args.noncausal_dim]
            p_semi_yes_avg_out = p_semi_avg_out[:, 0:args.causal_dim]
            p_semi_no_avg_out = p_semi_avg_out[:, args.causal_dim:args.noncausal_dim]

            pred_no_out = clsfier(torch.cat((a_no_avg_out, p_style_no_avg_out, p_semi_no_avg_out), 0))

           ###############  L_cls_init
            # L_cls_init = cross_ent(clsfier(torch.cat((a_yes_avg_out, p_style_yes_avg_out, p_semi_yes_avg_out), 0)),
            #                             torch.cat((a_labels, p_style_labels, p_semi_labels), 0))
            # + bin_ent(pred_no_out, torch.ones_like(pred_no_out) )
            num_classes = pred_no_out.shape[-1]
            uniform_target = torch.ones_like(pred_no_out) / num_classes
            # L_cls_init =  cross_ent(clsfier(p_style_yes_avg_out), p_style_labels) + bin_ent(pred_no_out, torch.zeros_like(pred_no_out) )

            # L_cls_init =  cross_ent(clsfier(torch.cat((a_yes_avg_out, p_style_yes_avg_out, p_semi_yes_avg_out), 0)), torch.cat((a_labels, p_style_labels, p_semi_labels), 0)) + F.kl_div(
            #                                                                             F.log_softmax(pred_no_out, dim=-1),
            #                                                                             uniform_target,
            #                                                                             reduction="sum"
            #                                                                         )
            combined_feat = torch.cat((a_yes_avg_out, p_style_yes_avg_out, p_semi_yes_avg_out), 0)
            combined_labels = torch.cat((a_labels, p_style_labels, p_semi_labels), 0)
            N = combined_feat.shape[0]
            # 生成随机打乱的索引
            perm = torch.randperm(N, device=combined_feat.device)

            # 用同样的 perm 打乱 pred_intervened 和 all_labels
            combined_feat_shuffled = combined_feat[perm]
            combined_labels_shuffled = combined_labels[perm]
            L_cls_init =  cross_ent(clsfier(combined_feat_shuffled), combined_labels_shuffled) + 100*F.kl_div(
                                                                                        F.log_softmax(pred_no_out, dim=-1),
                                                                                        uniform_target,
                                                                                        reduction="batchmean"
                                                                                    )

            yes_feat = torch.cat(
                [a_yes_avg_out, p_style_yes_avg_out, p_semi_yes_avg_out], dim=0
            )
            no_feat = torch.cat(
                [a_no_avg_out, p_style_no_avg_out, p_semi_no_avg_out], dim=0
            )

            yes_norm = F.normalize(yes_feat, dim=1)
            no_norm = F.normalize(no_feat, dim=1)

            ###############  L_ind_init
            L_ind_init = (yes_norm * no_norm).sum(dim=1).mean(dim=0)  # [3B]

            gen_a_avg_outs_ag, gen_a_avg_outs_ap = generator(a_yes_avg_out,
                                       p_style_yes_avg_out,
                                       p_semi_yes_avg_out,
                                        times=args.GEN_TIMES)


            gen_non_a_avg_outs_ag, gen_non_a_avg_outs_ap = generator(a_no_avg_out,
                                            p_style_no_avg_out,
                                            p_semi_no_avg_out,
                                             times=args.GEN_TIMES)
            pred_no_out_aug = clsfier(torch.cat((gen_non_a_avg_outs_ag, gen_non_a_avg_outs_ap), 0))
            num_classes = pred_no_out_aug.shape[-1]
            uniform_target = torch.ones_like(pred_no_out_aug) / num_classes
            ###############  L_cls_aug
            # L_cls_aug = (cross_ent(clsfier(torch.cat((gen_a_avg_outs_ag, gen_a_avg_outs_ap), 0)), a_labels.repeat(GEN_TIMES*2)) +
            #  bin_ent(pred_no_out_aug, torch.zeros_like(pred_no_out_aug) ))

            combined_feat = torch.cat((gen_a_avg_outs_ag, gen_a_avg_outs_ap), 0)
            combined_labels = a_labels.repeat(args.GEN_TIMES*2)
            N = combined_feat.shape[0]
            perm = torch.randperm(N, device=combined_feat.device)

            combined_feat_shuffled = combined_feat[perm]
            combined_labels_shuffled = combined_labels[perm]

            L_cls_aug = (cross_ent(clsfier(combined_feat_shuffled), combined_labels_shuffled) +
                         100*F.kl_div(
                             F.log_softmax(pred_no_out_aug, dim=-1),
                             uniform_target,
                             reduction="batchmean"
                         ))

            aug_yes_feat = torch.cat(
                [gen_a_avg_outs_ag, gen_a_avg_outs_ap], dim=0
            )
            aug_no_feat = torch.cat(
                [gen_non_a_avg_outs_ag, gen_non_a_avg_outs_ap], dim=0
            )
            aug_yes_norm = F.normalize(yes_feat, dim=1)
            aug_no_norm = F.normalize(no_feat, dim=1)
            L_ind_aug = (aug_yes_norm * aug_no_norm).sum(dim=1).mean(dim=0)
            cos = nn.CosineSimilarity(dim=1)
            sim_yes_ag = cos(a_yes_avg_out.repeat(args.GEN_TIMES, 1), gen_a_avg_outs_ag)
            sim_no_ag = cos(a_no_avg_out.repeat(args.GEN_TIMES, 1), gen_non_a_avg_outs_ag)

            dist_yes_ag = 1 - sim_yes_ag
            dist_no_ag = 1 - sim_no_ag

            loss_ag = torch.mean(dist_yes_ag) + torch.mean(F.relu(dist_yes_ag - dist_no_ag - args.delta))

            # AP 距离
            sim_yes_ap = cos(a_yes_avg_out.repeat(args.GEN_TIMES, 1), gen_a_avg_outs_ap)
            sim_no_ap = cos(a_no_avg_out.repeat(args.GEN_TIMES, 1), gen_non_a_avg_outs_ap)

            dist_yes_ap = 1 - sim_yes_ap
            dist_no_ap = 1 - sim_no_ap

            loss_ap = torch.mean(dist_yes_ap) + torch.mean(F.relu(dist_yes_ap - dist_no_ap - delta))

            L_aug = loss_ag + loss_ap

            all_no_feat = torch.cat([no_feat, aug_no_feat], dim=0)
            all_yes_feat = torch.cat([yes_feat, aug_yes_feat], dim=0)

            # all_labels = a_labels.repeat(all_yes_feat.shape[0] // a_labels.shape[0])

            shuffled_no_feat = all_no_feat[torch.randperm(all_no_feat.shape[0])]  # 打乱后维度不变

            # combine_feat = torch.cat((all_yes_feat, shuffled_no_feat ), 1)
            # all_labels = a_labels.repeat(all_yes_feat.shape[0] // a_labels.shape[0])
            # pred_intervened = clsfier(combine_feat, prevent=True)  # 干预后的预测：[N_c*N_b, num_classes]
            # pred_causal = clsfier(all_yes_feat)
            # # L_int = cross_ent(pred_intervened, all_labels) + 100*kl_divergence(pred_intervened, pred_causal)
            # L_int = cross_ent(pred_intervened, all_labels)
            N_c = all_yes_feat.shape[0]
            N_b = shuffled_no_feat.shape[0]
            yes_feat_expanded = all_yes_feat.unsqueeze(1).repeat(1, N_b, 1)  # [N_c, N_b, feat_dim_c]
            no_feat_expanded = shuffled_no_feat.unsqueeze(0).repeat(N_c, 1, 1)  # [N_c, N_b, feat_dim_b]

            combined_feat = torch.cat([yes_feat_expanded, no_feat_expanded], dim=-1)  # [N_c, N_b, feat_dim_c+feat_dim_b]
            combined_feat = combined_feat.reshape(-1, combined_feat.shape[-1])  # 展平为[N_c*N_b, total_dim]

            all_labels = a_labels.repeat(all_yes_feat.shape[0] // a_labels.shape[0]).unsqueeze(1).repeat(1, N_b).reshape(-1)
            pred_intervened = clsfier(combined_feat, prevent=True)  # 干预后的预测：[N_c*N_b, num_classes]


            pred_causal = clsfier(all_yes_feat)  # [N_c, num_classes]
            pred_causal_expanded = pred_causal.unsqueeze(1).repeat(1, N_b, 1).reshape(-1, pred_causal.shape[-1])  # [N_c*N_b, num_classes]
            N = pred_intervened.shape[0]
            perm = torch.randperm(N, device=pred_intervened.device)

            pred_intervened_shuffled = pred_intervened[perm]
            all_labels_shuffled = all_labels[perm]
            pred_causal_expanded_shuffled = pred_causal_expanded[perm]
            ################### L_int
            L_int = cross_ent(pred_intervened_shuffled, all_labels_shuffled) + kl_divergence(pred_intervened_shuffled, pred_causal_expanded_shuffled)


            L_cls = L_cls_init +  L_cls_aug
            L_ind = L_ind_init + L_ind_aug
            L_aug = L_aug
            L_int = L_int


            ### total loss
            loss = L_cls + args.a1*L_ind + args.a2*L_aug + args.a3*L_int

            f_extractor.zero_grad()
            clsfier.zero_grad()
            generator.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(f_extractor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(clsfier.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)

            f_optimizer.step()
            clsfier_optimizer.step()
            gen_optimizer.step()

            f_scheduler.step()
            clsfier_scheduler.step()
            gen_scheduler.step()

def main():
    args = get_args()
    train(args)

if __name__ == '__main__':
    main()