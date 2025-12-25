import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
import torchvision.models as models
import torchvision
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import torch.distributions as dist

GPU = 0
class Encoder(nn.Module):
    def __init__(self, feature_dim=256, encoder_size=[8192], z_dim=32, dropout=0.5, dropout_input=0.0, leak=0.2, gau_num=1):
        super(Encoder, self).__init__()

        self.gau_num = gau_num

        self.first_linear = nn.Linear(feature_dim*2, encoder_size[0])
        self.z_dim = z_dim

        linear = []
        for i in range(len(encoder_size) - 1):
            linear.append(nn.Linear(encoder_size[i], encoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)
        self.final_linear = nn.Linear(encoder_size[-1], z_dim * gau_num)
        self.lrelu = nn.LeakyReLU(leak)
        self.relu = nn.ReLU()
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, reference_features):
        # features = self.dropout_input(features)
        x = torch.cat([features, reference_features], 1)


        x = self.first_linear(x)
        x = self.relu(x)

        x = self.linear(x)

        x = self.final_linear(x)
        # x = self.relu(x)

        # x = x.reshape(-1, self.gau_num, self.z_dim)

        mu = x[:,  :self.z_dim//2]
        logvar = x[:, self.z_dim//2:]
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, feature_dim=256, decoder_size=[8192], z_dim=32, dropout=0.5, leak=0.2):
        super(Decoder, self).__init__()
        self.first_linear = nn.Linear(z_dim//2 + feature_dim, decoder_size[0])
        self.z_dim = z_dim
        #
        linear = []
        for i in range(len(decoder_size) - 1):
            linear.append(nn.Linear(decoder_size[i], decoder_size[i + 1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)

        self.final_linear = nn.Linear(decoder_size[-1], feature_dim)
        self.lrelu = nn.LeakyReLU(leak)
        self.relu = nn.ReLU()
        #
        self.dropout = nn.Dropout(dropout)

    def forward(self, reference_features, code):
        x = torch.cat([reference_features, code], 1)

        x = self.first_linear(x)
        x = self.relu(x)
        x = self.linear(x)

        x = self.final_linear(x)
        # x = self.relu(x)

        return x

class Gen(nn.Module):

    def __init__(self, feature_dim=256,  hidden=512 ):
        super(Gen, self).__init__()
        self.first_linear = nn.Linear(hidden*2, feature_dim)
        self.relu = nn.ReLU()
        #
        # self.dropout = nn.Dropout(dropout)

    def forward(self, h1, h2):
        x = torch.cat((h1, h2), 1)

        x = self.first_linear(x)
        x = self.relu(x)
        # x = self.linear(x)
        #
        # x = self.final_linear(x)
        # x = self.relu(x)

        return x


class Generator1(nn.Module):
    def __init__(self, f_d, z_d, hd=512):
        super(Generator1, self).__init__()
        self.f_d = f_d
        self.z_d = z_d
        self.hd = hd

        # Encoder -> latent
        self.fc_enc = nn.Linear(f_d, hd)
        self.fc_mean = nn.Linear(hd, z_d)
        self.fc_logvar = nn.Linear(hd, z_d)

        # Decoder -> residual, 输入为 z + 原始特征
        self.fc_dec = nn.Linear(z_d + f_d, hd)
        self.fc_res = nn.Linear(hd, f_d * 2)  # 输出 ag/ap residual
        self.norm = nn.LayerNorm(f_d)
        self.relu = nn.ReLU()

    def forward(self, a_feat, s_feat, semi_feat, times=1, noise_scale=1.0):
        B, f_d = a_feat.shape
        a_repeat = a_feat.repeat(times, 1)
        s_repeat = s_feat.repeat(times, 1)
        semi_repeat = semi_feat.repeat(times, 1)
        h = self.relu(self.fc_enc(a_repeat))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + noise_scale * epsilon * std
        dec_input = torch.cat([z, a_repeat], dim=1)
        h_dec = self.relu(self.fc_dec(dec_input))
        residual = self.fc_res(h_dec)
        res_ag, res_ap = residual[:, :f_d], residual[:, f_d:]
        gen_ag = res_ag
        gen_ap = res_ap
        return gen_ag, gen_ap


class Generator(nn.Module):
    def __init__(self, f_d, z_d, gau_num, hd):
        super(Generator,self).__init__()
        self.style_encoder = Encoder(feature_dim=f_d, encoder_size=[1024], z_dim=z_d, dropout=0.5, dropout_input=0.0, leak=0.2, gau_num=gau_num)
        self.semi_encoder = Encoder(feature_dim=f_d, encoder_size=[1024], z_dim=z_d, dropout=0.5, dropout_input=0.0, leak=0.2)
        self.gen = Decoder(feature_dim=f_d, decoder_size=[1024], z_dim=z_d, dropout=0.5, leak=0.2)

    def forward(self, a, pstyle, psemi, times):
        generate_a = None
        stylezs = None
        semizs = None
        gen_aps = None
        gen_ags = None
        style_mean, style_logvar = self.style_encoder(a, pstyle)
        semi_mean, semi_logvar = self.semi_encoder(a, psemi)
        for i in range(times):
            style_z = self.reparameterize(style_mean, style_logvar)
            semi_z = self.reparameterize(semi_mean, semi_logvar)
            if(stylezs == None):
                stylezs = style_z
            else:
                stylezs = torch.cat((stylezs, style_z), 0)
            if(semizs == None):
                semizs = semi_z
            else:
                semizs = torch.cat((semizs, semi_z), 0)
            gen_ap = self.gen(a, style_z)
            gen_ag = self.gen(a, semi_z)
            if (gen_aps == None):
                gen_aps = gen_ap
            else:
                gen_aps = torch.cat((gen_aps, gen_ap), 0)

            if(gen_ags == None):
                gen_ags = gen_ag
            else:
                gen_ags = torch.cat((gen_ags, gen_ag), 0)


        return gen_ags, gen_aps

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean



