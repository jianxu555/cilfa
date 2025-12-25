import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image,ImageStat
import cv2
from torchvision import transforms

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def DoShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def DoShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
def DoTranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
def DoTranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)
def DoRotate(img, v):  # [-30, 30]
    return img.rotate(v)


def AutoContrast(img, v):
    return PIL.ImageOps.autocontrast(img, v)
def DoAutoContrast(img, v):
    return PIL.ImageOps.autocontrast(img, v)

def Invert(img, _):
    return PIL.ImageOps.invert(img)
def DoInvert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)
def DoEqualize(img, _):
    return PIL.ImageOps.equalize(img)

def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)

def DoFlip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)
def DoSolarize(img, v):  # [0, 256]
    return PIL.ImageOps.solarize(img, v)

def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)
def DoSolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)
def DoPosterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def DoContrast(img, v):
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)

def DoColor(img, v):
    stat =ImageStat.Stat(img)
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def DoBrightness(img, v):  # obtain the brightness of image
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def DoSharpness(img, v):
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img
def DoCutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img

def NoiseSalt(img, noise_rate):
    """增加椒盐噪声
    args:
        noise_rate (float): noise rate
    """
    img_ = np.array(img).copy()
    h, w, c = img_.shape
    signal_pct = 1 - noise_rate
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_rate/2., noise_rate/2.])
    mask = np.repeat(mask, c, axis=2)
    img_[mask == 1] = 255   # 盐噪声
    img_[mask == 2] = 0     # 椒噪声
    return Image.fromarray(img_.astype('uint8'))

def DoNoiseSalt(img, noise_rate):
    """增加椒盐噪声
    args:
        noise_rate (float): noise rate
    """
    img_ = np.array(img).copy()
    h, w, c = img_.shape
    signal_pct = 1 - noise_rate
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_rate/2., noise_rate/2.])
    mask = np.repeat(mask, c, axis=2)
    img_[mask == 1] = 255   # 盐噪声
    img_[mask == 2] = 0     # 椒噪声
    return Image.fromarray(img_.astype('uint8'))
def NoiseGaussian(img, sigma):
    """增加高斯噪声
    传入:
        img   :  原图
        mean  :  均值默认0
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    """
    # 将图片灰度标准化
    img_ = np.array(img).copy()
    img_ = img_ / 255.0
    # 产生高斯 noise
    noise = np.random.normal(0, sigma, img_.shape)
    # 将噪声和图片叠加
    gaussian_out = img_ + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return Image.fromarray(gaussian_out)

def DoNoiseGaussian(img, sigma):
    """增加高斯噪声
    传入:
        img   :  原图
        mean  :  均值默认0
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    """
    # 将图片灰度标准化
    img_ = np.array(img).copy()
    img_ = img_ / 255.0
    # 产生高斯 noise
    noise = np.random.normal(0, sigma, img_.shape)
    # 将噪声和图片叠加
    gaussian_out = img_ + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return Image.fromarray(gaussian_out)

def causal_list(factor_num):  # 16 oeprations and their ranges
    # l = [
    #     # (NoiseSalt,0.0,0.1),
    #     # (NoiseGaussian,0.0,0.1),
    #     # (Rotate, 0, 30),
    #     # (NoiseSalt, 0.0, 0.1),
    #     # (Solarize, 0.0, 0.1),
    #     # (Rotate, 0, 30),
    # ]
    l = [
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (AutoContrast, 0, 100),
        (Invert, 0, 1),
        (Equalize, 0, 1),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Posterize, 0, 4),
        (Contrast, 0.1, 1.9),
        (Color, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (NoiseSalt,0.0,0.1),
        (NoiseGaussian,0.0,0.1),
        (Rotate, 0, 30),
        (Flip, 0, 1),
    ]
    # print(l[:factor_num])
    return l[:factor_num]

class RandAugment_all:
    def __init__(self, m, factor_num, randm=False):
        self.m = m      # [0, 30]
        self.causal_list = causal_list(factor_num)
        # print("---------------------------%d factors-----------------"%(len(self.causal_list)))
        self.randm = randm
        self.factor_num = factor_num

    def __call__(self, img):
        # print("%d factors-----------------"%(len(self.causal_list)))
        factor_choice = np.random.randint(0,2,self.factor_num)
        # ops = random.choices(self.causal_list, k=self.n)
        if self.randm:
            self.m = random.randint(0,30)
        for index, (op, minval, maxval) in enumerate(self.causal_list):
            if factor_choice[index] == 0:
                continue
            else:
                val = (float(self.m) / 30) * float(maxval - minval) + minval
                # print("val:",val)
                img = op(img, val)
        return img