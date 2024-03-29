import torch.nn as nn
import torch
import cv2
from kornia.filters import filter2d
import random
from module import *

##########################################################################






class OneBlock(nn.Module):  # 卷积层（1*1卷积核 步长1） 归一化 IS块中传入Gradinet的模块

    def __init__(self, in_channels, out_channels):
        super(OneBlock, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class Block(nn.Module):  # 卷积层（3*3卷积核 步长1） 归一化 IS块中传入MainNet的模块
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


# 效仿U-Net模型
class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EBlock, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class Down_Block(nn.Module):  # 下采样
    def __init__(self, in_channels, out_channels):
        super(Down_Block, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class Up_Block(nn.Module):  # 上采样
    def __init__(self, in_channels, out_channels):
        super(Up_Block, self).__init__()
        self.forw = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0,
                               bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x


class Encoder(nn.Module):  # EB   2*（卷积+批归一化+ELU)+(卷积+批归一化+RELU)
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()
        self.en1 = EBlock(in_channel, in_channel)
        self.en2 = EBlock(in_channel, in_channel)
        self.en3 = Down_Block(in_channel, out_channel)

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)

        return e3


class Decoder(nn.Module):  # DB
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()
        self.en1 = Up_Block(in_channel, in_channel)
        self.en2 = EBlock(in_channel, in_channel)
        self.en3 = EBlock(in_channel, out_channel)

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)

        return e3



class attention(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.head = nn.Sequential(
            BasicConv(in_channel, channel, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(channel, channel, 3, stride=1, padding=(3 - 1) // 2, relu=False))
        self.CIG_CA = CIG_CA_attention(channel)  ## Spatial Attention
        self.CIG_SA = CIG_SA_attention(channel)  ## Channel Attention
        self.cat = nn.Conv2d(2*channel, channel, 1, 1, 0)


    def forward(self, x):
        up_a = self.head(x)
        ca_branch = self.CIG_CA(up_a)
        sa_branch = self.CIG_SA(up_a)
        mix = self.cat(torch.cat((ca_branch, sa_branch), 1))
        return mix


class NFAE_Layer(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(NFAE_Layer, self).__init__()

        self.activaton = nn.Sigmoid()

    def forward(self, x):

        var = (x - x.mean(3, keepdim=True).mean(2, keepdim=True)).pow(2)
        spa_ave_var = var.mean(3, keepdim=True).mean(2, keepdim=True)
        cha_ave_var = var.mean(1, keepdim=True)

        y_spa = (10 * var) / (spa_ave_var + 1e-16)
        y_cha = (10 * var) / (cha_ave_var + 1e-16)

        weight_spa = self.activaton(y_spa)
        weight_cha = self.activaton(y_cha)

        weight = weight_spa * weight_cha

        return weight

def dwt_init1(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)



class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(15, 128, 3, 1, 1)   #12+3=15
        self.en1 = Encoder(128, 128)
        self.en2 = Encoder(128, 256)
        self.en3 = Encoder(256, 512)

        self.mid = Block(512, 1024)  # 512-1024
        self.mid1 = Block(1024, 512)

        self.de3 = Decoder(512, 256)
        self.de2 = Decoder(256, 128)
        self.de1 = Decoder(128, 128)
        self.sp = nn.Conv2d(128, 12, 3, 1, 1)


        self.attn = attention(27, 128)
        self.estimator = NFAE_Layer()
        self.en11 = Encoder(128, 128)
        self.en22 = Encoder(128, 256)
        self.en33 = Encoder(256, 512)
        self.mid00 = Block(512, 1024)

        self.mid11 = Block(1024, 512)
        self.de33 = Decoder(512, 256)
        self.de22 = Decoder(256, 128)
        self.de11 = Decoder(128, 128)  # 64
        self.enhance = attention(128, 128)
        self.tail = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)  # 1 for gray

        #self.enhance = map(4, 64)
    def forward(self, x, layer):
        weight = self.estimator(layer)   # (6, 39, 128, 128)
        ###################################### stage1  #######################################
        x_input1 = self.proj(torch.concat([x, layer], dim=1))
        en1 = self.en1(x_input1)
        en2 = self.en2(en1)
        en3 = self.en3(en2)

        mid = self.mid(en3)
        mid1 = self.mid1(mid)

        de3 = self.de3(mid1 + en3)
        de2 = self.de2(de3 + en2)
        de1 = self.de1(de2 + en1)
        sp = self.sp(de1)

        ############################################ stage3 ######################################


        x_input = self.attn(torch.concat([x, sp, x.repeat(1, 4, 1, 1) * weight], dim=1))   # (3+12+39=54)
        en11 = self.en11(x_input)

        en22 = self.en22(en11)
        en33 = self.en33(en22)

        mid00 = self.mid00(en33)
        mid11 = self.mid11(mid00)

        de33 = self.de33(mid11 + en33)
        de22 = self.de22(de33 + en22)
        de11 = self.de11(de22 + en11)
        de11 = self.enhance(de11 + x_input)
        res = self.tail(de11) + x
        return res, sp


import torch.nn.functional as F

from skimage.metrics import structural_similarity
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def sharpen(x):
    kernel1 = torch.tensor([[[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]], requires_grad=False)
    kernel2 = torch.tensor([[[2, 1, 0],
                            [1, 0, -1],
                            [0, -1, -2]]], requires_grad=False)
    kernel3 = torch.tensor([[[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]]], requires_grad=False)
    kernel4 = torch.tensor([[[0, -1, -2],
                            [1, 0, -1],
                            [2, 1, 0]]], requires_grad=False)

    img_sharpen1 = filter2d(x, kernel1)
    img_sharpen2 = filter2d(x, kernel2)
    img_sharpen3 = filter2d(x, kernel3)
    img_sharpen4 = filter2d(x, kernel4)

    return torch.concat([img_sharpen1, img_sharpen2, img_sharpen3, img_sharpen4], dim=1)

def sharpen2(x):
    kernel1 = torch.tensor([[[0,  0, -1, 0,  0],
                             [0, -1, -2, -1, 0],
                             [-1, -2, 16, -2, -1],
                             [0, -1, -2, -1, 0],
                             [0,  0, -1, 0,  0]]], requires_grad=False)

    img_sharpen1 = filter2d(x, kernel1)


    return img_sharpen1




def charbonnier_loss(noisy, target, eps=1e-6):
        diff = torch.add(noisy, -target)
        error = torch.sqrt(diff * diff + eps)
        return torch.mean(error)



def fixed_loss(gt_img, denoised_img, sp):
    gt_sharpen = sharpen2(gt_img)
    res_loss = charbonnier_loss(gt_img, denoised_img)                                  #+SSIM(gt_img, denoised_img)
    gt_edge = sharpen(gt_img)
    edge_loss = F.l1_loss(sp, gt_edge)
    sharpen_loss = charbonnier_loss(gt_sharpen+gt_img, denoised_img)
    return res_loss + edge_loss + 0.1*sharpen_loss


import torch
import torch.nn.functional as F




if __name__ == '__main__':
    inp = torch.randn((6, 15, 256, 256)).cuda()
    #inp = torch.randint(0, 256, (1, 1, 20, 20)).float()


    net = Network().cuda()
    sp, out = net(inp)

    #print(sp.shape)
    # inp = torch.randn((10, 13, 128, 128))
    # net = net()
    import time


    start = time.time()

    #summary(net, (1, 128, 128), batch_size=10, device='cpu')
    # sp, out = net(inp)

    end = time.time()
    print(end - start)



