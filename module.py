from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import numbers


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool_2D_1(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.Conv = BasicConv(channel, 2, 1, stride=1, padding=(1 - 1) // 2, relu=False)

    def forward(self, x):
        return torch.cat((torch.mean(x, 1).unsqueeze(1),
                          torch.max(x, 1)[0].unsqueeze(1),
                          self.Conv(x)), dim=1)

class ChannelPool_2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelPool_2D, self).__init__()
        self.Conv = BasicConv(channel, 2, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        self.Conv_weight = nn.Sequential(BasicConv(channel, channel, 1, 1, 0),
                                         nn.Sigmoid())
    def forward(self, x):
        y = x*self.Conv_weight(x)
        return torch.cat((torch.mean(x, 1).unsqueeze(1), torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(y, 1).unsqueeze(1), torch.max(y, 1)[0].unsqueeze(1)), dim=1)

class ChannelPool_1D(nn.Module):  # 空间维度的平均和最大值
    def forward(self, x):
        return torch.cat((x.mean(3).mean(2,keepdim=True), x.max(3)[0].max(2, keepdim=True)[0]), 2)


class CA(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.Conv = BasicConv(4, 4, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.Conv(y)
        return x + x*self.sig(y)

class inplaceCA(nn.Module):
    def __init__(self, channel):
        super(inplaceCA, self).__init__()
        self.Conv_qkv = BasicConv(4, 3*channel, 1, stride=1, padding=(1 - 1) // 2, relu=False)

    def forward(self, x):
        qkv = self.Conv_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return out

class CIG_SA_attention(nn.Module):
    def __init__(self, channel):
        super(CIG_SA_attention, self).__init__()
        self.compress = ChannelPool_2D(channel)
        self.inplaceCA = inplaceCA(channel)
        self.conv_du = nn.Sequential(
            self.inplaceCA,
            BasicConv(128, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.compress(x)
        y = self.conv_du(y)
        return x * y

class inplaceSA(nn.Module):
    def __init__(self, channel):
        super(inplaceSA, self).__init__()
        self.Conv = BasicConv(channel, 1, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.Conv(x)
        x = x * self.sig(y)
        return torch.cat((x.mean(3).mean(2, keepdim=True), x.max(3)[0].max(2, keepdim=True)[0]), 2)


class CIG_CA_attention(nn.Module):
    def __init__(self, channel):
        super(CIG_CA_attention, self).__init__()
        self.compress = ChannelPool_1D()
        self.inplaceSA = inplaceSA(channel)
        self.cat = nn.Sequential(
            BasicConv(channel, 4, 1, stride=1, padding=(1 - 1) // 2, relu=True),
            BasicConv(4, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        )
        self.conv_du = BasicConv(4, 8, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        self.sig = nn.Sequential(BasicConv(4, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False),
                                 nn.Sigmoid())

    def forward(self, x):   # (10, 128, 256, 256)
        a = self.compress(x)
        b = self.inplaceSA(x)
        y = self.cat(torch.cat((a, b), 2).unsqueeze(3))    # (10, 128, 4, 1)
        y = self.conv_du(y.transpose(1, 2))     # (10, 4, 128, 1) -> (10,8,128,1)
        y1, y2 = y.chunk(2, dim=1)
        attn = self.sig(y1 * F.gelu(y2)).transpose(1, 2)

        c = x * attn

        return c


