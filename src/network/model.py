'''
Model structures
'''
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d

from . import resnest


class UpTC(nn.Module):
    '''
    Upscaling | Short connection => conv2d * 2
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class UpLay(nn.Module):
    '''
    Upscaling | Short connection => conv2d * 2
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.conv = nn.Sequential(
            Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutLayer(nn.Module):
    '''
    conv => BN => (activative function)
    '''
    def __init__(self, in_ch, out_ch, ksz=1):
        super(OutLayer, self).__init__()
        self.out_lay = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ksz, padding=ksz//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.out_lay(x)


class ResNeSt50_TC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self._rsnst = resnest.resnest50(pretrained=False)
        self.up = nn.Sequential(
            UpTC(2048, 1024),
            UpTC(1024, 512),
            UpTC(512, 256)
        )
        self.up2 = nn.Sequential(
            UpLay(256, 128),
            UpLay(128, 64)
        )
        self.out = OutLayer(64, 1, ksz=1)

    def forward(self, x):
        m, n = x.shape[2:]
        x = self._rsnst(x)
        x = self.up(x)
        x = self.up2(x)

        if m != x.shape[2] or n != x.shape[3]:
            x = F.interpolate(x, (m, n), mode='bilinear', align_corners=False)
        
        return self.out(x)
