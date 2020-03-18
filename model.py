import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from resnext import ResNeXt101
savepath = 'vis_resnet50/'

def Increasing_Dimension(in_):
    out_ = in_.view(in_.size(0)//3, 3, in_.size(1), in_.size(2), in_.size(3)).permute(0, 2, 1, 3, 4)
    return out_

def Reducing_Dimension(in_):
    in_ = in_.permute(0, 2, 1, 3, 4)
    out_ = in_.contiguous().view(in_.size(0)*in_.size(1), in_.size(2), in_.size(3), in_.size(4))
    return out_

def STRF(in_):
    batchsize, num_channels, tem, height, width = in_.data.size()
    in_ = in_.contiguous().view(batchsize, num_channels*tem, height, width)
    channels_per_group = (num_channels*tem) // 64
    in_ = in_.contiguous().view(batchsize, 64, channels_per_group, height, width)
    in_ = torch.transpose(in_, 1, 2).contiguous()
    in_ = in_.contiguous().view(batchsize, num_channels, tem, height, width)
    return in_

class STRM(nn.Module):
    def __init__(self):
        super(STRM, self).__init__()
        self.refineST1 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 1, 1), padding=(0, 0, 0)), nn.BatchNorm3d(64)
        )
        self.refineST2 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 1, 1), padding=(0, 0, 0)), nn.BatchNorm3d(64)
        )
        self.refineST3 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 1, 1), padding=(0, 0, 0)), nn.BatchNorm3d(64)
        )

    def forward(self, in_0):
        in_0 = Increasing_Dimension(in_0)
        in_1 = in_0.repeat(1, 1, 3, 1, 1)
        in_1 = F.relu(STRF(self.refineST1(in_1[:, :, 2:7, :, :]) + in_0), True)
        in_2 = in_1.repeat(1, 1, 3, 1, 1)
        in_2 = F.relu(STRF(self.refineST2(in_2[:, :, 2:7, :, :]) + in_0), True)
        in_3 = in_2.repeat(1, 1, 3, 1, 1)
        in_3 = F.relu(STRF(self.refineST3(in_3[:, :, 2:7, :, :]) + in_0), True)
        in_3 = Reducing_Dimension(in_3)
        return in_3

class _ASPP(nn.Module):
    #  this module is proposed in deeplabv3 and we use it in all of our baselines
    def __init__(self):
        super(_ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * 64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                           align_corners=True)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


class STFM(nn.Module):
    def __init__(self):
        super(STFM, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        # --------------------------------------------------------------------------------
        self.ASPP = _ASPP()
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down0 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU()
        )
        # --------------------------------------------------------------------------------
        self.refine4 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )   
        self.refine3 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.refine0 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.predict0 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predict1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predict2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predict3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predict4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predictA = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        # -------------------------------------------------------------------------------- 
        self.strm0 = STRM()  
        self.strm1 = STRM()
        self.strm2 = STRM()
        self.strm3 = STRM()
        self.strm4 = STRM()
        self.strm4A = STRM()
        # --------------------------------------------------------------------------------
        self.predict0_a = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predict1_a = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predict2_a = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predict3_a = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.predict4_a = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        # --------------------------------------------------------------------------------

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        # --------------------------------------------------------------------------------
        aspp = self.ASPP(layer4)
        # --------------------------------------------------------------------------------
        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)
        down0 = self.down0(layer0)
        # --------------------------------------------------------------------------------
        refine4 = F.relu(self.refine4(torch.cat((aspp, aspp, down4), 1)) + aspp, True)
        refine4_a = F.relu(self.strm4(refine4) + refine4, True)

        refine4 = F.upsample(refine4, size=down3.size()[2:], mode='bilinear')
        refine4_a = F.upsample(refine4_a, size=down3.size()[2:], mode='bilinear')
        aspp = F.upsample(aspp, size=down3.size()[2:], mode='bilinear')
        refine3 = F.relu(self.refine3(torch.cat((refine4, aspp, down3), 1)) + refine4, True)
        refine3_a = F.relu(self.strm3(refine3+refine4_a) + refine3, True)

        refine3 = F.upsample(refine3, size=down2.size()[2:], mode='bilinear')
        refine3_a = F.upsample(refine3_a, size=down2.size()[2:], mode='bilinear')
        aspp = F.upsample(aspp, size=down2.size()[2:], mode='bilinear')
        refine2 = F.relu(self.refine2(torch.cat((refine3, aspp, down2), 1)) + refine3, True)
        refine2_a = F.relu(self.strm2(refine2+refine3_a) + refine2, True)

        refine2 = F.upsample(refine2, size=down1.size()[2:], mode='bilinear')
        refine2_a = F.upsample(refine2_a, size=down1.size()[2:], mode='bilinear')
        aspp = F.upsample(aspp, size=down1.size()[2:], mode='bilinear')
        refine1 = F.relu(self.refine1(torch.cat((refine2, aspp, down1), 1)) + refine2, True)
        refine1_a = F.relu(self.strm1(refine1+refine2_a) + refine1, True)

        refine1 = F.upsample(refine1, size=down0.size()[2:], mode='bilinear')
        refine1_a = F.upsample(refine1_a, size=down0.size()[2:], mode='bilinear')
        aspp = F.upsample(aspp, size=down0.size()[2:], mode='bilinear')
        refine0 = F.relu(self.refine0(torch.cat((refine1, aspp, down0), 1)) + refine1, True)
        refine0_a = F.relu(self.strm0(refine0+refine1_a) + refine0, True)
        # --------------------------------------------------------------------------------
        predict0_a = self.predict0_a(refine0_a)
        predict0 = self.predict0(refine0)
        predict0_a = F.upsample(predict0_a, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict0 = F.upsample(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_fuse = torch.sum(torch.cat((predict0, predict0_a), 1), 1, True)
        # ---------------------------------------------------------------------------------
        return F.sigmoid(predict_fuse)
