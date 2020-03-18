import torch
from torch import nn

import resnext_101_32x4d_
from config import resnext_101_32_path


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        net.load_state_dict(torch.load(resnext_101_32_path))  # 加载预训练好的模型参数

        net = list(net.children())  
        # children()与modules()都是返回网络模型里的组成元素，但是children()返回的是最外层的元素，modules()返回的是所有的元素，包括不同级别的子元素。
        self.layer0 = nn.Sequential(*net[:3])  # layer0,layer1,layer2
        self.layer1 = nn.Sequential(*net[3: 5])  # layer3,layer4
        self.layer2 = net[5]  # layer5
        self.layer3 = net[6]  # layer6
        self.layer4 = net[7]  # layer7

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4
