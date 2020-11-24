import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from Soundmodel import SoundNet
from VCalssModel import att_Net
import argparse

classes = 27


class att_Model(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_Model, self).__init__()
        Amodel = SoundNet()
        Amodel = torch.load('SingleAModel.pt')
        Amodel = list(Amodel.children())
        self.layerA_0 = Amodel[0]
        self.layerA_1 = Amodel[1]
        self.layerA_2 = Amodel[2]
        self.layerA_3 = Amodel[3]
        self.layerA_4 = Amodel[4]
        self.layerA_p = Amodel[5]

        Vmodel = att_Net()
        Vmodel = torch.load('9001.pt')
        Vmodel = list(Vmodel.children())
        self.layerV_0 = Vmodel[0]
        self.layerV_1 = Vmodel[1]
        self.layerV_2 = Vmodel[2]
        self.layerV_3 = Vmodel[3]
        self.layerV_4 = Vmodel[4]

        self.relu = nn.ReLU()
        self.Aup1 = nn.ConvTranspose2d(
            8192, 64, kernel_size=4, stride=2, padding=0)
        self.Aup2 = nn.ConvTranspose2d(
            64, 64, kernel_size=4, stride=2, padding=0)
        self.Aup3 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=1, padding=0)
        self.layerV_down = nn.Conv2d(2048, 64, 1)
        self.attention = nn.Conv2d(128, 1, 1)
        self.atten_conv = nn.Conv2d(128, 128, 1)
        self.atten_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.atten_fc = nn.Linear(128, classes)

        self.Vatten_conv = nn.Conv2d(2048, 2048, 1)
        self.Vatten_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.Vatten_fc = nn.Linear(2048, classes)

    def forward(self, audio, video):
        layerV_0 = self.layerV_0(video)
        layerV_1 = self.layerV_1(layerV_0)
        layerV_2 = self.layerV_2(layerV_1)
        layerV_3 = self.layerV_3(layerV_2)
        layerV_4 = self.layerV_4(layerV_3)

        layerA_0 = self.layerA_0(audio)
        layerA_1 = self.layerA_1(layerA_0)
        layerA_2 = self.layerA_2(layerA_1)
        layerA_3 = self.layerA_3(layerA_2)
        layerA_4 = self.layerA_4(layerA_3)
        layerA_p = self.layerA_p(layerA_4)
        layerA_p = layerA_p.reshape(layerA_p.size(0), -1)

        Aup1 = self.Aup1(layerA_p.unsqueeze(2).unsqueeze(3))
        Aup2 = self.Aup2(Aup1)
        Aup3 = self.Aup3(Aup2)
        fuse_conv = self.atten_conv(torch.cat((Aup3, self.layerV_down(layerV_4)), dim=1))
        atten_pool = self.atten_pool(fuse_conv)
        atten_pool = atten_pool.view(atten_pool.size(0), -1)
        atten_fc = self.atten_fc(atten_pool)

        attention = self.attention(fuse_conv)

        # F.softmax(Afc2, dim=1)[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3) *
        Vattention = F.relu((F.sigmoid(attention) * layerV_4) + layerV_4)
        Vattention = self.Vatten_conv(Vattention)
        Vatten_pool = self.Vatten_pool(Vattention)
        Vatten_pool = Vatten_pool.view(Vatten_pool.size(0), -1)
        Vatten_fc = self.Vatten_fc(Vatten_pool)

        predict = F.upsample(attention, size=video.size()[
                             2:], mode='bilinear', align_corners=True)

        return Vatten_fc, atten_fc, predict