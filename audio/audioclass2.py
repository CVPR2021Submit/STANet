import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from Soundmodel import SoundNet
import argparse

class att_Model(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_Model, self).__init__()
        Amodel= SoundNet()
        checkpoint = torch.load('vggsound_netvlad.pth.tar')
        Amodel.load_state_dict(checkpoint['model_state_dict'])
        Amodel = list(Amodel.audnet.children())
        self.layerA_0 = nn.Sequential(*Amodel[:4])
        self.layerA_1 = Amodel[4]
        self.layerA_2 = Amodel[5]
        self.layerA_3 = Amodel[6]
        self.layerA_4 = Amodel[7]
        self.layerA_p1 = Amodel[8]
        self.Afc1 = nn.Linear(8192, 2)

    def forward(self, audio):
        layerA_0 = self.layerA_0(audio)                         # 1,1,257,48->1,64,65,12
        layerA_1 = self.layerA_1(layerA_0)                      # 1,64,65,12->1,64,65,12
        layerA_2 = self.layerA_2(layerA_1)                      # 1,128,33,6
        layerA_3 = self.layerA_3(layerA_2)                      # 1,256,17,3
        layerA_4 = self.layerA_4(layerA_3)                      # 1,512,9,2
        layerA_p1 = self.layerA_p1(layerA_4)                    # 1,8192
        layerA_p1 = layerA_p1.reshape(layerA_p1.size(0), -1)
        Afc1 = self.Afc1(layerA_p1)# 1,28

        return Afc1
