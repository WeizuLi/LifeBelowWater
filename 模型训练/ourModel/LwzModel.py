import torch.nn as nn
import torch.nn.functional as F
from torch import squeeze
from torchvision import models
"""
自编写神经网络，用于训练模型
该模型由团队成员：李为祖编写
"""


"""
该模型，输入的图片为batch ?,channel 3,w 224,h 224
"""


class LwzModule(nn.Module):
    def __init__(self):
        super(LwzModule, self).__init__()
        self.conv1=nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu=nn.ReLU(inplace=True)
        # 56 56 64
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1=nn.Sequential(nn.Conv2d(64,64,kernel_size=(3,3),padding=1),nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU(),
                                  nn.Conv2d(64,64,kernel_size=(3,3),padding=1),nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU())
        self.layer2=nn.Sequential(nn.Conv2d(64,128,kernel_size=(3,3),stride=2,padding=1),nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU(),
                                  nn.Conv2d(128,128,kernel_size=(3,3),padding=1),nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU())
        self.layer2_2=nn.Sequential(nn.Conv2d(128,128,kernel_size=(3,3),stride=1,padding=1),nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU(),
                                  nn.Conv2d(128,128,kernel_size=(3,3),padding=1),nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU())
        self.layer3=nn.Sequential(nn.Conv2d(128,256,kernel_size=(3,3),stride=2,padding=1),nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU(),
                                  nn.Conv2d(256,256,kernel_size=(3,3),padding=1),nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU())
        self.layer3_2=nn.Sequential(nn.Conv2d(256,256,kernel_size=(3,3),stride=1,padding=1),nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU(),
                                  nn.Conv2d(256,256,kernel_size=(3,3),padding=1),nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU())
        self.layer4=nn.Sequential(nn.Conv2d(256,512,kernel_size=(3,3),stride=2,padding=1),nn.BatchNorm2d(512,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU(),
                                  nn.Conv2d(512,512,kernel_size=(3,3),padding=1),nn.BatchNorm2d(512,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU())
        self.layer4_2=nn.Sequential(nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=1),nn.BatchNorm2d(512,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU(),
                                  nn.Conv2d(512,512,kernel_size=(3,3),padding=1),nn.BatchNorm2d(512,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),nn.ReLU())
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,33)


    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        # 56 56 64
        x=self.layer1(x)
        # x=self.layer1(x)
        # 28 28 128
        x=self.layer2(x)
        # x=self.layer2_2(x)
        # 14 14 256
        x=self.layer3(x)
        # x=self.layer3_2(x)
        # 7 7 512
        x=self.layer4(x)
        # x=self.layer4_2(x)
        # print(x.size())
        x=self.avgpool(x)
        x=squeeze(x)
        # print(x.size())
        x=self.fc(x)
        return x





