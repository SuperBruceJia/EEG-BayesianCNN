import math
import torch
import torch.nn as nn
from layers.BBBConv import BBBConv2d
from layers.BBBLinear import BBBLinear
from layers.misc import FlattenLayer, ModuleWrapper

import utils
import metrics
import config_bayesian as cfg
# from .misc import ModuleWrapper

class BBB3Conv3FC(ModuleWrapper):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs):
        super(BBB3Conv3FC, self).__init__()

        self.num_classes = outputs

        self.layer1 = nn.Sequential(
            BBBConv2d(inputs, 16, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv1'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.25)
        )

        self.layer2 = nn.Sequential(
            BBBConv2d(16, 32, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv1_1'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.25)
        )

        self.layer3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            BBBConv2d(48, 64, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv2'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.25)
        )

        self.layer5 = nn.Sequential(
            BBBConv2d(64, 128, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv2_1'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.25)
        )

        self.layer6 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer7 = nn.Sequential(
            BBBConv2d(192, 256, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv3'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.25)
        )

        self.layer8 = nn.Sequential(
            BBBConv2d(256, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv3_1'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.25)
        )

        self.layer9 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer10 = nn.Sequential(
            BBBConv2d(768, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv4'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.25)
        )

        self.layer11 = nn.Sequential(
            BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv4_1'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.25)
        )

        self.layer12 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer13 = nn.Sequential(
            FlattenLayer(2 * 8 * 1024),

            BBBLinear(2 * 8 * 1024, 512, alpha_shape=(1, 1), bias=True, name='fc1'),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            BBBLinear(512, 256, alpha_shape=(1, 1), bias=True, name='fc2'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            BBBLinear(256, outputs, alpha_shape=(1, 1), bias=True, name='fc3')
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.layer3(x3)

        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = torch.cat((x4, x5), dim=1)
        x6 = self.layer6(x6)

        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = torch.cat((x7, x8), dim=1)
        x9 = self.layer9(x9)

        x10 = self.layer10(x9)
        x11 = self.layer11(x10)
        x12 = torch.cat((x10, x11), dim=1)
        x12 = self.layer12(x12)

        x13 = self.layer13(x12)

        return x13

    def kl_loss(self):
        return self.weight.nelement() / self.log_alpha.nelement() * metrics.calculate_kl(self.log_alpha)
