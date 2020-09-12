import math
import torch
import torch.nn as nn
from layers.BBBConv import BBBConv2d
from layers.BBBLinear import BBBLinear
from layers.misc import FlattenLayer, ModuleWrapper


class BBB3Conv3FC(ModuleWrapper):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs):
        super(BBB3Conv3FC, self).__init__()

        self.num_classes = outputs

        self.conv1 = BBBConv2d(inputs, 16, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv1')
        self.bn1 = nn.BatchNorm2d(16)
        self.activate1 = nn.PReLU()
        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv2 = BBBConv2d(16, 16, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv2')
        self.bn2 = nn.BatchNorm2d(16)
        self.activate2 = nn.PReLU()
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv3 = BBBConv2d(16, 32, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv3')
        self.bn3 = nn.BatchNorm2d(32)
        self.activate3 = nn.PReLU()
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = BBBConv2d(33, 32, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv4')
        self.bn4 = nn.BatchNorm2d(32)
        self.activate4 = nn.PReLU()
        self.dropout4 = nn.Dropout2d(p=0.25)

        self.conv5 = BBBConv2d(32, 32, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv5')
        self.bn5 = nn.BatchNorm2d(32)
        self.activate5 = nn.PReLU()
        self.dropout5 = nn.Dropout2d(p=0.25)

        self.conv6 = BBBConv2d(32, 64, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv6')
        self.bn6 = nn.BatchNorm2d(64)
        self.activate6 = nn.PReLU()
        self.dropout6 = nn.Dropout2d(p=0.25)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(8 * 5 * 97)

        self.fc1 = BBBLinear(8 * 5 * 97, 512, alpha_shape=(1, 1), bias=True, name='fc1')
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_activate = nn.PReLU()
        self.fc1_dropout = nn.Dropout(p=0.50)

        self.fc2 = BBBLinear(512, 256, alpha_shape=(1, 1), bias=True, name='fc2')
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc2_activate = nn.Softplus()
        self.fc2_dropout = nn.Dropout(p=0.50)
        #
        # self.fc3 = BBBLinear(256, 128, alpha_shape=(1, 1), bias=True, name='fc3')
        # self.fc3_bn = nn.BatchNorm1d(128)
        # self.fc3_activate = nn.Softplus()
        # self.fc3_dropout = nn.Dropout(p=0.50)
        #
        # self.fc4 = BBBLinear(128, 64, alpha_shape=(1, 1), bias=True, name='fc4')
        # self.fc4_bn = nn.BatchNorm1d(64)
        # self.fc4_activate = nn.Softplus()
        # self.fc4_dropout = nn.Dropout(p=0.50)

        self.fc5 = BBBLinear(256, outputs, alpha_shape=(1, 1), bias=True, name='fc5')

        # x1 = self.dropout1(self.activate1(self.bn1(self.conv1(x))))
        # x2 = self.dropout2(self.activate2(self.bn2(self.conv2(x1))))
        # x3 = torch.cat((x1, x2), dim=1)
        # x3 = self.pool1(x3)
        #
        # x4 = self.dropout4(self.activate4(self.bn4(self.conv4(x3))))
        # x5 = self.dropout5(self.activate5(self.bn5(self.conv5(x4))))
        # x6 = torch.cat((x4, x5), dim=1)
        # x6 = self.pool2(x6)
        #
        # x7 = self.dropout7(self.activate7(self.bn7(self.conv7(x6))))
        # x8 = self.dropout8(self.activate8(self.bn8(self.conv8(x7))))
        # x9 = torch.cat((x7, x8), dim=1)
        # x9 = self.pool3(x9)
        #
        # x10 = self.dropout10(self.activate10(self.bn10(self.conv10(x9))))
        # x11 = self.dropout11(self.activate11(self.bn11(self.conv11(x10))))
        # x12 = torch.cat((x10, x11), dim=1)
        # x12 = self.pool4(x12)
        #
        # x12 = self.flatten(x12)
        #
        # x13 = self.fc1_dropout(self.fc1_activate(self.fc1_bn(self.fc1(x12))))
        # x14 = self.fc2_dropout(self.fc2_activate(self.fc2_bn(self.fc2(x13))))
        # x15 = self.fc3_dropout(self.fc3_activate(self.fc3_bn(self.fc3(x14))))
        #
        # x16 = self.fc4(x15)

        # # ResNet Architecture
        # x1 = self.dropout1(self.activate1(self.bn1(self.conv1(x))))
        # x2 = self.bn2(self.conv2(x1))
        # x2 = torch.cat((x2, x), dim=1)
        # x2 = self.dropout2(self.activate2(x2))
        #
        # x3 = self.pool1(x2)
        #
        # x4 = self.dropout4(self.activate4(self.bn4(self.conv4(x3))))
        # x5 = self.bn5(self.conv5(x4))
        # x5 = torch.cat((x5, x3), dim=1)
        # x5 = self.dropout5(self.activate5(x5))
        #
        # x6 = self.pool2(x5)
        #
        # x7 = self.dropout7(self.activate7(self.bn7(self.conv7(x6))))
        # x8 = self.bn8(self.conv8(x7))
        # x8 = torch.cat((x8, x6), dim=1)
        # x8 = self.dropout8(self.activate8(x8))
        #
        # x9 = self.pool3(x8)
        #
        # x10 = self.dropout10(self.activate10(self.bn10(self.conv10(x9))))
        # x11 = self.bn11(self.conv11(x10))
        # x11 = torch.cat((x11, x9), dim=1)
        # x11 = self.dropout11(self.activate11(x11))
        #
        # x12 = self.pool4(x11)
        #
        # x12 = self.flatten(x12)
        #
        # x13 = self.fc1_dropout(self.fc1_activate(self.fc1_bn(self.fc1(x12))))
        # x14 = self.fc2_dropout(self.fc2_activate(self.fc2_bn(self.fc2(x13))))
        # x15 = self.fc3_dropout(self.fc3_activate(self.fc3_bn(self.fc3(x14))))
        #
        # x16 = self.fc4(x15)
        #











