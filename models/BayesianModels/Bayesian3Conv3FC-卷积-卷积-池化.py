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
        self.activate1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv2 = BBBConv2d(16, 32, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv1_1')
        self.bn2 = nn.BatchNorm2d(32)
        self.activate2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(32, 64, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv2')
        self.bn3 = nn.BatchNorm2d(64)
        self.activate3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.conv4 = BBBConv2d(64, 128, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv2_1')
        self.bn4 = nn.BatchNorm2d(128)
        self.activate4 = nn.ReLU()
        self.dropout4 = nn.Dropout2d(p=0.25)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = BBBConv2d(128, 256, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv3')
        self.bn5 = nn.BatchNorm2d(256)
        self.activate5 = nn.ReLU()
        self.dropout5 = nn.Dropout2d(p=0.25)

        self.conv6 = BBBConv2d(256, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv3_1')
        self.bn6 = nn.BatchNorm2d(512)
        self.activate6 = nn.ReLU()
        self.dropout6 = nn.Dropout2d(p=0.25)

        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv7 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv4')
        self.bn7 = nn.BatchNorm2d(512)
        self.activate7 = nn.ReLU()
        self.dropout7 = nn.Dropout2d(p=0.25)

        self.conv8 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv4_1')
        self.bn8 = nn.BatchNorm2d(512)
        self.activate8 = nn.ReLU()
        self.dropout8 = nn.Dropout2d(p=0.25)

        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(2 * 8 * 512)

        self.fc1 = BBBLinear(2 * 8 * 512, 512, alpha_shape=(1, 1), bias=True, name='fc1')
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_activate = nn.ReLU()
        self.fc1_dropout = nn.Dropout(p=0.25)

        self.fc2 = BBBLinear(512, 256, alpha_shape=(1, 1), bias=True, name='fc2')
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc2_activate = nn.ReLU()
        self.fc2_dropout = nn.Dropout(p=0.25)

        self.fc3 = BBBLinear(256, outputs, alpha_shape=(1, 1), bias=True, name='fc3')




