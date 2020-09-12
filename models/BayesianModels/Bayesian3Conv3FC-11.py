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

        self.conv1 = BBBConv2d(inputs, 32, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv1')
        self.bn1 = nn.BatchNorm2d(32)
        self.activate1 = nn.ELU()
        self.drop1 = nn.Dropout2d(0.25)

        self.conv2 = BBBConv2d(32, 64, 5, alpha_shape=(1, 1), stride=1, padding=2, bias=True, name='conv2')
        self.bn2 = nn.BatchNorm2d(64)
        self.activate2 = nn.ELU()
        self.drop2 = nn.Dropout2d(0.25)

        self.pool1 = BBBConv2d(64, 64, 2, alpha_shape=(1, 1), stride=2, padding=0, bias=True, name='pool1')
        self.pool1_bn = nn.BatchNorm2d(64)
        self.pool1_activate = nn.ELU()
        self.pool1_drop = nn.Dropout2d(0.25)

        self.conv4 = BBBConv2d(64, 128, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv4')
        self.bn4 = nn.BatchNorm2d(128)
        self.activate4 = nn.ELU()
        self.drop4 = nn.Dropout2d(0.25)

        self.conv5 = BBBConv2d(128, 256, 5, alpha_shape=(1, 1), stride=1, padding=2, bias=True, name='conv5')
        self.bn5 = nn.BatchNorm2d(256)
        self.activate5 = nn.ELU()
        self.drop5 = nn.Dropout2d(0.25)

        self.pool2 = BBBConv2d(256, 256, 2, alpha_shape=(1, 1), stride=2, padding=0, bias=True, name='pool2')
        self.pool2_bn = nn.BatchNorm2d(256)
        self.pool2_activate = nn.ELU()
        self.pool2_drop = nn.Dropout2d(0.25)

        self.conv7 = BBBConv2d(256, 384, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv7')
        self.bn7 = nn.BatchNorm2d(384)
        self.activate7 = nn.ELU()
        self.drop7 = nn.Dropout2d(0.25)

        self.conv8 = BBBConv2d(384, 512, 5, alpha_shape=(1, 1), stride=1, padding=2, bias=True, name='conv8')
        self.bn8 = nn.BatchNorm2d(512)
        self.activate8 = nn.ELU()
        self.drop8 = nn.Dropout2d(0.25)

        self.pool3 = BBBConv2d(512, 512, 2, alpha_shape=(1, 1), stride=2, padding=0, bias=True, name='pool3')
        self.pool3_bn = nn.BatchNorm2d(512)
        self.pool3_activate = nn.ELU()
        self.pool3_drop = nn.Dropout2d(0.25)

        self.conv10 = BBBConv2d(512, 640, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv10')
        self.bn10 = nn.BatchNorm2d(640)
        self.activate10 = nn.ELU()
        self.drop10 = nn.Dropout2d(0.25)

        self.conv11 = BBBConv2d(640, 768, 5, alpha_shape=(1, 1), stride=1, padding=2, bias=True, name='conv11')
        self.bn11 = nn.BatchNorm2d(768)
        self.activate11 = nn.ELU()
        self.drop11 = nn.Dropout2d(0.25)

        self.pool4 = BBBConv2d(768, 768, 2, alpha_shape=(1, 1), stride=2, padding=0, bias=True, name='pool4')
        self.pool4_bn = nn.BatchNorm2d(768)
        self.pool4_activate = nn.ELU()
        self.pool4_drop = nn.Dropout2d(0.25)

        self.flatten = FlattenLayer(2 * 8 * 768)

        self.fc1 = BBBLinear(2 * 8 * 768, 512, alpha_shape=(1, 1), bias=True, name='fc1')
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_activate = nn.ELU()
        self.fc1_drop = nn.Dropout(0.50)

        self.fc2 = BBBLinear(512, 256, alpha_shape=(1, 1), bias=True, name='fc2')
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc2_activate = nn.ELU()
        self.fc2_drop = nn.Dropout(0.50)

        self.fc3 = BBBLinear(256, 64, alpha_shape=(1, 1), bias=True, name='fc3')
        self.fc3_bn = nn.BatchNorm1d(64)
        self.fc3_activate = nn.ELU()
        self.fc3_drop = nn.Dropout(0.50)

        self.fc4 = BBBLinear(64, outputs, alpha_shape=(1, 1), bias=True, name='fc4')
