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

        self.conv2 = BBBConv2d(16, 32, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv2')
        self.bn2 = nn.BatchNorm2d(32)
        self.activate2 = nn.PReLU()
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv2_1 = BBBConv2d(32, 32, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv2_1')
        self.bn2_1 = nn.BatchNorm2d(32)
        self.activate2_1 = nn.PReLU()
        self.dropout2_1 = nn.Dropout2d(p=0.25)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(32, 64, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv3')
        self.bn3 = nn.BatchNorm2d(64)
        self.activate3 = nn.PReLU()
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.conv4 = BBBConv2d(64, 128, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv4')
        self.bn4 = nn.BatchNorm2d(128)
        self.activate4 = nn.PReLU()
        self.dropout4 = nn.Dropout2d(p=0.25)

        self.conv4_1 = BBBConv2d(128, 128, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv4_1')
        self.bn4_1 = nn.BatchNorm2d(128)
        self.activate4_1 = nn.PReLU()
        self.dropout4_1 = nn.Dropout2d(p=0.25)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = BBBConv2d(128, 256, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv5')
        self.bn5 = nn.BatchNorm2d(256)
        self.activate5 = nn.PReLU()
        self.dropout5 = nn.Dropout2d(p=0.25)

        self.conv6 = BBBConv2d(256, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv6')
        self.bn6 = nn.BatchNorm2d(512)
        self.activate6 = nn.PReLU()
        self.dropout6 = nn.Dropout2d(p=0.25)

        self.conv6_1 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv6_1')
        self.bn6_1 = nn.BatchNorm2d(512)
        self.activate6_1 = nn.PReLU()
        self.dropout6_1 = nn.Dropout2d(p=0.25)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv7')
        self.bn7 = nn.BatchNorm2d(512)
        self.activate7 = nn.PReLU()
        self.dropout7 = nn.Dropout2d(p=0.25)

        self.conv8 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv8')
        self.bn8 = nn.BatchNorm2d(512)
        self.activate8 = nn.PReLU()
        self.dropout8 = nn.Dropout2d(p=0.25)

        self.conv8_1 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv8_1')
        self.bn8_1 = nn.BatchNorm2d(512)
        self.activate8_1 = nn.PReLU()
        self.dropout8_1 = nn.Dropout2d(p=0.25)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv9')
        self.bn9 = nn.BatchNorm2d(512)
        self.activate9 = nn.PReLU()
        self.dropout9 = nn.Dropout2d(p=0.25)

        self.conv10 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv10')
        self.bn10 = nn.BatchNorm2d(512)
        self.activate10 = nn.PReLU()
        self.dropout10 = nn.Dropout2d(p=0.25)

        self.conv10_1 = BBBConv2d(512, 512, 3, alpha_shape=(1, 1), stride=1, padding=1, bias=True, name='conv10_1')
        self.bn10_1 = nn.BatchNorm2d(512)
        self.activate10_1 = nn.PReLU()
        self.dropout10_1 = nn.Dropout2d(p=0.25)

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(2 * 2 * 512)

        self.fc1 = BBBLinear(2 * 2 * 512, 1024, alpha_shape=(1, 1), bias=True, name='fc1')
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc1_activate = nn.PReLU()
        self.fc1_dropout = nn.Dropout(p=0.25)

        self.fc2 = BBBLinear(1024, 512, alpha_shape=(1, 1), bias=True, name='fc2')
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc2_activate = nn.PReLU()
        self.fc2_dropout = nn.Dropout(p=0.25)

        self.fc3 = BBBLinear(512, 256, alpha_shape=(1, 1), bias=True, name='fc3')
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc3_activate = nn.PReLU()
        self.fc3_dropout = nn.Dropout(p=0.25)

        self.fc4 = BBBLinear(256, outputs, alpha_shape=(1, 1), bias=True, name='fc4')




