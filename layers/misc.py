import torch
from torch import nn


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        # ResNet Architecture
        x1 = self.dropout1(self.activate1(self.bn1(self.conv1(x))))
        x2 = self.dropout2(self.activate2(self.bn2(self.conv2(x1))))
        x3 = self.bn3(self.conv3(x2))
        x3 = torch.cat((x3, x), dim=1)
        x3 = self.dropout3(self.activate3(x3))

        x3 = self.pool1(x3)

        x4 = self.dropout4(self.activate4(self.bn4(self.conv4(x3))))
        x5 = self.dropout5(self.activate5(self.bn5(self.conv5(x4))))
        x6 = self.bn6(self.conv6(x5))
        x6 = torch.cat((x6, x3), dim=1)
        x6 = self.dropout6(self.activate6(x6))

        x7 = self.pool2(x6)

        # x7 = self.dropout7(self.activate7(self.bn7(self.conv7(x6))))
        # x8 = self.dropout8(self.activate8(self.bn8(self.conv8(x7))))
        # x9 = self.bn9(self.conv9(x8))
        # x9 = torch.cat((x9, x6), dim=1)
        # x9 = self.dropout9(self.activate9(x9))

        # x9 = self.pool3(x9)

        # x10 = self.dropout10(self.activate10(self.bn10(self.conv10(x9))))
        # x11 = self.dropout11(self.activate11(self.bn11(self.conv11(x10))))
        # x12 = self.bn12(self.conv12(x11))
        # x12 = torch.cat((x12, x9), dim=1)
        # x12 = self.dropout12(self.activate12(x12))
        #
        # x12 = self.pool4(x12)

        x12 = self.flatten(x7)

        x13 = self.fc1_dropout(self.fc1_activate(self.fc1_bn(self.fc1(x12))))
        x14 = self.fc2_dropout(self.fc2_activate(self.fc2_bn(self.fc2(x13))))
        # x13 = self.fc1_dropout(self.fc1_activate(self.fc1(x12)))
        # x14 = self.fc2_dropout(self.fc2_activate(self.fc2(x13)))
        # x15 = self.fc3_dropout(self.fc3_activate(self.fc3_bn(self.fc3(x14))))
        # x16 = self.fc4_dropout(self.fc4_activate(self.fc4_bn(self.fc4(x15))))

        x17 = self.fc5(x14)

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

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x17, kl


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)
