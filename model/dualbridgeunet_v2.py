import math
import torch
import torch.nn as nn
from bridgeunet import BridgeUNet
from unet_utils_v2 import up_conv


class DualBridgeUNet(nn.Module):
    def __init__(self):
        super(DualBridgeUNet, self).__init__()
        self.bridgeunet = BridgeUNet()
        self.upconv = up_conv()

    def forward(self, img_pre, img_post):
        pre_x5d, pre_x4d, pre_x3d, pre_x2d, pre_x1d = self.bridgeunet(img_pre)  # batch, Channel, H, W
        post_x5d, post_x4d, post_x3d, post_x2d, post_x1d = self.bridgeunet(img_post)  # batch, Channel, H, W

        bridge = torch.cat((pre_x5d, post_x5d), dim=1)

        output = self.upconv(bridge, pre_x4d, pre_x3d, pre_x2d, pre_x1d, post_x4d, post_x3d, post_x2d, post_x1d)

        return output
