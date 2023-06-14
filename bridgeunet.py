import torch.nn as nn
from unet_utils_v2 import double_conv


class BridgeUNet(nn.Module):
    def __init__(self):
        super(BridgeUNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down_conv_4_64 = double_conv(4, 64)
        self.down_conv_64_128 = double_conv(64, 128)
        self.down_conv_128_256 = double_conv(128, 256)
        self.down_conv_256_512 = double_conv(256, 512)
        self.down_conv_512_1024 = double_conv(512, 1024)

    def forward(self, image):
        x1d = self.down_conv_4_64(image)
        # print('x1d: ', x1d.size())

        x1dm = self.max_pool_2x2(x1d)
        x2d = self.down_conv_64_128(x1dm)
        # print('x2d: ', x2d.size())

        x2dm = self.max_pool_2x2(x2d)
        x3d = self.down_conv_128_256(x2dm)
        # print('x3d: ', x3d.size())

        x3dm = self.max_pool_2x2(x3d)
        x4d = self.down_conv_256_512(x3dm)
        # print('x4d: ', x4d.size())

        x4dm = self.max_pool_2x2(x4d)
        x5d = self.down_conv_512_1024(x4dm)
        # print('x5d: ', x5d.size())

        return x5d, x4d, x3d, x2d, x1d
