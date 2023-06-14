import torch
import torch.nn as nn
import math


class double_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1, bias=True, groups=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1, bias=True, groups=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class out_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out_conv, self).__init__()
        self.out = nn.Conv2d(in_channels=128, out_channels=1, groups=1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.out(x)
        return torch.sigmoid(x)

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta / 2
    if delta.is_integer():
        return tensor[:, :, int(delta):tensor_size - (int(delta)), int(delta):tensor_size - (int(delta))]
    return tensor[:, :, int(delta):tensor_size - (int(delta) + 1), int(delta):tensor_size - (int(delta) + 1)]


class up_conv(nn.Module):
    def __init__(self, in_c=2048, out_c=1024):
        super(up_conv, self).__init__()

        self.up_trans_2048 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=5, stride=2, padding=0)
        self.up_conv_2048_1024 = double_conv(2048, 1024)

        self.up_trans_1024 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=0)
        self.up_conv_1024_512 = double_conv(1024, 512)

        self.up_trans_512 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=0)
        self.up_conv_512_256 = double_conv(512, 256)

        self.up_trans_256 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=0)
        self.up_conv_256_128 = double_conv(256, 128)

        # output layer
        self.out_conv = out_conv(128, 1)



    def forward(self, pp_bridge, pre_x4d, pre_x3d, pre_x2d, pre_x1d, post_x4d, post_x3d, post_x2d, post_x1d):
        x4tu = self.up_trans_2048(pp_bridge)  # 1

        x4tucrop = crop_img(x4tu, pre_x4d)
#         print('x4tucrop: ', x4tucrop.size())
        x4u = self.up_conv_2048_1024(torch.cat([x4tucrop, pre_x4d, post_x4d], 1))
#         print('x4u after concat: ', x4u.size())

        x3tu = self.up_trans_1024(x4u)
#         print('x3tu: ', x3tu.size())
#         print('pre_x3d: ', pre_x3d.size())
        x3tucrop = crop_img(x3tu, pre_x3d)

        x3u = self.up_conv_1024_512(torch.cat([x3tucrop, pre_x3d, post_x3d], 1))
#         print('x3u after concat: ', x3u.size())

        x2tu = self.up_trans_512(x3u)
        # print('x2tu: ', x2tu.size())
        # print('bridge_pre[3]: ', bridge_pre[3].size())
        x2tucrop = crop_img(x2tu, pre_x2d)

        x2u = self.up_conv_512_256(torch.cat([x2tucrop, pre_x2d, post_x2d], 1))
        # print('concat x2tu, x2dc_pre, x2dc_post = x2u after concat: ', x2u.size())

        x1tu = self.up_trans_256(x2u)
        # print('x1tu: ', x1tu.size())
        # print('bridge_pre[4]: ', bridge_pre[4].size())
        x1tucrop = crop_img(x1tu, pre_x1d)

        x1u = self.up_conv_256_128(torch.cat([x1tucrop, pre_x1d, post_x1d], 1))
        # print('x1u after concat: ', x1u.size())

        x = self.out_conv(x1u)

        return x
