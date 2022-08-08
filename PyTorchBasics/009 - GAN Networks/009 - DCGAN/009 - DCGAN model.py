"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N * channels_img * 64 * 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ), # In the first layer we don't use BatchNorm (as said in the paper)
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),       # 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1), # 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1), # 4 x 4
            nn.Conv2d(features_d * 8, 1,  kernel_size=4, stride=2, padding=0), # 1 x 1
            nn.Sigmoid(),
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),  # Bias se pone a false por el uso de BatchNorm
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(

        )

    def _block(self, in_channels):


