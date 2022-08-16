"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
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
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),  # Bias se pone a false por el uso de BatchNorm
            nn.InstanceNorm2d(out_channels, affine=True), # Layer Norm --- Instante norm
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g * 16, 4, 1, 0), # N x f_g*16 x 4 x 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  #[-1, 1]
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)




def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)