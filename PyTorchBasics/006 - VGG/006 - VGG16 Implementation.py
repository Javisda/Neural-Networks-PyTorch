# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# VGG16 resumé
# Input -> (224 x 224 RGB image), then:
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# Then flatten and create classifier 4096x4096x1000 Linear Layers

class VGG_16(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Conv layers part
        self.conv_layers = self.create_conv_layers(VGG16)

        # Fully connected part. Images are 7*7px and 512 conv layers
        self.fcs = nn.Sequential(nn.Linear(512*7*7, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.conv_layers(x) # Le pasamos la x a la función ya que esta devuelve un nn.Sequential
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        return x

    def create_conv_layers(self, vgg_architecture):
        layers = []
        in_channels = self.in_channels

        for x in vgg_architecture:
            if type(x) == int: # Si es un numero entero sabemos que es una capa convolucional
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = out_channels
            elif x == 'M': # MaxPool
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VGG_16(in_channels=3, num_classes=1000)
model.to(device)
# Simulate an image
x = torch.randn(1, 3, 224, 224).to(device) # También hay que enviar los datos a la gpu
print(model)
print(model(x).shape)