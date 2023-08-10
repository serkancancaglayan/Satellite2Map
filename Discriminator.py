import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)
    

class Discriminator(nn.Module):
    def __init__(self, in_channels, features):
        super(Discriminator, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = list()

        in_channels = features[0]
        for feature in features[1:]:
            stride = 1 if feature == features[-1] else 2
            layers.append(ConvBlock(in_channels=in_channels, out_channels=feature, stride=stride))
            in_channels = feature

        self.model = nn.Sequential(*layers)


    def forward(self, x, y):
        input_ = torch.cat([x, y], dim=1)
        input_ = self.first_block(input_)
        input_ = self.model(input_)
        return input_
    
