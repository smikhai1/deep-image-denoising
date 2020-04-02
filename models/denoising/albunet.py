import torch
from torch import nn
from torchvision.models import resnet34

class DecoderBlock(nn.Module):

    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters

        internal_filters = self.in_filters//4

        self.conv_1 = nn.Conv2d(self.in_filters, internal_filters, kernel_size=1)
        self.bn_1 = nn.BatchNorm2d(internal_filters)
        self.deconv = nn.ConvTranspose2d(internal_filters,
                                         internal_filters,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1
                                         )
        self.conv_2 = nn.Conv2d(internal_filters, self.out_filters, kernel_size=1)
        self.bn_2 = nn.BatchNorm2d(self.out_filters)
        self.relu = nn.ReLU(True)

        self.decoder = nn.Sequential(self.conv_1,
                                     self.bn_1,
                                     self.deconv,
                                     self.bn_1,
                                     self.relu,
                                     self.conv_2,
                                     self.bn_2
                                     )

    def forward(self, x):

        out = self.decoder(x)

        return out


class AlbuNet(nn.Module):

    def __init__(self, num_filters=64, pretrained=True):
        super().__init__()
        resnet = resnet34(pretrained=pretrained)
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(num_filters//2, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.deconv = nn.ConvTranspose2d(num_filters,
                                         num_filters//2,
                                         kernel_size=3,
                                         stride=2,
                                         output_padding=1,
                                         padding=1
                                         )


        self.encoder_1 = nn.Sequential(self.conv1,
                                       resnet.bn1,
                                       resnet.relu,
                                       resnet.maxpool
                                      )

        self.encoder_2 = nn.ModuleList([resnet.layer1,
                                        resnet.layer2,
                                        resnet.layer3
                                       ]
                                      )

        self.bottleneck = nn.Sequential(resnet.layer4,
                                        DecoderBlock(num_filters * 2 ** 3, num_filters * 2 ** 2)
                                        )

        self.decoder_1 = nn.ModuleList([DecoderBlock(num_filters * 2 ** 2, num_filters * 2 ** 1),
                                        DecoderBlock(num_filters * 2 ** 1, num_filters),
                                        DecoderBlock(num_filters, num_filters)
                                       ]
                                      )

        self.decoder_2 = nn.Sequential(self.deconv,
                                       self.relu,
                                       self.conv2,
                                       self.sigmoid
                                       )
    def forward(self, x):

        skip_connects = []

        x = self.encoder_1(x)
        for idx, layer in enumerate(self.encoder_2):
            x = layer(x)
            skip_connects.append(x)

        x = self.bottleneck(x)

        for idx, layer in enumerate(self.decoder_1):
            x = layer(torch.add(skip_connects[-idx-1], x))

        x = self.decoder_2(x)

        return x