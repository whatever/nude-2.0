import torch
import torch.nn as nn


from nude2.train_constants import *


def size(w, h, filters, stride, padding):
    return (
        filters,
        (w - filters + 2*padding) / stride + 1,
        (h - filters + 2*padding) / stride + 1,
    )


class Discriminator(nn.Module):
    """Module to discriminate between real and fake images"""

    def __init__(self):
        super(Discriminator, self).__init__()

        #  (W - F + 2P) / S + 1
        # (128 - 4

        end = 4+1

        nc = 3
        ndf = 128

        layers = []

        W = 128
        H = 128

        # input is (nc) x 256 x 256
        layers += [
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        self.main = nn.Sequential(*layers)

        for i in range(1, end):
            c = ndf * 2**(i-1)
            f = ndf * 2**i
            layers.extend([
                nn.Conv2d(c, f, 4, 2, 1, bias=False),
                nn.BatchNorm2d(f),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        layers.extend([
            nn.Conv2d(ndf * 2**(end-1), 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        ])

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    """Module to generate an image from a feature vector"""

    def __init__(self):
        super(Generator, self).__init__()

        ngf = 128

        print("nz = ", nz)

        layers = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        ]

        layers += [
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``,
        ]
        # """

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)
