"""
DCGAN stolen from 
"""


from datetime import datetime


import logging
import torch.utils.data as data
import torch
import torch.nn as nn
import torchvision


import torch.optim

import nude2.data

from nude2.progress import ProgressBar
from nude2.utils import splash


LOG_FORMAT = "\033[95m%(asctime)s\033[00m [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Root directory for dataset
dataroot = "celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 1024

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Set device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Checkpoint path
checkpoint_path = "drive/MyDrive/Colab Checkpoints/nude2-dcgan.pt"


def weights_init(m):
    """Normalize weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    """Module to discriminate between real and fake images"""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    """Module to generate an image from a feature vector"""

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
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
        )

    def forward(self, input):
        return self.main(input)




images_path = "~/.cache/nude2/celeba"



def main(num_epochs, batch_size):

    splash("splash")
    print("\n\n")
    print("NUDE 2.0")
    print("========")
    print(f"epochs ........ {num_epochs}")
    print(f"batch size .... {batch_size}")
    print(f"data path ..... {images_path}")

    dataset = nude2.data.MetCenterCroppedDataset()

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    g = Generator()
    g.apply(weights_init)

    d = Discriminator()
    d.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1.0
    fake_label = 0.0

    optimizerD = torch.optim.Adam(d.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(g.parameters(), lr=lr, betas=(beta1, 0.999))

    try:
        states = torch.load(checkpoint_path)
        g.load_state_dict(states["g"])
        d.load_state_dict(states["d"])
        epoch = states["epoch"] + 1
        logger.info(f"Loaded at {epoch}")
    except FileNotFoundError:
        logger.warn("Could not find specified checkpoint... starting GAN at epoch=0")
        epoch = 0

    for epoch in range(epoch, epoch + num_epochs):

        start = datetime.utcnow()

        logger.info(f"Epoch {epoch}")

        with ProgressBar(len(dataloader)) as progress:
            for i, imgs in enumerate(dataloader):

                # Discriminate against real data
                d.zero_grad()
                real_cpu = imgs.to(device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,),
                    real_label,
                    dtype=torch.float,
                    device=device,
                )
                output = d(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # Discriminate against fake data
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = g(noise)
                label.fill_(fake_label)
                output = d(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                # Increment generator
                g.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output = d(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                # Print out progress
                progress.inc()

        dur = datetime.utcnow() - start
    


