"""
DCGAN stolen from 
"""


import json
import logging
import nude2.data
import os
import os.path
import random
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as data
import torchvision

from datetime import datetime
from nude2.progress import ProgressBar
from nude2.utils import splash

from nude2.model import Generator, Discriminator
from nude2.train_constants import *

import nude2.model_v1 as model_v1


LOG_FORMAT = "\033[95m%(asctime)s\033[00m [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def weights_init(m):
    """Normalize weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




def main(data_folder, num_epochs, batch_size, checkpoint_path, samples_path, seed=None):

    if seed is not None:
        manualSeed = 999
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.use_deterministic_algorithms(True)

    data_dir = os.path.expanduser(data_folder)

    splash("splash")
    print("\n\n")
    print("NUDE 2.0")
    print("========")
    print(f"data .......... \033[95m{data_dir}\033[00m")
    print(f"epochs ........ \033[96m{num_epochs}\033[00m")
    print(f"batch size .... \033[97m{batch_size}\033[00m")
    print(f"device ........ \033[98m{device}\033[00m")
    print(f"checkpoint .... \033[99m{checkpoint_path}\033[00m")
    print(f"samples path .. \033[100m{samples_path}\033[00m")
    print()

    if samples_path is not None:
        os.makedirs(samples_path, exist_ok=True)
        with open(os.path.join(samples_path, "meta.json"), "w") as f:
            json.dump({
                "data_folder": data_folder,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "checkpoint_path": checkpoint_path,
                "samples_path": samples_path,
                "lr": lr,
                "beta1": beta1,
            }, f)

    # dataset = nude2.data.MetCenterCroppedDataset(data_dir)

    dataset = nude2.data.CachedDataset(
        data_dir,
        "~/.cache/nude2/images-random-crop-256x256",
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    g = Generator().to(device)
    g.apply(weights_init)

    d = Discriminator().to(device)
    d.apply(weights_init)
    
    total_params = sum(
        param.numel()
        for param in g.parameters()
    )

    print("Total params =", total_params)

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
        logger.info(f"Loaded at {epoch-1}")
    except FileNotFoundError:
        logger.warn("Could not find specified checkpoint... starting GAN at epoch=0")
        epoch = 0

    last_epoch = epoch + num_epochs

    for epoch in range(epoch, epoch + num_epochs):

        start = datetime.utcnow()

        with ProgressBar(len(dataloader), prefix=f"[epoch={epoch:04d}/{last_epoch}] ", size=40) as progress:
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

                # inc
                progress.inc()

        sample = g(fixed_noise).detach().cpu()

        if samples_path is not None:
            for i in range(sample.size(0)):
                img = nude2.data.MetCenterCroppedDataset.pilify(sample[i])
                fname = f"sample-{epoch:04d}-{i:02d}.jpg"
                img.save(os.path.join(samples_path, fname))

        dur = datetime.utcnow() - start

        res = torch.save({
            "g": g.state_dict(),
            "d": d.state_dict(),
            "epoch": epoch,
        }, checkpoint_path)
