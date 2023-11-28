import PIL
import os.path
import torch

import torch.nn as nn

from glob import glob

from collections import defaultdict
import itertools
import subprocess
import random

from nude2.model import Generator198x198
import nude2.data


def gridify(images, nrows, ncols):
    WIDTH = HEIGHT = 64
    PADDING = 4

    w = (ncols+1)*PADDING + ncols*WIDTH
    h = (nrows+1)*PADDING + nrows*HEIGHT

    elpapa = PIL.Image.new("RGB", (w, h))
    elpapa.paste((255, 255, 255), (0, 0, w, h))

    for n, img in enumerate(images):
        i = n // nrows
        j = n % ncols
        x = PADDING + i*(WIDTH+PADDING)
        y = PADDING + j*(HEIGHT+PADDING)
        elpapa.paste(img, (x, y))

    return elpapa


def main2(checkpoint_path):

    NCOLS = NROWS = 8

    images = [
        PIL.Image.new("RGB", (64, 64))
        for i in range(NROWS*NCOLS)
    ]

    for i, img in enumerate(images):
        x = i % NCOLS
        y = i // NCOLS

        r = int(x/(NCOLS-1)*255)
        g = 0
        b = int(y/(NROWS-1)*255)

        img.paste(
            (r, g, b),
            (0, 0, 64, 64),
        )

    elpapa = gridify(images, NROWS, NCOLS)
    
    elpapa.save("colsamps/yikes.png")


def main2(samples_path, out):

    fnames = sorted(glob(os.path.join(samples_path, "*.jpg")))

    image_set = defaultdict(list)

    images = defaultdict(list)

    for fname in sorted(fnames):
        fname = os.path.basename(fname)
        g = fname.split("-")[1]
        image_set[g].append(fname)

    for epoch in sorted(image_set.keys()):
        fnames = sorted(image_set[epoch])[0:64]
        for fname in fnames:
            with PIL.Image.open(os.path.join(samples_path, fname)) as img:
                images[epoch].append(img.resize((64, 64)))

    for epoch in images.keys():
        elpapa = gridify(images[epoch], 8, 8)
        print(f"saving epoch={epoch}")
        elpapa.save(f"colsamps/{epoch}.png")


    subprocess.run([
        "ffmpeg",
        "-framerate", "30",
        "-pattern_type", "glob",
        "-i", "colsamps/*.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out,
    ])







    # Useful with:
    """
    ffmpeg -framerate 30 \
           -pattern_type glob \
           -i '*.jpg' \
           -c:v libx264 \
           -pix_fmt yuv420p \
           out.mp4
    """


def main(checkpoint_path, samples_path):
    states = torch.load(checkpoint_path)

    g = Generator198x198()
    g.load_state_dict(states["g"])

    noise = torch.rand(68, 100, 1, 1)

    for i in range(68):
        out = g(noise)
        img = nude2.data.MetCenterCroppedDataset.pilify(out[i])
        img.save(os.path.join(samples_path, f"sample-{i}.jpg"))
