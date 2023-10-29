import torch
import torchvision


from nude2.train import Generator, weights_init
from nude2.data import MetCenterCroppedDataset

def main(checkpoint):

    states = torch.load(checkpoint)

    g = Generator()
    g.apply(weights_init)
    g.load_state_dict(states["g"])

    fixed_noise = torch.randn(1, 100, 1, 1)
    vec = g(fixed_noise)
    img = MetCenterCroppedDataset.pilify(vec[0])
    img.save("generated.png")
