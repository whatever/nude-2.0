import torch
import torchvision


from nude2.train import Generator
from nude2.data import MetCenterCroppedDataset

def main(checkpoint):

    states = torch.load(checkpoint)

    g = Generator()
    g.load_state_dict(states["g"])
    torch.rand(1, 1, 100, 100)

    fixed_noise = torch.randn(1, 100, 1, 1)
    vec = g(fixed_noise)
    img = MetCenterCroppedDataset.pilify(vec[0])
    img.save("generated.png")
