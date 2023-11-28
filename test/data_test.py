import PIL
import torch
import unittest


from nude2.data import CachedDataset
from nude2.train import Generator, Discriminator


from train_test import WIDTH, HEIGHT


class DataTest(unittest.TestCase):
    def test_resize(self):

        dataset = CachedDataset("./whatever", "./no-cache")
        img = PIL.Image.new("RGB", (8*WIDTH, 8*HEIGHT))

        for t in dataset.transforms:
            img = t(img)
            w, h = img.size
            self.assertEqual(w, WIDTH)
            self.assertEqual(h, HEIGHT)
