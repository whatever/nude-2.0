import torch
import unittest


from nude2.train import Generator, Discriminator


WIDTH = HEIGHT = 128


class ModuleShapeTest(unittest.TestCase):
    """Test that the generator and discrimator compute the correct shapes"""

    def test_generator_shape(self):
        """Ensure that g(Nx100x1x1) -> Nx3x256, 256)"""

        g = Generator()

        r = g(torch.rand(10, 100, 1, 1))

        self.assertEqual(
            r.shape,
            (10, 3, WIDTH, HEIGHT),
        )

    def test_discriminator_shape(self):
        """Ensure that d(Nx3x256, 256) -> Nx1x1x1"""

        d = Discriminator()

        r = d(torch.rand(8, 3, WIDTH, HEIGHT))

        self.assertEqual(
            r.shape,
            (8, 1, 1, 1),
        )
