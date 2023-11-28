import tempfile
import PIL
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

import onnx
from onnx import compose

from nude2.model import Generator198x198
import nude2.data

class ImageGenerator(Generator198x198):


    def __init__(self):
        super().__init__()
        s = 1/255.
        self.pilify = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
            transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[s, s, s]),
        ])

    def forward(self, x):
        return self.pilify(super().forward(x))


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        s = 1.0
        self.normalize1 = transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5])
        self.normalize2 = transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[s, s, s])

    def forward(self, x):
        x = self.normalize1(x)
        x = self.normalize2(x)
        # x = self.to_pil(x)
        return x



def main(checkpoint, output):

    fname = "checkpoints/nude2-dcgan-met-random-crop-198x198.pt"
    fname = checkpoint

    states = torch.load(fname, map_location="cpu")

    with tempfile.NamedTemporaryFile("w") as f:
        t = Transform()
        x = torch.randn(1, 3, 1, 1)
        torch.onnx.export(t, x, f.name, opset_version=10, input_names=["raw"], output_names=["output"])
        t_onnx = onnx.load(f.name)

    with tempfile.NamedTemporaryFile("w") as f:
        g = Generator198x198()
        g.load_state_dict(states["g"])
        g.cpu()
        g.eval()
        g.train()
        x = torch.randn(1, 100, 1, 1)
        x = torch.ones(1, 100, 1, 1)
        torch.onnx.export(
            g,
            x,
            f.name,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["vector"],
            output_names=["unnormalized_output"],
        )

        g_onnx = onnx.load(f.name)
        onnx.checker.check_model(g_onnx)

        # Session
        ort_sess = ort.InferenceSession(f.name)
        ort_outs = ort_sess.run(None, {"vector": x.numpy()})[0]
        g_outs = g(x).detach().numpy()

        # print(g_outs[0, 0, :10, :10])

        if not np.allclose(ort_outs, g_outs, rtol=1e-6, atol=1e-6):
            print("WARNING: predicted matrices aren't close!")

        model = compose.merge_models(g_onnx, t_onnx, io_map=[("unnormalized_output", "raw")])
        onnx.save_model(model, output)

        # XXX: GENERATE SAMPLE
        # session = ort.InferenceSession("checkpoints/nice.onnx")
        # res = session.run(None, {"vector": x.numpy()})
        # res = res[0][0]
        # res = res.astype(np.uint8)
        # res = np.moveaxis(res, 0, -1)
        # print(res.shape)
        # img = PIL.Image.fromarray(res, mode="RGB")
        # img.save("cmon.png")
        # print(img)


        # XXX: GENERATE PYTORCH SAMPLE
        # batch_size = 64
        # x = torch.randn(batch_size, 100, 1, 1)
        # y = g(x)
        # y = t(y)
        # img = nude2.data.MetCenterCroppedDataset.pilify(y[0])
        # img.save("yea.png")

        # print((ort_outs - g_outs).mean())
        # print((ort_outs - g_outs).max())
