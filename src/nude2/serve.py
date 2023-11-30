import io
import json
import http.server
import torch

from nude2.model import Generator198x198, Generator
from nude2.data import MetCenterCroppedDataset


GENERATOR = Generator198x198()


class Handler(http.server.SimpleHTTPRequestHandler):

    def write_image(self):
        noise = torch.randn(1, 100, 1, 1).to("cpu")
        noise = torch.ones(1, 100, 1, 1).to("cpu")
        arr = GENERATOR(noise)
        img = MetCenterCroppedDataset.pilify(arr[0])

        res = io.BytesIO()
        img.save(res, format="png")
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        self.end_headers()
        self.wfile.write(res.getvalue())

    def do_GET(self):

        if self.path == "/api/v1/image":
            self.write_image()
            return

        if self.path != "/api/v1/query":
            return

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        res = json.dumps({
            "ok": True,
            "data": {
            },
        })
        self.wfile.write(bytes(res.encode("ascii")))


def main(checkpoint, port):


    print(f"loading checkpoint from {checkpoint}")
    STATES = torch.load(checkpoint, map_location=torch.device('cpu'))
    GENERATOR.load_state_dict(STATES["g"])
    GENERATOR.train(False)


    noise = torch.randn(1, 100, 1, 1).to("cpu")
    arr = GENERATOR(noise)
    img = MetCenterCroppedDataset.pilify(arr[0])
    img.save("/checkpoints/test.png")
    print(img)

    print(f"serving at port {port}")
    httpd = http.server.HTTPServer(("", port), Handler)
    httpd.serve_forever()
