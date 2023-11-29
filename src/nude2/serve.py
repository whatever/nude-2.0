import json
import http.server
import torch

from nude2.model import Generator198x198, Generator
from nude2.data import MetCenterCroppedDataset


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
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
    states = torch.load(checkpoint, map_location=torch.device('cpu'))

    g = Generator198x198()
    g = Generator()
    g = g.to("cpu")
    # g.load_state_dict(states["g"])

    noise = torch.randn(1, 100, 1, 1).to("cpu")
    print("<<<")
    try:
        arr = g(noise)
    except Exception as e:
        print(e)
    print(">>>")
    print(arr)
    # img = MetCenterCroppedDataset.pilify(arr[0])
    # img.save("/checkpoints/test.png")

    print(f"serving at port {port}")
    httpd = http.server.HTTPServer(("", port), Handler)
    httpd.serve_forever()
