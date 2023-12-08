import io
import json
import http.server
import torch

from nude2.model import Generator198x198, Generator
from nude2.data import MetCenterCroppedDataset


GENERATOR = Generator198x198()


class Handler(http.server.SimpleHTTPRequestHandler):

    def write_image(self, vec, train=False):

        GENERATOR.train(train)

        with torch.no_grad():
            arr = GENERATOR(vec)

        img = MetCenterCroppedDataset.pilify(arr[0])

        res = io.BytesIO()
        img.save(res, format="png")
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        self.end_headers()
        self.wfile.write(res.getvalue())

    def do_GET(self):

        if self.path == "/api/v1/image":
            vec = torch.randn(1, 100, 1, 1).to("cpu")
            self.write_image(vec)
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

    def do_POST(self):

        if self.path != "/api/v1/image":
            return

        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body.decode("ascii"))

        if "vec" not in data:
            self.send_response(400)
            self.end_headers()
            return

        vec = torch.tensor(data["vec"]).reshape(1, 100, 1, 1).to("cpu")
        self.write_image(vec, train=data.get("train", True))


def main(checkpoint, port):

    print(f"loading checkpoint from {checkpoint}")
    STATES = torch.load(checkpoint, map_location=torch.device('cpu'))
    GENERATOR.load_state_dict(STATES["g"])

    print(f"serving at port {port}")
    httpd = http.server.HTTPServer(("", port), Handler)
    httpd.serve_forever()
