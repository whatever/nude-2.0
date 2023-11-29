import json
import http.server
import torch


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
    torch.load(checkpoint, map_location=torch.device('cpu'))

    print(f"serving at port {port}")
    httpd = http.server.HTTPServer(("", port), Handler)
    httpd.serve_forever()
