import webbrowser
import http.server
import os
import threading
import time


from http.server import HTTPServer, SimpleHTTPRequestHandler


HOST = "127.0.0.1"
"""Open server locally"""

PORT = 8181
"""Default to port 8181"""

DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "static"))
"""Use /static/ directory"""


class StaticDirServer(SimpleHTTPRequestHandler):
    """Serve a directory of static files"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)
        print(DIR)


def serve():
    """Start the static server"""
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, StaticDirServer)
    httpd.serve_forever()


def view():
    threading.Thread(target=start_server).start()
    time.sleep(10)


def start_server():
    print("Starting server...")
    threading.Thread(target=serve).start()

    print("Sleeping for 1 second...")
    time.sleep(1)

    print("Opening browser...")
    webbrowser.open(f"http://{HOST}:{PORT}")
