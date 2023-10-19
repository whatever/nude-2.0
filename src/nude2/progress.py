import random
import time
import sys

class ProgressBar(object):
    """..."""

    def __init__(self, limit, size=100):
        self.size = size
        self.limit = limit
        self.current = 0
        self.line = "[" + " "*self.size + "]"
        self.show()

    def x_x(self):
        eye = ["x", "-", "^", "o", "O"]
        return f"{random.choice(eye)}_{random.choice(eye)}"

    def show(self):
        pct = self.current / self.limit
        pct = round(pct * self.size)
        sys.stdout.write("\b"*len(self.line))
        self.line = "[" + "*"*pct + " "*(self.size-pct) + "] " + self.x_x()
        self.line += f" {100*self.current/self.limit:0.2f}%\r"
        sys.stdout.write(self.line)
        sys.stdout.flush()

    def inc(self):
        self.current += 1
        self.show()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print(self.line)

def main():
    with ProgressBar(300) as progress:
        for _ in range(300):
            time.sleep(0.02)
            progress.inc()
