import random
import time
import sys

from datetime import datetime

class ProgressBar(object):
    """..."""

    def __init__(self, x, prefix="", size=100, clear=True):

        if isinstance(x, int):
            self.iter = range(x)
        else:
            self.iter = iter(x)

        self.start_time = datetime.now()
        self.clear = clear
        self.prefix = prefix
        self.size = size
        self.current = 0
        self.line = self.prefix + "[" + " "*self.size + "]"
        self.show()

    def x_x(self):
        eye = ["x", "-", "^", "o", "O"]
        return f"{random.choice(eye)}_{random.choice(eye)}"

    def show(self):

        # Percentage complete
        pct = self.current / len(self.iter)
        pct = round(pct * self.size)

        # Time elapsed
        dur = datetime.now() - self.start_time

        t = int(dur.total_seconds())
        r = dur.total_seconds() - t

        h = t // 3600
        m = (t - h*3600) // 60
        s = t - h*3600 - m*60
        f = t - h*3600 - m*60 - s

        sys.stdout.write("\b"*len(self.line))
        self.line = self.prefix + "[" + "*"*pct + " "*(self.size-pct) + "] " + self.x_x()
        self.line += f" {100*self.current/len(self.iter):0.2f}% [{h}:{m:02}:{s:02}] @ {datetime.now().strftime('%H:%M:%S')}\r"
        sys.stdout.write(self.line)
        sys.stdout.flush()

    def inc(self):
        self.current += 1
        self.show()

    def set(self, value):
        self.current = value
        self.show()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print(self.line)
        if self.clear:
            sys.stdout.write("\b"*len(self.line))

class I(object):
    def __iter__(self):
        return self
    def __next__(self):
        return 1

def main():
    for _ in range(5):
        with ProgressBar(66) as progress:
            for _ in range(66):
                time.sleep(0.02)
                progress.inc()
