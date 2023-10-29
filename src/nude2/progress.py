import random
import time
import sys


class Progress(object):
    """..."""

    def __init__(self, iterable, prefix="", size=100, clear=True):
        pass

    def __iter__(self):
        for i in self.iterable:
            yield i
            self.show()

    def show(self):
        pass


class ProgressBar(object):
    """..."""

    def __init__(self, limit, prefix="", size=100, clear=True):
        self.clear = clear
        self.prefix = prefix
        self.size = size
        self.limit = limit
        self.current = 0
        self.line = self.prefix + "[" + " "*self.size + "]"
        self.show()

    def x_x(self):
        eye = ["x", "-", "^", "o", "O"]
        return f"{random.choice(eye)}_{random.choice(eye)}"

    def show(self):
        pct = self.current / self.limit
        pct = round(pct * self.size)
        sys.stdout.write("\b"*len(self.line))
        self.line = self.prefix + "[" + "*"*pct + " "*(self.size-pct) + "] " + self.x_x()
        self.line += f" {100*self.current/self.limit:0.2f}%\r"
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

def main():
    with ProgressBar(300) as progress:
        for _ in range(300):
            time.sleep(0.02)
            progress.inc()
