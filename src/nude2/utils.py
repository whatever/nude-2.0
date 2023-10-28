"""
Feels lazy to name something "utils.py", but here we are.
"""


import os.path


def splash(splash_name):
    """Print a splash message."""
    fname = os.path.join(os.path.dirname(__file__), "splashes", f"{splash_name}.txt")
    with open(fname, "r") as f:
        txt = f.read()
        print(f"\033[95m{txt}\033[00m")
