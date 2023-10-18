import argparse

import nude2.benchmark
import nude2.browse
import nude2.data

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    browse_parser = subparsers.add_parser("browse")

    data_parser = subparsers.add_parser("data")

    benchmark_parser = subparsers.add_parser("benchmark")

    args = parser.parse_args()

    if args.command == "browse":
       nude2.browse.view()
    elif args.command == "data":
        nude2.data.main()
    elif args.command == "benchmark":
        nude2.benchmark.main()
