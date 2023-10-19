import argparse

import nude2.benchmark
import nude2.browse
import nude2.data
import nude2.progress

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    browse_parser = subparsers.add_parser("browse")

    data_parser = subparsers.add_parser("data")
    data_parser.add_argument("--concurrency", type=int, default=8)
    data_parser.add_argument("--limit", type=int, default=1000)

    benchmark_parser = subparsers.add_parser("benchmark")

    progress_parser = subparsers.add_parser("progress")

    args = parser.parse_args()

    if args.command == "browse":
       nude2.browse.view()
    elif args.command == "data":
        nude2.data.main(args.concurrency, args.limit)
    elif args.command == "benchmark":
        nude2.benchmark.main()
    elif args.command == "progress":
        nude2.progress.main()
