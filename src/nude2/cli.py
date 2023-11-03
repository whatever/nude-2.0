import argparse

import nude2.benchmark
import nude2.browse
import nude2.data
import nude2.generate
import nude2.progress
import nude2.sample
import nude2.train

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    browse_parser = subparsers.add_parser("browse")

    data_parser = subparsers.add_parser("data")
    data_parser.add_argument("--concurrency", type=int, default=8)
    data_parser.add_argument("--limit", type=int, default=1000)

    benchmark_parser = subparsers.add_parser("benchmark")

    progress_parser = subparsers.add_parser("progress")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--checkpoint", type=str, required=True)
    train_parser.add_argument("--samples-path", type=str)
    train_parser.add_argument("--seed", type=int, default=42069)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--checkpoint", type=str)
    generate_parser.add_argument("--samples", type=str)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--checkpoint", type=str)
    sample_parser.add_argument("--samples", type=str)
    sample_parser.add_argument("-o", "--out", type=str)

    args = parser.parse_args()

    if args.command == "browse":
       nude2.browse.view()
    elif args.command == "data":
        nude2.data.main(args.concurrency, args.limit)
    elif args.command == "benchmark":
        nude2.benchmark.main()
    elif args.command == "progress":
        nude2.progress.main()
    elif args.command == "train":
        nude2.train.main(
            args.data,
            args.epochs,
            args.batch_size,
            args.checkpoint,
            args.samples_path,
            seed=args.seed,
        )
    elif args.command == "generate":
        nude2.generate.main(args.checkpoint)
    elif args.command == "sample":
        nude2.sample.main(args.samples, args.out)

