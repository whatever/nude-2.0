import argparse

import nude2.browse

def main():
    parser = argparse.ArgumentParser()


    subparsers = parser.add_subparsers(dest="command")

    browse_parser = subparsers.add_parser("browse")

    # data_parser = subparsers.add_parser("wrangle-data")
    # data_parser.add_argument("input_file", type=str)


    args = parser.parse_args()


    if args.command == "browse":
        nude2.browse.view()
