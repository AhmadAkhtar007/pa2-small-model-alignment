import argparse

from alignment.io import read_rows
from alignment.training import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_training(read_rows(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
