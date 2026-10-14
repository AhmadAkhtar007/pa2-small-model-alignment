import argparse
from pathlib import Path

from alignment.evaluation import build_pairs
from alignment.io import append_jsonl, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairs = build_pairs(read_rows(args.baseline), read_rows(args.candidate), args.seed)
    path = Path(args.output)
    if path.exists():
        raise ValueError("Output already exists; select a new path")
    for pair in pairs:
        append_jsonl(path, pair)


if __name__ == "__main__":
    main()
