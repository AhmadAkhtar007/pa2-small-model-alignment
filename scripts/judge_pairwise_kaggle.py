"""Run in a Kaggle utility benchmark notebook with model access enabled."""

import argparse
from pathlib import Path

from alignment.evaluation import judge_pairs, summarize
from alignment.io import read_rows, write_json


def evaluate(pairs_path, output_path, candidate, judge_model="google/gemini-3.6-flash", mode="general",
             batch_size=5, sleep_seconds=5, max_retries=3):
    import kaggle_benchmarks as kbench

    model = kbench.llms[judge_model]
    def request(prompt):
        with kbench.chats.new("pairwise_judgment"):
            return model.prompt(prompt)
    pairs = read_rows(pairs_path)
    judgments = judge_pairs(pairs, output_path, request, judge_model, batch_size, sleep_seconds, max_retries, mode)
    failure_path = Path(output_path).with_suffix(".failures.jsonl")
    failures = read_rows(failure_path) if failure_path.exists() else []
    summary = summarize(pairs, judgments, failures, candidate)
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--mode", choices=["general", "safety", "persona"], default="general")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--reuse_judgments", action="store_true", help="Judgments are always reused when their provenance matches")
    args = parser.parse_args()
    evaluate(args.pairs_path, args.output_path, args.candidate, args.judge_model, args.mode, args.batch_size, args.sleep_seconds, args.max_retries)


if __name__ == "__main__":
    main()
