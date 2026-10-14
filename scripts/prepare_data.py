"""Download source data once, save exact seeded subsets and provenance."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from alignment.data import normalize_hh, normalize_sft, normalize_uf, split_rows
from alignment.io import file_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-size", type=int, default=30000)
    parser.add_argument("--dpo-size", type=int, default=15000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=300)
    args = parser.parse_args()
    output = Path(args.output_dir)
    metadata = {"seed": args.seed, "sources": {}}
    sources = [
        ("sft", "HuggingFaceH4/ultrachat_200k", "train_sft", normalize_sft, args.sft_size),
        ("hh", "Anthropic/hh-rlhf", "train", normalize_hh, args.dpo_size),
        ("uf", "HuggingFaceH4/ultrafeedback_binarized", "train_prefs", normalize_uf, args.dpo_size),
    ]
    for name, source, split, normalize, size in sources:
        dataset = load_dataset(source, split=split)
        normalized = [normalize(row) for row in dataset]
        train, val = split_rows(normalized, size, args.val_size, args.seed)
        for label, rows in (("train", train), ("val", val)):
            write_json(output / f"{name}_{label}.json", rows)
        metadata["sources"][name] = {"dataset": source, "split": split, "fingerprint": dataset._fingerprint,
                                      "source_rows": len(dataset), "valid_rows": sum(r is not None for r in normalized),
                                      "train": len(train), "validation": len(val)}
    # Read the published JSON directly instead of running a legacy dataset script.
    revision = HfApi().dataset_info("tatsu-lab/alpaca_eval").sha
    source_path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset", revision=revision)
    dataset = load_dataset("json", data_files=source_path, split="train")
    subset = dataset.shuffle(seed=args.seed).select(range(args.eval_size))
    rows = [{"id": f"alpaca-{args.seed}-{i:04d}", "instruction": r["instruction"], "dataset": r["dataset"]} for i, r in enumerate(subset)]
    write_json(output / "alpaca_eval.json", rows)
    metadata["sources"]["alpaca"] = {"dataset": "tatsu-lab/alpaca_eval", "revision": revision,
                                     "source_sha256": file_hash(source_path), "fingerprint": dataset._fingerprint, "size": len(rows)}
    metadata["sha256"] = {path.name: file_hash(path) for path in output.glob("*.json") if path.name != "manifest.json"}
    metadata["format"] = "### Instruction:\\n{instruction}\\n\\n### Response:\\n"
    metadata["filtering"] = "SFT first user/assistant turn; preferences strictly single-turn; duplicate prompts removed before splitting."
    write_json(output / "manifest.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
