import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alignment.io import read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    rows = read_rows(root / "metrics.jsonl")
    metrics = [("train_loss", "Training objective"), ("validation_loss", "Validation objective"),
               ("reward_accuracy", "Preference accuracy"), ("reward_margin", "Reward margin")]
    if any("loss" in row for row in rows):
        metrics = [("loss", "Training objective"), ("eval_loss", "Validation objective"),
                   ("eval_rewards/accuracies", "Preference accuracy"), ("eval_rewards/margins", "Reward margin")]
    present = [(key, title) for key, title in metrics if any(key in row for row in rows)]
    if not present:
        raise ValueError("No plottable training metrics")
    figure, axes = plt.subplots(len(present), 1, figsize=(7, 2.7 * len(present)), squeeze=False)
    for axis, (key, title) in zip(axes.flat, present):
        # On resume the same step may be logged again; use its last saved value.
        values = {row["step"]: row[key] for row in rows if key in row}
        steps = sorted(values)
        axis.plot(steps, [values[s] for s in steps], marker="." if len(steps) < 20 else None)
        axis.set(xlabel="Optimizer step", ylabel=title, title=title)
        axis.grid(alpha=.25)
        if "accurac" in key:
            axis.set_ylim(0, 1)
        best_file = root / "best_metrics.json"
        if best_file.exists():
            best = read_rows(best_file)["step"]
            axis.axvline(best, linestyle="--", color="#a64b2a", label="Selected best checkpoint")
            axis.legend()
    figure.suptitle(f"{root.name}: raw metrics, no smoothing")
    figure.tight_layout()
    figure.savefig(root / "training_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
