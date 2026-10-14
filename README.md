# Small-model alignment: SFT and DPO

Implementation of the PA2 assignment on top of the [course starter](https://github.com/ahmad903501/small-model-alignment-starter).

**Status:** SFT and both full manual DPO runs completed on Kaggle T4 x2. Persona, evaluation, and the final report are in progress. See `experiment-status.json` and `ARTIFACTS.md` for verified stage status and saved checkpoint links. Earlier verification included 31 tests; this snapshot has not yet been freshly retested.

## What is included

```text
alignment/core.py             Response-only SFT and DPO objectives
alignment/training.py         Shared manual training loop
alignment/data.py             Single-turn conversion and seeded splitting
alignment/evaluation.py       Blind judging, retries, resume, summaries
submission_adapter.py         Six unchanged required signatures
scripts/                      Data preparation, training, generation, judging, plots
configs/                      Explicit settings for all four training stages
notebooks/                    Separate Kaggle training, generation, judging
tests/                        Public tests and additional offline verification
docs/IMPLEMENTATION.md        Decisions and deviations to disclose
docs/REPORT_GUIDE.md          Evidence required for the final report
data/persona_lora/            Provided persona preferences
eval_sets/                    Fixed HH and persona evaluation prompts
```

## Run on Kaggle

1. Upload `small-model-alignment.zip` as a private Kaggle dataset.
2. Import `notebooks/01_training.ipynb`, attach the project dataset, enable internet, and select **GPU T4 x2**. Save a notebook version after each completed stage.
3. Import `notebooks/02_generation.ipynb`, attach the project and saved training outputs, and enable a GPU. Inspect the three sanity responses per model before full generation. Save a notebook version.
4. Create a **utility benchmark** notebook from `notebooks/03_evaluation.ipynb`, attach the project and saved generation outputs, and enable supported model access. The judge setting is `google/gemini-3.6-flash`; choose another supported judge if unavailable and record it.
5. Preserve plots, actual configurations, generations and metadata, pairs, judgments, failures, and summaries. Write the report from real results using `docs/REPORT_GUIDE.md`.

The notebooks include setup and handoff instructions. They have not been executed top-to-bottom on Kaggle. Publishing to GitHub is optional for running the packaged notebooks, but the assignment submission requires your completed repository URL.

## Local verification

Use Python 3.10 or newer and an existing suitable PyTorch installation. Requirements do not pin torch; preserve Kaggle's CUDA build.

```bash
python -m pip install -r requirements.txt -r requirements-persona.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Offline tests construct local models and tokenizers; no model or assignment dataset download is required. The full suite needs TRL and PEFT too.

## Training commands

Run from the repository root. The default manual DPO configuration uses the policy on GPU 0 and frozen reference on GPU 1.

```sh
python scripts/prepare_data.py
python scripts/train.py --config configs/sft.json
python scripts/train.py --config configs/dpo_hh.json
python scripts/train.py --config configs/dpo_uf.json
CUDA_VISIBLE_DEVICES=0 python scripts/train_persona.py --config configs/persona.json
```

The final command uses Kaggle/Linux shell syntax; on Windows set `CUDA_VISIBLE_DEVICES` in the environment first. Manual runs never use Trainer or TRL. Both DPO runs start from the same SFT best checkpoint.

Manual runs save best/final models, durable metrics, actual configuration/data hashes, and checkpoint-selection metrics. Add `--resume` after an interruption to restore the last completed validation checkpoint, optimizer, scheduler, scaler, RNG state, and batch cursor. Configurations and dataset hashes must match. Completed stages remove the large optimizer state and retain best/final models. Use a new output directory for a new experiment.

Persona runs save Trainer checkpoints, the selected best adapter, exact train/validation subsets, metrics, and the final adapter path in `complete.json`. Resume with `--resume-from-checkpoint PATH`.

```sh
python scripts/plot_metrics.py --run-dir outputs/sft
python scripts/plot_metrics.py --run-dir outputs/dpo_hh
python scripts/plot_metrics.py --run-dir outputs/dpo_uf
python scripts/plot_metrics.py --run-dir outputs/persona
```

## Submission adapter and public tests

The six function signatures are unchanged and call `alignment/core.py`.

Run the public tests from the repository root:

```bash
python -m pytest -q tests/test_public_adapter_smoke.py
```

SFT uses mean negative log probability over scored response tokens. DPO sums response log probabilities per example and takes the batch mean of the sigmoid objective. EOS is appended before truncation, and prompt/padding labels are masked. See `docs/IMPLEMENTATION.md` for the exact prompt format, truncation, precision, and parameter decisions to disclose in the report.

## Evaluation interpretation

Judge-facing prompts receive only the instruction and Response A/B; model mapping stays in metadata. Pair construction rejects mismatched prompt IDs, instructions, or datasets. Generation resume rejects changed settings, local weight hashes, or evaluation data.

Decisive win rate is wins divided by wins plus losses. Ties-half win rate is wins plus half of ties divided by wins plus losses plus ties. Both exclude unknowns and failures, which are reported separately against the complete sample count. Keep HH, UltraFeedback, and persona results separate.

Persona summaries separately report style, usefulness, and safety assessments for the instruct baseline and LoRA adapter. Inspect representative responses manually, including factuality and safety regressions in the stylized data. Judge assessments remain subjective.

## References

Objectives are implemented directly in PyTorch rather than copied from a public DPO training loop.

- [DPO paper](https://arxiv.org/abs/2305.18290)
- [PyTorch mixed precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Transformers documentation](https://huggingface.co/docs/transformers/v4.49.0/en/index)
- [TRL 0.15.2 DPOTrainer](https://huggingface.co/docs/trl/v0.15.2/dpo_trainer)
- [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [Kaggle Benchmarks quick start](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md)
