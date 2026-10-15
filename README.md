# Small-model alignment: SFT and DPO

Implementation of the PA2 assignment on top of the [course starter](https://github.com/ahmad903501/small-model-alignment-starter).

## Submission guide

| What to review | Location |
|---|---|
| Final report (6 pages) | [PDF](report/PA2_report.pdf) |
| SFT and DPO training plots | [Training results](results/training/) |
| DPO-versus-SFT evaluation | [Summary](results/evaluation-summary.csv) |
| Required grading interface | [submission_adapter.py](submission_adapter.py) |
| Submission checklist | [Submission contents](docs/SUBMISSION.md) |
| Saved Kaggle runs and checkpoints | [Artifact links](docs/ARTIFACTS.md) |

The repository is organized by purpose: `report/` contains the report and its figures, `results/` contains the experiment evidence, and `docs/` contains supporting documentation. Implementation is in `alignment/` and `scripts/`, with settings in `configs/`, workflows in `notebooks/`, and checks in `tests/`. Provided training data and fixed evaluation prompts are in `data/` and `eval_sets/`. Large model weights are retained in saved Kaggle outputs.

**Completed experiments:** SFT, HH DPO, UF DPO, persona LoRA, 1,524 evaluation answers, and all 612 model judgments. Final unresolved failures and unknowns: zero. See the [LaTeX report](report/PA2_report.pdf), [evaluation summary](results/evaluation-summary.json), and [saved-artifact links](docs/ARTIFACTS.md). The earlier Kaggle suite reported 31 passing tests; final evidence validation independently recomputed summaries and verified pair hashes.

| Candidate vs baseline | Wins | Losses | Ties | Win rate, ties counted half |
|---|---:|---:|---:|---:|
| HH DPO vs SFT | 168 | 95 | 37 | 62.17% |
| UF DPO vs SFT | 95 | 138 | 67 | 42.83% |
| Persona LoRA vs instruct base | 4 | 4 | 4 | 50.00% |

HH improved relative judge preference, UF regressed, and the persona adapter showed weak style fidelity. These scores do not establish that outputs are correct or safe. Full weights are excluded; saved Kaggle checkpoint links are retained separately. The report and `results/` provide the small reviewable deliverables.

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

1. Clone this repository in Kaggle: `git clone https://github.com/AhmadAkhtar007/pa2-small-model-alignment.git`. The submitted runs originally used a Google Drive ZIP; that workflow deviation is disclosed in the report.
2. Import `notebooks/01_training.ipynb`, attach the project dataset, enable internet, and select **GPU T4 x2**. Save a notebook version after each completed stage.
3. Import `notebooks/02_generation.ipynb`, attach the project and saved training outputs, and enable a GPU. Inspect the three sanity responses per model before full generation. Save a notebook version.
4. Create a **utility benchmark** notebook from `notebooks/03_evaluation.ipynb`, attach the project and saved generation outputs, and enable supported model access. The judge setting is `google/gemini-3.6-flash`; choose another supported judge if unavailable and record it.
5. Preserve plots, actual configurations, generations and metadata, pairs, judgments, failures, and summaries. Write the report from real results using `docs/REPORT_GUIDE.md`.

The three numbered notebooks are generic workflow templates and were not executed top-to-bottom. Actual saved-stage orchestration and recovery helpers are in `notebooks/saved_batches/`; exact saved run versions are linked in `ARTIFACTS.md`. Generation uses two workers, one per GPU. Judging uses five-pair batches, then individual retries only for five failed persona pairs. The generation helper additionally includes a public-adapter preflight for future reruns; it was not part of the submitted generation run.

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

The final repository passed all **7 public adapter smoke tests** on CPU. See the [test output](results/public-adapter-tests.txt) and [environment record](results/public-adapter-tests.json). This verifies the public checklist requirement; private staff checks remain separate.

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
