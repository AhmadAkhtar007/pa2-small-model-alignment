# Local verification

Verified on 2026-10-12 with Python 3.12, CPU PyTorch 2.14.1, Transformers 4.49.0, TRL 0.15.2, and PEFT 0.14.0.

- 31 tests passed, including all seven public adapter checks.
- Numeric checks cover prompt/padding masks, EOS-as-padding, truncation, response-only sums, causal shifting, row order, the sigmoid objective, and policy/reference gradient separation.
- A locally constructed tiny Transformers model completed SFT and DPO, saved best/final checkpoints, handled the final partial accumulation group, and resumed an interrupted run to the same weights as an uninterrupted run.
- A real TRL/PEFT tiny-model training run saved a selected LoRA adapter.
- Generation tests saved three outputs, resumed without duplicating them, and rejected changed decoding settings.
- Judge tests checked blind metadata handling, retries, spacing, resume, unresolved failures, and identity-correct persona ratings.
- All three notebooks passed notebook-schema and Python-cell syntax validation. Every script's command-line entry point was checked.
- Source spot checks sampled five rows each from UltraChat, HH-RLHF, and UltraFeedback. Five, two, and five rows respectively passed their intended normalizers. This validates sampled schemas, not entire datasets or final subset sizes.
- The published AlpacaEval JSON contained 805 rows; the checked revision was `2edc6fad8be6b14ea7230aabfd08188da6b8b814`.

The editable package was built and installed locally. Full-model training, GPU memory/runtime, complete data subsets, actual model-proxy calls, and empirical model quality remain unverified until Kaggle runs complete. The three local test warnings concern GPT2's Conv1D LoRA test fixture and CPU-only pinned memory.
