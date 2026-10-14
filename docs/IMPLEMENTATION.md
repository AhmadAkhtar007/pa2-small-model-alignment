# Implementation and verification plan

The assignment manual defines the requirements. This implementation uses PyTorch for the manual SFT and two manual DPO runs, Transformers for model loading, and TRL/PEFT only for the separate SmolLM2 persona experiment.

1. Implement the six submission adapter functions and response-only objectives. Verify against the public suite, explicit token-level calculations, truncation boundaries, causal shifting, and a small preference optimization step.
2. Normalize data into single turns, remove duplicate prompts, split with seed 42, and save exact subsets and source fingerprints. Verify source-format filtering and train/validation separation.
3. Implement SFT and shared DPO training with FP32 trainable weights, FP16 CUDA autocast, clipping, accumulation, validation, and best/final checkpoints. Verify with a locally constructed tiny Transformers model, including a final incomplete accumulation group and interrupted-run resume.
4. Implement deterministic generation, blind seeded pair construction, and resumable judging with request spacing, retries, failures, and explicit denominators. Verify identity mapping and failure/resume behavior offline.
5. Provide separate Kaggle training, generation, and evaluation notebooks. Validate notebook structure and cell syntax locally. Actual T4 runs and model-proxy calls require Kaggle access and remain a separate empirical validation step.

## Decisions

- Manual models share the exact text template `### Instruction:\n{instruction}\n\n### Response:\n`. Responses are tokenized separately with no automatic special tokens; one EOS token is appended before truncation.
- SFT uses mean negative log probability over all scored response tokens. Accumulation groups are normalized by their total token count, including the final incomplete group. Validation uses the same reduction.
- DPO sums response log probabilities per example and minimizes the batch mean of the sigmoid objective. Reference computations use `no_grad`; reference weights are frozen. Strict positive reward margin defines preference accuracy; identical initial policy/reference models have zero accuracy in exact arithmetic. The FP16 reference on T4 may introduce small numerical differences at initialization.
- Manual training caps formatted prompts at 256 tokens by retaining their tail. This is also applied to generation. SFT therefore has the same prompt cap as DPO, a documented addition to the reference SFT configuration. Long instructions can lose context and should be discussed in the report.
- HH and UltraFeedback preference rows must be genuinely single-turn and agree on the user prompt. SFT extracts the first user/assistant turn from UltraChat. Prompts are deduplicated before selecting train and validation sets.
- AdamW is used for both manual DPO runs; the DPO learning rate is fixed at 1e-6 and beta is 0.1. Both policies and their references start from the same SFT best checkpoint.
- LoRA initially uses rank 8, alpha 16, zero dropout, and all attention projection modules, with learning rate 5e-5, batch size 1, accumulation 8, and three epochs. These are pilot settings, not measured optimal choices. The low rank limits trainable capacity on the small style dataset; zero dropout avoids stochastic differences in preference scoring. The frozen SmolLM2 instruct model is the reference and evaluation baseline. A seeded 10% holdout is saved.
- Transformers is constrained to 4.46-4.49 and TRL/PEFT to 0.15.2/0.14.0 so the tested trainer interface is reproducible. PyTorch is deliberately unpinned and Kaggle's CUDA build is preserved. AlpacaEval is loaded from its published JSON at a recorded revision rather than through its legacy dataset script.
- Greedy generation uses 256 new tokens, no repetition penalty beyond 1.0, and no n-gram constraint. All settings and checkpoint fingerprints are saved. Any later decoding change requires fresh outputs for both models in that comparison.
- Judge failures and unknown decisions are never counted as losses or wins. Decisive win rate excludes ties; ties-half win rate includes ties with weight 0.5. Both exclude failures and unknowns, which are reported separately against the full sample count.
- Persona judging also records per-response style and usefulness ratings (1-5) and safety assessments. These are reported separately for the baseline and adapter; they remain subjective judge measurements and need representative manual review.

## Remaining empirical work

Run the three notebooks with T4 x2 and model access, inspect sanity responses, preserve Kaggle versions, then write the report using actual configuration files, metrics, generations, judgments, and plots. No final model quality or win-rate claims can be made from offline toy tests.
