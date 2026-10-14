# Final report: evidence to collect

Write at most eight pages, including figures, tables, and references, after the real runs finish. This guide is not a completed report and contains no fabricated experimental results.

## Repository, models, and data

Include your GitHub URL. Identify base models, source datasets, actual row counts, filtering, fingerprints, hashes, and seed from saved manifests. Identify each evaluation file. Disclose the manual prompt template, 256-token prompt-tail cap, separate response tokenization, EOS append, and persona chat template.

## Configuration table

Use one row for each of the four training stages. Read saved run configurations rather than proposed defaults. Record trainable strategy, LoRA parameters, optimizer, learning rate, schedule, warmup, weight decay, clipping, batch size, accumulation, effective batch size, precision, sequence/prompt lengths, epochs/steps, seed, beta, and exact reference checkpoint. Include generation settings from the `.meta.json` files.

## Choices and deviations

Explain each changed parameter with measured evidence. Separate memory-driven changes from performance choices. Report token-mean SFT loss and FP32 trainable weights with FP16 autocast. Preserve results for any attempted settings you discuss.

## Training behavior

Use each run's `training_metrics.png` and raw logs. Show SFT training/validation loss and DPO loss plus reward accuracy or margin. Explain checkpoint selection, instability, or overfitting. Label any smoothing. Do not infer generation quality solely from training loss.

## DPO versus SFT

Report HH and UltraFeedback separately: sample count, wins, ties, losses, unknowns, unresolved failures, both documented win-rate denominators, judge model, and retry information. Verify identical baseline/candidate decoding settings and prompts. Cite summary, judgment, and attempt-log paths.

## Persona result

Compare LoRA with the frozen SmolLM2 instruct baseline on the persona prompt set. Report style, usefulness, and safety separately. Include representative improvements and factuality, tone, helpfulness, or safety regressions. Keep this evaluation separate from HH and general helpfulness scores.

## Interpretation and limitations

State which checkpoint did better on each set and by how much. Separate observations from possible causes. Connect explanations to preference-data characteristics, beta, training duration, reward diagnostics, and response examples. Discuss judge bias, the small persona sample, one seed, synthetic HH prompts, truncation, failed runs, and compute constraints.

## Reproducibility

List actual paths to configurations, prepared manifests/subsets, selected checkpoints, plots, generation metadata, pairs, judgments, failures, and summaries. Cite external documentation and papers used. Export the finished report to PDF and verify the eight-page limit and figure readability.
