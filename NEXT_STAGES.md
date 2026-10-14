# Remaining verified workflow

1. Inspect saved controlled sanity version 358235518 at https://www.kaggle.com/code/ahmadakhtar007/pa2-controlled-generation-sanity?scriptVersionId=358235518 . Confirm success and inspect three responses from each of seven model/evaluation combinations. It uses repetition penalty 1.1 and no-repeat trigram size 3 for all models.
2. Launch a fresh saved generation notebook with `notebooks/saved_batches/generation_batch.py`. Attach only the completed DPO version 358022133 and persona version 358223311. Use GPU T4 x2 and internet. It does not retrain any model. Verify `generation-complete.json` and save output.
3. Launch a separate Kaggle utility benchmark notebook with `notebooks/saved_batches/judging_batch.py`, attaching the completed generation output. Model access must be enabled. Verify the configured judge exists before calls. Judgments are checkpointed after each batch; failures and unknowns must be counted rather than hidden.
4. Download the small plots, metrics, generation metadata, response files, pairwise files, judgments, summaries and test log. Keep full model weights in their existing saved Kaggle version.
5. Write an at-most-eight-page PDF report from actual results. Include the configurations, checkpoint selection, decoding adjustment, separate HH/UF/persona comparisons, representative failures, and reproducibility links. Validation preference accuracy is not a judge win rate.
6. Add final artifacts to this repository. Grant course reviewers access to the private GitHub repository and private Kaggle outputs before submitting the links.

Current saved training source: https://www.kaggle.com/code/ahmadakhtar007/pa2-dpo-saved-batch/output?scriptVersionId=358022133
Current saved persona/plots source: https://www.kaggle.com/code/ahmadakhtar007/pa2-persona-plots-sanity/output?scriptVersionId=358223311

The current source ZIP is a progress backup, not a finished assignment submission. Evaluation and report are not yet complete.
