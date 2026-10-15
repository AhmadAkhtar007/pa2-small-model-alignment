# Saved assignment artifacts

The full model weights are retained in Kaggle outputs and excluded from this repository.

- SFT and both DPO runs: https://www.kaggle.com/code/ahmadakhtar007/pa2-dpo-saved-batch/output?scriptVersionId=358022133
- Persona and original plots: https://www.kaggle.com/code/ahmadakhtar007/pa2-persona-plots-sanity?scriptVersionId=358223311
- Controlled decoding sanity (21 inspected answers): https://www.kaggle.com/code/ahmadakhtar007/pa2-controlled-generation-sanity?scriptVersionId=358235518
- Full generation (1,524 answers, 612 pairs): https://www.kaggle.com/code/ahmadakhtar007/pa2-controlled-generation-sanity?scriptVersionId=358554825
- Initial judging (600 main judgments and seven persona judgments): https://www.kaggle.com/code/ahmadakhtar007/pa2-saved-response-judging?scriptVersionId=358601648
- Persona retry (five missing judgments recovered): https://www.kaggle.com/code/ahmadakhtar007/pa2-saved-response-judging?scriptVersionId=358676439

All experiment stages completed successfully. The final 612 judgments have zero unknowns, pending items, or unresolved failures. The DPO version contains model checksums and both DPO completion records. The persona version contains the adapter and original plots. The initial five persona failures and their successful individual retries are retained under `results/judging-history/`.

Reviewable evidence is included in `results/`, the final PDF/source in `report/`, and the grading adapter at the repository root. Kaggle versions remain private and require separate access if full weights are requested; model weights are not included in the source ZIP. Do not infer generation quality from validation reward accuracy.
