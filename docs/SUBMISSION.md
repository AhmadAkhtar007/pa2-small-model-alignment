# Submission contents

- Repository URL: https://github.com/AhmadAkhtar007/pa2-small-model-alignment
- Final report: `report/PA2_report.pdf`; editable LaTeX: `report/PA2_report.tex`.
- Original training plots: `results/training/{sft,dpo_hh,dpo_uf,persona}/training_metrics.png`.
- Measured evaluation summary: `results/evaluation-summary.json`.
- All generated answers, pair mappings, and 612 final judgments: `results/evaluation/`.
- Historical retries/failures: `results/judging-history/`.
- Required grading interface: root `submission_adapter.py`.
- Configurations and implementation: `configs/`, `alignment/`, `scripts/`, and `tests/`.
- Saved stage versions and full-weight locations: `docs/ARTIFACTS.md`.
- Evidence integrity: `results/SHA256SUMS.txt`.

Submit the repository URL and final report PDF using the course submission form. If plots and summary are requested as separate uploads, use the paths above. Do not upload the 10 GB model checkpoint bundle. This package does not itself constitute a submission to the course platform.

All training, generation, and judging are complete. Final counts: HH 168/95/37 wins/losses/ties; UF 95/138/67; persona 4/4/4. No final unknowns or unresolved judgments. The persona adapter did not improve style fidelity, and model outputs contain correctness and safety failures; these limitations are documented in the report.
