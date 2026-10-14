from pathlib import Path
import os, sys, json, shutil, subprocess

WORK = Path('/kaggle/working')
sources = [p.parent / 'small-model-alignment' for p in Path('/kaggle/input').rglob('generation-complete.json')]
assert len(sources) == 1, sources
REPO = WORK / 'small-model-alignment'
shutil.copytree(sources[0], REPO, ignore=shutil.ignore_patterns('__pycache__', '.git'))
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import kaggle_benchmarks as kbench
from scripts.judge_pairwise_kaggle import evaluate

JUDGE = 'google/gemini-3.6-flash'
assert JUDGE in kbench.llms, f'Configured judge unavailable: {JUDGE}'
summaries = {}
for name, candidate, mode in [('hh', 'dpo-hh', 'safety'), ('uf', 'dpo-uf', 'general'), ('persona', 'persona-lora', 'persona')]:
    print('Judging', name, 'with', JUDGE, flush=True)
    summaries[name] = evaluate(f'outputs/eval/{name}_pairs.jsonl', f'outputs/eval/{name}_judged.jsonl',
                               candidate, JUDGE, mode, batch_size=5, sleep_seconds=5, max_retries=3)
    (WORK / 'evaluation-summaries.json').write_text(json.dumps(summaries, indent=2))
(WORK / 'judging-complete.json').write_text(json.dumps({'judge': JUDGE, 'tracks': list(summaries)}, indent=2))
print('Judging finished. Inspect failure/unknown counts before writing the final report.', flush=True)
