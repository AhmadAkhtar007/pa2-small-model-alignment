from pathlib import Path
import os, sys, json, shutil, subprocess
import threading, time, zipfile

WORK = Path('/kaggle/working')
markers = list(Path('/kaggle/input').rglob('generation-complete.json'))
assert len(markers) == 1, markers
manifest = json.loads(markers[0].read_text())
source = markers[0].parent / manifest['project']
REPO = WORK / 'small-model-alignment'
if not REPO.exists():
    shutil.copytree(source, REPO, ignore=shutil.ignore_patterns('__pycache__', '.git', '*.safetensors', '*.pt', '*.bin'))
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import kaggle_benchmarks as kbench
from scripts.judge_pairwise_kaggle import evaluate

JUDGE = 'google/gemini-3.6-flash'
kbench.llms[JUDGE]
def bundle():
    with zipfile.ZipFile(WORK / 'judging-results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in (REPO / 'outputs/final_eval').glob('*'):
            if path.is_file():
                archive.write(path, 'final_eval/' + path.name)
        for path in [WORK / 'evaluation-summaries.json', WORK / 'judging-complete.json']:
            if path.exists():
                archive.write(path, path.name)

@kbench.task(name='pa2_saved_response_judging')
def pa2_judging(llm) -> bool:
    stop = threading.Event()
    def progress():
        while not stop.wait(30):
            counts = {name: len(path.read_text().splitlines()) for name in ('hh', 'uf', 'persona')
                      if (path := REPO / f'outputs/final_eval/{name}_judged.jsonl').exists()}
            print('Saved judgments:', counts, flush=True)
    monitor = threading.Thread(target=progress, daemon=True)
    monitor.start()
    summaries = {}
    try:
        for name, candidate, mode in [('hh', 'dpo-hh', 'safety'), ('uf', 'dpo-uf', 'general'), ('persona', 'persona-lora', 'persona')]:
            print('Judging', name, 'with', JUDGE, flush=True)
            summaries[name] = evaluate(f'outputs/final_eval/{name}_pairs.jsonl', f'outputs/final_eval/{name}_judged.jsonl',
                                       candidate, JUDGE, mode, batch_size=5, sleep_seconds=5, max_retries=3)
            (WORK / 'evaluation-summaries.json').write_text(json.dumps(summaries, indent=2))
            bundle()
        (WORK / 'judging-complete.json').write_text(json.dumps({'judge': JUDGE, 'summaries': summaries}, indent=2))
        bundle()
        print('JUDGING COMPLETE', json.dumps(summaries, indent=2), flush=True)
        return True
    finally:
        stop.set()

pa2_judging.run(kbench.llms[JUDGE])
