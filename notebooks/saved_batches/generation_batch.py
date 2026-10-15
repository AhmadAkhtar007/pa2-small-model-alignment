from pathlib import Path
from importlib.metadata import version
import os, sys, json, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from zipfile import ZipFile, ZIP_DEFLATED

WORK = Path('/kaggle/working')
SANITY_ONLY = False
dpo_sources = [p.parent / 'small-model-alignment' for p in Path('/kaggle/input').rglob('dpo-complete.json')]
persona_sources = [p.parent / 'small-model-alignment' for p in Path('/kaggle/input').rglob('persona-sanity-complete.json')]
assert len(dpo_sources) == len(persona_sources) == 1
DPO, PERSONA = dpo_sources[0], persona_sources[0]
REPO = WORK / f'evaluation-{time.time_ns()}'
shutil.copytree(PERSONA, REPO, ignore=shutil.ignore_patterns('__pycache__', '.git'))
os.chdir(REPO)
env = os.environ.copy()
env.update(PYTHONPATH=str(REPO), HF_HOME='/kaggle/temp/pa2-cache', HF_HUB_DISABLE_XET='1', PYTHONUNBUFFERED='1')
constraint = WORK / 'keep-torch.txt'
constraint.write_text('torch==' + version('torch') + '\n')
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', str(constraint), 'transformers==4.49.0', 'datasets==3.6.0', 'peft==0.14.0', 'sentencepiece', 'pyyaml', 'pytest'], check=True)
test_env = env.copy()
test_env['CUDA_VISIBLE_DEVICES'] = ''
test_result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_public_adapter_smoke.py'],
                             cwd=REPO, env=test_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(WORK / 'public-adapter-tests.log').write_text(test_result.stdout)
print(test_result.stdout, flush=True)
assert test_result.returncode == 0, 'Public adapter verification failed'

def run(script, *args):
    subprocess.run([sys.executable, '-u', script, *map(str, args)], cwd=REPO, env=env, check=True)

jobs = [
    ('qwen-base', 'Qwen/Qwen2.5-0.5B', DPO / 'data/prepared/alpaca_eval.json', 'base_alpaca', []),
    ('sft', DPO / 'outputs/sft/best', DPO / 'data/prepared/alpaca_eval.json', 'sft_alpaca', []),
    ('sft', DPO / 'outputs/sft/best', REPO / 'eval_sets/hh_rlhf_eval_300_seed42.json', 'sft_hh', []),
    ('dpo-uf', DPO / 'outputs/dpo_uf/best', DPO / 'data/prepared/alpaca_eval.json', 'dpo_uf', []),
    ('dpo-hh', DPO / 'outputs/dpo_hh/best', REPO / 'eval_sets/hh_rlhf_eval_300_seed42.json', 'dpo_hh', []),
    ('smollm-instruct', 'HuggingFaceTB/SmolLM2-360M-Instruct', REPO / 'eval_sets/persona_eval_seed42.json', 'persona_base', ['--chat-template']),
    ('persona-lora', 'HuggingFaceTB/SmolLM2-360M-Instruct', REPO / 'eval_sets/persona_eval_seed42.json', 'persona_lora', ['--chat-template', '--adapter', REPO / 'outputs/persona/best']),
]
pending = Queue()
for job in jobs:
    pending.put(job)

def generate_job(job, device):
    label, model, prompts, filename, extra = job
    print('Generating full evaluation subset:', filename, flush=True)
    suffix = '_controlled_sanity' if SANITY_ONLY else ''
    path = REPO / f'outputs/final_eval/{filename}{suffix}.json'
    run('scripts/generate.py', '--model', model, '--prompts', prompts, '--output', path, '--generator', label,
        '--device', f'cuda:{device}', '--repetition-penalty', '1.1', '--no-repeat-ngram-size', '3',
        *(['--limit', '3'] if SANITY_ONLY else []), *extra)
    rows = json.loads(path.read_text())
    expected = json.loads(Path(prompts).read_text())
    if SANITY_ONLY:
        expected = expected[:3]
        for row in rows:
            print(label, row['instruction'], '\nANSWER:', row['output'], flush=True)
    assert len(rows) == len(expected)
    assert {r['id'] for r in rows} == {str(r['id']) for r in expected}

def worker(device):
    while True:
        try:
            job = pending.get_nowait()
        except Empty:
            return
        generate_job(job, device)

with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(worker, device) for device in range(2)]
    for future in futures:
        future.result()

if SANITY_ONLY:
    (WORK / 'controlled-sanity-complete.json').write_text(json.dumps({'models': len(jobs), 'repetition_penalty': 1.1, 'no_repeat_ngram_size': 3}, indent=2))
    print('Controlled sanity responses saved. Inspect before full generation.', flush=True)
    sys.exit(0)

for baseline, candidate, name in [('sft_alpaca', 'dpo_uf', 'uf_pairs'), ('sft_hh', 'dpo_hh', 'hh_pairs'), ('persona_base', 'persona_lora', 'persona_pairs')]:
    run('scripts/build_pairs.py', '--baseline', f'outputs/final_eval/{baseline}.json', '--candidate', f'outputs/final_eval/{candidate}.json', '--output', f'outputs/final_eval/{name}.jsonl')
(WORK / 'generation-complete.json').write_text(json.dumps({'project': REPO.name, 'models': len(jobs), 'responses': 1524, 'response_directory': 'outputs/final_eval', 'hh_pairs': 300, 'uf_pairs': 300, 'persona_pairs': 12}, indent=2))
print('FULL GENERATION AND BLIND PAIRS COMPLETE. Save outputs before judging.', flush=True)
with ZipFile(WORK / 'report-artifacts.zip', 'w', ZIP_DEFLATED) as archive:
    for path in (REPO / 'outputs').rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.jsonl', '.png', '.csv'}:
            archive.write(path, path.relative_to(REPO))
    archive.write(WORK / 'generation-complete.json', 'generation-complete.json')
    archive.write(WORK / 'public-adapter-tests.log', 'public-adapter-tests.log')
print('Compact report artifacts saved.', flush=True)
