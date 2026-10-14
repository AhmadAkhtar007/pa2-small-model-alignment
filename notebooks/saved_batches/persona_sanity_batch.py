from pathlib import Path
from importlib.metadata import version
import os, sys, json, shutil, subprocess

WORK = Path('/kaggle/working')
sources = [p.parent / 'small-model-alignment' for p in Path('/kaggle/input').rglob('dpo-complete.json')]
assert len(sources) == 1, sources
SOURCE = sources[0]
REPO = WORK / 'small-model-alignment'
assert not REPO.exists()
shutil.copytree(SOURCE, REPO, ignore=shutil.ignore_patterns('outputs', '__pycache__', '.git'))
os.chdir(REPO)
env = os.environ.copy()
env.update(PYTHONPATH=str(REPO), HF_HOME='/kaggle/temp/pa2-cache', HF_HUB_DISABLE_XET='1', PYTHONUNBUFFERED='1')
constraint = WORK / 'keep-torch.txt'
constraint.write_text('torch==' + version('torch') + '\n')
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', str(constraint), 'transformers==4.49.0', 'datasets==3.6.0', 'trl==0.15.2', 'peft==0.14.0', 'accelerate', 'sentencepiece', 'matplotlib', 'pyyaml'], check=True)

def run(script, *args, gpu=None):
    child_env = env.copy()
    if gpu is not None:
        child_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    subprocess.run([sys.executable, '-u', script, *map(str, args)], cwd=REPO, env=child_env, check=True)

print('Generating plots from completed saved runs...', flush=True)
for stage in ('sft', 'dpo_hh', 'dpo_uf'):
    target = REPO / 'outputs' / stage
    target.mkdir(parents=True)
    for name in ('metrics.jsonl', 'run.json', 'complete.json', 'best_metrics.json'):
        path = SOURCE / 'outputs' / stage / name
        if path.exists():
            shutil.copy2(path, target / name)
    run('scripts/plot_metrics.py', '--run-dir', target)

print('Training required persona LoRA experiment...', flush=True)
run('scripts/train_persona.py', '--config', 'configs/persona.json', gpu=0)
assert (REPO / 'outputs/persona/best/adapter_model.safetensors').is_file()
run('scripts/plot_metrics.py', '--run-dir', 'outputs/persona')

jobs = [
    ('qwen-base', 'Qwen/Qwen2.5-0.5B', SOURCE / 'data/prepared/alpaca_eval.json', 'base_alpaca', []),
    ('sft', SOURCE / 'outputs/sft/best', SOURCE / 'data/prepared/alpaca_eval.json', 'sft_alpaca', []),
    ('sft', SOURCE / 'outputs/sft/best', REPO / 'eval_sets/hh_rlhf_eval_300_seed42.json', 'sft_hh', []),
    ('dpo-uf', SOURCE / 'outputs/dpo_uf/best', SOURCE / 'data/prepared/alpaca_eval.json', 'dpo_uf', []),
    ('dpo-hh', SOURCE / 'outputs/dpo_hh/best', REPO / 'eval_sets/hh_rlhf_eval_300_seed42.json', 'dpo_hh', []),
    ('smollm-instruct', 'HuggingFaceTB/SmolLM2-360M-Instruct', REPO / 'eval_sets/persona_eval_seed42.json', 'persona_base', ['--chat-template']),
    ('persona-lora', 'HuggingFaceTB/SmolLM2-360M-Instruct', REPO / 'eval_sets/persona_eval_seed42.json', 'persona_lora', ['--chat-template', '--adapter', REPO / 'outputs/persona/best']),
]
for label, model, prompts, filename, extra in jobs:
    path = REPO / f'outputs/eval/{filename}_sanity.json'
    run('scripts/generate.py', '--model', model, '--prompts', prompts, '--output', path, '--generator', label, '--limit', 3, *extra)
    for row in json.loads(path.read_text()):
        print(label, row['instruction'], '\nANSWER:', row['output'], flush=True)
(WORK / 'persona-sanity-complete.json').write_text(json.dumps({'source_version': 358022133, 'persona_complete': True, 'sanity_models': len(jobs)}, indent=2))
print('Persona, plots and sanity responses COMPLETE. Save these outputs before full generation.', flush=True)
