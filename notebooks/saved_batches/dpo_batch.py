from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

WORK = Path('/kaggle/working')
REPO = WORK / 'small-model-alignment'
ENV = os.environ.copy()
ENV.update(PYTHONUNBUFFERED='1', PYTHONPATH=str(REPO), HF_HOME='/kaggle/temp/dpo-cache', HF_HUB_DISABLE_XET='1')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

print('1. Locating saved HH checkpoint and completed SFT...', flush=True)
candidates = [p.parents[3] for p in Path('/kaggle/input').rglob('training_state.pt')
              if p.parent.parent.name == 'dpo_hh']
assert len(candidates) == 1, f'Expected one saved HH source: {candidates}'
source = candidates[0]
assert read(source / 'outputs/sft/complete.json')['optimizer_steps'] == 938
assert not REPO.exists(), 'Fresh batch working directory required.'

def ignore(folder, names):
    excluded = {'__pycache__'}
    if Path(folder) == source / 'outputs/sft':
        excluded.add('final')
    return excluded

shutil.copytree(source, REPO, ignore=ignore)
print('Saved project restored:', source, flush=True)
print('2. Installing tested dependencies while preserving PyTorch...', flush=True)
torch_version = version('torch')
constraints = WORK / 'keep-torch.txt'
constraints.write_text(f'torch=={torch_version}\n')
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', str(constraints),
                'transformers==4.49.0', 'datasets==3.6.0', 'peft==0.14.0', 'trl==0.15.2',
                'accelerate', 'sentencepiece', 'safetensors', 'matplotlib', 'pyyaml', 'tqdm'], check=True)
assert version('torch') == torch_version

check = '''
import torch
from pathlib import Path
assert torch.cuda.device_count() == 2, 'Select T4 x2 for this saved run.'
for i in range(2):
    print('GPU', i, torch.cuda.get_device_name(i), flush=True)
    assert 'T4' in torch.cuda.get_device_name(i)
state = torch.load('outputs/dpo_hh/resume/training_state.pt', map_location='cpu', weights_only=False)
assert state['step'] == 100
assert state['optimizer']['state']
print('Resuming HH step', state['step'], 'batch', state['batch'], flush=True)
'''
subprocess.run([sys.executable, '-u', '-c', check], cwd=REPO, env=ENV, check=True)

for track in ('hh', 'uf'):
    config = read(REPO / f'configs/dpo_{track}.json')
    assert config['model'] == config['reference_model'] == 'outputs/sft/best'
    assert len(read(REPO / config['train_path'])) == 15000
    assert len(read(REPO / config['val_path'])) == 1000

def train(track, resume):
    run = REPO / f'outputs/dpo_{track}'
    log = WORK / f'dpo-{track}-training.log'
    command = [sys.executable, '-u', 'scripts/train.py', '--config', f'configs/dpo_{track}.json']
    if resume:
        command.append('--resume')
    print(f'Starting full {track.upper()} DPO. Resume={resume}', flush=True)
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                time.sleep(30)
                metrics = run / 'metrics.jsonl'
                latest = metrics.read_text().splitlines()[-1] if metrics.exists() else 'Loading model / initial validation'
                print(f'[{track.upper()} {(time.monotonic()-started)/60:.1f} min] {latest}', flush=True)
            print(log.read_text()[-6000:], flush=True)
            assert process.returncode == 0, f'{track} failed; see {log}'
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    result = read(run / 'complete.json')
    for checkpoint in ('best', 'final'):
        weights = list((run / checkpoint).glob('*.safetensors'))
        assert weights and all(p.stat().st_size > 100_000_000 for p in weights)
        assert (run / checkpoint / 'config.json').is_file()
    print(f'{track.upper()} completed and checkpoints verified: {result}', flush=True)
    return result

results = {'hh': train('hh', True)}
(WORK / 'hh-complete.json').write_text(json.dumps(results['hh'], indent=2))
results['uf'] = train('uf', False)
checksums = {str(p.relative_to(REPO)): sha(p) for p in (REPO / 'outputs').rglob('*.safetensors')}
(WORK / 'checkpoint-sha256.json').write_text(json.dumps(checksums, indent=2))
(WORK / 'dpo-complete.json').write_text(json.dumps(results, indent=2))
print('BOTH DPO RUNS COMPLETE. Best/final weights, logs and checksums are in saved outputs.', flush=True)
