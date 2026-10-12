"""Linux counterpart of run_all.ps1; completed summaries are skipped."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT / 'src')
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
jobs = []

def add(name, lang, seed=1, extra=()):
    batch, accum = (4, 4) if lang == 'bn' else (8, 2)
    overrides = [f'seed={seed}', f'gen.seed={seed}', f'train.seed={seed}',
                 'model.name=Qwen/Qwen3-1.7B', 'loop.generations=6',
                 f'train.per_device_bs={batch}', f'train.grad_accum={accum}']
    jobs.append((name, ['-m', 'collapse.cli', 'run', '--config',
                        f'configs/main_{lang}.yaml', '--run-name', name,
                        '--set', *overrides, *extra]))

for lang in ('en', 'bn'):
    for seed in (1, 2, 3):
        add(f'main_{lang}_r100_s{seed}', lang, seed, ['loop.ratio=1.0'])
for lang in ('en', 'bn'):
    for ratio in (0.75, 0.5, 0.0):
        add(f'main_{lang}_r{int(ratio*100)}_s1', lang, extra=[f'loop.ratio={ratio}'])
full = ['train.mode=full', 'train.lr=2.0e-5', 'train.per_device_bs=1', 'train.grad_accum=16']
for lang in ('en', 'bn'):
    add(f'ctrl_{lang}_fullft_s1', lang, extra=full)
    add(f'ctrl_{lang}_model2_s1', lang, extra=['model.name=Qwen/Qwen2.5-1.5B'])
for lang in ('en', 'bn'):
    add(f'h4_{lang}_selfverify_s1', lang, extra=['loop.filter=selfverify', 'loop.filter_threshold=0.5', 'loop.sft_size=1500'])
    add(f'h4_{lang}_cal_s1', lang, extra=['train.loss=cal', 'train.cal_gamma=1.0'])
for lang in ('en', 'bn'):
    for seed in (2, 3):
        add(f'main_{lang}_r50_s{seed}', lang, seed, ['loop.ratio=0.5'])
for seed in (2, 3):
    add(f'ctrl_bn_fullft_s{seed}', 'bn', seed, full)

dry = '--dry-run' in sys.argv
failed = []
for name, command in jobs:
    if (ROOT / 'runs' / name / 'summary.json').is_file():
        print(f'SKIP complete: {name}', flush=True)
        continue
    print(f'RUN: {name}\n{sys.executable} {" ".join(command)}', flush=True)
    if not dry and subprocess.run([sys.executable, *command]).returncode:
        failed.append(name)
        # Stop on a failure so shared setup problems do not waste the rental.
        break

if not dry and not failed:
    for lang in ('en', 'bn'):
        subprocess.run([sys.executable, '-m', 'collapse.cli', 'h1', '--prefix', f'main_{lang}_r100_s'], check=True)
    for run in sorted((ROOT / 'runs').glob('main_*')):
        if (run / 'summary.json').exists():
            subprocess.run([sys.executable, '-m', 'collapse.cli', 'analyze', '--run', str(run)], check=True)
if not dry:
    (ROOT / 'logs' / 'linux_runner_status.json').write_text(json.dumps({'failed': failed, 'complete': not failed}, indent=2))
sys.exit(bool(failed))
