"""Launch fresh validation pilots, not the final controlled ACL study."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--size', choices=['1.7B', '8B'], required=True)
    ap.add_argument('--lang', choices=['en', 'bn'], required=True)
    ap.add_argument('--stage', choices=['g0', 'smoke', 'pilot'], default='g0')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--gen-batch-size', type=int, default=4,
                    help='Generation/evaluation batch size; training batch is unchanged.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.gen_batch_size < 1:
        ap.error('--gen-batch-size must be positive')
    if not args.tag.replace('_', '').replace('-', '').isalnum():
        ap.error('--tag must contain only letters, digits, underscores or hyphens')
    os.chdir(ROOT)
    from collapse.config import load_config
    from collapse.cli import cmd_run
    # Keep data selection constant across training seeds and model sizes.
    # These are historical task choices, not a resource-controlled language study.
    overrides = {
        'model.name': 'Qwen/Qwen3-' + args.size,
        'model.max_len': 1024, 'gen.max_new_tokens': 128,
        'gen.batch_size': args.gen_batch_size, 'gen.sc_samples': 1,
        'train.mode': 'lora', 'train.per_device_bs': 1,
        'train.grad_accum': 16, 'train.epochs': 1,
        'train.max_len': 1024, 'train.optim': 'adamw',
        'data.lang': args.lang,
        'data.seed_pool': ['triviaqa', 'nq_open'] if args.lang == 'en' else ['squad_bn'],
        'data.eval_sets': ['global_mmlu_' + args.lang],
        'data.prompt_pool_size': 512, 'data.real_pool_size': 512,
        'data.eval_size': 400, 'data.probe_size': 0, 'probe.max_items': 0,
        'loop.generations': {'g0': 1, 'smoke': 2, 'pilot': 6}[args.stage],
        'loop.sft_size': 256, 'loop.ratio': 1.0,
        'loop.strict_ratio': True, 'loop.keep_checkpoints': True,
        'seed': 0, 'gen.seed': args.seed, 'train.seed': args.seed,
    }
    if args.stage == 'smoke':
        overrides.update({'data.prompt_pool_size': 32, 'data.real_pool_size': 32,
                          'data.eval_size': 16, 'loop.sft_size': 16})
    if args.stage == 'g0':
        overrides['data.eval_size'] = 100
    name = f'validation_{args.tag}_{args.stage}_{args.size}_{args.lang}_s{args.seed}'
    cfg = load_config(ROOT / 'configs/base.yaml', overrides)
    cfg.run_name = name
    config_path = ROOT / 'configs' / (name + '.yaml')
    if args.dry_run:
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(cfg), indent=2))
        return
    cfg.save(config_path)
    print(f'VALIDATION ONLY: {name}. Probes disabled; existing MCQ scoring remains exploratory.', flush=True)
    cmd_run(argparse.Namespace(config=str(config_path), set=[], run_name=None,
                               force=False, offline=False))


if __name__ == '__main__':
    main()
