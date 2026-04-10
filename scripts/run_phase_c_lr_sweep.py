#!/usr/bin/env python
"""Phase C: LR sweep for 10x-wide (512ch/48b) model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'
os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

try:
    import toml
except ImportError:
    import pip._vendor.tomli as toml


def run_lr_probe(*, name, peak_lr, final_lr, warmup_ratio,
                 base_config, steps, python_bin, torchrun_bin, cuda_vis,
                 batch_size_override=2048, grad_accum_override=4):
    run_dir = ROOT / 'artifacts' / 'tmp' / 'phase_c_lr_sweep' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = toml.loads(Path(base_config).read_text(encoding='utf-8'))

    # Model: 10x-wide
    cfg['resnet']['conv_channels'] = 512
    cfg['resnet']['num_blocks'] = 48
    cfg['resnet']['bottleneck_channels'] = 64
    cfg['resnet']['hidden_dim'] = 2048

    # Training
    cfg['bc']['control']['max_steps'] = steps
    cfg['bc']['control']['max_runtime_seconds'] = 14400
    cfg['bc']['control']['save_every'] = steps
    cfg['bc']['control']['train_log_every'] = 25
    cfg['bc']['control']['val_steps'] = 8
    cfg['bc']['control']['best_eval_every'] = 0
    cfg['bc']['control']['batch_size'] = batch_size_override
    cfg['bc']['control']['grad_accum_steps'] = grad_accum_override
    cfg['bc']['control']['state_file'] = str(run_dir / 'state.pth')
    cfg['bc']['control']['best_state_file'] = str(run_dir / 'best.pth')
    cfg['bc']['control']['tensorboard_dir'] = str(run_dir / 'tensorboard')
    cfg['bc']['control']['metrics_jsonl'] = str(run_dir / 'metrics.jsonl')

    # LR overrides
    cfg['bc']['optim']['scheduler']['peak'] = peak_lr
    cfg['bc']['optim']['scheduler']['final'] = final_lr
    cfg['bc']['optim']['scheduler']['warm_up_ratio'] = warmup_ratio

    cfg['bc']['preflight']['min_steps_before_stop'] = steps
    cfg['bc']['preflight']['min_runtime_seconds'] = 120
    cfg['bc']['preflight']['summary_json'] = str(run_dir / 'preflight_summary.json')
    cfg['bc']['preflight']['disable_wandb'] = True
    cfg['bc']['preflight']['min_samples_per_second'] = 300
    cfg['bc']['preflight']['preferred_samples_per_second'] = 1000
    cfg['bc']['preflight']['max_loader_wait_fraction'] = 0.30
    cfg['bc']['preflight']['min_steady_gpu_ratio'] = 0.50

    patched_config = run_dir / 'config.toml'
    patched_config.write_text(toml.dumps(cfg), encoding='utf-8')

    log_path = run_dir / 'run.log'
    command = [
        python_bin, 'scripts/run_bc_loader_preflight.py',
        '--config', str(patched_config),
        '--python-bin', python_bin,
        '--torchrun-bin', torchrun_bin,
    ]
    env = dict(os.environ)
    env['MORTAL_CFG'] = str(patched_config)
    env['CUDA_VISIBLE_DEVICES'] = cuda_vis

    print(f'\n{"=" * 90}')
    print(f'LR SWEEP: {name}  peak={peak_lr}  final={final_lr}  warmup={warmup_ratio}')
    print(f'{"=" * 90}')

    t0 = time.monotonic()
    with log_path.open('w', encoding='utf-8') as fh:
        fh.write(f'$ {" ".join(command)}\n')
        fh.flush()
        result = subprocess.run(command, cwd=str(ROOT), env=env, check=False, stdout=fh, stderr=fh)
    wall = time.monotonic() - t0

    # Parse metrics
    train_metrics = []
    metrics_path = run_dir / 'metrics.jsonl'
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('event') == 'train_live':
                train_metrics.append(event)

    last_n = min(5, len(train_metrics))
    recent = train_metrics[-last_n:] if train_metrics else []

    def avg(events, *keys):
        vals = []
        for e in events:
            v = e
            for k in keys:
                v = (v or {}).get(k)
                if v is None: break
            if v is not None: vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

    # Collect curve
    curve = []
    for e in train_metrics:
        tm = e.get('train_metrics') or {}
        curve.append({
            'step': e.get('step', 0),
            'nll': round(float(tm.get('nll', 0)), 4),
            'top1': round(float(tm.get('accuracy', 0)), 4),
        })

    r = {
        'name': name,
        'peak_lr': peak_lr,
        'final_lr': final_lr,
        'warmup_ratio': warmup_ratio,
        'batch_size': batch_size_override,
        'grad_accum': grad_accum_override,
        'completed_step': int(train_metrics[-1].get('step', 0)) if train_metrics else 0,
        'nll': round(avg(recent, 'train_metrics', 'nll'), 4),
        'top1': round(avg(recent, 'train_metrics', 'accuracy'), 4),
        'topk': round(avg(recent, 'train_metrics', 'topk_accuracy'), 4),
        'wall': round(wall, 1),
        'rc': result.returncode,
        'curve': curve,
    }
    print(f'  steps: {r["completed_step"]}/{steps}  nll: {r["nll"]}  top1: {r["top1"]}  '
          f'wall: {r["wall"]}s  rc: {r["rc"]}')
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--cuda-visible-devices', default='0,1')
    parser.add_argument('--python-bin', default=sys.executable)
    args = parser.parse_args()

    torchrun_bin = str(Path(args.python_bin).with_name('torchrun'))
    base_config = str(ROOT / 'configs' / 'step6_bc_phase_c_probe_baseline.toml')

    variants = [
        # (name, peak_lr, final_lr, warmup_ratio, batch_size, grad_accum)
        # bs=3072, ga=2 → eff_batch=12,288
        ('bs3072_lr_4e-4', 4e-4, 4e-5, 0.05, 3072, 2),
        ('bs3072_lr_2e-4', 2e-4, 2e-5, 0.05, 3072, 2),
        ('bs3072_lr_1e-4', 1e-4, 1e-5, 0.05, 3072, 2),
        # bs=2048, ga=4 → eff_batch=16,384 (reference)
        ('bs2048_lr_4e-4', 4e-4, 4e-5, 0.05, 2048, 4),
        ('bs2048_lr_2e-4', 2e-4, 2e-5, 0.05, 2048, 4),
        ('bs2048_lr_1e-4', 1e-4, 1e-5, 0.05, 2048, 4),
    ]

    results = []
    for name, peak, final, warmup, bs, ga in variants:
        r = run_lr_probe(
            name=name, peak_lr=peak, final_lr=final, warmup_ratio=warmup,
            base_config=base_config, steps=args.steps,
            python_bin=args.python_bin, torchrun_bin=torchrun_bin,
            cuda_vis=args.cuda_visible_devices,
            batch_size_override=bs, grad_accum_override=ga,
        )
        results.append(r)

    print('\n' + '=' * 110)
    print(f'PHASE C LR SWEEP RESULTS ({args.steps} steps, 512ch/48b 83.4M params)')
    print('=' * 110)
    print(f'{"Config":<25} {"Peak_LR":>9} {"Final_LR":>9} {"Warmup":>7} '
          f'{"BS":>5} {"GA":>2} {"EffBS":>6} '
          f'{"NLL":>8} {"Top1":>7} {"TopK":>7} {"Steps":>5}')
    print('-' * 120)
    for r in results:
        bs = r.get('batch_size', 2048)
        ga = r.get('grad_accum', 4)
        eff = bs * ga * 2
        print(f'{r["name"]:<25} {r["peak_lr"]:>9.1e} {r["final_lr"]:>9.1e} {r["warmup_ratio"]:>7.2f} '
              f'{bs:>5} {ga:>2} {eff:>6} '
              f'{r["nll"]:>8.4f} {r["top1"]:>7.4f} {r["topk"]:>7.4f} {r["completed_step"]:>5}')

    # Print curves side by side at key steps
    print(f'\nLearning curves (NLL):')
    step_set = sorted(set(pt['step'] for r in results for pt in r['curve']))
    header = f'{"Step":>6}'
    for r in results:
        header += f'  {r["name"][:18]:>18}'
    print(header)
    for step in step_set:
        if step % 50 != 0 and step != 25:
            continue
        row = f'{step:>6}'
        for r in results:
            pt = next((p for p in r['curve'] if p['step'] == step), None)
            row += f'  {pt["nll"]:>18.4f}' if pt else f'  {"":>18}'
        print(row)

    output_path = ROOT / 'artifacts' / 'reports' / 'phase_c_probes' / 'lr_sweep_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
