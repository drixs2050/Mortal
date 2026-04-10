#!/usr/bin/env python
"""Phase C: LR sweep for bs=2560/ga=1 (eff_batch=5120), 1000 steps."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

try:
    import toml
except ImportError:
    import pip._vendor.tomli as toml


def kill_stale_workers():
    for line in subprocess.run(
        ['ps', '-eo', 'pid=,cmd='], capture_output=True, text=True
    ).stdout.splitlines():
        if 'mortal/train_bc.py' in line:
            pid = int(line.strip().split()[0])
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(2)


def run_variant(*, name, peak_lr, final_lr, warmup_ratio,
                base_config, steps, torchrun_bin, cuda_vis):
    run_dir = ROOT / 'artifacts' / 'tmp' / 'phase_c_lr_sweep_ga1' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = toml.loads(Path(base_config).read_text(encoding='utf-8'))

    # Model: 10x-wide
    cfg['resnet']['conv_channels'] = 512
    cfg['resnet']['num_blocks'] = 48
    cfg['resnet']['bottleneck_channels'] = 64
    cfg['resnet']['hidden_dim'] = 2048

    # Training: bs=2560, ga=1
    cfg['bc']['control']['max_steps'] = steps
    cfg['bc']['control']['max_runtime_seconds'] = 14400
    cfg['bc']['control']['save_every'] = steps
    cfg['bc']['control']['train_log_every'] = 25
    cfg['bc']['control']['val_steps'] = 8
    cfg['bc']['control']['best_eval_every'] = 0
    cfg['bc']['control']['batch_size'] = 2560
    cfg['bc']['control']['grad_accum_steps'] = 1
    cfg['bc']['control']['state_file'] = str(run_dir / 'state.pth')
    cfg['bc']['control']['best_state_file'] = str(run_dir / 'best.pth')
    cfg['bc']['control']['tensorboard_dir'] = str(run_dir / 'tensorboard')
    cfg['bc']['control']['metrics_jsonl'] = str(run_dir / 'metrics.jsonl')

    # LR overrides
    cfg['bc']['optim']['scheduler']['peak'] = peak_lr
    cfg['bc']['optim']['scheduler']['final'] = final_lr
    cfg['bc']['optim']['scheduler']['warm_up_ratio'] = warmup_ratio

    # Disable preflight and wandb
    cfg['bc']['preflight']['enabled'] = False
    cfg['bc']['preflight']['disable_wandb'] = True
    cfg['bc']['wandb']['enabled'] = False

    patched_config = run_dir / 'config.toml'
    patched_config.write_text(toml.dumps(cfg), encoding='utf-8')

    nproc = cfg['bc']['launch'].get('nproc_per_node', 2)
    master_port = cfg['bc']['launch'].get('master_port', 29530)

    log_path = run_dir / 'run.log'
    command = [
        torchrun_bin,
        '--standalone',
        '--nproc_per_node', str(nproc),
        '--master-port', str(master_port),
        'mortal/train_bc.py',
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
        result = subprocess.run(
            command, cwd=str(ROOT), env=env, check=False,
            stdout=fh, stderr=subprocess.STDOUT,
        )
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

    def avg_field(events, *keys):
        vals = []
        for e in events:
            v = e
            for k in keys:
                v = (v or {}).get(k)
                if v is None:
                    break
            if v is not None:
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

    last5 = train_metrics[-5:] if len(train_metrics) >= 5 else train_metrics

    # Learning curve
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
        'completed_step': int(train_metrics[-1].get('step', 0)) if train_metrics else 0,
        'nll': round(avg_field(last5, 'train_metrics', 'nll'), 4),
        'top1': round(avg_field(last5, 'train_metrics', 'accuracy'), 4),
        'topk': round(avg_field(last5, 'train_metrics', 'topk_accuracy'), 4),
        'wall': round(wall, 1),
        'rc': result.returncode,
        'curve': curve,
    }
    print(f'  steps={r["completed_step"]}  nll={r["nll"]}  top1={r["top1"]}  '
          f'topk={r["topk"]}  wall={r["wall"]}s  rc={r["rc"]}')
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--cuda-visible-devices', default='0,1')
    args = parser.parse_args()

    torchrun_bin = str(Path(sys.executable).with_name('torchrun'))
    base_config = str(ROOT / 'configs' / 'step6_bc_phase_c_probe_baseline.toml')

    # All at bs=2560, ga=1, eff_batch=5120
    # Linear scaling from 4e-4 @ eff=16384 → ~1.25e-4 @ eff=5120
    # Sweep around that: 2e-4 (aggressive), 1e-4 (scaled), 5e-5 (conservative)
    variants = [
        ('lr_2e-4', 2e-4, 2e-5, 0.05),
        ('lr_1e-4', 1e-4, 1e-5, 0.05),
        ('lr_5e-5', 5e-5, 5e-6, 0.05),
    ]

    results = []
    for name, peak, final, warmup in variants:
        kill_stale_workers()
        r = run_variant(
            name=name, peak_lr=peak, final_lr=final, warmup_ratio=warmup,
            base_config=base_config, steps=args.steps,
            torchrun_bin=torchrun_bin,
            cuda_vis=args.cuda_visible_devices,
        )
        results.append(r)

    print('\n' + '=' * 110)
    print(f'LR SWEEP RESULTS ({args.steps} steps, bs=2560 ga=1 eff=5120, 512ch/48b 83.4M)')
    print('=' * 110)
    print(f'{"Config":<12} {"Peak_LR":>9} {"Final_LR":>9} '
          f'{"NLL":>8} {"Top1":>7} {"TopK":>7} {"Steps":>5} {"Wall":>6}')
    print('-' * 110)
    for r in results:
        print(f'{r["name"]:<12} {r["peak_lr"]:>9.1e} {r["final_lr"]:>9.1e} '
              f'{r["nll"]:>8.4f} {r["top1"]:>7.4f} {r["topk"]:>7.4f} '
              f'{r["completed_step"]:>5} {r["wall"]:>6.0f}')

    # Print curves side by side
    print(f'\nLearning curves (NLL):')
    step_set = sorted(set(pt['step'] for r in results for pt in r['curve']))
    header = f'{"Step":>6}'
    for r in results:
        header += f'  {r["name"][:14]:>14}'
    print(header)
    for step in step_set:
        if step % 50 != 0 and step != 25:
            continue
        row = f'{step:>6}'
        for r in results:
            pt = next((p for p in r['curve'] if p['step'] == step), None)
            row += f'  {pt["nll"]:>14.4f}' if pt else f'  {"":>14}'
        print(row)

    # Also include the ga=3 reference from the previous test
    print(f'\nReference: ga=3 @ same 1000 steps reached NLL=0.5373, top1=0.7956')
    print(f'(but took 2623s vs ~945s per run here)')

    output_path = ROOT / 'artifacts' / 'reports' / 'phase_c_probes' / 'lr_sweep_ga1_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
