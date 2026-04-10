#!/usr/bin/env python
"""Phase C: Compare 10x-wide vs 20x model with ga=1, 1000 steps."""

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


def run_variant(*, name, conv_channels, num_blocks, bottleneck_channels,
                hidden_dim, batch_size, base_config, steps,
                torchrun_bin, cuda_vis):
    run_dir = ROOT / 'artifacts' / 'tmp' / 'phase_c_20x_ga1_test' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = toml.loads(Path(base_config).read_text(encoding='utf-8'))

    cfg['resnet']['conv_channels'] = conv_channels
    cfg['resnet']['num_blocks'] = num_blocks
    cfg['resnet']['bottleneck_channels'] = bottleneck_channels
    cfg['resnet']['hidden_dim'] = hidden_dim

    # Training: ga=1, peak_lr=4e-4
    cfg['bc']['control']['max_steps'] = steps
    cfg['bc']['control']['max_runtime_seconds'] = 14400
    cfg['bc']['control']['save_every'] = steps
    cfg['bc']['control']['train_log_every'] = 25
    cfg['bc']['control']['val_steps'] = 8
    cfg['bc']['control']['best_eval_every'] = 0
    cfg['bc']['control']['batch_size'] = batch_size
    cfg['bc']['control']['grad_accum_steps'] = 1
    cfg['bc']['control']['state_file'] = str(run_dir / 'state.pth')
    cfg['bc']['control']['best_state_file'] = str(run_dir / 'best.pth')
    cfg['bc']['control']['tensorboard_dir'] = str(run_dir / 'tensorboard')
    cfg['bc']['control']['metrics_jsonl'] = str(run_dir / 'metrics.jsonl')

    cfg['bc']['optim']['scheduler']['peak'] = 4e-4
    cfg['bc']['optim']['scheduler']['final'] = 4e-5
    cfg['bc']['optim']['scheduler']['warm_up_ratio'] = 0.05

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
    print(f'{name}: {conv_channels}ch/{num_blocks}b bn={bottleneck_channels} hd={hidden_dim} '
          f'bs={batch_size} ga=1')
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

    half = max(1, len(train_metrics) // 2)
    recent = train_metrics[half:]
    last5 = train_metrics[-5:] if len(train_metrics) >= 5 else train_metrics

    sps_steady = avg_field(recent, 'runtime_metrics', 'samples_per_second')

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
        'conv_channels': conv_channels,
        'num_blocks': num_blocks,
        'conv_channels': conv_channels,
        'batch_size': batch_size,
        'completed_step': int(train_metrics[-1].get('step', 0)) if train_metrics else 0,
        'nll': round(avg_field(last5, 'train_metrics', 'nll'), 4),
        'top1': round(avg_field(last5, 'train_metrics', 'accuracy'), 4),
        'topk': round(avg_field(last5, 'train_metrics', 'topk_accuracy'), 4),
        'sps_steady': round(sps_steady, 1),
        'wall': round(wall, 1),
        'rc': result.returncode,
        'curve': curve,
    }
    print(f'  steps={r["completed_step"]}  nll={r["nll"]}  top1={r["top1"]}  '
          f'sps={r["sps_steady"]}  wall={r["wall"]}s  rc={r["rc"]}')
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--cuda-visible-devices', default='0,1')
    args = parser.parse_args()

    torchrun_bin = str(Path(sys.executable).with_name('torchrun'))
    base_config = str(ROOT / 'configs' / 'step6_bc_phase_c_probe_baseline.toml')

    # 10x-wide: 512ch/48b = 83.4M, fits bs=2560
    # 20x: 768ch/40b = 189M, may need smaller batch due to VRAM
    # Try 20x at bs=2048 first (safe) and bs=2560 if it fits
    variants = [
        {
            'name': '10x_wide_512ch_48b',
            'conv_channels': 512, 'num_blocks': 48,
            'bottleneck_channels': 64, 'hidden_dim': 2048,
            'batch_size': 2560,
        },
        {
            'name': '20x_768ch_40b_bs2048',
            'conv_channels': 768, 'num_blocks': 40,
            'bottleneck_channels': 96, 'hidden_dim': 2048,
            'batch_size': 2048,
        },
        {
            'name': '20x_768ch_40b_bs2560',
            'conv_channels': 768, 'num_blocks': 40,
            'bottleneck_channels': 96, 'hidden_dim': 2048,
            'batch_size': 2560,
        },
    ]

    results = []
    for v in variants:
        kill_stale_workers()
        r = run_variant(
            **v,
            base_config=base_config, steps=args.steps,
            torchrun_bin=torchrun_bin,
            cuda_vis=args.cuda_visible_devices,
        )
        results.append(r)

    print('\n' + '=' * 120)
    print(f'10x-WIDE vs 20x MODEL COMPARISON ({args.steps} steps, ga=1, peak_lr=4e-4)')
    print('=' * 120)
    print(f'{"Config":<28} {"BS":>5} {"SPS":>7} '
          f'{"NLL":>8} {"Top1":>7} {"TopK":>7} {"Wall":>6}')
    print('-' * 120)
    for r in results:
        print(f'{r["name"]:<28} {r["batch_size"]:>5} {r["sps_steady"]:>7.1f} '
              f'{r["nll"]:>8.4f} {r["top1"]:>7.4f} {r["topk"]:>7.4f} {r["wall"]:>6.0f}')

    # Learning curves
    print(f'\nLearning curves (NLL):')
    step_set = sorted(set(pt['step'] for r in results for pt in r['curve']))
    header = f'{"Step":>6}'
    for r in results:
        header += f'  {r["name"][:24]:>24}'
    print(header)
    for step in step_set:
        if step % 50 != 0 and step != 25:
            continue
        row = f'{step:>6}'
        for r in results:
            pt = next((p for p in r['curve'] if p['step'] == step), None)
            row += f'  {pt["nll"]:>24.4f}' if pt else f'  {"":>24}'
        print(row)

    output_path = ROOT / 'artifacts' / 'reports' / 'phase_c_probes' / '20x_ga1_comparison.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
