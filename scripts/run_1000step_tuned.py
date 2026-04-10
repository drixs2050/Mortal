#!/usr/bin/env python
"""Test w4 with tuned buffer/prefetch settings at 1000 steps."""

from __future__ import annotations

import copy
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


def load_toml(path: Path) -> dict:
    return toml.loads(path.read_text(encoding='utf-8'))


def write_toml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(data), encoding='utf-8')


def run_variant(*, name, workers, steps, overrides, base_config, cuda_vis, python_bin):
    torchrun_bin = str(Path(python_bin).with_name('torchrun'))
    run_dir = ROOT / 'artifacts' / 'tmp' / 'step6_w4_tuning' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_toml(ROOT / base_config)
    cfg['bc']['control']['max_steps'] = steps
    cfg['bc']['control']['max_runtime_seconds'] = 7200
    cfg['bc']['control']['save_every'] = steps
    cfg['bc']['control']['train_log_every'] = 25
    cfg['bc']['control']['val_steps'] = 0
    cfg['bc']['control']['best_eval_every'] = 0
    cfg['bc']['control']['state_file'] = str(run_dir / 'state.pth')
    cfg['bc']['control']['best_state_file'] = str(run_dir / 'best.pth')
    cfg['bc']['control']['tensorboard_dir'] = str(run_dir / 'tensorboard')
    cfg['bc']['control']['metrics_jsonl'] = str(run_dir / 'metrics.jsonl')
    cfg['bc']['dataset']['num_workers'] = workers
    cfg['bc']['dataset']['persistent_workers'] = True
    cfg['bc']['dataset']['prefetch_factor'] = 2
    cfg['bc']['dataset']['multiprocessing_context'] = 'spawn'
    cfg['bc']['preflight']['min_steps_before_stop'] = steps
    cfg['bc']['preflight']['min_runtime_seconds'] = 120
    cfg['bc']['preflight']['summary_json'] = str(run_dir / 'preflight_summary.json')
    cfg['bc']['preflight']['disable_wandb'] = True

    # Apply variant-specific overrides
    for dotted_key, value in overrides.items():
        parts = dotted_key.split('.')
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value

    config_path = run_dir / 'config.toml'
    write_toml(config_path, cfg)

    log_path = run_dir / 'run.log'
    command = [
        python_bin, 'scripts/run_bc_loader_preflight.py',
        '--config', str(config_path),
        '--python-bin', python_bin,
        '--torchrun-bin', torchrun_bin,
    ]
    env = dict(os.environ)
    env['MORTAL_CFG'] = str(config_path)
    env['CUDA_VISIBLE_DEVICES'] = cuda_vis

    print(f'\n=== {name} ===')
    desc = {k: v for k, v in overrides.items() if 'dataset' in k or 'ready' in k}
    print(f'  overrides: {desc}')

    t0 = time.monotonic()
    with log_path.open('w', encoding='utf-8') as fh:
        fh.write(f'$ {" ".join(command)}\n')
        fh.flush()
        result = subprocess.run(command, cwd=str(ROOT), env=env, check=False, stdout=fh, stderr=fh)
    wall = time.monotonic() - t0

    summary = {}
    if (run_dir / 'preflight_summary.json').exists():
        summary = json.loads((run_dir / 'preflight_summary.json').read_text())

    cw = summary.get('completed_window_metrics') or {}
    sw = summary.get('sustained_metrics') or {}
    rss = summary.get('max_combined_train_worker_rss_kib', 0) / 1024 / 1024

    r = {
        'name': name,
        'completed_step': summary.get('completed_step', 0),
        'completed_sps': round(cw.get('samples_per_second', 0), 1),
        'completed_wait': round(cw.get('wait_fraction', 0), 4),
        'cpu_pipe_wait': round(cw.get('cpu_pipe_wait_fraction', 0), 4),
        'cpu_ready_batches': cw.get('cpu_ready_batches', 0),
        'producer_blocked': round(cw.get('cpu_producer_blocked_put_fraction', 0), 4),
        'sustained_sps': round(sw.get('samples_per_second', 0), 1),
        'sustained_wait': round(sw.get('wait_fraction', 0), 4),
        'peak_rss_gib': round(rss, 1),
        'wall_seconds': round(wall, 1),
        'rc': result.returncode,
    }
    print(f'  steps: {r["completed_step"]}/{steps}  sps: {r["completed_sps"]}  '
          f'wait: {r["completed_wait"]}  pipe_wait: {r["cpu_pipe_wait"]}  '
          f'ready: {r["cpu_ready_batches"]}  blocked: {r["producer_blocked"]}  '
          f'rss: {r["peak_rss_gib"]} GiB  wall: {r["wall_seconds"]}s')
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--base-config', default='configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml')
    parser.add_argument('--cuda-visible-devices', default='0,1')
    parser.add_argument('--python-bin', default=sys.executable)
    args = parser.parse_args()

    variants = [
        ('w4_baseline', 4, {}),
        ('w4_ready8', 4, {
            'bc.dataset.cpu_ready_batches': 8,
        }),
        ('w4_prefetch4', 4, {
            'bc.dataset.prefetch_factor': 4,
        }),
        ('w4_ready8_prefetch4', 4, {
            'bc.dataset.cpu_ready_batches': 8,
            'bc.dataset.prefetch_factor': 4,
        }),
        ('w4_ready8_pf4_fb96', 4, {
            'bc.dataset.cpu_ready_batches': 8,
            'bc.dataset.prefetch_factor': 4,
            'bc.dataset.file_batch_size': 96,
        }),
    ]

    results = []
    for name, workers, overrides in variants:
        r = run_variant(
            name=name, workers=workers, steps=args.steps,
            overrides=overrides, base_config=args.base_config,
            cuda_vis=args.cuda_visible_devices, python_bin=args.python_bin,
        )
        results.append(r)

    print('\n' + '=' * 90)
    print('W4 TUNING RESULTS')
    print('=' * 90)
    headers = ['Config', 'Steps', 'SPS', 'Wait', 'PipeWait', 'Ready', 'Blocked', 'RSS GiB', 'Wall']
    print(f'{"Config":<25} {"Steps":>5} {"SPS":>8} {"Wait":>6} {"Pipe":>6} {"Ready":>5} {"Block":>6} {"RSS":>6} {"Wall":>6}')
    print('-' * 90)
    for r in results:
        print(f'{r["name"]:<25} {r["completed_step"]:>5} {r["completed_sps"]:>8.1f} '
              f'{r["completed_wait"]:>6.4f} {r["cpu_pipe_wait"]:>6.4f} '
              f'{r["cpu_ready_batches"]:>5.0f} {r["producer_blocked"]:>6.4f} '
              f'{r["peak_rss_gib"]:>6.1f} {r["wall_seconds"]:>6.0f}')

    output_path = ROOT / 'artifacts' / 'reports' / 'step6_w4_tuning' / 'results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
