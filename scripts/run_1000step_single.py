#!/usr/bin/env python
"""Run a single 1000-step preflight for a given worker count."""

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--run-id', default='')
    parser.add_argument('--base-config', default='configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml')
    parser.add_argument('--cuda-visible-devices', default='0,1')
    parser.add_argument('--python-bin', default=sys.executable)
    args = parser.parse_args()

    torchrun_bin = str(Path(args.python_bin).with_name('torchrun'))
    name = args.run_id or f'w{args.workers}_{args.steps}step'
    run_dir = ROOT / 'artifacts' / 'tmp' / 'step6_1000step_comparison' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_toml(ROOT / args.base_config)
    cfg['bc']['control']['max_steps'] = args.steps
    cfg['bc']['control']['max_runtime_seconds'] = 7200
    cfg['bc']['control']['save_every'] = args.steps
    cfg['bc']['control']['train_log_every'] = 25
    cfg['bc']['control']['val_steps'] = 0
    cfg['bc']['control']['best_eval_every'] = 0
    cfg['bc']['control']['state_file'] = str(run_dir / 'state.pth')
    cfg['bc']['control']['best_state_file'] = str(run_dir / 'best.pth')
    cfg['bc']['control']['tensorboard_dir'] = str(run_dir / 'tensorboard')
    cfg['bc']['control']['metrics_jsonl'] = str(run_dir / 'metrics.jsonl')
    cfg['bc']['dataset']['num_workers'] = args.workers
    if args.workers > 0:
        cfg['bc']['dataset']['persistent_workers'] = True
        cfg['bc']['dataset']['prefetch_factor'] = 2
        cfg['bc']['dataset']['multiprocessing_context'] = 'spawn'
    cfg['bc']['preflight']['min_steps_before_stop'] = args.steps
    cfg['bc']['preflight']['min_runtime_seconds'] = 120
    cfg['bc']['preflight']['summary_json'] = str(run_dir / 'preflight_summary.json')
    cfg['bc']['preflight']['disable_wandb'] = True

    config_path = run_dir / 'config.toml'
    write_toml(config_path, cfg)

    log_path = run_dir / 'run.log'
    command = [
        args.python_bin, 'scripts/run_bc_loader_preflight.py',
        '--config', str(config_path),
        '--python-bin', args.python_bin,
        '--torchrun-bin', torchrun_bin,
    ]
    env = dict(os.environ)
    env['MORTAL_CFG'] = str(config_path)
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    print(f'Running {name}: {args.workers} workers, {args.steps} steps')
    t0 = time.monotonic()
    with log_path.open('w', encoding='utf-8') as fh:
        fh.write(f'$ {" ".join(command)}\n')
        fh.flush()
        result = subprocess.run(command, cwd=str(ROOT), env=env, check=False, stdout=fh, stderr=fh)
    wall = time.monotonic() - t0

    summary = {}
    summary_path = run_dir / 'preflight_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    cw = summary.get('completed_window_metrics') or {}
    sw = summary.get('sustained_metrics') or {}
    rss = summary.get('max_combined_train_worker_rss_kib', 0) / 1024 / 1024

    print(f'  completed_step: {summary.get("completed_step", "?")}')
    print(f'  completed: {cw.get("samples_per_second", 0):.1f} sps, wait={cw.get("wait_fraction", 0):.4f}')
    print(f'  sustained: {sw.get("samples_per_second", 0):.1f} sps, wait={sw.get("wait_fraction", 0):.4f}')
    print(f'  rss: {rss:.1f} GiB, wall: {wall:.1f}s, rc: {result.returncode}')


if __name__ == '__main__':
    main()
