#!/usr/bin/env python
"""Phase C: Compare batch size / grad_accum combinations — direct torchrun."""

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


def run_variant(*, name, batch_size, grad_accum, base_config, steps,
                torchrun_bin, cuda_vis):
    run_dir = ROOT / 'artifacts' / 'tmp' / 'phase_c_batch_compare_v2' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = toml.loads(Path(base_config).read_text(encoding='utf-8'))

    # Model: 10x-wide
    cfg['resnet']['conv_channels'] = 512
    cfg['resnet']['num_blocks'] = 48
    cfg['resnet']['bottleneck_channels'] = 64
    cfg['resnet']['hidden_dim'] = 2048

    # Training
    cfg['bc']['control']['max_steps'] = steps
    cfg['bc']['control']['max_runtime_seconds'] = 7200
    cfg['bc']['control']['save_every'] = steps
    cfg['bc']['control']['train_log_every'] = 25
    cfg['bc']['control']['val_steps'] = 0
    cfg['bc']['control']['best_eval_every'] = 0
    cfg['bc']['control']['batch_size'] = batch_size
    cfg['bc']['control']['grad_accum_steps'] = grad_accum
    cfg['bc']['control']['state_file'] = str(run_dir / 'state.pth')
    cfg['bc']['control']['best_state_file'] = str(run_dir / 'best.pth')
    cfg['bc']['control']['tensorboard_dir'] = str(run_dir / 'tensorboard')
    cfg['bc']['control']['metrics_jsonl'] = str(run_dir / 'metrics.jsonl')

    # Disable preflight gates (we just want raw training)
    cfg['bc']['preflight']['enabled'] = False
    cfg['bc']['preflight']['disable_wandb'] = True

    # Disable wandb
    cfg['bc']['wandb']['enabled'] = False

    patched_config = run_dir / 'config.toml'
    patched_config.write_text(toml.dumps(cfg), encoding='utf-8')

    nproc = cfg['bc']['launch'].get('nproc_per_node', 2)
    master_port = cfg['bc']['launch'].get('master_port', 29530)
    eff_global_batch = batch_size * grad_accum * nproc

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
    print(f'{name}: bs={batch_size} ga={grad_accum} eff_batch={eff_global_batch}')
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

    # Compute average SPS from last 50% of steps (skip warmup)
    half = max(1, len(train_metrics) // 2)
    recent = train_metrics[half:]

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

    sps_all = avg_field(train_metrics, 'runtime_metrics', 'samples_per_second')
    sps_steady = avg_field(recent, 'runtime_metrics', 'samples_per_second')
    wait_all = avg_field(train_metrics, 'loader_metrics', 'wait_fraction')

    r = {
        'name': name,
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        'eff_global_batch': eff_global_batch,
        'completed_step': int(train_metrics[-1].get('step', 0)) if train_metrics else 0,
        'sps_all': round(sps_all, 1),
        'sps_steady': round(sps_steady, 1),
        'wait': round(wait_all, 4),
        'wall': round(wall, 1),
        'rc': result.returncode,
    }
    print(f'  steps={r["completed_step"]}  sps_all={r["sps_all"]}  '
          f'sps_steady={r["sps_steady"]}  wait={r["wait"]}  '
          f'wall={r["wall"]}s  rc={r["rc"]}')
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--cuda-visible-devices', default='0,1')
    args = parser.parse_args()

    torchrun_bin = str(
        Path(sys.executable).with_name('torchrun')
    )
    base_config = str(ROOT / 'configs' / 'step6_bc_phase_c_probe_baseline.toml')

    variants = [
        ('bs2048_ga4', 2048, 4),   # eff=16384, current default
        ('bs2560_ga3', 2560, 3),   # eff=15360
    ]

    results = []
    for name, bs, ga in variants:
        kill_stale_workers()
        r = run_variant(
            name=name, batch_size=bs, grad_accum=ga,
            base_config=base_config, steps=args.steps,
            torchrun_bin=torchrun_bin,
            cuda_vis=args.cuda_visible_devices,
        )
        results.append(r)

    print('\n' + '=' * 100)
    print('BATCH SIZE COMPARISON')
    print('=' * 100)
    print(f'{"Config":<18} {"BS":>5} {"GA":>2} {"EffBS":>6} {"SPS_all":>8} {"SPS_steady":>10} '
          f'{"Wait":>6} {"Wall":>6} {"RC":>3}')
    print('-' * 100)
    for r in results:
        print(f'{r["name"]:<18} {r["batch_size"]:>5} {r["grad_accum"]:>2} {r["eff_global_batch"]:>6} '
              f'{r["sps_all"]:>8.1f} {r["sps_steady"]:>10.1f} '
              f'{r["wait"]:>6.4f} {r["wall"]:>6.0f} {r["rc"]:>3}')


if __name__ == '__main__':
    main()
