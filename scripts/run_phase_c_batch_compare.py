#!/usr/bin/env python
"""Phase C: Compare batch size / grad_accum combinations for SPS."""

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


def run_variant(*, name, batch_size, grad_accum, base_config, steps,
                python_bin, torchrun_bin, cuda_vis):
    run_dir = ROOT / 'artifacts' / 'tmp' / 'phase_c_batch_compare' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = toml.loads(Path(base_config).read_text(encoding='utf-8'))

    cfg['resnet']['conv_channels'] = 512
    cfg['resnet']['num_blocks'] = 48
    cfg['resnet']['bottleneck_channels'] = 64
    cfg['resnet']['hidden_dim'] = 2048

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

    cfg['bc']['preflight']['min_steps_before_stop'] = steps
    cfg['bc']['preflight']['min_runtime_seconds'] = 60
    cfg['bc']['preflight']['summary_json'] = str(run_dir / 'preflight_summary.json')
    cfg['bc']['preflight']['disable_wandb'] = True
    cfg['bc']['preflight']['min_samples_per_second'] = 100
    cfg['bc']['preflight']['preferred_samples_per_second'] = 500
    cfg['bc']['preflight']['max_loader_wait_fraction'] = 0.50
    cfg['bc']['preflight']['min_steady_gpu_ratio'] = 0.30

    patched_config = run_dir / 'config.toml'
    patched_config.write_text(toml.dumps(cfg), encoding='utf-8')

    nproc = cfg['bc']['launch'].get('nproc_per_node', 2)
    eff_global_batch = batch_size * grad_accum * nproc

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
    print(f'{name}: bs={batch_size} ga={grad_accum} eff_batch={eff_global_batch}')
    print(f'{"=" * 90}')

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
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        'eff_global_batch': eff_global_batch,
        'completed_step': summary.get('completed_step', 0),
        'completed_sps': round(cw.get('samples_per_second', 0), 1),
        'sustained_sps': round(sw.get('samples_per_second', 0), 1),
        'wait': round(cw.get('wait_fraction', 0), 4),
        'rss_gib': round(rss, 1),
        'wall': round(wall, 1),
        'rc': result.returncode,
    }
    print(f'  steps={r["completed_step"]}  sps={r["completed_sps"]}  '
          f'sustained={r["sustained_sps"]}  wait={r["wait"]}  '
          f'rss={r["rss_gib"]}G  wall={r["wall"]}s  rc={r["rc"]}')
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--cuda-visible-devices', default='0,1')
    parser.add_argument('--python-bin', default=sys.executable)
    args = parser.parse_args()

    torchrun_bin = str(Path(args.python_bin).with_name('torchrun'))
    base_config = str(ROOT / 'configs' / 'step6_bc_phase_c_probe_baseline.toml')

    variants = [
        ('bs2048_ga4', 2048, 4),   # eff=16384, current default
        ('bs2560_ga3', 2560, 3),   # eff=15360
    ]

    results = []
    for name, bs, ga in variants:
        # Kill any stale train_bc workers between runs
        import signal
        for line in subprocess.run(['ps', '-eo', 'pid=,cmd='], capture_output=True, text=True).stdout.splitlines():
            if 'mortal/train_bc.py' in line:
                pid = int(line.strip().split()[0])
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(2)

        r = run_variant(
            name=name, batch_size=bs, grad_accum=ga,
            base_config=base_config, steps=args.steps,
            python_bin=args.python_bin, torchrun_bin=torchrun_bin,
            cuda_vis=args.cuda_visible_devices,
        )
        results.append(r)

    print('\n' + '=' * 100)
    print('BATCH SIZE COMPARISON')
    print('=' * 100)
    print(f'{"Config":<18} {"BS":>5} {"GA":>2} {"EffBS":>6} {"SPS":>7} {"Sust_SPS":>8} '
          f'{"Wait":>6} {"RSS":>5} {"Wall":>6} {"RC":>3}')
    print('-' * 100)
    for r in results:
        print(f'{r["name"]:<18} {r["batch_size"]:>5} {r["grad_accum"]:>2} {r["eff_global_batch"]:>6} '
              f'{r["completed_sps"]:>7.1f} {r["sustained_sps"]:>8.1f} '
              f'{r["wait"]:>6.4f} {r["rss_gib"]:>5.1f} {r["wall"]:>6.0f} {r["rc"]:>3}')


if __name__ == '__main__':
    main()
