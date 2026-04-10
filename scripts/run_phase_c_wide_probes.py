#!/usr/bin/env python
"""Phase C: Run wider 15x/20x/25x model probes."""

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


def run_probe(*, name, resnet_overrides, batch_size, grad_accum_steps,
              base_config, steps, python_bin, torchrun_bin, cuda_vis):
    run_dir = ROOT / 'artifacts' / 'tmp' / 'phase_c_wide_probes' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = toml.loads(Path(base_config).read_text(encoding='utf-8'))

    # Apply resnet overrides
    for k, v in resnet_overrides.items():
        cfg['resnet'][k] = v

    # Control overrides
    cfg['bc']['control']['max_steps'] = steps
    cfg['bc']['control']['max_runtime_seconds'] = 14400
    cfg['bc']['control']['save_every'] = steps
    cfg['bc']['control']['train_log_every'] = 25
    cfg['bc']['control']['val_steps'] = 8
    cfg['bc']['control']['best_eval_every'] = 0
    cfg['bc']['control']['batch_size'] = batch_size
    cfg['bc']['control']['grad_accum_steps'] = grad_accum_steps
    cfg['bc']['control']['state_file'] = str(run_dir / 'state.pth')
    cfg['bc']['control']['best_state_file'] = str(run_dir / 'best.pth')
    cfg['bc']['control']['tensorboard_dir'] = str(run_dir / 'tensorboard')
    cfg['bc']['control']['metrics_jsonl'] = str(run_dir / 'metrics.jsonl')

    cfg['bc']['preflight']['min_steps_before_stop'] = steps
    cfg['bc']['preflight']['min_runtime_seconds'] = 120
    cfg['bc']['preflight']['summary_json'] = str(run_dir / 'preflight_summary.json')
    cfg['bc']['preflight']['disable_wandb'] = True
    # Relax gates for large models
    cfg['bc']['preflight']['min_samples_per_second'] = 300
    cfg['bc']['preflight']['preferred_samples_per_second'] = 1000
    cfg['bc']['preflight']['max_loader_wait_fraction'] = 0.30
    cfg['bc']['preflight']['min_steady_gpu_ratio'] = 0.50

    patched_config = run_dir / 'config.toml'
    patched_config.write_text(toml.dumps(cfg), encoding='utf-8')

    resnet_cfg = cfg['resnet']
    nproc = cfg['bc']['launch'].get('nproc_per_node', 2)
    eff_global_batch = batch_size * grad_accum_steps * nproc

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
    print(f'PROBE: {name}')
    print(f'  conv_channels={resnet_cfg.get("conv_channels")} '
          f'num_blocks={resnet_cfg.get("num_blocks")} '
          f'bottleneck_channels={resnet_cfg.get("bottleneck_channels", 32)} '
          f'hidden_dim={resnet_cfg.get("hidden_dim", 1024)}')
    print(f'  batch_size={batch_size} grad_accum={grad_accum_steps} eff_global_batch={eff_global_batch}')
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

    # Parse training quality
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

    cw = summary.get('completed_window_metrics') or {}
    rss = summary.get('max_combined_train_worker_rss_kib', 0) / 1024 / 1024

    r = {
        'name': name,
        'conv_channels': resnet_cfg.get('conv_channels'),
        'num_blocks': resnet_cfg.get('num_blocks'),
        'batch_size': batch_size,
        'grad_accum_steps': grad_accum_steps,
        'eff_global_batch': eff_global_batch,
        'completed_step': summary.get('completed_step', 0),
        'sps': round(cw.get('samples_per_second', 0), 1),
        'wait': round(cw.get('wait_fraction', 0), 4),
        'fw_bw_opt': round(avg(recent, 'runtime_metrics', 'fw_bw_opt_fraction'), 4),
        'nll': round(avg(recent, 'train_metrics', 'nll'), 4),
        'top1': round(avg(recent, 'train_metrics', 'accuracy'), 4),
        'topk': round(avg(recent, 'train_metrics', 'topk_accuracy'), 4),
        'rss_gib': round(rss, 1),
        'wall': round(wall, 1),
        'rc': result.returncode,
    }
    print(f'  steps: {r["completed_step"]}/{steps}  sps: {r["sps"]}  '
          f'nll: {r["nll"]}  top1: {r["top1"]}  wall: {r["wall"]}s  rc: {r["rc"]}')
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

    # Wider variants (all use baseline as template, override resnet + batch settings)
    variants = [
        # Re-run current best for reference
        ('ref_20x_768ch_40b', dict(conv_channels=768, num_blocks=40, bottleneck_channels=96, hidden_dim=2048), 2048, 4),
        # Wider 15x
        ('15x_wide_768ch_32b', dict(conv_channels=768, num_blocks=32, bottleneck_channels=96, hidden_dim=2048), 2048, 4),
        # Wider 20x
        ('20x_wide_1024ch_30b', dict(conv_channels=1024, num_blocks=30, bottleneck_channels=96, hidden_dim=2048), 2048, 4),
        ('20x_wide_1024ch_32b', dict(conv_channels=1024, num_blocks=32, bottleneck_channels=96, hidden_dim=2048), 2048, 4),
        # 25x wide
        ('25x_wide_1024ch_40b', dict(conv_channels=1024, num_blocks=40, bottleneck_channels=128, hidden_dim=2048), 1536, 5),
    ]

    results = []
    for name, resnet_overrides, bs, ga in variants:
        r = run_probe(
            name=name, resnet_overrides=resnet_overrides,
            batch_size=bs, grad_accum_steps=ga,
            base_config=base_config, steps=args.steps,
            python_bin=args.python_bin, torchrun_bin=torchrun_bin,
            cuda_vis=args.cuda_visible_devices,
        )
        results.append(r)

    print('\n' + '=' * 130)
    print('PHASE C WIDE PROBE RESULTS')
    print('=' * 130)
    print(f'{"Config":<28} {"ch":>4} {"blk":>3} {"BS":>5} {"GA":>2} {"EffBS":>6} '
          f'{"SPS":>7} {"NLL":>7} {"Top1":>7} {"TopK":>7} {"fw/bw%":>6} {"RSS":>5} {"Wall":>5}')
    print('-' * 130)
    for r in results:
        print(f'{r["name"]:<28} {r["conv_channels"]:>4} {r["num_blocks"]:>3} '
              f'{r["batch_size"]:>5} {r["grad_accum_steps"]:>2} {r["eff_global_batch"]:>6} '
              f'{r["sps"]:>7.1f} {r["nll"]:>7.4f} {r["top1"]:>7.4f} {r["topk"]:>7.4f} '
              f'{r["fw_bw_opt"]:>6.4f} {r["rss_gib"]:>5.1f} {r["wall"]:>5.0f}')

    output_path = ROOT / 'artifacts' / 'reports' / 'phase_c_probes' / 'wide_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
