#!/usr/bin/env python
"""Phase C: Long (1000-step) runs for the two best model configs."""

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


def run_probe(*, name, resnet_overrides, batch_size, grad_accum_steps,
              base_config, steps, python_bin, torchrun_bin, cuda_vis):
    run_dir = ROOT / 'artifacts' / 'tmp' / 'phase_c_long' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = toml.loads(Path(base_config).read_text(encoding='utf-8'))

    for k, v in resnet_overrides.items():
        cfg['resnet'][k] = v

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
    print(f'LONG PROBE: {name} ({steps} steps)')
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
    sw = summary.get('sustained_metrics') or {}
    rss = summary.get('max_combined_train_worker_rss_kib', 0) / 1024 / 1024

    # Collect NLL/accuracy curve from all train_live events
    curve = []
    for e in train_metrics:
        step = e.get('step', 0)
        tm = e.get('train_metrics') or {}
        curve.append({
            'step': step,
            'nll': round(float(tm.get('nll', 0)), 4),
            'top1': round(float(tm.get('accuracy', 0)), 4),
        })

    r = {
        'name': name,
        'conv_channels': resnet_cfg.get('conv_channels'),
        'num_blocks': resnet_cfg.get('num_blocks'),
        'bottleneck_channels': resnet_cfg.get('bottleneck_channels', 32),
        'hidden_dim': resnet_cfg.get('hidden_dim', 1024),
        'batch_size': batch_size,
        'grad_accum_steps': grad_accum_steps,
        'eff_global_batch': eff_global_batch,
        'completed_step': summary.get('completed_step', 0),
        'completed_sps': round(cw.get('samples_per_second', 0), 1),
        'sustained_sps': round(sw.get('samples_per_second', 0), 1),
        'completed_wait': round(cw.get('wait_fraction', 0), 4),
        'sustained_wait': round(sw.get('wait_fraction', 0), 4),
        'fw_bw_opt': round(avg(recent, 'runtime_metrics', 'fw_bw_opt_fraction'), 4),
        'train_nll': round(avg(recent, 'train_metrics', 'nll'), 4),
        'train_top1': round(avg(recent, 'train_metrics', 'accuracy'), 4),
        'train_topk': round(avg(recent, 'train_metrics', 'topk_accuracy'), 4),
        'train_legal_rate': round(avg(recent, 'train_metrics', 'legal_rate'), 4),
        'peak_rss_gib': round(rss, 1),
        'wall_seconds': round(wall, 1),
        'rc': result.returncode,
        'curve': curve,
    }

    print(f'  steps: {r["completed_step"]}/{steps}  '
          f'sps: {r["completed_sps"]}  '
          f'nll: {r["train_nll"]}  top1: {r["train_top1"]}  '
          f'rss: {r["peak_rss_gib"]} GiB  wall: {r["wall_seconds"]}s  rc: {r["rc"]}')
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--cuda-visible-devices', default='0,1')
    parser.add_argument('--python-bin', default=sys.executable)
    args = parser.parse_args()

    torchrun_bin = str(Path(args.python_bin).with_name('torchrun'))
    base_config = str(ROOT / 'configs' / 'step6_bc_phase_c_probe_baseline.toml')

    variants = [
        ('10x_wide_512ch_48b', dict(conv_channels=512, num_blocks=48, bottleneck_channels=64, hidden_dim=2048), 2048, 4),
        ('20x_768ch_40b', dict(conv_channels=768, num_blocks=40, bottleneck_channels=96, hidden_dim=2048), 2048, 4),
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
    print(f'PHASE C LONG PROBE RESULTS ({args.steps} steps)')
    print('=' * 130)
    print(f'{"Config":<28} {"ch":>4} {"blk":>3} {"BS":>5} {"GA":>2} {"EffBS":>6} '
          f'{"SPS":>7} {"NLL":>7} {"Top1":>7} {"TopK":>7} {"Wait":>6} {"RSS":>5} {"Wall":>6}')
    print('-' * 130)
    for r in results:
        print(f'{r["name"]:<28} {r["conv_channels"]:>4} {r["num_blocks"]:>3} '
              f'{r["batch_size"]:>5} {r["grad_accum_steps"]:>2} {r["eff_global_batch"]:>6} '
              f'{r["completed_sps"]:>7.1f} {r["train_nll"]:>7.4f} {r["train_top1"]:>7.4f} '
              f'{r["train_topk"]:>7.4f} {r["completed_wait"]:>6.4f} '
              f'{r["peak_rss_gib"]:>5.1f} {r["wall_seconds"]:>6.0f}')

    # Print learning curves
    for r in results:
        print(f'\n  {r["name"]} learning curve:')
        print(f'  {"Step":>6} {"NLL":>8} {"Top1":>7}')
        for pt in r['curve']:
            print(f'  {pt["step"]:>6} {pt["nll"]:>8.4f} {pt["top1"]:>7.4f}')

    output_path = ROOT / 'artifacts' / 'reports' / 'phase_c_probes' / 'long_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
