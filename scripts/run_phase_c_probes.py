#!/usr/bin/env python
"""Phase C: Run all model scaling probes sequentially and collect results.

Each probe runs a 200-step DDP preflight with full metric logging:
  - SPS (samples/second), steps/second
  - Training loss (NLL), accuracy (top1, topk), legal_rate
  - Loader wait fraction, GPU utilization, RSS
  - Wall time, startup latency

Results are aggregated into a single comparison report.
"""

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


def run_probe(*, name, config_path, steps, python_bin, torchrun_bin, cuda_vis):
    """Run one preflight probe and return results dict."""
    run_dir = ROOT / 'artifacts' / 'tmp' / 'phase_c_probes' / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load and patch config for this run
    try:
        import toml
    except ImportError:
        import pip._vendor.tomli as toml

    cfg = toml.loads(Path(config_path).read_text(encoding='utf-8'))

    # Override paths to use run_dir
    cfg['bc']['control']['max_steps'] = steps
    cfg['bc']['control']['max_runtime_seconds'] = 14400
    cfg['bc']['control']['save_every'] = steps
    cfg['bc']['control']['train_log_every'] = 25
    cfg['bc']['control']['val_steps'] = 8
    cfg['bc']['control']['best_eval_every'] = 0
    cfg['bc']['control']['state_file'] = str(run_dir / 'state.pth')
    cfg['bc']['control']['best_state_file'] = str(run_dir / 'best.pth')
    cfg['bc']['control']['tensorboard_dir'] = str(run_dir / 'tensorboard')
    cfg['bc']['control']['metrics_jsonl'] = str(run_dir / 'metrics.jsonl')

    cfg['bc']['preflight']['min_steps_before_stop'] = steps
    cfg['bc']['preflight']['min_runtime_seconds'] = 120
    cfg['bc']['preflight']['summary_json'] = str(run_dir / 'preflight_summary.json')
    cfg['bc']['preflight']['disable_wandb'] = True

    # Write patched config
    patched_config = run_dir / 'config.toml'
    patched_config.write_text(toml.dumps(cfg), encoding='utf-8')

    resnet_cfg = cfg.get('resnet', {})
    batch_size = cfg['bc']['control'].get('batch_size', 8192)
    grad_accum = cfg['bc']['control'].get('grad_accum_steps', 1)
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
    print(f'PROBE: {name}')
    print(f'  conv_channels={resnet_cfg.get("conv_channels", 192)} '
          f'num_blocks={resnet_cfg.get("num_blocks", 40)} '
          f'bottleneck_channels={resnet_cfg.get("bottleneck_channels", 32)} '
          f'hidden_dim={resnet_cfg.get("hidden_dim", 1024)}')
    print(f'  batch_size={batch_size} grad_accum={grad_accum} eff_global_batch={eff_global_batch}')
    print(f'{"=" * 90}')

    t0 = time.monotonic()
    with log_path.open('w', encoding='utf-8') as fh:
        fh.write(f'$ {" ".join(command)}\n')
        fh.flush()
        result = subprocess.run(command, cwd=str(ROOT), env=env, check=False, stdout=fh, stderr=fh)
    wall = time.monotonic() - t0

    # Parse summary
    summary = {}
    summary_path = run_dir / 'preflight_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    # Parse metrics.jsonl for training quality metrics
    metrics_path = run_dir / 'metrics.jsonl'
    train_metrics = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('event') == 'train_live':
                train_metrics.append(event)

    # Extract quality metrics from last few training windows
    last_n = min(5, len(train_metrics))
    recent_metrics = train_metrics[-last_n:] if train_metrics else []

    def avg_metric(events, *keys):
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

    cw = summary.get('completed_window_metrics') or {}
    sw = summary.get('sustained_metrics') or {}
    rss = summary.get('max_combined_train_worker_rss_kib', 0) / 1024 / 1024
    startup = summary.get('startup') or {}
    startup_seconds = float(startup.get('elapsed_seconds', 0) or 0)
    gate = summary.get('gate') or {}

    r = {
        'name': name,
        'conv_channels': resnet_cfg.get('conv_channels', 192),
        'num_blocks': resnet_cfg.get('num_blocks', 40),
        'bottleneck_channels': resnet_cfg.get('bottleneck_channels', 32),
        'hidden_dim': resnet_cfg.get('hidden_dim', 1024),
        'batch_size': batch_size,
        'grad_accum_steps': grad_accum,
        'eff_global_batch': eff_global_batch,

        # Throughput
        'completed_step': summary.get('completed_step', 0),
        'completed_sps': round(cw.get('samples_per_second', 0), 1),
        'sustained_sps': round(sw.get('samples_per_second', 0), 1),
        'steps_per_second': round(cw.get('steps_per_second', 0), 3),

        # Data pipeline
        'completed_wait': round(cw.get('wait_fraction', 0), 4),
        'sustained_wait': round(sw.get('wait_fraction', 0), 4),
        'cpu_pipe_wait': round(cw.get('cpu_pipe_wait_fraction', 0), 4),
        'cpu_ready_batches': round(cw.get('cpu_ready_batches', 0), 1),
        'producer_blocked': round(cw.get('cpu_producer_blocked_put_fraction', 0), 4),

        # Training quality (from last few windows)
        'train_nll': round(avg_metric(recent_metrics, 'train_metrics', 'nll'), 4),
        'train_top1': round(avg_metric(recent_metrics, 'train_metrics', 'accuracy'), 4),
        'train_topk': round(avg_metric(recent_metrics, 'train_metrics', 'topk_accuracy'), 4),
        'train_legal_rate': round(avg_metric(recent_metrics, 'train_metrics', 'legal_rate'), 4),

        # GPU utilization
        'fw_bw_opt_fraction': round(avg_metric(recent_metrics, 'runtime_metrics', 'fw_bw_opt_fraction'), 4),

        # Resources
        'peak_rss_gib': round(rss, 1),
        'startup_seconds': round(startup_seconds, 1),
        'wall_seconds': round(wall, 1),

        # Status
        'rc': result.returncode,
        'gate_passed': gate.get('passed', False),
        'gate_reasons': gate.get('reasons', []),
    }

    print(f'  steps: {r["completed_step"]}/{steps}  '
          f'sps: {r["completed_sps"]}  '
          f'wait: {r["completed_wait"]}  '
          f'nll: {r["train_nll"]}  '
          f'top1: {r["train_top1"]}  '
          f'rss: {r["peak_rss_gib"]} GiB  '
          f'wall: {r["wall_seconds"]}s  '
          f'rc: {r["rc"]}')
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--cuda-visible-devices', default='0,2')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--probes', nargs='*', default=None,
                        help='Subset of probe names to run (default: all)')
    args = parser.parse_args()

    torchrun_bin = str(Path(args.python_bin).with_name('torchrun'))

    all_probes = [
        ('baseline', 'configs/step6_bc_phase_c_probe_baseline.toml'),
        ('2x_256ch_40b', 'configs/step6_bc_phase_c_probe_2x.toml'),
        ('3x_256ch_60b', 'configs/step6_bc_phase_c_probe_3x.toml'),
        ('5x_384ch_40b', 'configs/step6_bc_phase_c_probe_5x.toml'),
        ('10x_384ch_80b', 'configs/step6_bc_phase_c_probe_10x.toml'),
        ('10x_wide_512ch_48b', 'configs/step6_bc_phase_c_probe_10x_wide.toml'),
        ('15x_512ch_60b', 'configs/step6_bc_phase_c_probe_15x.toml'),
        ('20x_768ch_40b', 'configs/step6_bc_phase_c_probe_20x.toml'),
    ]

    if args.probes:
        probes = [(n, c) for n, c in all_probes if n in args.probes]
    else:
        probes = all_probes

    results = []
    for name, config_path in probes:
        full_config = str(ROOT / config_path)
        if not Path(full_config).exists():
            print(f'\nSKIPPING {name}: config not found at {full_config}')
            continue
        r = run_probe(
            name=name,
            config_path=full_config,
            steps=args.steps,
            python_bin=args.python_bin,
            torchrun_bin=torchrun_bin,
            cuda_vis=args.cuda_visible_devices,
        )
        results.append(r)

    # Print comparison table
    print('\n' + '=' * 140)
    print('PHASE C MODEL SCALING PROBE RESULTS')
    print('=' * 140)

    # Table 1: Throughput & Resources
    print(f'\n{"Config":<22} {"Params":>6} {"BS":>5} {"GA":>2} {"EffBS":>6} '
          f'{"SPS":>8} {"Wait":>6} {"Blocked":>7} '
          f'{"fw/bw%":>6} {"RSS_G":>5} {"Start":>5} {"Wall":>6} {"RC":>3}')
    print('-' * 140)
    for r in results:
        ch = r['conv_channels']
        nb = r['num_blocks']
        # Rough param count
        params_approx = f'{ch}c{nb}b'
        print(f'{r["name"]:<22} {params_approx:>6} {r["batch_size"]:>5} {r["grad_accum_steps"]:>2} '
              f'{r["eff_global_batch"]:>6} '
              f'{r["completed_sps"]:>8.1f} {r["completed_wait"]:>6.4f} '
              f'{r["producer_blocked"]:>7.4f} '
              f'{r["fw_bw_opt_fraction"]:>6.4f} {r["peak_rss_gib"]:>5.1f} '
              f'{r["startup_seconds"]:>5.0f} {r["wall_seconds"]:>6.0f} {r["rc"]:>3}')

    # Table 2: Training Quality
    print(f'\n{"Config":<22} {"Steps":>5} {"NLL":>8} {"Top1":>7} {"TopK":>7} {"Legal":>7} {"Gate":>6}')
    print('-' * 80)
    for r in results:
        gate = 'PASS' if r['gate_passed'] else 'FAIL'
        print(f'{r["name"]:<22} {r["completed_step"]:>5} '
              f'{r["train_nll"]:>8.4f} {r["train_top1"]:>7.4f} '
              f'{r["train_topk"]:>7.4f} {r["train_legal_rate"]:>7.4f} {gate:>6}')

    # Save full results
    output_path = ROOT / 'artifacts' / 'reports' / 'phase_c_probes' / 'results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'\nFull results saved to: {output_path}')

    # Print any gate failures
    failures = [r for r in results if not r['gate_passed']]
    if failures:
        print(f'\nGate failures ({len(failures)}):')
        for r in failures:
            print(f'  {r["name"]}: {r["gate_reasons"]}')


if __name__ == '__main__':
    main()
