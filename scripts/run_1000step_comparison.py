#!/usr/bin/env python
"""1000-step thorough comparison: control_b4 (num_workers=0) vs w6 (num_workers=6).

Runs each config sequentially on dual A100 DDP, collecting full metrics.
"""

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
    text = path.read_text(encoding='utf-8')
    if hasattr(toml, 'loads'):
        return toml.loads(text)
    raise RuntimeError('No TOML parser available')


def write_toml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(toml, 'dumps'):
        path.write_text(toml.dumps(data), encoding='utf-8')
    else:
        # Fallback: manual write
        import json as _json
        path.write_text(_json.dumps(data, indent=2), encoding='utf-8')


def deep_update(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def subprocess_env(config_path: Path) -> dict:
    env = dict(os.environ)
    env['MORTAL_CFG'] = str(config_path)
    return env


def main():
    import argparse
    parser = argparse.ArgumentParser(description='1000-step control vs w6 comparison')
    parser.add_argument('--base-config', default='configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--cuda-visible-devices', default='0,1')
    parser.add_argument('--python-bin', default=sys.executable)
    args = parser.parse_args()

    torchrun_bin = str(Path(args.python_bin).with_name('torchrun'))
    run_root = ROOT / 'artifacts' / 'tmp' / 'step6_1000step_comparison'
    report_root = ROOT / 'artifacts' / 'reports' / 'step6_1000step_comparison'
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    base_config = load_toml(ROOT / args.base_config)

    # --- Build configs ---
    variants = {
        'control_b4': {
            'bc.control.max_steps': args.steps,
            'bc.control.max_runtime_seconds': 3600,
            'bc.control.save_every': args.steps,
            'bc.control.train_log_every': 25,
            'bc.control.val_steps': 0,
            'bc.control.best_eval_every': 0,
            'bc.dataset.num_workers': 0,
            'bc.preflight.min_steps_before_stop': args.steps,
            'bc.preflight.min_runtime_seconds': 120,
            'bc.preflight.disable_wandb': True,
        },
        'w6': {
            'bc.control.max_steps': args.steps,
            'bc.control.max_runtime_seconds': 3600,
            'bc.control.save_every': args.steps,
            'bc.control.train_log_every': 25,
            'bc.control.val_steps': 0,
            'bc.control.best_eval_every': 0,
            'bc.dataset.num_workers': 6,
            'bc.dataset.persistent_workers': True,
            'bc.dataset.prefetch_factor': 2,
            'bc.dataset.multiprocessing_context': 'spawn',
            'bc.preflight.min_steps_before_stop': args.steps,
            'bc.preflight.min_runtime_seconds': 120,
            'bc.preflight.disable_wandb': True,
        },
    }

    results = {}

    for name, overrides_flat in variants.items():
        print(f'\n=== Running {name} ({args.steps} steps) ===')

        # Apply overrides
        cfg = copy.deepcopy(base_config)
        for dotted_key, value in overrides_flat.items():
            parts = dotted_key.split('.')
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value

        # Set paths for this variant
        variant_dir = run_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        cfg['bc']['control']['state_file'] = str(variant_dir / 'state.pth')
        cfg['bc']['control']['best_state_file'] = str(variant_dir / 'best.pth')
        cfg['bc']['control']['tensorboard_dir'] = str(variant_dir / 'tensorboard')
        cfg['bc']['control']['metrics_jsonl'] = str(variant_dir / 'metrics.jsonl')
        cfg['bc']['preflight']['summary_json'] = str(variant_dir / 'preflight_summary.json')

        config_path = run_root / 'configs' / f'{name}.toml'
        write_toml(config_path, cfg)

        log_path = run_root / 'logs' / f'{name}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Run via preflight runner
        command = [
            args.python_bin,
            'scripts/run_bc_loader_preflight.py',
            '--config', str(config_path),
            '--python-bin', args.python_bin,
            '--torchrun-bin', torchrun_bin,
        ]
        env = subprocess_env(config_path)
        env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

        t0 = time.monotonic()
        with log_path.open('w', encoding='utf-8') as log_fh:
            log_fh.write(f'$ {" ".join(command)}\n')
            log_fh.flush()
            result = subprocess.run(
                command, cwd=str(ROOT), env=env, check=False,
                stdout=log_fh, stderr=log_fh,
            )
        wall_seconds = time.monotonic() - t0

        # Load summary
        summary_path = variant_dir / 'preflight_summary.json'
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))

        cw = summary.get('completed_window_metrics') or {}
        sw = summary.get('sustained_metrics') or {}
        gpu = summary.get('steady_gpu') or {}
        rss = summary.get('max_combined_train_worker_rss_kib', 0) / 1024 / 1024

        results[name] = {
            'return_code': result.returncode,
            'wall_seconds': round(wall_seconds, 1),
            'completed_step': summary.get('completed_step', 0),
            'completed_sps': round(cw.get('samples_per_second', 0), 1),
            'completed_wait': round(cw.get('wait_fraction', 0), 4),
            'completed_cpu_pipe_wait': round(cw.get('cpu_pipe_wait_fraction', 0), 4),
            'sustained_sps': round(sw.get('samples_per_second', 0), 1),
            'sustained_wait': round(sw.get('wait_fraction', 0), 4),
            'gpu_ratio': round(gpu.get('pass_ratio', 0), 4),
            'peak_rss_gib': round(rss, 1),
            'startup_seconds': round(float(summary.get('startup_seconds') or 0), 1),
        }

        # Also extract per-window breakdown from metrics.jsonl
        metrics_path = variant_dir / 'metrics.jsonl'
        if metrics_path.exists():
            lines = metrics_path.read_text(encoding='utf-8').strip().split('\n')
            train_lines = [json.loads(l) for l in lines if '"train_window"' in l or '"train"' in l]
            if train_lines:
                first = train_lines[0]
                last = train_lines[-1]
                results[name]['first_window_sps'] = round(first.get('samples_per_second', 0), 1)
                results[name]['first_window_wait'] = round(first.get('loader_wait_fraction', first.get('wait_fraction', 0)), 4)
                results[name]['last_window_sps'] = round(last.get('samples_per_second', 0), 1)
                results[name]['last_window_wait'] = round(last.get('loader_wait_fraction', last.get('wait_fraction', 0)), 4)

        print(f'  {name}: {results[name]["completed_sps"]} sps, '
              f'wait={results[name]["completed_wait"]}, '
              f'rss={results[name]["peak_rss_gib"]} GiB, '
              f'wall={results[name]["wall_seconds"]}s')

    # --- Print comparison ---
    print('\n' + '=' * 70)
    print(f'1000-STEP COMPARISON RESULTS')
    print('=' * 70)

    ctrl = results.get('control_b4', {})
    w6 = results.get('w6', {})

    def pct(new, old):
        if old and old > 0:
            return f'+{(new/old - 1)*100:.1f}%'
        return 'N/A'

    headers = ['Metric', 'control_b4', 'w6', 'Improvement']
    rows = [
        ['Completed Steps', ctrl.get('completed_step', '?'), w6.get('completed_step', '?'), ''],
        ['Wall Clock (s)', ctrl.get('wall_seconds', '?'), w6.get('wall_seconds', '?'),
         pct(ctrl.get('wall_seconds', 1), w6.get('wall_seconds', 1)) + ' faster' if w6.get('wall_seconds') else ''],
        ['Throughput (sps)', ctrl.get('completed_sps', '?'), w6.get('completed_sps', '?'),
         pct(w6.get('completed_sps', 0), ctrl.get('completed_sps', 1))],
        ['Loader Wait Frac', ctrl.get('completed_wait', '?'), w6.get('completed_wait', '?'), ''],
        ['CPU Pipe Wait', ctrl.get('completed_cpu_pipe_wait', '?'), w6.get('completed_cpu_pipe_wait', '?'), ''],
        ['Sustained SPS', ctrl.get('sustained_sps', '?'), w6.get('sustained_sps', '?'),
         pct(w6.get('sustained_sps', 0), ctrl.get('sustained_sps', 1))],
        ['Sustained Wait', ctrl.get('sustained_wait', '?'), w6.get('sustained_wait', '?'), ''],
        ['GPU Ratio', ctrl.get('gpu_ratio', '?'), w6.get('gpu_ratio', '?'), ''],
        ['Startup (s)', ctrl.get('startup_seconds', '?'), w6.get('startup_seconds', '?'), ''],
        ['Peak RSS (GiB)', ctrl.get('peak_rss_gib', '?'), w6.get('peak_rss_gib', '?'), ''],
    ]

    # Print table
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(4)]
    fmt = ' | '.join(f'{{:<{w}}}' for w in col_widths)
    print(fmt.format(*headers))
    print('-+-'.join('-' * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))

    # Save results
    output_path = report_root / 'comparison_1000step.json'
    output_path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
