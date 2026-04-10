#!/usr/bin/env python
"""
Step 6 Phase 5: Multi-Worker DataLoader comparison.

Runs the confirmed control_b4 baseline against multiple num_workers > 0 variants.
Phase 5a: 192-file module benchmark on single GPU for all worker counts.
Phase 5b: 200-step dual-A100 preflight for the top candidates.

Usage:
    python scripts/run_step6_phase5_worker_comparison.py
    python scripts/run_step6_phase5_worker_comparison.py --phase module_only
    python scripts/run_step6_phase5_worker_comparison.py --phase preflight_only --workers 4,8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'
os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import load_full_config, subprocess_env, utc_now_iso  # noqa: E402
from step6_experiments import (  # noqa: E402
    average_wait_fraction_after_step,
    deep_update,
    phase5_candidate_beats_control,
    phase5_worker_overrides,
    phase_runtime_overrides,
    render_markdown_table,
    summarize_preflight_row,
    write_config,
)


DEFAULT_CONFIG = 'configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml'
WORKER_COUNTS = [2, 4, 8, 12, 16, 24]
TOP_N_FOR_PREFLIGHT = 3

MODULE_COLUMNS = [
    ('name', 'Experiment'),
    ('num_workers', 'Workers'),
    ('module_raw_to_cpu_sps', 'Producer SPS'),
    ('module_training_sps', 'Train SPS'),
    ('module_can_hide_producer', 'Can Hide'),
    ('module_sps_ratio', 'Prod/Train'),
    ('status', 'Status'),
]

PREFLIGHT_COLUMNS = [
    ('name', 'Experiment'),
    ('num_workers', 'Workers'),
    ('module_raw_to_cpu_sps', 'Module Producer SPS'),
    ('module_training_sps', 'Module Train SPS'),
    ('module_can_hide_producer', 'Module Hide'),
    ('completed_step', 'Completed Step'),
    ('samples_per_second', 'Completed SPS'),
    ('loader_wait_fraction', 'Completed Wait'),
    ('late_window_wait_fraction', 'Late Wait'),
    ('raw_read_fraction', 'Raw Read Frac'),
    ('collate_or_assemble_fraction', 'Collate Frac'),
    ('cpu_pipe_wait_fraction', 'CPU Pipe Wait'),
    ('gate_samples_per_second', 'Sustained SPS'),
    ('gate_loader_wait_fraction', 'Sustained Wait'),
    ('steady_gpu_ratio', 'GPU Ratio'),
    ('startup_seconds', 'Startup Sec'),
    ('peak_combined_rss_gib', 'Peak RSS GiB'),
    ('promoted', 'Promoted'),
    ('status', 'Status'),
]


def default_torchrun_bin() -> str:
    sibling = Path(sys.executable).with_name('torchrun')
    if sibling.exists():
        return str(sibling)
    return shutil.which('torchrun') or 'torchrun'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 6 Phase 5: multi-worker DataLoader comparison.',
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Base control config.')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--torchrun-bin', default=default_torchrun_bin())
    parser.add_argument(
        '--module-cuda-visible-devices', default='0',
        help='CUDA_VISIBLE_DEVICES for single-GPU module benchmark.',
    )
    parser.add_argument(
        '--preflight-cuda-visible-devices', default='0,1',
        help='CUDA_VISIBLE_DEVICES for dual-GPU preflight.',
    )
    parser.add_argument('--run-id', default='latest')
    parser.add_argument(
        '--work-root', default='artifacts/tmp/step6_phase5_worker',
    )
    parser.add_argument(
        '--report-root', default='artifacts/reports/step6_phase5_worker',
    )
    parser.add_argument(
        '--workers', default='',
        help='Comma-separated worker counts to test. Default: 2,4,8,12,16,24',
    )
    parser.add_argument(
        '--phase', default='full',
        choices=('full', 'module_only', 'preflight_only'),
        help='Which phases to run.',
    )
    parser.add_argument(
        '--top-n', type=int, default=TOP_N_FOR_PREFLIGHT,
        help='Number of top module-benchmark winners to send to preflight.',
    )
    parser.add_argument(
        '--sample-size', type=int, default=192,
        help='Number of files for module benchmark.',
    )
    parser.add_argument(
        '--mp-context', default='spawn',
        choices=('spawn', 'fork', 'forkserver'),
        help='Multiprocessing context for worker processes. forkserver uses COW memory sharing.',
    )
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def save_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def run_command(*, command: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Write directly to file instead of capture_output=True.
    # This prevents orphaned forkserver workers from blocking
    # subprocess.run by holding inherited pipe file descriptors open.
    with log_path.open('w', encoding='utf-8') as log_fh:
        log_fh.write(f'$ {" ".join(command)}\n')
        log_fh.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            stdout=log_fh,
            stderr=log_fh,
        )
    return int(result.returncode)


def build_variant_config(
    *,
    source_config_path: Path,
    run_root: Path,
    experiment_name: str,
    num_workers: int,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    cpu_ready_batches: int = 4,
    multiprocessing_context: str = 'spawn',
) -> tuple[Path, Path, Path, dict]:
    _resolved, full_config = load_full_config(source_config_path)
    config_payload = deep_update(
        full_config,
        phase_runtime_overrides(run_root=run_root, experiment_name=experiment_name),
    )
    if num_workers > 0:
        config_payload = deep_update(
            config_payload,
            phase5_worker_overrides(
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                cpu_ready_batches=cpu_ready_batches,
                multiprocessing_context=multiprocessing_context,
            ),
        )
    config_payload = deep_update(
        config_payload,
        {
            'bc': {
                'wandb': {
                    'enabled': False,
                    'group': 'step6_phase5_worker',
                    'name': experiment_name,
                },
                'preflight': {
                    'disable_wandb': True,
                },
            },
        },
    )
    config_path = run_root / 'configs' / f'{experiment_name}.toml'
    return (
        write_config(config_path, config_payload),
        resolve_path(config_payload['bc']['preflight']['summary_json']),
        resolve_path(config_payload['bc']['control']['metrics_jsonl']),
        config_payload,
    )


def run_module_benchmark(
    *,
    config_path: Path,
    python_bin: str,
    cuda_visible_devices: str,
    output_json: Path,
    log_path: Path,
    sample_size: int = 192,
) -> tuple[int, dict]:
    command = [
        python_bin,
        'scripts/benchmark_bc_conversion_vs_training.py',
        '--config',
        str(config_path),
        '--split',
        'train',
        '--sample-size',
        str(sample_size),
        '--sample-strategy',
        'round_robin',
        '--output-json',
        str(output_json),
    ]
    env = subprocess_env(config_path)
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    return_code = run_command(command=command, env=env, log_path=log_path)
    if not output_json.exists():
        return return_code, {}
    with output_json.open(encoding='utf-8') as f:
        return return_code, json.load(f)


def run_preflight(
    *,
    config_path: Path,
    python_bin: str,
    torchrun_bin: str,
    cuda_visible_devices: str,
    log_path: Path,
) -> int:
    command = [
        python_bin,
        'scripts/run_bc_loader_preflight.py',
        '--config',
        str(config_path),
        '--python-bin',
        python_bin,
        '--torchrun-bin',
        torchrun_bin,
    ]
    env = subprocess_env(config_path)
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    return run_command(command=command, env=env, log_path=log_path)


def extract_module_row(
    *,
    experiment_name: str,
    num_workers: int,
    module_return_code: int,
    module_summary: dict,
) -> dict:
    raw_to_cpu = module_summary.get('raw_to_cpu_batches') or {}
    training = module_summary.get('training_on_produced_batches') or {}
    comparison = module_summary.get('comparison') or {}
    producer_sps = float(raw_to_cpu.get('samples_per_second') or 0.0)
    training_sps = float(training.get('samples_per_second') or 0.0)
    return {
        'name': experiment_name,
        'num_workers': num_workers,
        'module_raw_to_cpu_sps': producer_sps,
        'module_training_sps': training_sps,
        'module_can_hide_producer': bool(comparison.get('can_hide_producer', False)),
        'module_sps_ratio': round(producer_sps / training_sps, 4) if training_sps > 0 else 0.0,
        'module_return_code': module_return_code,
        'status': 'ok' if module_return_code == 0 else f'exit_{module_return_code}',
    }


def run_full_variant(
    *,
    source_config_path: Path,
    run_root: Path,
    python_bin: str,
    torchrun_bin: str,
    module_cuda_visible_devices: str,
    preflight_cuda_visible_devices: str,
    experiment_name: str,
    num_workers: int,
    sample_size: int = 192,
    module_row: dict | None = None,
    multiprocessing_context: str = 'spawn',
) -> dict:
    config_path, summary_path, metrics_jsonl_path, config_payload = build_variant_config(
        source_config_path=source_config_path,
        run_root=run_root,
        experiment_name=experiment_name,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
    )

    if module_row is None:
        module_output_json = run_root / 'module' / f'{experiment_name}.json'
        module_log_path = run_root / 'logs' / f'{experiment_name}.module.log'
        module_return_code, module_summary = run_module_benchmark(
            config_path=config_path,
            python_bin=python_bin,
            cuda_visible_devices=module_cuda_visible_devices,
            output_json=module_output_json,
            log_path=module_log_path,
            sample_size=sample_size,
        )
        module_row = extract_module_row(
            experiment_name=experiment_name,
            num_workers=num_workers,
            module_return_code=module_return_code,
            module_summary=module_summary,
        )

    preflight_log_path = run_root / 'logs' / f'{experiment_name}.preflight.log'
    wrapper_return_code = run_preflight(
        config_path=config_path,
        python_bin=python_bin,
        torchrun_bin=torchrun_bin,
        cuda_visible_devices=preflight_cuda_visible_devices,
        log_path=preflight_log_path,
    )

    preflight_summary = {}
    if summary_path.exists():
        with summary_path.open(encoding='utf-8') as f:
            preflight_summary = json.load(f)

    row = summarize_preflight_row(
        phase='phase5',
        name=experiment_name,
        knobs={'num_workers': num_workers},
        stage_summary=None,
        preflight_summary=preflight_summary,
        backend='raw_threaded',
    )
    row['num_workers'] = num_workers
    row['late_window_wait_fraction'] = average_wait_fraction_after_step(metrics_jsonl_path, min_step=125)
    row['wrapper_return_code'] = wrapper_return_code
    row.update({
        'module_raw_to_cpu_sps': module_row.get('module_raw_to_cpu_sps', 0.0),
        'module_training_sps': module_row.get('module_training_sps', 0.0),
        'module_can_hide_producer': module_row.get('module_can_hide_producer', False),
        'module_return_code': module_row.get('module_return_code', 0),
    })
    row['source_config'] = str(source_config_path)
    row['config_path'] = str(config_path)
    row['summary_path'] = str(summary_path)
    row['metrics_jsonl_path'] = str(metrics_jsonl_path)
    row['preflight_log_path'] = str(preflight_log_path)
    return row


def select_top_module_candidates(module_rows: list[dict], *, top_n: int) -> list[dict]:
    """Select top N candidates by producer SPS, excluding control and failed runs."""
    candidates = [
        r for r in module_rows
        if r.get('num_workers', 0) > 0 and r.get('module_return_code', 1) == 0
    ]
    candidates.sort(key=lambda r: float(r.get('module_raw_to_cpu_sps') or 0.0), reverse=True)
    return candidates[:top_n]


def main():
    args = parse_args()
    source_config_path = resolve_path(args.config)
    run_root = resolve_path(Path(args.work_root) / args.run_id)
    report_root = resolve_path(Path(args.report_root) / args.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    worker_counts = [int(w) for w in args.workers.split(',') if w.strip()] if args.workers else WORKER_COUNTS
    run_module = args.phase in ('full', 'module_only')
    run_preflights = args.phase in ('full', 'preflight_only')

    print(f'=== Step 6 Phase 5: Multi-Worker DataLoader Comparison ===')
    print(f'Control config: {source_config_path}')
    print(f'Worker counts to test: {worker_counts}')
    print(f'Multiprocessing context: {args.mp_context}')
    print(f'Run root: {run_root}')
    print(f'Report root: {report_root}')
    print()

    # --- Phase 5a: Module benchmarks ---
    module_rows = []
    if run_module:
        print('--- Phase 5a: Module Benchmarks (single GPU, 192 files) ---')
        print()

        # Control first
        print(f'  Running control_b4 (num_workers=0)...')
        t0 = time.monotonic()
        control_config_path, _, _, _ = build_variant_config(
            source_config_path=source_config_path,
            run_root=run_root,
            experiment_name='control_b4',
            num_workers=0,
            multiprocessing_context=args.mp_context,
        )
        control_module_json = run_root / 'module' / 'control_b4.json'
        control_module_log = run_root / 'logs' / 'control_b4.module.log'
        control_rc, control_module = run_module_benchmark(
            config_path=control_config_path,
            python_bin=args.python_bin,
            cuda_visible_devices=args.module_cuda_visible_devices,
            output_json=control_module_json,
            log_path=control_module_log,
            sample_size=args.sample_size,
        )
        control_module_row = extract_module_row(
            experiment_name='control_b4',
            num_workers=0,
            module_return_code=control_rc,
            module_summary=control_module,
        )
        module_rows.append(control_module_row)
        elapsed = time.monotonic() - t0
        print(f'    control_b4: producer={control_module_row["module_raw_to_cpu_sps"]:.1f} sps, '
              f'train={control_module_row["module_training_sps"]:.1f} sps, '
              f'can_hide={control_module_row["module_can_hide_producer"]}  ({elapsed:.0f}s)')

        # Worker variants
        for nw in worker_counts:
            name = f'w{nw}'
            print(f'  Running {name} (num_workers={nw})...')
            t0 = time.monotonic()
            variant_config_path, _, _, _ = build_variant_config(
                source_config_path=source_config_path,
                run_root=run_root,
                experiment_name=name,
                num_workers=nw,
                multiprocessing_context=args.mp_context,
            )
            variant_module_json = run_root / 'module' / f'{name}.json'
            variant_module_log = run_root / 'logs' / f'{name}.module.log'
            variant_rc, variant_module = run_module_benchmark(
                config_path=variant_config_path,
                python_bin=args.python_bin,
                cuda_visible_devices=args.module_cuda_visible_devices,
                output_json=variant_module_json,
                log_path=variant_module_log,
                sample_size=args.sample_size,
            )
            variant_row = extract_module_row(
                experiment_name=name,
                num_workers=nw,
                module_return_code=variant_rc,
                module_summary=variant_module,
            )
            module_rows.append(variant_row)
            elapsed = time.monotonic() - t0
            print(f'    {name}: producer={variant_row["module_raw_to_cpu_sps"]:.1f} sps, '
                  f'train={variant_row["module_training_sps"]:.1f} sps, '
                  f'can_hide={variant_row["module_can_hide_producer"]}  ({elapsed:.0f}s)')

        # Save module results
        module_markdown = render_markdown_table(module_rows, columns=MODULE_COLUMNS)
        module_payload = {
            'created_at': utc_now_iso(),
            'phase': 'phase5a_module',
            'run_root': str(run_root),
            'rows': module_rows,
            'columns': MODULE_COLUMNS,
            'markdown_table': module_markdown,
        }
        save_json(report_root / 'phase5a_module_summary.json', module_payload)
        (report_root / 'phase5a_module_summary.md').write_text(module_markdown, encoding='utf-8')
        print()
        print('Phase 5a Module Results:')
        print(module_markdown)
        print()

    # --- Phase 5b: Preflights ---
    if run_preflights:
        print('--- Phase 5b: 200-step Dual-A100 Preflights ---')
        print()

        # If we have module results, select top N; otherwise use the provided --workers
        if module_rows:
            top_candidates = select_top_module_candidates(module_rows, top_n=args.top_n)
            preflight_worker_counts = [r['num_workers'] for r in top_candidates]
            print(f'  Top {args.top_n} by module producer SPS: {preflight_worker_counts}')
        else:
            preflight_worker_counts = worker_counts[:args.top_n]
            print(f'  No module results available. Using first {args.top_n} worker counts: {preflight_worker_counts}')

        preflight_rows = []

        # Get module row lookup for reuse
        module_row_map = {r['num_workers']: r for r in module_rows}

        # Control preflight
        print(f'  Running preflight: control_b4 (num_workers=0)...')
        t0 = time.monotonic()
        control_preflight_row = run_full_variant(
            source_config_path=source_config_path,
            run_root=run_root,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            module_cuda_visible_devices=args.module_cuda_visible_devices,
            preflight_cuda_visible_devices=args.preflight_cuda_visible_devices,
            experiment_name='control_b4',
            num_workers=0,
            sample_size=args.sample_size,
            module_row=module_row_map.get(0),
            multiprocessing_context=args.mp_context,
        )
        preflight_rows.append(control_preflight_row)
        elapsed = time.monotonic() - t0
        print(f'    control_b4: completed={float(control_preflight_row.get("samples_per_second") or 0):.1f} sps, '
              f'wait={float(control_preflight_row.get("loader_wait_fraction") or 0):.4f}, '
              f'gpu_ratio={float(control_preflight_row.get("steady_gpu_ratio") or 0):.4f}, '
              f'rss={float(control_preflight_row.get("peak_combined_rss_gib") or 0):.1f} GiB  ({elapsed:.0f}s)')

        # Worker preflights
        for nw in preflight_worker_counts:
            name = f'w{nw}'
            print(f'  Running preflight: {name} (num_workers={nw})...')
            t0 = time.monotonic()
            variant_row = run_full_variant(
                source_config_path=source_config_path,
                run_root=run_root,
                python_bin=args.python_bin,
                torchrun_bin=args.torchrun_bin,
                module_cuda_visible_devices=args.module_cuda_visible_devices,
                preflight_cuda_visible_devices=args.preflight_cuda_visible_devices,
                experiment_name=name,
                num_workers=nw,
                sample_size=args.sample_size,
                module_row=module_row_map.get(nw),
                multiprocessing_context=args.mp_context,
            )
            preflight_rows.append(variant_row)
            elapsed = time.monotonic() - t0
            print(f'    {name}: completed={float(variant_row.get("samples_per_second") or 0):.1f} sps, '
                  f'wait={float(variant_row.get("loader_wait_fraction") or 0):.4f}, '
                  f'gpu_ratio={float(variant_row.get("steady_gpu_ratio") or 0):.4f}, '
                  f'rss={float(variant_row.get("peak_combined_rss_gib") or 0):.1f} GiB  ({elapsed:.0f}s)')

        # Promotion decisions
        control_row = preflight_rows[0]
        for row in preflight_rows[1:]:
            row['promoted'] = bool(phase5_candidate_beats_control(
                control=control_row,
                candidate=row,
            ))
        control_row['promoted'] = False

        # Save preflight results
        preflight_markdown = render_markdown_table(preflight_rows, columns=PREFLIGHT_COLUMNS)
        preflight_payload = {
            'created_at': utc_now_iso(),
            'phase': 'phase5b_preflight',
            'run_root': str(run_root),
            'rows': preflight_rows,
            'control': control_row,
            'columns': PREFLIGHT_COLUMNS,
            'markdown_table': preflight_markdown,
        }
        save_json(report_root / 'phase5b_preflight_summary.json', preflight_payload)
        (report_root / 'phase5b_preflight_summary.md').write_text(preflight_markdown, encoding='utf-8')
        print()
        print('Phase 5b Preflight Results:')
        print(preflight_markdown)

        # Overall summary
        promoted = [r for r in preflight_rows if r.get('promoted')]
        print()
        if promoted:
            print(f'PROMOTED: {[r["name"] for r in promoted]}')
        else:
            print('NO CANDIDATES PROMOTED. control_b4 remains the Step 6 baseline.')

    print()
    print(f'All results saved to: {report_root}')


if __name__ == '__main__':
    main()
