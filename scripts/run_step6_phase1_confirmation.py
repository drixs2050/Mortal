#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
    phase1_candidate_beats_control,
    phase_runtime_overrides,
    render_markdown_table,
    summarize_phase1_queue_row,
    write_config,
)


DEFAULT_CONFIGS = [
    'configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml',
    'configs/step6_bc_large_preflight_full8dan_r5.toml',
    'configs/step6_bc_large_preflight_full8dan_r5_phase1b_cap8g.toml',
]

DEFAULT_COLUMNS = [
    ('name', 'Experiment'),
    ('source_config', 'Source Config'),
    ('cpu_ready_batches_target', 'Ready Batches'),
    ('cpu_ready_bytes_target_gib', 'Ready GiB Cap'),
    ('completed_step', 'Completed Step'),
    ('samples_per_second', 'Completed SPS'),
    ('loader_wait_fraction', 'Completed Wait'),
    ('late_window_wait_fraction', 'Late Wait'),
    ('gate_samples_per_second', 'Sustained SPS'),
    ('gate_loader_wait_fraction', 'Sustained Wait'),
    ('steady_gpu_ratio', 'GPU Ratio'),
    ('cpu_ready_bytes_gib', 'Observed Ready GiB'),
    ('startup_seconds', 'Startup Sec'),
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
        description=(
            'Run the Step 6 Phase 1 confirmation set on the historical b4 control, '
            'the promoted b16 baseline, and the retained b16_cap8g comparison branch.'
        ),
    )
    parser.add_argument(
        '--config',
        action='append',
        default=[],
        help='Config to include in the confirmation set. Repeat to add multiple configs.',
    )
    parser.add_argument(
        '--python-bin',
        default=sys.executable,
        help='Python interpreter used for the preflight wrapper.',
    )
    parser.add_argument(
        '--torchrun-bin',
        default=default_torchrun_bin(),
        help='Torchrun executable used by the preflight wrapper.',
    )
    parser.add_argument(
        '--cuda-visible-devices',
        default='0,1',
        help='CUDA_VISIBLE_DEVICES to use for the confirmation runs.',
    )
    parser.add_argument(
        '--run-id',
        default='latest',
        help='Run id under the work/report roots.',
    )
    parser.add_argument(
        '--work-root',
        default='artifacts/tmp/step6_phase1_confirmation',
        help='Root directory for generated configs, logs, and per-run outputs.',
    )
    parser.add_argument(
        '--report-root',
        default='artifacts/reports/step6_phase1_confirmation',
        help='Root directory for the confirmation JSON and Markdown reports.',
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


def run_preflight(
    *,
    config_path: Path,
    python_bin: str,
    torchrun_bin: str,
    env: dict[str, str],
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    combined = [f'$ {" ".join(command)}\n']
    if result.stdout:
        combined.append(result.stdout)
        if not result.stdout.endswith('\n'):
            combined.append('\n')
    if result.stderr:
        combined.append('\n[stderr]\n')
        combined.append(result.stderr)
        if not result.stderr.endswith('\n'):
            combined.append('\n')
    log_path.write_text(''.join(combined), encoding='utf-8')
    return int(result.returncode)


def experiment_name_from_config(config_path: Path) -> str:
    return config_path.stem


def build_run_config(
    *,
    source_config_path: Path,
    run_root: Path,
) -> tuple[Path, Path, Path, dict]:
    _resolved_config_path, full_config = load_full_config(source_config_path)
    experiment_name = experiment_name_from_config(source_config_path)
    config_payload = deep_update(
        full_config,
        phase_runtime_overrides(run_root=run_root, experiment_name=experiment_name),
    )
    config_payload = deep_update(
        config_payload,
        {
            'bc': {
                'wandb': {
                    'enabled': False,
                    'group': 'step6_phase1_confirmation',
                    'name': experiment_name,
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


def run_config(
    *,
    source_config_path: Path,
    run_root: Path,
    python_bin: str,
    torchrun_bin: str,
    cuda_visible_devices: str,
) -> dict:
    config_path, summary_path, metrics_jsonl_path, config_payload = build_run_config(
        source_config_path=source_config_path,
        run_root=run_root,
    )
    env = subprocess_env(config_path)
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    experiment_name = experiment_name_from_config(source_config_path)
    log_path = run_root / 'logs' / f'{experiment_name}.log'
    wrapper_return_code = run_preflight(
        config_path=config_path,
        python_bin=python_bin,
        torchrun_bin=torchrun_bin,
        env=env,
        log_path=log_path,
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            f'preflight summary was not written for {experiment_name}: expected {summary_path}'
        )
    with summary_path.open(encoding='utf-8') as f:
        preflight_summary = json.load(f)
    dataset_cfg = (config_payload.get('bc') or {}).get('dataset') or {}
    row = summarize_phase1_queue_row(
        name=experiment_name,
        knobs={
            'cpu_ready_batches': int(dataset_cfg.get('cpu_ready_batches', 0) or 0),
            'cpu_ready_bytes_gib': float(dataset_cfg.get('cpu_ready_bytes_gib', 0.0) or 0.0),
        },
        preflight_summary=preflight_summary,
    )
    row['late_window_wait_fraction'] = average_wait_fraction_after_step(metrics_jsonl_path, min_step=125)
    row['wrapper_return_code'] = wrapper_return_code
    row['source_config'] = str(source_config_path)
    row['config_path'] = str(config_path)
    row['summary_path'] = str(summary_path)
    row['metrics_jsonl_path'] = str(metrics_jsonl_path)
    row['log_path'] = str(log_path)
    return row


def main():
    args = parse_args()
    config_paths = [resolve_path(path) for path in (args.config or DEFAULT_CONFIGS)]
    run_root = resolve_path(Path(args.work_root) / args.run_id)
    report_root = resolve_path(Path(args.report_root) / args.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for config_path in config_paths:
        rows.append(
            run_config(
                source_config_path=config_path,
                run_root=run_root,
                python_bin=args.python_bin,
                torchrun_bin=args.torchrun_bin,
                cuda_visible_devices=args.cuda_visible_devices,
            )
        )

    if not rows:
        raise ValueError('no rows were produced')

    control_row = rows[0]
    control_row['promoted'] = False
    promoted_rows = []
    for row in rows[1:]:
        row['promoted'] = bool(phase1_candidate_beats_control(control=control_row, candidate=row))
        if row['promoted']:
            promoted_rows.append(row)

    best_by_completed_sps = max(rows, key=lambda row: float(row.get('samples_per_second') or 0.0))
    best_by_sustained_sps = max(rows, key=lambda row: float(row.get('gate_samples_per_second') or 0.0))
    best_by_steady_gpu = max(rows, key=lambda row: float(row.get('steady_gpu_ratio') or 0.0))

    markdown = render_markdown_table(rows, columns=DEFAULT_COLUMNS)
    report_payload = {
        'created_at': utc_now_iso(),
        'run_root': str(run_root),
        'rows': rows,
        'control': control_row,
        'promoted_rows': promoted_rows,
        'best_by_completed_sps': best_by_completed_sps,
        'best_by_sustained_sps': best_by_sustained_sps,
        'best_by_steady_gpu': best_by_steady_gpu,
        'columns': DEFAULT_COLUMNS,
        'markdown_table': markdown,
    }
    save_json(report_root / 'summary.json', report_payload)
    (report_root / 'summary.md').write_text(markdown, encoding='utf-8')


if __name__ == '__main__':
    main()
