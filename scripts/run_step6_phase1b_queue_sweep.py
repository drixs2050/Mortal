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
    raw_threaded_queue_overrides,
    render_markdown_table,
    select_phase1_batch_count_winner,
    summarize_phase1_queue_row,
    write_config,
)


DEFAULT_COLUMNS = [
    ('name', 'Experiment'),
    ('cpu_ready_batches_target', 'Ready Batches'),
    ('cpu_ready_bytes_target_gib', 'Ready GiB Cap'),
    ('completed_step', 'Completed Step'),
    ('samples_per_second', 'Samples/Sec'),
    ('loader_wait_fraction', 'Wait'),
    ('late_window_wait_fraction', 'Late Wait'),
    ('steady_gpu_ratio', 'GPU Ratio'),
    ('cpu_ready_bytes_gib', 'Observed Ready GiB'),
    ('peak_combined_rss_gib', 'Peak RSS GiB'),
    ('startup_seconds', 'Startup Sec'),
    ('promoted', 'Promoted'),
    ('status', 'Status'),
]

DEFAULT_BATCH_GRID = (4, 6, 8, 10, 12, 16, 20, 24)
DEFAULT_BYTE_CAP_GRIDS = {
    8: (4.0, 6.0, 8.0),
    12: (6.0, 8.0, 10.0, 12.0),
    16: (8.0, 10.0, 12.0, 14.0),
}


def default_torchrun_bin() -> str:
    sibling = Path(sys.executable).with_name('torchrun')
    if sibling.exists():
        return str(sibling)
    return shutil.which('torchrun') or 'torchrun'


def parse_batch_grid(raw_value: str) -> list[int]:
    values = []
    for part in str(raw_value or '').split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError('batch grid must contain at least one batch count')
    if sorted(values) != values:
        raise ValueError('batch grid must be sorted in ascending order')
    if len(set(values)) != len(values):
        raise ValueError('batch grid must not contain duplicate values')
    if values[0] != 4:
        raise ValueError('batch grid must begin with the control depth 4')
    return values


def parse_byte_cap_spec(specs: list[str]) -> dict[int, list[float]]:
    if not specs:
        return {
            depth: list(caps)
            for depth, caps in DEFAULT_BYTE_CAP_GRIDS.items()
        }
    parsed: dict[int, list[float]] = {}
    for spec in specs:
        raw = str(spec or '').strip()
        if not raw:
            continue
        if ':' not in raw:
            raise ValueError(
                f"invalid byte-cap spec {raw!r}; expected '<ready_batches>:<cap1>,<cap2>,...'"
            )
        depth_raw, caps_raw = raw.split(':', 1)
        depth = int(depth_raw.strip())
        caps = []
        for cap_part in caps_raw.split(','):
            cap_part = cap_part.strip()
            if not cap_part:
                continue
            caps.append(float(cap_part))
        if not caps:
            raise ValueError(f'invalid byte-cap spec {raw!r}; no cap values were provided')
        if sorted(caps) != caps:
            raise ValueError(f'byte caps for ready_batches={depth} must be sorted ascending')
        if len(set(caps)) != len(caps):
            raise ValueError(f'byte caps for ready_batches={depth} must not contain duplicates')
        parsed[depth] = caps
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run the corrected Step 6 Phase 1b pageable CPU queue sweep with a broader '
            'ready-batch grid and byte-cap rungs that can actually bind.'
        ),
    )
    parser.add_argument(
        '--base-config',
        default='configs/step6_bc_large_preflight_full8dan_r5.toml',
        help='Baseline Step 6 preflight config to clone for each Phase 1b rung.',
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
        help='CUDA_VISIBLE_DEVICES to use for the sweep.',
    )
    parser.add_argument(
        '--run-id',
        default='latest',
        help='Run id under the work/report roots.',
    )
    parser.add_argument(
        '--work-root',
        default='artifacts/tmp/step6_phase1b_queue_sweep',
        help='Root directory for generated configs, logs, and per-rung outputs.',
    )
    parser.add_argument(
        '--report-root',
        default='artifacts/reports/step6_phase1b_queue_sweep',
        help='Root directory for the queue-sweep JSON and Markdown reports.',
    )
    parser.add_argument(
        '--batch-grid',
        default=','.join(str(value) for value in DEFAULT_BATCH_GRID),
        help='Comma-separated ready-batch depths to sweep. Must begin with 4.',
    )
    parser.add_argument(
        '--byte-cap-grid',
        action='append',
        default=[],
        help=(
            "Optional byte-cap sweep spec in the form '<ready_batches>:<cap1>,<cap2>,...'. "
            "Defaults to 8:(4,6,8), 12:(6,8,10,12), and 16:(8,10,12,14). "
            'Repeat the flag to provide multiple specs.'
        ),
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


def build_rung_config(
    *,
    base_config: dict,
    run_root: Path,
    experiment_name: str,
    knobs: dict,
) -> tuple[Path, Path, Path]:
    config_payload = deep_update(
        base_config,
        phase_runtime_overrides(run_root=run_root, experiment_name=experiment_name),
    )
    config_payload = deep_update(
        config_payload,
        raw_threaded_queue_overrides(
            cpu_ready_batches=int(knobs['cpu_ready_batches']),
            cpu_ready_bytes_gib=float(knobs.get('cpu_ready_bytes_gib', 0.0)),
        ),
    )
    config_payload = deep_update(
        config_payload,
        {
            'bc': {
                'wandb': {
                    'enabled': False,
                    'group': 'step6_phase1b_queue_sweep',
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
    )


def run_rung(
    *,
    base_config: dict,
    run_root: Path,
    python_bin: str,
    torchrun_bin: str,
    cuda_visible_devices: str,
    name: str,
    knobs: dict,
) -> dict:
    config_path, summary_path, metrics_jsonl_path = build_rung_config(
        base_config=base_config,
        run_root=run_root,
        experiment_name=name,
        knobs=knobs,
    )
    env = subprocess_env(config_path)
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    log_path = run_root / 'logs' / f'{name}.log'
    wrapper_return_code = run_preflight(
        config_path=config_path,
        python_bin=python_bin,
        torchrun_bin=torchrun_bin,
        env=env,
        log_path=log_path,
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            f'preflight summary was not written for {name}: expected {summary_path}'
        )
    with summary_path.open(encoding='utf-8') as f:
        preflight_summary = json.load(f)
    row = summarize_phase1_queue_row(
        name=name,
        knobs=knobs,
        preflight_summary=preflight_summary,
    )
    row['late_window_wait_fraction'] = average_wait_fraction_after_step(metrics_jsonl_path, min_step=125)
    row['wrapper_return_code'] = wrapper_return_code
    row['config_path'] = str(config_path)
    row['summary_path'] = str(summary_path)
    row['metrics_jsonl_path'] = str(metrics_jsonl_path)
    row['log_path'] = str(log_path)
    return row


def promoted_rows(*, control_row: dict, candidate_rows: list[dict]) -> list[dict]:
    promoted = []
    for row in candidate_rows:
        flagged = phase1_candidate_beats_control(control=control_row, candidate=row)
        row['promoted'] = bool(flagged)
        if flagged:
            promoted.append(row)
    control_row['promoted'] = False
    return promoted


def build_batch_rungs(batch_grid: list[int]) -> list[tuple[str, dict]]:
    rungs = []
    for idx, ready_batches in enumerate(batch_grid):
        if idx == 0:
            name = f'control_b{ready_batches}'
        else:
            name = f'queue_b{ready_batches}'
        rungs.append(
            (
                name,
                {
                    'cpu_ready_batches': int(ready_batches),
                    'cpu_ready_bytes_gib': 0.0,
                },
            )
        )
    return rungs


def build_byte_cap_rungs(
    *,
    batch_grid: list[int],
    byte_cap_grids: dict[int, list[float]],
) -> list[tuple[str, dict]]:
    batch_depths = set(int(value) for value in batch_grid)
    rungs = []
    for ready_batches in sorted(byte_cap_grids):
        if ready_batches not in batch_depths:
            continue
        for cap_gib in byte_cap_grids[ready_batches]:
            cap_label = str(cap_gib).replace('.', 'p').rstrip('0').rstrip('p')
            rungs.append(
                (
                    f'queue_b{ready_batches}_cap{cap_label}g',
                    {
                        'cpu_ready_batches': int(ready_batches),
                        'cpu_ready_bytes_gib': float(cap_gib),
                    },
                )
            )
    return rungs


def main():
    args = parse_args()
    batch_grid = parse_batch_grid(args.batch_grid)
    byte_cap_grids = parse_byte_cap_spec(args.byte_cap_grid)

    base_config_path, base_config = load_full_config(args.base_config)
    run_root = resolve_path(Path(args.work_root) / args.run_id)
    report_root = resolve_path(Path(args.report_root) / args.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    batch_rungs = build_batch_rungs(batch_grid)
    byte_cap_rungs = build_byte_cap_rungs(
        batch_grid=batch_grid,
        byte_cap_grids=byte_cap_grids,
    )

    rows = []
    for name, knobs in [*batch_rungs, *byte_cap_rungs]:
        rows.append(
            run_rung(
                base_config=base_config,
                run_root=run_root,
                python_bin=args.python_bin,
                torchrun_bin=args.torchrun_bin,
                cuda_visible_devices=args.cuda_visible_devices,
                name=name,
                knobs=knobs,
            )
        )

    control_row = rows[0]
    batch_count_rows = rows[: len(batch_rungs)]
    batch_count_winner = select_phase1_batch_count_winner(batch_count_rows)
    promoted = promoted_rows(control_row=control_row, candidate_rows=rows[1:])
    if promoted:
        recommended = select_phase1_batch_count_winner(promoted)
        recommendation_reason = 'promoted_by_phase1_rule'
    else:
        recommended = control_row
        recommendation_reason = 'control_retained'

    markdown = render_markdown_table(rows, columns=DEFAULT_COLUMNS)
    report_payload = {
        'created_at': utc_now_iso(),
        'base_config_path': str(base_config_path),
        'run_root': str(run_root),
        'batch_grid': batch_grid,
        'byte_cap_grids': byte_cap_grids,
        'rows': rows,
        'batch_count_winner': batch_count_winner,
        'recommended': recommended,
        'recommendation_reason': recommendation_reason,
        'promotion_rule': {
            'relative_sps_gain': 0.03,
            'wait_fraction_drop': 0.03,
            'max_sps_regression_for_wait_win': 0.01,
        },
        'columns': DEFAULT_COLUMNS,
        'markdown_table': markdown,
    }
    save_json(report_root / 'summary.json', report_payload)
    (report_root / 'summary.md').write_text(markdown, encoding='utf-8')


if __name__ == '__main__':
    main()
