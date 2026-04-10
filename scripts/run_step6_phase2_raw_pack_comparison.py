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
    phase2_candidate_beats_control,
    phase_runtime_overrides,
    raw_source_backend_overrides,
    render_markdown_table,
    summarize_preflight_row,
    write_config,
)


DEFAULT_CONFIG = 'configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml'
DEFAULT_COLUMNS = [
    ('name', 'Experiment'),
    ('source_config', 'Source Config'),
    ('raw_source_backend', 'Raw Backend'),
    ('completed_step', 'Completed Step'),
    ('samples_per_second', 'Completed SPS'),
    ('loader_wait_fraction', 'Completed Wait'),
    ('late_window_wait_fraction', 'Late Wait'),
    ('raw_read_fraction', 'Raw Read'),
    ('gate_samples_per_second', 'Sustained SPS'),
    ('gate_loader_wait_fraction', 'Sustained Wait'),
    ('steady_gpu_ratio', 'GPU Ratio'),
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
        description='Compare the confirmed Step 6 control against a raw_pack backend variant.',
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Base control config.')
    parser.add_argument('--raw-pack-path', required=True, help='Packed raw-byte payload file.')
    parser.add_argument('--raw-pack-index-path', required=True, help='Packed raw-byte index JSON.')
    parser.add_argument('--python-bin', default=sys.executable, help='Python interpreter used for the wrapper.')
    parser.add_argument('--torchrun-bin', default=default_torchrun_bin(), help='Torchrun executable.')
    parser.add_argument('--cuda-visible-devices', default='0,1', help='CUDA_VISIBLE_DEVICES for the comparison runs.')
    parser.add_argument('--run-id', default='latest', help='Run id under the work/report roots.')
    parser.add_argument('--work-root', default='artifacts/tmp/step6_phase2_raw_pack', help='Root directory for generated configs and outputs.')
    parser.add_argument('--report-root', default='artifacts/reports/step6_phase2_raw_pack', help='Root directory for summary artifacts.')
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


def run_preflight(*, config_path: Path, python_bin: str, torchrun_bin: str, env: dict[str, str], log_path: Path) -> int:
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


def build_variant_config(*, source_config_path: Path, run_root: Path, experiment_name: str, backend: str, raw_pack_path: str, raw_pack_index_path: str) -> tuple[Path, Path, Path, dict]:
    _resolved_config_path, full_config = load_full_config(source_config_path)
    config_payload = deep_update(
        full_config,
        phase_runtime_overrides(run_root=run_root, experiment_name=experiment_name),
    )
    config_payload = deep_update(
        config_payload,
        raw_source_backend_overrides(
            backend=backend,
            raw_pack_path=raw_pack_path if backend == 'raw_pack' else '',
            raw_pack_index_path=raw_pack_index_path if backend == 'raw_pack' else '',
        ),
    )
    config_payload = deep_update(
        config_payload,
        {
            'bc': {
                'wandb': {
                    'enabled': False,
                    'group': 'step6_phase2_raw_pack',
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


def run_variant(*, source_config_path: Path, run_root: Path, python_bin: str, torchrun_bin: str, cuda_visible_devices: str, experiment_name: str, backend: str, raw_pack_path: str, raw_pack_index_path: str) -> dict:
    config_path, summary_path, metrics_jsonl_path, config_payload = build_variant_config(
        source_config_path=source_config_path,
        run_root=run_root,
        experiment_name=experiment_name,
        backend=backend,
        raw_pack_path=raw_pack_path,
        raw_pack_index_path=raw_pack_index_path,
    )
    env = subprocess_env(config_path)
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
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
    row = summarize_preflight_row(
        phase='phase2',
        name=experiment_name,
        knobs={
            'cpu_ready_batches': int(((config_payload.get('bc') or {}).get('dataset') or {}).get('cpu_ready_batches', 0) or 0),
            'cpu_ready_bytes_gib': float(((config_payload.get('bc') or {}).get('dataset') or {}).get('cpu_ready_bytes_gib', 0.0) or 0.0),
        },
        stage_summary=None,
        preflight_summary=preflight_summary,
        backend='raw_threaded',
    )
    row['raw_source_backend'] = backend
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
    source_config_path = resolve_path(args.config)
    run_root = resolve_path(Path(args.work_root) / args.run_id)
    report_root = resolve_path(Path(args.report_root) / args.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    rows = [
        run_variant(
            source_config_path=source_config_path,
            run_root=run_root,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            experiment_name='control_files',
            backend='files',
            raw_pack_path='',
            raw_pack_index_path='',
        ),
        run_variant(
            source_config_path=source_config_path,
            run_root=run_root,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            experiment_name='candidate_raw_pack',
            backend='raw_pack',
            raw_pack_path=args.raw_pack_path,
            raw_pack_index_path=args.raw_pack_index_path,
        ),
    ]

    control_row = rows[0]
    candidate_row = rows[1]
    candidate_row['promoted'] = bool(phase2_candidate_beats_control(control=control_row, candidate=candidate_row))
    control_row['promoted'] = False

    markdown = render_markdown_table(rows, columns=DEFAULT_COLUMNS)
    report_payload = {
        'created_at': utc_now_iso(),
        'run_root': str(run_root),
        'rows': rows,
        'control': control_row,
        'candidate': candidate_row,
        'columns': DEFAULT_COLUMNS,
        'markdown_table': markdown,
    }
    save_json(report_root / 'summary.json', report_payload)
    (report_root / 'summary.md').write_text(markdown, encoding='utf-8')


if __name__ == '__main__':
    main()
