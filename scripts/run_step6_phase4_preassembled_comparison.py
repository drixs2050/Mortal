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
    phase4_candidate_beats_control,
    phase4_preassembled_overrides,
    phase_runtime_overrides,
    render_markdown_table,
    summarize_preflight_row,
    write_config,
)


DEFAULT_CONFIG = 'configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml'
DEFAULT_COLUMNS = [
    ('name', 'Experiment'),
    ('loader_mode', 'Loader Mode'),
    ('loader_block_target_samples', 'Block Target'),
    ('module_raw_to_cpu_sps', 'Module Producer SPS'),
    ('module_training_sps', 'Module Train SPS'),
    ('module_can_hide_producer', 'Module Hide Producer'),
    ('completed_step', 'Completed Step'),
    ('samples_per_second', 'Completed SPS'),
    ('loader_wait_fraction', 'Completed Wait'),
    ('late_window_wait_fraction', 'Late Wait'),
    ('collate_or_assemble_fraction', 'Collate Frac'),
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
        description=(
            'Run the Step 6 Phase 4 comparison: exact 192-file module benchmark first, '
            'then the real 200-step dual-A100 preflight for the control and preassembled-batch variants.'
        ),
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Confirmed Step 6 control config.')
    parser.add_argument('--python-bin', default=sys.executable, help='Python interpreter used for the wrapper.')
    parser.add_argument('--torchrun-bin', default=default_torchrun_bin(), help='Torchrun executable.')
    parser.add_argument('--module-cuda-visible-devices', default='0', help='CUDA_VISIBLE_DEVICES for the single-GPU 192-file module benchmark.')
    parser.add_argument('--preflight-cuda-visible-devices', default='0,1', help='CUDA_VISIBLE_DEVICES for the dual-GPU 200-step preflight.')
    parser.add_argument('--run-id', default='latest', help='Run id under the work/report roots.')
    parser.add_argument('--work-root', default='artifacts/tmp/step6_phase4_preassembled', help='Root directory for generated configs and outputs.')
    parser.add_argument('--report-root', default='artifacts/reports/step6_phase4_preassembled', help='Root directory for summary artifacts.')
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


def build_variant_config(
    *,
    source_config_path: Path,
    run_root: Path,
    experiment_name: str,
    loader_mode: str,
    loader_block_target_samples: int,
) -> tuple[Path, Path, Path, dict]:
    _resolved_config_path, full_config = load_full_config(source_config_path)
    config_payload = deep_update(
        full_config,
        phase_runtime_overrides(run_root=run_root, experiment_name=experiment_name),
    )
    if loader_mode == 'preassembled_batches':
        config_payload = deep_update(
            config_payload,
            phase4_preassembled_overrides(
                loader_block_target_samples=loader_block_target_samples,
            ),
        )
    config_payload = deep_update(
        config_payload,
        {
            'bc': {
                'wandb': {
                    'enabled': False,
                    'group': 'step6_phase4_preassembled',
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


def run_module_benchmark(
    *,
    config_path: Path,
    python_bin: str,
    cuda_visible_devices: str,
    output_json: Path,
    log_path: Path,
) -> tuple[int, dict]:
    command = [
        python_bin,
        'scripts/benchmark_bc_conversion_vs_training.py',
        '--config',
        str(config_path),
        '--split',
        'train',
        '--sample-size',
        '192',
        '--sample-strategy',
        'round_robin',
        '--output-json',
        str(output_json),
    ]
    env = subprocess_env(config_path)
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    return_code = run_command(command=command, env=env, log_path=log_path)
    if not output_json.exists():
        raise FileNotFoundError(f'module benchmark did not write {output_json}')
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


def run_variant(
    *,
    source_config_path: Path,
    run_root: Path,
    python_bin: str,
    torchrun_bin: str,
    module_cuda_visible_devices: str,
    preflight_cuda_visible_devices: str,
    experiment_name: str,
    loader_mode: str,
    loader_block_target_samples: int,
) -> dict:
    config_path, summary_path, metrics_jsonl_path, config_payload = build_variant_config(
        source_config_path=source_config_path,
        run_root=run_root,
        experiment_name=experiment_name,
        loader_mode=loader_mode,
        loader_block_target_samples=loader_block_target_samples,
    )

    module_output_json = run_root / 'module' / f'{experiment_name}.json'
    module_log_path = run_root / 'logs' / f'{experiment_name}.module.log'
    module_return_code, module_summary = run_module_benchmark(
        config_path=config_path,
        python_bin=python_bin,
        cuda_visible_devices=module_cuda_visible_devices,
        output_json=module_output_json,
        log_path=module_log_path,
    )

    preflight_log_path = run_root / 'logs' / f'{experiment_name}.preflight.log'
    wrapper_return_code = run_preflight(
        config_path=config_path,
        python_bin=python_bin,
        torchrun_bin=torchrun_bin,
        cuda_visible_devices=preflight_cuda_visible_devices,
        log_path=preflight_log_path,
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            f'preflight summary was not written for {experiment_name}: expected {summary_path}'
        )
    with summary_path.open(encoding='utf-8') as f:
        preflight_summary = json.load(f)

    row = summarize_preflight_row(
        phase='phase4',
        name=experiment_name,
        knobs={
            'loader_block_target_samples': int(loader_block_target_samples),
        },
        stage_summary=None,
        preflight_summary=preflight_summary,
        backend='raw_threaded',
    )
    row['loader_mode'] = loader_mode
    row['loader_block_target_samples'] = int(loader_block_target_samples)
    row['late_window_wait_fraction'] = average_wait_fraction_after_step(metrics_jsonl_path, min_step=125)
    row['wrapper_return_code'] = wrapper_return_code
    row['module_return_code'] = module_return_code
    row['module_raw_to_cpu_sps'] = float(((module_summary.get('raw_to_cpu_batches') or {}).get('samples_per_second') or 0.0))
    row['module_training_sps'] = float(((module_summary.get('training_on_produced_batches') or {}).get('samples_per_second') or 0.0))
    row['module_can_hide_producer'] = bool(((module_summary.get('comparison') or {}).get('can_hide_producer', False)))
    row['module_summary_path'] = str(module_output_json)
    row['module_log_path'] = str(module_log_path)
    row['source_config'] = str(source_config_path)
    row['config_path'] = str(config_path)
    row['summary_path'] = str(summary_path)
    row['metrics_jsonl_path'] = str(metrics_jsonl_path)
    row['preflight_log_path'] = str(preflight_log_path)
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
            module_cuda_visible_devices=args.module_cuda_visible_devices,
            preflight_cuda_visible_devices=args.preflight_cuda_visible_devices,
            experiment_name='control_b4',
            loader_mode='baseline',
            loader_block_target_samples=65536,
        ),
        run_variant(
            source_config_path=source_config_path,
            run_root=run_root,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            module_cuda_visible_devices=args.module_cuda_visible_devices,
            preflight_cuda_visible_devices=args.preflight_cuda_visible_devices,
            experiment_name='preassembled_32768',
            loader_mode='preassembled_batches',
            loader_block_target_samples=32768,
        ),
        run_variant(
            source_config_path=source_config_path,
            run_root=run_root,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            module_cuda_visible_devices=args.module_cuda_visible_devices,
            preflight_cuda_visible_devices=args.preflight_cuda_visible_devices,
            experiment_name='preassembled_65536',
            loader_mode='preassembled_batches',
            loader_block_target_samples=65536,
        ),
    ]

    control_row = rows[0]
    for row in rows[1:]:
        row['promoted'] = bool(phase4_candidate_beats_control(control=control_row, candidate=row))
    control_row['promoted'] = False

    markdown = render_markdown_table(rows, columns=DEFAULT_COLUMNS)
    payload = {
        'created_at': utc_now_iso(),
        'run_root': str(run_root),
        'rows': rows,
        'control': control_row,
        'columns': DEFAULT_COLUMNS,
        'markdown_table': markdown,
    }
    save_json(report_root / 'summary.json', payload)
    (report_root / 'summary.md').write_text(markdown, encoding='utf-8')


if __name__ == '__main__':
    main()
