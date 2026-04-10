#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'
os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import (  # noqa: E402
    build_stage_command,
    load_full_config,
    subprocess_env,
    utc_now_iso,
    write_summary,
)
from step6_experiments import (  # noqa: E402
    PHASE_ORDER,
    candidate_improves,
    checkpoint_best_perf,
    deep_update,
    latest_train_live_metrics,
    load_json,
    loader_knob_overrides,
    phase_in_window,
    phase_runtime_overrides,
    prepare_subset_artifacts,
    render_markdown_table,
    row_meets_loader_decision_gate,
    row_meets_loader_gate,
    should_run_device_prefetch_comparison,
    should_run_shard_size_comparison,
    should_run_thread_comparison,
    summarize_preflight_row,
    validate_phase_window,
    write_config,
)


DEFAULT_COLUMNS_PHASE_A = [
    ('phase', 'Phase'),
    ('name', 'Experiment'),
    ('backend', 'Backend'),
    ('completed_step', 'Completed Step'),
    ('target_chunk_gib', 'Chunk GiB'),
    ('decode_threads', 'Decode Threads'),
    ('max_inflight_chunk_builders', 'Inflight Builders'),
    ('raw_lru_budget_gib', 'Raw LRU GiB'),
    ('device_prefetch_batches', 'Device Prefetch'),
    ('startup_seconds', 'Startup Sec'),
    ('samples_per_second', 'Samples/Sec'),
    ('loader_wait_fraction', 'Wait'),
    ('steady_gpu_ratio', 'GPU Ratio'),
    ('ready_bytes_gib', 'Ready GiB'),
    ('inflight_bytes_gib', 'Inflight GiB'),
    ('peak_combined_rss_gib', 'Peak RSS GiB'),
    ('peak_single_rss_gib', 'Peak Single RSS GiB'),
    ('file_count', 'Files'),
    ('gate_passed', 'Gate'),
    ('fail_reasons', 'Reasons'),
]

DEFAULT_COLUMNS_PHASE_BD = [
    ('phase', 'Phase'),
    ('name', 'Experiment'),
    ('status', 'Status'),
    ('samples_per_second', 'Samples/Sec'),
    ('steps_per_second', 'Steps/Sec'),
    ('best_accuracy', 'Best Top1'),
    ('best_nll', 'Best NLL'),
    ('resume_passed', 'Resume'),
    ('startup_seconds', 'Startup Sec'),
    ('peak_combined_rss_gib', 'Peak RSS GiB'),
    ('notes', 'Notes'),
]


def default_torchrun_bin() -> str:
    sibling = Path(sys.executable).with_name('torchrun')
    if sibling.exists():
        return str(sibling)
    return shutil.which('torchrun') or 'torchrun'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the Step 6 staged-loader experiment ladder as a gated sequence.',
    )
    parser.add_argument(
        '--preflight-config',
        default='configs/step6_bc_large_preflight_full8dan_r4.toml',
        help='Base preflight config for loader tuning and final staged preflights.',
    )
    parser.add_argument(
        '--large-config',
        default='configs/step6_bc_large_bounded_full8dan_8192_r4.toml',
        help='Base large bounded DDP config.',
    )
    parser.add_argument(
        '--width-config',
        default='configs/step6_bc_large_model_probe_width.toml',
        help='Base width-probe config.',
    )
    parser.add_argument(
        '--depth-config',
        default='configs/step6_bc_large_model_probe_depth.toml',
        help='Base depth-probe config.',
    )
    parser.add_argument(
        '--full-9dan-config',
        default='configs/step6_bc_full_9dan.toml',
        help='Base full 9dan config.',
    )
    parser.add_argument(
        '--python-bin',
        default=sys.executable,
        help='Python interpreter for stage/preflight/campaign commands.',
    )
    parser.add_argument(
        '--torchrun-bin',
        default=default_torchrun_bin(),
        help='Torchrun executable for DDP launches.',
    )
    parser.add_argument(
        '--cuda-visible-devices',
        default='0,1',
        help='Canonical visible devices for the dual-A100 pair.',
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=10_000,
        help='Representative train subset size for Phase A.',
    )
    parser.add_argument(
        '--phase-a-max-stage-size-gib',
        type=float,
        default=64.0,
        help='Hard on-disk cap for each Phase A staged train cache. Prevents temporary subset caches from growing into hundreds of GiB.',
    )
    parser.add_argument(
        '--run-id',
        default='latest',
        help='Experiment run id under the work/report roots.',
    )
    parser.add_argument(
        '--work-root',
        default='artifacts/tmp/step6_experiment_ladder',
        help='Root directory for generated configs, logs, checkpoints, and temp artifacts.',
    )
    parser.add_argument(
        '--report-root',
        default='artifacts/reports/step6_experiment_ladder',
        help='Root directory for ladder summaries and comparison tables.',
    )
    parser.add_argument(
        '--start-at',
        choices=PHASE_ORDER,
        default='phase_a',
        help='Phase to start executing.',
    )
    parser.add_argument(
        '--stop-after',
        choices=PHASE_ORDER,
        default='phase_a',
        help='Final phase to execute in this invocation. Defaults to Phase A for safe loader tuning.',
    )
    parser.add_argument(
        '--allow-zarr',
        action='store_true',
        help='Allow the optional Zarr benchmark branch when the environment supports it.',
    )
    parser.add_argument(
        '--resume-checkpoint-step',
        type=int,
        default=100,
        help='Minimum saved step to interrupt at for the large-run resume proof.',
    )
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def command_to_string(command: list[str]) -> str:
    return shlex.join([str(part) for part in command])


def format_optional_float(value: object, *, digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def save_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_state(path: Path) -> dict:
    if path.exists():
        return load_json(path)
    return {
        'created_at': utc_now_iso(),
        'phases': {},
    }


def save_state(path: Path, state: dict) -> None:
    save_json(path, state)


def experiment_env(*, config_path: Path, cuda_visible_devices: str) -> dict[str, str]:
    env = subprocess_env(config_path)
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    return env


def run_logged_command(
    *,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    capture_output: bool = False,
    stream_output: bool = False,
    allow_nonzero: bool = False,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    started = time.perf_counter()
    if capture_output and stream_output:
        raise ValueError('capture_output and stream_output are mutually exclusive')
    if capture_output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        combined = []
        combined.append(f'$ {command_to_string(command)}\n')
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
    elif stream_output:
        collected = []
        with log_path.open('w', encoding='utf-8') as f:
            f.write(f'$ {command_to_string(command)}\n')
            f.flush()
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()
                collected.append(line)
            return_code = proc.wait()
        result = subprocess.CompletedProcess(
            args=command,
            returncode=return_code,
            stdout=''.join(collected),
            stderr='',
        )
    else:
        with log_path.open('w', encoding='utf-8') as f:
            f.write(f'$ {command_to_string(command)}\n')
            f.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                check=False,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
    finished_at = utc_now_iso()
    duration_seconds = time.perf_counter() - started
    if result.returncode != 0 and not allow_nonzero:
        raise subprocess.CalledProcessError(result.returncode, command)
    return {
        'command': command,
        'started_at': started_at,
        'finished_at': finished_at,
        'duration_seconds': duration_seconds,
        'return_code': result.returncode,
        'stdout': result.stdout if capture_output else '',
        'stderr': result.stderr if capture_output else '',
        'log_path': str(log_path),
    }


def run_stage_command(
    *,
    config_path: Path,
    full_config: dict,
    python_bin: str,
    env: dict[str, str],
    log_path: Path,
    splits: list[str],
    force: bool,
) -> tuple[dict, dict]:
    stage_summary_path = log_path.parent / 'stage_summary.json'
    command = build_stage_command(
        config_path=config_path,
        full_config=full_config,
        python_bin=python_bin,
        splits=splits,
        force=force,
    )
    if not command:
        return {}, {
            'command': [],
            'started_at': utc_now_iso(),
            'finished_at': utc_now_iso(),
            'duration_seconds': 0.0,
            'return_code': 0,
            'stdout': '',
            'stderr': '',
            'log_path': str(log_path),
        }
    command.extend(['--output-json', str(stage_summary_path)])
    print(f'[step6] staging {log_path.parent.name}: {command_to_string(command)}', flush=True)
    result = run_logged_command(
        command=command,
        env=env,
        log_path=log_path,
        stream_output=True,
        allow_nonzero=False,
    )
    payload = load_json(stage_summary_path)
    return payload, result


def preflight_summary_path(full_config: dict) -> Path:
    summary_json = str((((full_config.get('bc') or {}).get('preflight') or {}).get('summary_json') or '')).strip()
    if not summary_json:
        raise ValueError('expected bc.preflight.summary_json in preflight config')
    return resolve_path(summary_json)


def metrics_jsonl_path(full_config: dict) -> Path:
    metrics_jsonl = str((((full_config.get('bc') or {}).get('control') or {}).get('metrics_jsonl') or '')).strip()
    if not metrics_jsonl:
        raise ValueError('expected bc.control.metrics_jsonl in experiment config')
    return resolve_path(metrics_jsonl)


def state_checkpoint_path(full_config: dict) -> Path:
    state_file = str((((full_config.get('bc') or {}).get('control') or {}).get('state_file') or '')).strip()
    if not state_file:
        raise ValueError('expected bc.control.state_file in experiment config')
    return resolve_path(state_file)


def best_checkpoint_path(full_config: dict) -> Path:
    state_file = str((((full_config.get('bc') or {}).get('control') or {}).get('best_state_file') or '')).strip()
    if not state_file:
        raise ValueError('expected bc.control.best_state_file in experiment config')
    return resolve_path(state_file)


def campaign_summary_path(full_config: dict) -> Path:
    summary_json = str((((full_config.get('bc') or {}).get('launch') or {}).get('campaign_summary_json') or '')).strip()
    if not summary_json:
        raise ValueError('expected bc.launch.campaign_summary_json in experiment config')
    return resolve_path(summary_json)


def final_eval_report_paths(full_config: dict) -> dict[str, Path]:
    launch_cfg = ((full_config.get('bc') or {}).get('launch') or {})
    return {
        'val': resolve_path(str(launch_cfg.get('final_val_json') or '').strip()),
        'test': resolve_path(str(launch_cfg.get('final_test_json') or '').strip()),
    }


def checkpoint_current_step(checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        return 0
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return int(payload.get('current_step', 0))


def build_variant_config(
    *,
    base_config_path: str | Path,
    run_root: Path,
    experiment_name: str,
    overrides: dict,
) -> tuple[Path, dict]:
    _, base_config = load_full_config(base_config_path)
    merged = deep_update(base_config, overrides)
    merged = deep_update(merged, phase_runtime_overrides(run_root=run_root, experiment_name=experiment_name))
    config_path = run_root / experiment_name / 'config.toml'
    write_config(config_path, merged)
    return config_path, merged


def dataset_list_overrides(subset_metadata: dict) -> dict:
    return {
        'bc': {
            'dataset': {
                'train_list': subset_metadata['train_list'],
                'val_list': subset_metadata['val_list'],
                'test_list': subset_metadata['test_list'],
                'path_cache': subset_metadata['path_cache'],
            },
        },
    }


def extract_resnet_overrides(base_config_path: str | Path) -> dict:
    _, full_config = load_full_config(base_config_path)
    return {
        'resnet': dict(full_config.get('resnet') or {}),
    }


def campaign_row(
    *,
    phase: str,
    name: str,
    knobs: dict,
    full_config: dict,
    summary: dict,
    notes: str = '',
    resume_passed: bool = False,
    startup_seconds: float | None = None,
    preflight_summary: dict | None = None,
) -> dict:
    metrics = latest_train_live_metrics(metrics_jsonl_path(full_config))
    best_perf = checkpoint_best_perf(best_checkpoint_path(full_config))
    if best_perf is None:
        best_perf = checkpoint_best_perf(state_checkpoint_path(full_config))
    row = {
        'phase': phase,
        'name': name,
        'backend': knobs.get('backend', ''),
        'target_chunk_gib': knobs.get('target_chunk_gib'),
        'decode_threads': knobs.get('decode_threads'),
        'max_inflight_chunk_builders': knobs.get('max_inflight_chunk_builders'),
        'device_prefetch_batches': knobs.get('device_prefetch_batches'),
        'status': summary.get('status', ''),
        'samples_per_second': (metrics or {}).get('samples_per_second'),
        'steps_per_second': (metrics or {}).get('steps_per_second'),
        'best_accuracy': (best_perf or {}).get('accuracy'),
        'best_nll': (best_perf or {}).get('nll'),
        'resume_passed': resume_passed,
        'startup_seconds': startup_seconds,
        'peak_combined_rss_gib': (
            float((preflight_summary or {}).get('max_combined_train_worker_rss_kib', 0)) / (1024 ** 2)
            if preflight_summary else None
        ),
        'notes': notes,
        'summary_path': str(campaign_summary_path(full_config)),
    }
    return row


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def terminate_process_group(proc: subprocess.Popen, *, signal_type: int, grace_seconds: float = 10.0) -> int:
    try:
        os.killpg(proc.pid, signal_type)
    except ProcessLookupError:
        return proc.wait()
    deadline = time.time() + max(grace_seconds, 0.0)
    while time.time() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            return return_code
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + max(grace_seconds, 0.0)
    while time.time() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            return return_code
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return proc.wait()


def run_preflight_experiment(
    *,
    config_path: Path,
    full_config: dict,
    experiment_name: str,
    knobs: dict,
    python_bin: str,
    torchrun_bin: str,
    cuda_visible_devices: str,
    log_dir: Path,
) -> dict:
    env = experiment_env(config_path=config_path, cuda_visible_devices=cuda_visible_devices)
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
    print(f'[step6] preflight {experiment_name}: {command_to_string(command)}', flush=True)
    result = run_logged_command(
        command=command,
        env=env,
        log_path=log_dir / 'preflight.log',
        stream_output=True,
        allow_nonzero=True,
    )
    summary_path = preflight_summary_path(full_config)
    if not summary_path.exists():
        raise RuntimeError(
            f'preflight summary was not written for {experiment_name}; '
            f'return_code={result["return_code"]} log_path={log_dir / "preflight.log"}'
        )
    summary = load_json(summary_path)
    row = summarize_preflight_row(
        phase='phase_a',
        name=experiment_name,
        knobs=knobs,
        stage_summary=None,
        preflight_summary=summary,
    )
    row['preflight_return_code'] = int(result['return_code'])
    row['preflight_log_path'] = str(log_dir / 'preflight.log')
    row['config_path'] = str(config_path)
    row['config_sha256'] = file_sha256(config_path)
    return {
        'row': row,
        'preflight_summary': summary,
        'preflight_result': result,
        'config_path': str(config_path),
    }


def run_campaign_experiment(
    *,
    config_path: Path,
    full_config: dict,
    experiment_name: str,
    knobs: dict,
    python_bin: str,
    torchrun_bin: str,
    cuda_visible_devices: str,
    log_dir: Path,
) -> dict:
    env = experiment_env(config_path=config_path, cuda_visible_devices=cuda_visible_devices)
    command = [
        python_bin,
        'scripts/run_bc_campaign.py',
        '--config',
        str(config_path),
        '--python-bin',
        python_bin,
        '--torchrun-bin',
        torchrun_bin,
    ]
    print(f'[step6] campaign {experiment_name}: {command_to_string(command)}', flush=True)
    result = run_logged_command(
        command=command,
        env=env,
        log_path=log_dir / 'campaign.log',
        stream_output=True,
        allow_nonzero=True,
    )
    summary = load_json(campaign_summary_path(full_config))
    row = campaign_row(
        phase='phase_b',
        name=experiment_name,
        knobs=knobs,
        full_config=full_config,
        summary=summary,
        notes='',
    )
    row['campaign_return_code'] = int(result['return_code'])
    row['campaign_log_path'] = str(log_dir / 'campaign.log')
    row['config_path'] = str(config_path)
    row['final_val_json'] = str(final_eval_report_paths(full_config)['val'])
    row['final_test_json'] = str(final_eval_report_paths(full_config)['test'])
    return {
        'row': row,
        'summary': summary,
        'result': result,
        'config_path': str(config_path),
    }


def run_campaign_with_resume_probe(
    *,
    config_path: Path,
    full_config: dict,
    experiment_name: str,
    knobs: dict,
    python_bin: str,
    torchrun_bin: str,
    cuda_visible_devices: str,
    log_dir: Path,
    interrupt_at_step: int,
) -> dict:
    env = experiment_env(config_path=config_path, cuda_visible_devices=cuda_visible_devices)
    command = [
        python_bin,
        'scripts/run_bc_campaign.py',
        '--config',
        str(config_path),
        '--python-bin',
        python_bin,
        '--torchrun-bin',
        torchrun_bin,
    ]
    print(
        f'[step6] resume probe {experiment_name}: start run and interrupt after checkpoint step >= {interrupt_at_step}',
        flush=True,
    )
    first_log_path = log_dir / 'campaign_interrupt.log'
    first_log_path.parent.mkdir(parents=True, exist_ok=True)
    with first_log_path.open('w', encoding='utf-8') as f:
        f.write(f'$ {command_to_string(command)}\n')
        f.flush()
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        interrupted_step = 0
        timeout_deadline = time.time() + 1800.0
        try:
            while proc.poll() is None:
                current_step = checkpoint_current_step(state_checkpoint_path(full_config))
                if current_step >= interrupt_at_step:
                    interrupted_step = current_step
                    break
                if time.time() > timeout_deadline:
                    raise RuntimeError(
                        f'resume probe failed to reach checkpoint step {interrupt_at_step} within 1800 seconds'
                    )
                time.sleep(5.0)
        finally:
            if proc.poll() is None:
                terminate_process_group(proc, signal_type=signal.SIGINT)

    interrupted_summary_path = campaign_summary_path(full_config)
    interrupted_summary_snapshot = log_dir / 'campaign_interrupted_summary.json'
    copy_if_exists(interrupted_summary_path, interrupted_summary_snapshot)

    resumed = run_campaign_experiment(
        config_path=config_path,
        full_config=full_config,
        experiment_name=experiment_name,
        knobs=knobs,
        python_bin=python_bin,
        torchrun_bin=torchrun_bin,
        cuda_visible_devices=cuda_visible_devices,
        log_dir=log_dir,
    )
    resumed_step = checkpoint_current_step(state_checkpoint_path(full_config))
    resume_passed = interrupted_step > 0 and resumed_step > interrupted_step and resumed['summary'].get('status') == 'completed'
    resumed['row']['resume_passed'] = resume_passed
    resumed['row']['notes'] = (
        f'interrupted_at_step={interrupted_step}, resumed_to_step={resumed_step}'
    )
    resumed['resume_proof'] = {
        'passed': resume_passed,
        'interrupted_step': interrupted_step,
        'resumed_step': resumed_step,
        'interrupted_summary_path': str(interrupted_summary_snapshot),
    }
    return resumed


def write_phase_report(
    *,
    report_dir: Path,
    phase_name: str,
    rows: list[dict],
    columns: list[tuple[str, str]],
    extra_payload: dict | None = None,
) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown_table(rows, columns=columns)
    markdown_path = report_dir / f'{phase_name}.md'
    markdown_path.write_text(markdown, encoding='utf-8')
    payload = {
        'phase': phase_name,
        'row_count': len(rows),
        'rows': rows,
        'markdown_path': str(markdown_path),
    }
    if extra_payload:
        payload.update(extra_payload)
    json_path = report_dir / f'{phase_name}.json'
    save_json(json_path, payload)
    return {
        'json_path': str(json_path),
        'markdown_path': str(markdown_path),
    }


def persist_phase_progress(
    *,
    state: dict,
    state_path: Path,
    report_dir: Path,
    phase_name: str,
    rows: list[dict],
    experiments: list[dict],
    winner: dict,
    columns: list[tuple[str, str]],
    extra_payload: dict | None = None,
) -> dict:
    payload = {
        'winner': winner,
        'rows': rows,
        'experiments': experiments,
    }
    if extra_payload:
        payload.update(extra_payload)
    report_paths = write_phase_report(
        report_dir=report_dir,
        phase_name=phase_name,
        rows=rows,
        columns=columns,
        extra_payload=payload,
    )
    payload['reports'] = report_paths
    state['phases'][phase_name] = payload
    save_state(state_path, state)
    return payload


def find_phase_a_experiment_checkpoint(*, phase_state: dict | None, experiment_name: str) -> tuple[dict | None, dict | None]:
    phase_state = phase_state or {}
    row = next(
        (
            candidate
            for candidate in (phase_state.get('rows') or [])
            if candidate.get('name') == experiment_name
        ),
        None,
    )
    experiment = next(
        (
            candidate
            for candidate in (phase_state.get('experiments') or [])
            if candidate.get('name') == experiment_name
        ),
        None,
    )
    return row, experiment


def load_phase_a_preflight_checkpoint(
    *,
    phase_state: dict | None,
    experiment_name: str,
    config_path: Path,
    full_config: dict,
) -> dict | None:
    row, experiment = find_phase_a_experiment_checkpoint(
        phase_state=phase_state,
        experiment_name=experiment_name,
    )
    if row is None or experiment is None:
        return None
    if str(row.get('config_path', '')) != str(config_path):
        return None
    if not config_path.exists():
        return None
    expected_config_sha256 = str(row.get('config_sha256', '') or '').strip()
    if not expected_config_sha256:
        return None
    if expected_config_sha256 != file_sha256(config_path):
        return None
    summary_path = preflight_summary_path(full_config)
    if not summary_path.exists():
        return None
    preflight_summary = load_json(summary_path)
    recorded_return_code = int(row.get('preflight_return_code', 0) or 0)
    has_meaningful_measurement = (
        row.get('startup_seconds') is not None
        or row.get('samples_per_second') is not None
        or bool(preflight_summary.get('startup'))
        or int(preflight_summary.get('metrics_event_count', 0) or 0) > 0
    )
    if recorded_return_code != 0 and not has_meaningful_measurement:
        return None
    knobs = dict(experiment.get('knobs') or {})
    refreshed_row = summarize_preflight_row(
        phase='phase_a',
        name=experiment_name,
        knobs=knobs,
        stage_summary=None,
        preflight_summary=preflight_summary,
    )
    refreshed_row['preflight_return_code'] = int(preflight_summary.get('return_code', row.get('preflight_return_code', 0)) or 0)
    refreshed_row['preflight_log_path'] = row.get('preflight_log_path', str(config_path.parent / 'preflight.log'))
    refreshed_row['config_path'] = str(config_path)
    refreshed_row['config_sha256'] = expected_config_sha256
    print(f'[step6] reusing checkpointed preflight {experiment_name}', flush=True)
    return {
        'row': refreshed_row,
        'preflight_summary': preflight_summary,
        'preflight_result': None,
        'config_path': str(config_path),
        'reused': True,
    }


def runtime_cache_knobs(
    *,
    target_chunk_gib: float,
    decode_threads: int,
    max_inflight_chunk_builders: int,
    raw_lru_budget_gib: int,
    device_prefetch_batches: int,
    node_ram_budget_gib: int = 160,
    node_pinned_budget_gib: int = 8,
    node_inflight_budget_gib: int = 8,
    low_watermark: float = 0.65,
    high_watermark: float = 0.85,
    max_chunk_gib: float | None = None,
    startup_ready_chunks: int = 2,
    producer_threads: int = 1,
    min_files_per_chunk: int = 16,
    max_files_per_chunk: int = 96,
    eval_node_ram_budget_gib: int = 16,
    eval_target_chunk_gib: float = 1.0,
    eval_decode_threads: int = 2,
) -> dict:
    return {
        'backend': 'prepared_ram',
        'target_chunk_gib': float(target_chunk_gib),
        'max_chunk_gib': float(max_chunk_gib if max_chunk_gib is not None else max(float(target_chunk_gib) * 2.0, float(target_chunk_gib))),
        'decode_threads': int(decode_threads),
        'max_inflight_chunk_builders': int(max_inflight_chunk_builders),
        'raw_lru_budget_gib': int(raw_lru_budget_gib),
        'device_prefetch_batches': int(device_prefetch_batches),
        'node_ram_budget_gib': int(node_ram_budget_gib),
        'node_pinned_budget_gib': int(node_pinned_budget_gib),
        'node_inflight_budget_gib': int(node_inflight_budget_gib),
        'low_watermark': float(low_watermark),
        'high_watermark': float(high_watermark),
        'startup_ready_chunks': int(startup_ready_chunks),
        'producer_threads': int(producer_threads),
        'min_files_per_chunk': int(min_files_per_chunk),
        'max_files_per_chunk': int(max_files_per_chunk),
        'eval_node_ram_budget_gib': int(eval_node_ram_budget_gib),
        'eval_target_chunk_gib': float(eval_target_chunk_gib),
        'eval_decode_threads': int(eval_decode_threads),
    }


def visible_device_count(cuda_visible_devices: str | None) -> int:
    entries = [entry.strip() for entry in str(cuda_visible_devices or '').split(',') if entry.strip()]
    return max(len(entries), 1)


def inflight_builder_budget_supported(*, knobs: dict, world_size: int) -> bool:
    target_chunk_gib = float(knobs.get('target_chunk_gib') or 0.0)
    builder_count = max(int(knobs.get('max_inflight_chunk_builders', 1) or 1), 1)
    node_inflight_budget_gib = float(knobs.get('node_inflight_budget_gib') or 0.0)
    if target_chunk_gib <= 0.0:
        return True
    per_rank_inflight_budget_gib = node_inflight_budget_gib / max(int(world_size), 1)
    required_inflight_budget_gib = target_chunk_gib * builder_count
    return per_rank_inflight_budget_gib + 1e-9 >= required_inflight_budget_gib


def phase_a_overrides(
    *,
    subset_metadata: dict,
    knobs: dict,
) -> dict:
    return deep_update(
        dataset_list_overrides(subset_metadata),
        loader_knob_overrides(knobs=knobs, cache_root=None, required_splits=None),
    )


def choose_thread_candidate(current: dict, candidate_8: dict, candidate_2: dict | None) -> dict:
    if candidate_improves(
        baseline=current,
        candidate=candidate_8,
        min_relative_gain=0.03,
        max_rss_growth_ratio=0.05,
        max_startup_growth_seconds=2.0,
    ):
        return candidate_8
    if candidate_2 is None:
        return current
    if not row_meets_loader_gate(candidate_8):
        if row_meets_loader_gate(candidate_2):
            candidate_2_wait = float(candidate_2.get('loader_wait_fraction') or 1.0)
            current_wait = float(current.get('loader_wait_fraction') or 1.0)
            candidate_2_gpu = float(candidate_2.get('steady_gpu_ratio') or 0.0)
            current_gpu = float(current.get('steady_gpu_ratio') or 0.0)
            if candidate_2_wait < current_wait or candidate_2_gpu > current_gpu:
                return candidate_2
    return current


def candidate_beats_for_prefetch(*, baseline: dict, candidate: dict) -> bool:
    if not row_meets_loader_decision_gate(candidate):
        return False
    baseline_rss = float(baseline.get('peak_combined_rss_gib') or 0.0)
    candidate_rss = float(candidate.get('peak_combined_rss_gib') or 0.0)
    if baseline_rss > 0 and candidate_rss > baseline_rss * 1.05:
        return False
    baseline_startup = float(baseline.get('startup_seconds') or 0.0)
    candidate_startup = float(candidate.get('startup_seconds') or 0.0)
    if baseline_startup > 0 and candidate_startup > baseline_startup + 2.0:
        return False
    if candidate_improves(
        baseline=baseline,
        candidate=candidate,
        min_relative_gain=0.03,
        max_rss_growth_ratio=0.05,
        max_startup_growth_seconds=2.0,
    ):
        return True
    baseline_gpu = float(baseline.get('steady_gpu_ratio') or 0.0)
    candidate_gpu = float(candidate.get('steady_gpu_ratio') or 0.0)
    return candidate_gpu >= baseline_gpu + 0.05


def candidate_better_for_phase_a(*, baseline: dict, candidate: dict) -> bool:
    baseline_return = int(baseline.get('preflight_return_code', 0) or 0)
    candidate_return = int(candidate.get('preflight_return_code', 0) or 0)
    if baseline_return == 0 and candidate_return != 0:
        return False
    if candidate_return == 0 and baseline_return != 0:
        return True
    baseline_gate = bool(baseline.get('gate_passed'))
    candidate_gate = bool(candidate.get('gate_passed'))
    if candidate_gate != baseline_gate:
        return candidate_gate

    baseline_wait = float(baseline.get('loader_wait_fraction') or 1.0)
    candidate_wait = float(candidate.get('loader_wait_fraction') or 1.0)
    if abs(candidate_wait - baseline_wait) > 1e-6:
        return candidate_wait < baseline_wait

    baseline_sps = float(baseline.get('samples_per_second') or 0.0)
    candidate_sps = float(candidate.get('samples_per_second') or 0.0)
    if abs(candidate_sps - baseline_sps) > 1e-6:
        return candidate_sps > baseline_sps

    baseline_rss = float(baseline.get('peak_combined_rss_gib') or float('inf'))
    candidate_rss = float(candidate.get('peak_combined_rss_gib') or float('inf'))
    if abs(candidate_rss - baseline_rss) > 1e-6:
        return candidate_rss < baseline_rss

    baseline_startup = float(baseline.get('startup_seconds') or float('inf'))
    candidate_startup = float(candidate.get('startup_seconds') or float('inf'))
    return candidate_startup < baseline_startup


def build_phase_a_variant(
    *,
    base_config_path: str | Path,
    run_root: Path,
    subset_metadata: dict,
    experiment_name: str,
    knobs: dict,
) -> tuple[Path, dict]:
    overrides = phase_a_overrides(
        subset_metadata=subset_metadata,
        knobs=knobs,
    )
    return build_variant_config(
        base_config_path=base_config_path,
        run_root=run_root,
        experiment_name=experiment_name,
        overrides=overrides,
    )


def base_loader_knobs() -> dict:
    return runtime_cache_knobs(
        target_chunk_gib=2.0,
        decode_threads=4,
        max_inflight_chunk_builders=2,
        raw_lru_budget_gib=0,
        device_prefetch_batches=2,
        startup_ready_chunks=4,
    )


def cleanup_phase_a_cache_dirs(*, phase_cache_root: Path, winner_stage_summary: dict | None = None) -> list[str]:
    if not phase_cache_root.exists():
        return []
    removed = []
    for child in sorted(phase_cache_root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            removed.append(str(child))
    return removed


def phase_a(
    *,
    args,
    state: dict,
    state_path: Path,
    work_root: Path,
    report_root: Path,
) -> dict:
    phase_root = work_root / 'phase_a'
    report_dir = report_root / 'phase_a'
    runs_root = phase_root / 'runs'
    subset_dir = phase_root / 'subset'

    subset_metadata = prepare_subset_artifacts(
        config_path=args.preflight_config,
        output_dir=subset_dir,
        subset_size=args.subset_size,
    )
    existing_phase_state = state.get('phases', {}).get('phase_a')

    rows = []
    experiments = []
    winner = None

    def persist() -> dict:
        assert winner is not None
        return persist_phase_progress(
            state=state,
            state_path=state_path,
            report_dir=report_dir,
            phase_name='phase_a',
            rows=rows,
            experiments=experiments,
            winner=winner,
            columns=DEFAULT_COLUMNS_PHASE_A,
            extra_payload={
                'subset_metadata': subset_metadata,
            },
        )

    phase_a_world_size = visible_device_count(args.cuda_visible_devices)
    baseline_knobs = base_loader_knobs()
    baseline_name = 'phase_a_subset_ram_c2_d4_i2_sr4_dp2'
    baseline_config_path, baseline_full_config = build_phase_a_variant(
        base_config_path=args.preflight_config,
        run_root=runs_root,
        subset_metadata=subset_metadata,
        experiment_name=baseline_name,
        knobs=baseline_knobs,
    )
    baseline = load_phase_a_preflight_checkpoint(
        phase_state=existing_phase_state,
        experiment_name=baseline_name,
        config_path=baseline_config_path,
        full_config=baseline_full_config,
    ) or run_preflight_experiment(
        config_path=baseline_config_path,
        full_config=baseline_full_config,
        experiment_name=baseline_name,
        knobs=baseline_knobs,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        cuda_visible_devices=args.cuda_visible_devices,
        log_dir=runs_root / baseline_name,
    )
    rows.append(baseline['row'])
    experiments.append({
        'name': baseline_name,
        'knobs': baseline_knobs,
        'config_path': str(baseline_config_path),
    })
    winner = {
        'row': baseline['row'],
        'knobs': baseline_knobs,
        'config_path': str(baseline_config_path),
    }
    persist()

    if (
        float(winner['row'].get('loader_wait_fraction') or 1.0) > 0.20
        or float(winner['row'].get('ready_bytes_gib') or 0.0) < 1.0
    ):
        inflight_knobs = dict(winner['knobs'])
        inflight_knobs['max_inflight_chunk_builders'] = 4
        if inflight_builder_budget_supported(knobs=inflight_knobs, world_size=phase_a_world_size):
            inflight_name = f"{Path(winner['config_path']).parent.name}_i4".replace('_i2', '')
            inflight_config_path, inflight_full_config = build_phase_a_variant(
                base_config_path=args.preflight_config,
                run_root=runs_root,
                subset_metadata=subset_metadata,
                experiment_name=inflight_name,
                knobs=inflight_knobs,
            )
            inflight_candidate = load_phase_a_preflight_checkpoint(
                phase_state=existing_phase_state,
                experiment_name=inflight_name,
                config_path=inflight_config_path,
                full_config=inflight_full_config,
            ) or run_preflight_experiment(
                config_path=inflight_config_path,
                full_config=inflight_full_config,
                experiment_name=inflight_name,
                knobs=inflight_knobs,
                python_bin=args.python_bin,
                torchrun_bin=args.torchrun_bin,
                cuda_visible_devices=args.cuda_visible_devices,
                log_dir=runs_root / inflight_name,
            )
            rows.append(inflight_candidate['row'])
            experiments.append({
                'name': inflight_name,
                'knobs': inflight_knobs,
                'config_path': str(inflight_config_path),
            })
            if candidate_better_for_phase_a(baseline=winner['row'], candidate=inflight_candidate['row']):
                winner = {
                    'row': inflight_candidate['row'],
                    'knobs': inflight_knobs,
                    'config_path': str(inflight_config_path),
                }
            persist()
        else:
            per_rank_inflight_budget_gib = float(inflight_knobs.get('node_inflight_budget_gib') or 0.0) / max(phase_a_world_size, 1)
            required_inflight_budget_gib = float(inflight_knobs.get('target_chunk_gib') or 0.0) * max(
                int(inflight_knobs.get('max_inflight_chunk_builders', 1) or 1),
                1,
            )
            print(
                '[step6] skipping inflight-builder comparison: '
                f'per-rank inflight budget {per_rank_inflight_budget_gib:.2f} GiB '
                f'cannot support {int(inflight_knobs["max_inflight_chunk_builders"])} '
                f'builders at {float(inflight_knobs["target_chunk_gib"]):.2f} GiB '
                f'(requires {required_inflight_budget_gib:.2f} GiB)',
                flush=True,
            )

    if (
        float(winner['row'].get('ready_bytes_gib') or 0.0) >= 1.0
        and float(winner['row'].get('loader_wait_fraction') or 1.0) > 0.15
    ):
        shard_knobs = dict(winner['knobs'])
        shard_knobs['target_chunk_gib'] = 4.0
        shard_knobs['max_chunk_gib'] = 8.0
        shard_name = f"{Path(winner['config_path']).parent.name}_c4".replace('_c2', '')
        shard_config_path, shard_full_config = build_phase_a_variant(
            base_config_path=args.preflight_config,
            run_root=runs_root,
            subset_metadata=subset_metadata,
            experiment_name=shard_name,
            knobs=shard_knobs,
        )
        shard_candidate = load_phase_a_preflight_checkpoint(
            phase_state=existing_phase_state,
            experiment_name=shard_name,
            config_path=shard_config_path,
            full_config=shard_full_config,
        ) or run_preflight_experiment(
            config_path=shard_config_path,
            full_config=shard_full_config,
            experiment_name=shard_name,
            knobs=shard_knobs,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            log_dir=runs_root / shard_name,
        )
        rows.append(shard_candidate['row'])
        experiments.append({
            'name': shard_name,
            'knobs': shard_knobs,
            'config_path': str(shard_config_path),
        })
        if candidate_better_for_phase_a(baseline=winner['row'], candidate=shard_candidate['row']):
            winner = {
                'row': shard_candidate['row'],
                'knobs': shard_knobs,
                'config_path': str(shard_config_path),
            }
        persist()

    if float(winner['row'].get('chunk_parse_fraction') or 0.0) > 0.60:
        thread8_knobs = dict(winner['knobs'])
        thread8_knobs['decode_threads'] = 8
        thread8_name = f"{Path(winner['config_path']).parent.name}_d8".replace('_d4', '')
        thread8_config_path, thread8_full_config = build_phase_a_variant(
            base_config_path=args.preflight_config,
            run_root=runs_root,
            subset_metadata=subset_metadata,
            experiment_name=thread8_name,
            knobs=thread8_knobs,
        )
        thread8 = load_phase_a_preflight_checkpoint(
            phase_state=existing_phase_state,
            experiment_name=thread8_name,
            config_path=thread8_config_path,
            full_config=thread8_full_config,
        ) or run_preflight_experiment(
            config_path=thread8_config_path,
            full_config=thread8_full_config,
            experiment_name=thread8_name,
            knobs=thread8_knobs,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            log_dir=runs_root / thread8_name,
        )
        rows.append(thread8['row'])
        experiments.append({
            'name': thread8_name,
            'knobs': thread8_knobs,
            'config_path': str(thread8_config_path),
        })
        if candidate_better_for_phase_a(baseline=winner['row'], candidate=thread8['row']):
            winner = {
                'row': thread8['row'],
                'knobs': thread8_knobs,
                'config_path': str(thread8_config_path),
            }
        persist()

    if (
        float(winner['row'].get('loader_wait_fraction') or 1.0) > 0.20
        and float(winner['row'].get('ready_bytes_gib') or 0.0) < 1.0
    ):
        startup_knobs = dict(winner['knobs'])
        startup_knobs['startup_ready_chunks'] = 8
        startup_name = f"{Path(winner['config_path']).parent.name}_sr8".replace('_sr4', '')
        startup_config_path, startup_full_config = build_phase_a_variant(
            base_config_path=args.preflight_config,
            run_root=runs_root,
            subset_metadata=subset_metadata,
            experiment_name=startup_name,
            knobs=startup_knobs,
        )
        startup_candidate = load_phase_a_preflight_checkpoint(
            phase_state=existing_phase_state,
            experiment_name=startup_name,
            config_path=startup_config_path,
            full_config=startup_full_config,
        ) or run_preflight_experiment(
            config_path=startup_config_path,
            full_config=startup_full_config,
            experiment_name=startup_name,
            knobs=startup_knobs,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            log_dir=runs_root / startup_name,
        )
        rows.append(startup_candidate['row'])
        experiments.append({
            'name': startup_name,
            'knobs': startup_knobs,
            'config_path': str(startup_config_path),
        })
        if candidate_better_for_phase_a(baseline=winner['row'], candidate=startup_candidate['row']):
            winner = {
                'row': startup_candidate['row'],
                'knobs': startup_knobs,
                'config_path': str(startup_config_path),
            }
        persist()

    if should_run_device_prefetch_comparison(winner['row']):
        device_knobs = dict(winner['knobs'])
        device_knobs['device_prefetch_batches'] = 3
        device_name = f"{Path(winner['config_path']).parent.name}_dp3".replace('_dp2', '')
        device_config_path, device_full_config = build_phase_a_variant(
            base_config_path=args.preflight_config,
            run_root=runs_root,
            subset_metadata=subset_metadata,
            experiment_name=device_name,
            knobs=device_knobs,
        )
        device_candidate = load_phase_a_preflight_checkpoint(
            phase_state=existing_phase_state,
            experiment_name=device_name,
            config_path=device_config_path,
            full_config=device_full_config,
        ) or run_preflight_experiment(
            config_path=device_config_path,
            full_config=device_full_config,
            experiment_name=device_name,
            knobs=device_knobs,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            log_dir=runs_root / device_name,
        )
        rows.append(device_candidate['row'])
        experiments.append({
            'name': device_name,
            'knobs': device_knobs,
            'config_path': str(device_config_path),
        })
        if candidate_better_for_phase_a(baseline=winner['row'], candidate=device_candidate['row']):
            winner = {
                'row': device_candidate['row'],
                'knobs': device_knobs,
                'config_path': str(device_config_path),
            }
        persist()

    if float(winner['row'].get('chunk_read_fraction') or 0.0) + float(winner['row'].get('chunk_decompress_fraction') or 0.0) > 0.15:
        raw_lru_knobs = dict(winner['knobs'])
        raw_lru_knobs['raw_lru_budget_gib'] = 4
        raw_lru_name = f"{Path(winner['config_path']).parent.name}_raw4"
        raw_lru_config_path, raw_lru_full_config = build_phase_a_variant(
            base_config_path=args.preflight_config,
            run_root=runs_root,
            subset_metadata=subset_metadata,
            experiment_name=raw_lru_name,
            knobs=raw_lru_knobs,
        )
        raw_lru_candidate = load_phase_a_preflight_checkpoint(
            phase_state=existing_phase_state,
            experiment_name=raw_lru_name,
            config_path=raw_lru_config_path,
            full_config=raw_lru_full_config,
        ) or run_preflight_experiment(
            config_path=raw_lru_config_path,
            full_config=raw_lru_full_config,
            experiment_name=raw_lru_name,
            knobs=raw_lru_knobs,
            python_bin=args.python_bin,
            torchrun_bin=args.torchrun_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            log_dir=runs_root / raw_lru_name,
        )
        rows.append(raw_lru_candidate['row'])
        experiments.append({
            'name': raw_lru_name,
            'knobs': raw_lru_knobs,
            'config_path': str(raw_lru_config_path),
        })
        if candidate_better_for_phase_a(baseline=winner['row'], candidate=raw_lru_candidate['row']):
            winner = {
                'row': raw_lru_candidate['row'],
                'knobs': raw_lru_knobs,
                'config_path': str(raw_lru_config_path),
            }
        persist()

    phase_payload = persist()
    return phase_payload


def phase_b(
    *,
    args,
    state: dict,
    work_root: Path,
    report_root: Path,
) -> dict:
    phase_a_state = state['phases'].get('phase_a')
    if not phase_a_state:
        raise ValueError('phase_b requires an existing phase_a winner')
    winner_knobs = dict(phase_a_state['winner']['knobs'])

    phase_root = work_root / 'phase_b'
    report_dir = report_root / 'phase_b'
    runs_root = phase_root / 'runs'
    rows = []

    smoke_name = 'phase_b_smoke_full8dan'
    smoke_overrides = deep_update(
        loader_knob_overrides(
            knobs=winner_knobs,
            cache_root=None,
            required_splits=None,
        ),
        {
            'bc': {
                'control': {
                    'batch_size': 2048,
                    'val_batch_size': 4096,
                    'best_eval_batch_size': 4096,
                    'eval_batch_size': 4096,
                    'max_steps': 50,
                    'max_runtime_seconds': 600,
                    'train_log_every': 1,
                    'save_every': 25,
                    'val_steps': 4,
                    'best_eval_every': 50,
                    'best_eval_max_batches': 4,
                    'eval_max_batches': 4,
                },
                'launch': {
                    'nproc_per_node': 1,
                    'eval_device': 'cuda:0',
                },
            },
        },
    )
    smoke_config_path, smoke_config = build_variant_config(
        base_config_path=args.large_config,
        run_root=runs_root,
        experiment_name=smoke_name,
        overrides=smoke_overrides,
    )
    smoke = run_campaign_experiment(
        config_path=smoke_config_path,
        full_config=smoke_config,
        experiment_name=smoke_name,
        knobs=winner_knobs,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        cuda_visible_devices='0',
        log_dir=runs_root / smoke_name,
    )
    smoke['row']['phase'] = 'phase_b'
    rows.append(smoke['row'])
    if smoke['summary'].get('status') != 'completed':
        raise RuntimeError('phase_b smoke campaign failed before the large rehearsal')

    full_preflight_name = 'phase_b_full_preflight_full8dan'
    full_preflight_overrides = loader_knob_overrides(
        knobs=winner_knobs,
        cache_root=None,
        required_splits=None,
    )
    full_preflight_config_path, full_preflight_config = build_variant_config(
        base_config_path=args.preflight_config,
        run_root=runs_root,
        experiment_name=full_preflight_name,
        overrides=full_preflight_overrides,
    )
    full_preflight = run_preflight_experiment(
        config_path=full_preflight_config_path,
        full_config=full_preflight_config,
        experiment_name=full_preflight_name,
        knobs=winner_knobs,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        cuda_visible_devices=args.cuda_visible_devices,
        log_dir=runs_root / full_preflight_name,
    )
    full_preflight['row']['phase'] = 'phase_b'
    rows.append(full_preflight['row'])
    if not row_meets_loader_gate(full_preflight['row']):
        raise RuntimeError('phase_b full-corpus staged preflight did not clear the loader gate')

    large_name = 'phase_b_large_bounded_full8dan'
    large_overrides = deep_update(
        loader_knob_overrides(
            knobs=winner_knobs,
            cache_root=None,
            required_splits=None,
        ),
        {
            'bc': {
                'control': {
                    'train_log_every': 1,
                    'save_every': 100,
                    'best_eval_every': 400,
                    'max_runtime_seconds': 7200,
                },
            },
        },
    )
    large_config_path, large_config = build_variant_config(
        base_config_path=args.large_config,
        run_root=runs_root,
        experiment_name=large_name,
        overrides=large_overrides,
    )
    large = run_campaign_with_resume_probe(
        config_path=large_config_path,
        full_config=large_config,
        experiment_name=large_name,
        knobs=winner_knobs,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        cuda_visible_devices=args.cuda_visible_devices,
        log_dir=runs_root / large_name,
        interrupt_at_step=max(int(args.resume_checkpoint_step), 100),
    )
    large['row']['phase'] = 'phase_b'
    rows.append(large['row'])
    if large['summary'].get('status') != 'completed':
        raise RuntimeError('phase_b large bounded rehearsal did not complete successfully')
    if not large['resume_proof']['passed']:
        raise RuntimeError('phase_b large bounded rehearsal failed the resume proof')

    report_paths = write_phase_report(
        report_dir=report_dir,
        phase_name='phase_b',
        rows=rows,
        columns=DEFAULT_COLUMNS_PHASE_BD,
        extra_payload={
            'winner_knobs': winner_knobs,
            'smoke': smoke,
            'full_preflight': full_preflight,
            'large': large,
        },
    )
    phase_payload = {
        'winner_knobs': winner_knobs,
        'smoke': {
            'config_path': str(smoke_config_path),
            'summary': smoke['summary'],
            'row': smoke['row'],
        },
        'full_preflight': {
            'config_path': str(full_preflight_config_path),
            'summary': full_preflight['preflight_summary'],
            'row': full_preflight['row'],
        },
        'large': {
            'config_path': str(large_config_path),
            'summary': large['summary'],
            'row': large['row'],
            'resume_proof': large['resume_proof'],
        },
        'rows': rows,
        'reports': report_paths,
    }
    state['phases']['phase_b'] = phase_payload
    return phase_payload


def model_beats_baseline(*, baseline_row: dict, candidate_row: dict) -> bool:
    if candidate_row.get('status') != 'completed':
        return False
    baseline_acc = float(baseline_row.get('best_accuracy') or 0.0)
    candidate_acc = float(candidate_row.get('best_accuracy') or 0.0)
    baseline_sps = float(baseline_row.get('samples_per_second') or 0.0)
    candidate_sps = float(candidate_row.get('samples_per_second') or 0.0)
    if candidate_acc < baseline_acc + 0.003:
        return False
    if baseline_sps > 0 and candidate_sps < baseline_sps * 0.75:
        return False
    return True


def phase_c(
    *,
    args,
    state: dict,
    work_root: Path,
    report_root: Path,
) -> dict:
    phase_a_state = state['phases'].get('phase_a')
    phase_b_state = state['phases'].get('phase_b')
    if not phase_a_state or not phase_b_state:
        raise ValueError('phase_c requires completed phase_a and phase_b state')
    winner_knobs = dict(phase_a_state['winner']['knobs'])
    baseline_row = dict(phase_b_state['large']['row'])
    baseline_row['phase'] = 'phase_c'
    baseline_row['name'] = 'phase_c_baseline_large'

    phase_root = work_root / 'phase_c'
    report_dir = report_root / 'phase_c'
    runs_root = phase_root / 'runs'

    rows = [baseline_row]

    width_name = 'phase_c_width_probe_full8dan'
    width_overrides = deep_update(
        loader_knob_overrides(
            knobs=winner_knobs,
            cache_root=None,
            required_splits=None,
        ),
        {
            'bc': {
                'control': {
                    'train_log_every': 1,
                    'save_every': 100,
                    'best_eval_every': 400,
                    'max_runtime_seconds': 7200,
                },
            },
        },
    )
    width_config_path, width_config = build_variant_config(
        base_config_path=args.width_config,
        run_root=runs_root,
        experiment_name=width_name,
        overrides=width_overrides,
    )
    width = run_campaign_experiment(
        config_path=width_config_path,
        full_config=width_config,
        experiment_name=width_name,
        knobs=winner_knobs,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        cuda_visible_devices=args.cuda_visible_devices,
        log_dir=runs_root / width_name,
    )
    width['row']['phase'] = 'phase_c'
    rows.append(width['row'])

    depth_name = 'phase_c_depth_probe_full8dan'
    depth_overrides = deep_update(
        loader_knob_overrides(
            knobs=winner_knobs,
            cache_root=None,
            required_splits=None,
        ),
        {
            'bc': {
                'control': {
                    'train_log_every': 1,
                    'save_every': 100,
                    'best_eval_every': 400,
                    'max_runtime_seconds': 7200,
                },
            },
        },
    )
    depth_config_path, depth_config = build_variant_config(
        base_config_path=args.depth_config,
        run_root=runs_root,
        experiment_name=depth_name,
        overrides=depth_overrides,
    )
    depth = run_campaign_experiment(
        config_path=depth_config_path,
        full_config=depth_config,
        experiment_name=depth_name,
        knobs=winner_knobs,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        cuda_visible_devices=args.cuda_visible_devices,
        log_dir=runs_root / depth_name,
    )
    depth['row']['phase'] = 'phase_c'
    rows.append(depth['row'])

    selected_model = {
        'source': 'baseline',
        'config_path': phase_b_state['large']['config_path'],
        'row': baseline_row,
    }
    for candidate, source in ((width, 'width'), (depth, 'depth')):
        if model_beats_baseline(baseline_row=baseline_row, candidate_row=candidate['row']) and (
            selected_model['source'] == 'baseline'
            or float(candidate['row'].get('best_accuracy') or 0.0)
            > float(selected_model['row'].get('best_accuracy') or 0.0)
        ):
            selected_model = {
                'source': source,
                'config_path': candidate['config_path'],
                'row': candidate['row'],
            }

    report_paths = write_phase_report(
        report_dir=report_dir,
        phase_name='phase_c',
        rows=rows,
        columns=DEFAULT_COLUMNS_PHASE_BD,
        extra_payload={
            'winner_knobs': winner_knobs,
            'selected_model': selected_model,
            'width': width,
            'depth': depth,
        },
    )
    phase_payload = {
        'winner_knobs': winner_knobs,
        'selected_model': selected_model,
        'rows': rows,
        'width': {
            'config_path': str(width_config_path),
            'summary': width['summary'],
            'row': width['row'],
        },
        'depth': {
            'config_path': str(depth_config_path),
            'summary': depth['summary'],
            'row': depth['row'],
        },
        'reports': report_paths,
    }
    state['phases']['phase_c'] = phase_payload
    return phase_payload


def phase_d(
    *,
    args,
    state: dict,
    work_root: Path,
    report_root: Path,
) -> dict:
    phase_a_state = state['phases'].get('phase_a')
    phase_c_state = state['phases'].get('phase_c')
    if not phase_a_state or not phase_c_state:
        raise ValueError('phase_d requires completed phase_a and phase_c state')
    winner_knobs = dict(phase_a_state['winner']['knobs'])
    selected_model_path = phase_c_state['selected_model']['config_path']

    phase_root = work_root / 'phase_d'
    report_dir = report_root / 'phase_d'
    runs_root = phase_root / 'runs'

    resnet_overrides = extract_resnet_overrides(selected_model_path)

    _, full_9dan_dataset_source = load_full_config(args.full_9dan_config)
    dataset_source_cfg = ((full_9dan_dataset_source.get('bc') or {}).get('dataset') or {})
    dataset_override = {
        'bc': {
            'dataset': {
                key: dataset_source_cfg.get(key)
                for key in (
                    'root_dir',
                    'train_list',
                    'val_list',
                    'test_list',
                    'path_cache',
                    'actor_filter_index',
                    'step_count_summary',
                    'actor_filter_manifest',
                    'min_actor_dan',
                    'train_globs',
                    'val_globs',
                    'test_globs',
                    'oracle',
                    'trust_seed',
                    'always_include_kan_select',
                    'player_names_files',
                    'exclude_names_files',
                )
            },
        },
    }

    preflight_name = 'phase_d_9dan_preflight'
    preflight_overrides = deep_update(
        loader_knob_overrides(
            knobs=winner_knobs,
            cache_root=None,
            required_splits=None,
        ),
        resnet_overrides,
    )
    preflight_overrides = deep_update(preflight_overrides, dataset_override)
    preflight_config_path, preflight_config = build_variant_config(
        base_config_path=args.preflight_config,
        run_root=runs_root,
        experiment_name=preflight_name,
        overrides=preflight_overrides,
    )
    final_preflight = run_preflight_experiment(
        config_path=preflight_config_path,
        full_config=preflight_config,
        experiment_name=preflight_name,
        knobs=winner_knobs,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        cuda_visible_devices=args.cuda_visible_devices,
        log_dir=runs_root / preflight_name,
    )
    final_preflight['row']['phase'] = 'phase_d'
    if not row_meets_loader_gate(final_preflight['row']):
        raise RuntimeError('phase_d 9dan preflight did not clear the loader gate')

    frozen_overrides = deep_update(
        loader_knob_overrides(
            knobs=winner_knobs,
            cache_root=None,
            required_splits=None,
        ),
        resnet_overrides,
    )
    frozen_config_path, frozen_config = build_variant_config(
        base_config_path=args.full_9dan_config,
        run_root=runs_root,
        experiment_name='phase_d_step7_frozen',
        overrides=frozen_overrides,
    )
    canonical_config_path = resolve_path(args.full_9dan_config)
    write_config(canonical_config_path, frozen_config)

    rows = [final_preflight['row']]
    report_paths = write_phase_report(
        report_dir=report_dir,
        phase_name='phase_d',
        rows=rows,
        columns=DEFAULT_COLUMNS_PHASE_BD,
        extra_payload={
            'winner_knobs': winner_knobs,
            'selected_model': phase_c_state['selected_model'],
            'final_preflight': final_preflight,
            'frozen_config_path': str(frozen_config_path),
            'canonical_config_path': str(canonical_config_path),
            'step7_command': 'python scripts/run_bc_campaign.py --config configs/step6_bc_full_9dan.toml',
        },
    )
    phase_payload = {
        'winner_knobs': winner_knobs,
        'selected_model': phase_c_state['selected_model'],
        'final_preflight': {
            'config_path': str(preflight_config_path),
            'summary': final_preflight['preflight_summary'],
            'row': final_preflight['row'],
        },
        'frozen_config_path': str(frozen_config_path),
        'canonical_config_path': str(canonical_config_path),
        'step7_command': 'python scripts/run_bc_campaign.py --config configs/step6_bc_full_9dan.toml',
        'rows': rows,
        'reports': report_paths,
    }
    state['phases']['phase_d'] = phase_payload
    return phase_payload


def main():
    args = parse_args()
    validate_phase_window(start_at=args.start_at, stop_after=args.stop_after)

    work_root = resolve_path(args.work_root) / args.run_id
    report_root = resolve_path(args.report_root) / args.run_id
    work_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = report_root / 'ladder_state.json'
    state = load_state(state_path)
    state['args'] = {
        'preflight_config': str(args.preflight_config),
        'large_config': str(args.large_config),
        'width_config': str(args.width_config),
        'depth_config': str(args.depth_config),
        'full_9dan_config': str(args.full_9dan_config),
        'python_bin': str(args.python_bin),
        'torchrun_bin': str(args.torchrun_bin),
        'cuda_visible_devices': str(args.cuda_visible_devices),
        'subset_size': int(args.subset_size),
        'phase_a_max_stage_size_gib': float(args.phase_a_max_stage_size_gib),
        'run_id': str(args.run_id),
        'start_at': str(args.start_at),
        'stop_after': str(args.stop_after),
        'allow_zarr': bool(args.allow_zarr),
    }

    if phase_in_window(phase='phase_a', start_at=args.start_at, stop_after=args.stop_after):
        phase_a(args=args, state=state, state_path=state_path, work_root=work_root, report_root=report_root)
        save_state(state_path, state)
    if phase_in_window(phase='phase_b', start_at=args.start_at, stop_after=args.stop_after):
        phase_b(args=args, state=state, work_root=work_root, report_root=report_root)
        save_state(state_path, state)
    if phase_in_window(phase='phase_c', start_at=args.start_at, stop_after=args.stop_after):
        phase_c(args=args, state=state, work_root=work_root, report_root=report_root)
        save_state(state_path, state)
    if phase_in_window(phase='phase_d', start_at=args.start_at, stop_after=args.stop_after):
        phase_d(args=args, state=state, work_root=work_root, report_root=report_root)
        save_state(state_path, state)

    final_summary = {
        'run_id': args.run_id,
        'completed_at': utc_now_iso(),
        'state_path': str(state_path),
        'phases': sorted(state.get('phases', {}).keys()),
        'work_root': str(work_root),
        'report_root': str(report_root),
    }
    write_summary(report_root / 'summary.json', final_summary)

    if args.stop_after == 'phase_a' and 'phase_a' in state.get('phases', {}):
        winner = ((state.get('phases', {}).get('phase_a') or {}).get('winner') or {})
        winner_row = dict(winner.get('row') or {})
        report_path = (((state.get('phases', {}).get('phase_a') or {}).get('reports') or {}).get('markdown_path', ''))
        print(
            '[step6] phase_a complete: '
            f"winner={winner_row.get('name', 'n/a')} "
            f"samples_per_second={format_optional_float(winner_row.get('samples_per_second'))} "
            f"loader_wait_fraction={format_optional_float(winner_row.get('loader_wait_fraction'))} "
            f"startup_seconds={format_optional_float(winner_row.get('startup_seconds'))} "
            f"report={report_path or 'n/a'}",
            flush=True,
        )


if __name__ == '__main__':
    main()
