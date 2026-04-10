#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import (  # noqa: E402
    build_stage_command,
    build_train_command,
    expand_runtime_path,
    launcher_payload,
    subprocess_env,
    validate_torch_visible_launch_gpus,
    utc_now_iso,
    write_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a warmup-aware BC loader preflight and write a benchmark summary.',
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the Step 6 BC preflight TOML.',
    )
    parser.add_argument(
        '--torchrun-bin',
        default='torchrun',
        help='Torchrun executable to use for distributed training.',
    )
    parser.add_argument(
        '--python-bin',
        default=sys.executable,
        help='Python executable to use for staging commands and CUDA inventory checks.',
    )
    return parser.parse_args()


def parse_visible_gpu_indices(raw_value: str) -> list[int]:
    value = str(raw_value or '').strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def normalize_pci_bus_id(raw_value: str | int | None) -> str:
    value = str(raw_value or '').strip()
    if not value:
        return ''
    if ':' in value:
        parts = value.split(':')
        if len(parts) >= 2:
            try:
                return str(int(parts[-2], 16))
            except ValueError:
                return value.lower()
    try:
        return str(int(value, 16 if any(ch.isalpha() for ch in value.lower()) else 10))
    except ValueError:
        return value.lower()


def sample_gpu_state(*, target_gpu_indices: list[int], target_pci_bus_ids: list[str] | None = None) -> dict | None:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,pci.bus_id,name,utilization.gpu,power.draw,memory.used,memory.total',
                '--format=csv,noheader,nounits',
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    except Exception:
        return None

    rows = {}
    normalized_targets = {
        normalize_pci_bus_id(value)
        for value in (target_pci_bus_ids or [])
        if normalize_pci_bus_id(value)
    }
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 7:
            continue
        index = int(parts[0])
        pci_bus_id = parts[1]
        normalized_pci_bus_id = normalize_pci_bus_id(pci_bus_id)
        if normalized_targets:
            if normalized_pci_bus_id not in normalized_targets:
                continue
        elif target_gpu_indices and index not in target_gpu_indices:
            continue
        rows[index] = {
            'pci_bus_id': pci_bus_id,
            'normalized_pci_bus_id': normalized_pci_bus_id,
            'name': parts[2],
            'utilization_gpu': float(parts[3]),
            'power_draw_watts': float(parts[4]),
            'memory_used_mib': float(parts[5]),
            'memory_total_mib': float(parts[6]),
        }
    return rows


def list_train_worker_processes() -> list[dict]:
    try:
        result = subprocess.run(
            ['ps', '-eo', 'pid=,ppid=,rss=,cmd='],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    except Exception:
        return []

    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        rss_kib = int(parts[2])
        cmd = parts[3]
        if 'mortal/train_bc.py' not in cmd:
            continue
        rows.append({
            'pid': pid,
            'ppid': ppid,
            'rss_kib': rss_kib,
            'cmd': cmd,
        })
    return rows


def sample_train_worker_rss_kib() -> dict:
    workers = list_train_worker_processes()
    rss_values = [row['rss_kib'] for row in workers]

    return {
        'workers': workers,
        'worker_count': len(rss_values),
        'rss_sum_kib': sum(rss_values),
        'rss_max_kib': max(rss_values) if rss_values else 0,
    }


def kib_from_gib(raw_value: float | int | None) -> int:
    return max(int(float(raw_value or 0) * (1024 ** 2)), 0)


def terminate_process_group(proc: subprocess.Popen, *, grace_seconds: float = 5.0) -> int:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.wait()
    deadline = time.time() + max(float(grace_seconds), 0.0)
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


def load_metrics_events(metrics_jsonl_path: Path) -> list[dict]:
    if not metrics_jsonl_path.exists():
        return []
    events = []
    with metrics_jsonl_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def aggregate_window_metrics(events: list[dict]) -> dict | None:
    if not events:
        return None
    count = len(events)
    recent = events[-count:]
    last = recent[-1]
    return {
        'window_count': count,
        'samples_per_second': sum(
            float((event.get('runtime_metrics') or {}).get('samples_per_second', 0.0))
            for event in recent
        ) / count,
        'steps_per_second': sum(
            float((event.get('runtime_metrics') or {}).get('steps_per_second', 0.0))
            for event in recent
        ) / count,
        'wait_fraction': sum(
            float((event.get('loader_metrics') or {}).get('wait_fraction', 0.0))
            for event in recent
        ) / count,
        'cpu_pipe_wait_fraction': sum(
            float((event.get('loader_metrics') or {}).get('cpu_pipe_wait_fraction', 0.0))
            for event in recent
        ) / count,
        'cpu_pipe_wait_seconds': sum(
            float((event.get('loader_metrics') or {}).get('cpu_pipe_wait_seconds', 0.0))
            for event in recent
        ) / count,
        'cpu_ready_batches': sum(
            float((event.get('loader_metrics') or {}).get('cpu_ready_batches', 0.0))
            for event in recent
        ) / count,
        'cpu_ready_bytes_gib': sum(
            float((event.get('loader_metrics') or {}).get('cpu_ready_bytes_gib', 0.0))
            for event in recent
        ) / count,
        'cpu_producer_blocked_put_fraction': sum(
            float((event.get('loader_metrics') or {}).get('cpu_producer_blocked_put_fraction', 0.0))
            for event in recent
        ) / count,
        'cpu_producer_blocked_put_seconds': sum(
            float((event.get('loader_metrics') or {}).get('cpu_blocked_put_seconds', 0.0))
            for event in recent
        ) / count,
        'queued_bytes_gib': sum(
            float((event.get('loader_metrics') or {}).get('queued_bytes_gib', 0.0))
            for event in recent
        ) / count,
        'ready_bytes_gib': sum(
            float((event.get('loader_metrics') or {}).get('ready_bytes_gib', 0.0))
            for event in recent
        ) / count,
        'inflight_bytes_gib': sum(
            float((event.get('loader_metrics') or {}).get('inflight_bytes_gib', 0.0))
            for event in recent
        ) / count,
        'pinned_batch_bytes_gib': sum(
            float((event.get('loader_metrics') or {}).get('pinned_batch_bytes_gib', 0.0))
            for event in recent
        ) / count,
        'raw_lru_bytes_gib': sum(
            float((event.get('loader_metrics') or {}).get('raw_lru_bytes_gib', 0.0))
            for event in recent
        ) / count,
        'chunk_parse_fraction': sum(
            float((event.get('loader_metrics') or {}).get('chunk_parse_fraction', 0.0))
            for event in recent
        ) / count,
        'chunk_read_fraction': sum(
            float((event.get('loader_metrics') or {}).get('chunk_read_fraction', 0.0))
            for event in recent
        ) / count,
        'chunk_decompress_fraction': sum(
            float((event.get('loader_metrics') or {}).get('chunk_decompress_fraction', 0.0))
            for event in recent
        ) / count,
        'chunk_assemble_fraction': sum(
            float((event.get('loader_metrics') or {}).get('chunk_assemble_fraction', 0.0))
            for event in recent
        ) / count,
        'prefill_complete': all(
            bool((event.get('loader_metrics') or {}).get('prefill_complete', False))
            for event in recent
        ),
        'producer_blocked_reason': str((last.get('loader_metrics') or {}).get('producer_blocked_reason', '')),
        'step': int(last.get('step', 0) or 0),
        'runtime_seconds_total': float(
            last.get('runtime_seconds_total', (last.get('runtime_metrics') or {}).get('runtime_seconds_total', 0.0))
        ),
    }


def sustained_train_live_metrics(*, events: list[dict], min_runtime_seconds: float, required_windows: int) -> dict | None:
    post_warmup = [
        event
        for event in events
        if event.get('event') == 'train_live'
        and float(event.get('runtime_seconds_total', 0.0)) >= float(min_runtime_seconds)
    ]
    if len(post_warmup) < required_windows:
        return None
    return aggregate_window_metrics(post_warmup[-required_windows:])


def completed_save_window_metrics(*, events: list[dict], min_step: int) -> dict | None:
    selected = next(
        (
            event
            for event in events
            if event.get('event') == 'save_window' and int(event.get('step', 0) or 0) >= int(min_step)
        ),
        None,
    )
    if selected is None:
        selected = next(
            (event for event in reversed(events) if event.get('event') == 'save_window'),
            None,
        )
    if selected is None:
        return None
    return aggregate_window_metrics([selected])


def last_train_live_step(events: list[dict]) -> int:
    last = next(
        (event for event in reversed(events) if event.get('event') == 'train_live'),
        None,
    )
    return int((last or {}).get('step', 0) or 0)


def realized_completed_step(*, completed_metrics: dict | None, events: list[dict]) -> int:
    if completed_metrics is not None:
        return int(completed_metrics.get('step', 0) or 0)
    return last_train_live_step(events)


def steady_gpu_ratio(
    *,
    samples: list[dict],
    target_gpu_indices: list[int],
    target_pci_bus_ids: list[str] | None,
    min_runtime_seconds: float,
    min_power_watts: float,
    min_utilization: float,
) -> dict:
    post_warmup = [
        sample
        for sample in samples
        if float(sample.get('elapsed_seconds', 0.0)) >= float(min_runtime_seconds)
    ]
    if not post_warmup:
        return {
            'sample_count': 0,
            'matched_sample_count': 0,
            'passing_count': 0,
            'pass_ratio': 0.0,
        }

    matched_sample_count = 0
    passing_count = 0
    normalized_targets = {
        normalize_pci_bus_id(value)
        for value in (target_pci_bus_ids or [])
        if normalize_pci_bus_id(value)
    }
    for sample in post_warmup:
        gpus = sample.get('gpus', {})
        if normalized_targets:
            gpu_rows = [
                row
                for row in gpus.values()
                if normalize_pci_bus_id(row.get('normalized_pci_bus_id') or row.get('pci_bus_id')) in normalized_targets
            ]
        elif target_gpu_indices and all(index in gpus for index in target_gpu_indices):
            gpu_rows = [gpus[index] for index in target_gpu_indices]
        else:
            gpu_rows = list(gpus.values())
        if not gpu_rows:
            continue
        matched_sample_count += 1
        if gpu_rows and all(
            row['power_draw_watts'] >= min_power_watts or row['utilization_gpu'] >= min_utilization
            for row in gpu_rows
        ):
            passing_count += 1

    return {
        'sample_count': len(post_warmup),
        'matched_sample_count': matched_sample_count,
        'passing_count': passing_count,
        'pass_ratio': (passing_count / matched_sample_count) if matched_sample_count else 0.0,
    }


def summarize_preflight(
    *,
    config_path: str,
    config_fingerprint_value: str,
    started_at: str,
    finished_at: str,
    command: list[str],
    return_code: int,
    metrics_events: list[dict],
    gpu_samples: list[dict],
    target_gpu_indices: list[int],
    target_pci_bus_ids: list[str] | None,
    preflight_cfg: dict,
    max_worker_rss_kib: int,
    max_combined_worker_rss_kib: int,
    rss_guard_abort: dict | None = None,
) -> dict:
    min_runtime_seconds = float(preflight_cfg.get('min_runtime_seconds', 0) or 0)
    min_steps_before_stop = int(preflight_cfg.get('min_steps_before_stop', 200) or 0)
    required_windows = int(preflight_cfg.get('required_stable_windows', 2) or 2)
    min_samples_per_second = float(preflight_cfg.get('min_samples_per_second', 5900.0) or 5900.0)
    preferred_samples_per_second = float(preflight_cfg.get('preferred_samples_per_second', 7000.0) or 7000.0)
    max_wait_fraction = float(preflight_cfg.get('max_loader_wait_fraction', 0.15) or 0.15)
    min_power_watts = float(preflight_cfg.get('min_steady_gpu_watts', 150.0) or 150.0)
    min_utilization = float(preflight_cfg.get('min_steady_gpu_utilization', 60.0) or 60.0)
    min_gpu_ratio = float(preflight_cfg.get('min_steady_gpu_ratio', 0.70) or 0.70)

    sustained = sustained_train_live_metrics(
        events=metrics_events,
        min_runtime_seconds=min_runtime_seconds,
        required_windows=required_windows,
    )
    completed = completed_save_window_metrics(
        events=metrics_events,
        min_step=min_steps_before_stop,
    )
    completed_step = realized_completed_step(completed_metrics=completed, events=metrics_events)
    gpu_steady = steady_gpu_ratio(
        samples=gpu_samples,
        target_gpu_indices=target_gpu_indices,
        target_pci_bus_ids=target_pci_bus_ids,
        min_runtime_seconds=min_runtime_seconds,
        min_power_watts=min_power_watts,
        min_utilization=min_utilization,
    )
    last_save_event = next(
        (event for event in reversed(metrics_events) if event.get('event') == 'save_window'),
        None,
    )
    startup_event = next(
        (
            event for event in metrics_events
            if event.get('event') == 'loader_priming' and event.get('split') == 'train'
        ),
        None,
    )

    reasons = []
    if return_code != 0:
        reasons.append(f'train_return_code={return_code}')
    if min_steps_before_stop > 0 and completed_step < min_steps_before_stop:
        reasons.append(f'completed_step_below_floor={completed_step}<{min_steps_before_stop}')
    if sustained is None:
        reasons.append('insufficient_post_warmup_windows')
    else:
        if sustained['samples_per_second'] < min_samples_per_second:
            reasons.append(
                f"samples_per_second_below_gate={sustained['samples_per_second']:.1f}<{min_samples_per_second:.1f}"
            )
        if sustained['wait_fraction'] > max_wait_fraction:
            reasons.append(
                f"loader_wait_fraction_above_gate={sustained['wait_fraction']:.3f}>{max_wait_fraction:.3f}"
            )
    if gpu_steady['sample_count'] == 0:
        reasons.append('no_post_warmup_gpu_samples')
    elif gpu_steady['pass_ratio'] < min_gpu_ratio:
        reasons.append(
            f"steady_gpu_ratio_below_gate={gpu_steady['pass_ratio']:.3f}<{min_gpu_ratio:.3f}"
        )
    if rss_guard_abort is not None:
        reasons.append(rss_guard_abort['reason'])

    return {
        'config_path': config_path,
        'config_fingerprint': config_fingerprint_value,
        'started_at': started_at,
        'finished_at': finished_at,
        'status': 'passed' if not reasons else 'failed',
        'return_code': return_code,
        'command': command,
        'target_gpu_indices': target_gpu_indices,
        'preflight': {
            'min_runtime_seconds': min_runtime_seconds,
            'min_steps_before_stop': min_steps_before_stop,
            'required_stable_windows': required_windows,
            'stability_tolerance': float(preflight_cfg.get('stability_tolerance', 0.05) or 0.05),
            'min_samples_per_second': min_samples_per_second,
            'preferred_samples_per_second': preferred_samples_per_second,
            'max_loader_wait_fraction': max_wait_fraction,
            'min_steady_gpu_watts': min_power_watts,
            'min_steady_gpu_utilization': min_utilization,
            'min_steady_gpu_ratio': min_gpu_ratio,
        },
        'sustained_metrics': sustained,
        'completed_window_metrics': completed,
        'completed_step': completed_step,
        'last_train_live_step': last_train_live_step(metrics_events),
        'steady_gpu': gpu_steady,
        'last_save_event': last_save_event,
        'startup': startup_event,
        'metrics_event_count': len(metrics_events),
        'gpu_sample_count': len(gpu_samples),
        'max_train_worker_rss_kib': max_worker_rss_kib,
        'max_combined_train_worker_rss_kib': max_combined_worker_rss_kib,
        'rss_guard_abort': rss_guard_abort,
        'gate': {
            'passed': not reasons,
            'reasons': reasons,
        },
    }


def main():
    args = parse_args()
    config_path, full_config, launch_settings, config_fingerprint_value = launcher_payload(args.config)
    preflight_cfg = dict((full_config.get('bc') or {}).get('preflight') or {})
    if not preflight_cfg.get('enabled', False):
        raise ValueError('bc.preflight.enabled must be true for the loader preflight runner')

    control_cfg = (full_config.get('bc') or {}).get('control') or {}
    metrics_jsonl = str(control_cfg.get('metrics_jsonl', '') or '').strip()
    if not metrics_jsonl:
        raise ValueError('bc.control.metrics_jsonl is required for the loader preflight runner')

    summary_json = str(preflight_cfg.get('summary_json', '') or '').strip()
    if not summary_json:
        raise ValueError('bc.preflight.summary_json is required for the loader preflight runner')

    metrics_jsonl_path = expand_runtime_path(metrics_jsonl)
    summary_json_path = expand_runtime_path(summary_json)
    metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (metrics_jsonl_path, summary_json_path):
        if path.exists():
            path.unlink()

    env = subprocess_env(config_path)
    if bool(preflight_cfg.get('disable_wandb', True)):
        env['WANDB_DISABLED'] = 'true'
        env['WANDB_MODE'] = 'disabled'
    stale_workers = sample_train_worker_rss_kib()
    if stale_workers['worker_count'] > 0:
        stale_desc = ', '.join(
            f"pid={row['pid']} rss_gib={row['rss_kib'] / (1024 ** 2):.1f}"
            for row in stale_workers['workers']
        )
        raise RuntimeError(
            'refusing to start loader preflight while existing mortal/train_bc.py workers are still live: '
            f'{stale_desc}'
        )
    visible_gpu_inventory = validate_torch_visible_launch_gpus(
        env=env,
        python_bin=args.python_bin,
        expected_count=launch_settings['nproc_per_node'],
        required_name_substring='A100' if launch_settings['nproc_per_node'] > 1 else '',
    )
    stage_command = build_stage_command(
        config_path=config_path,
        full_config=full_config,
        python_bin=args.python_bin,
        splits=['train'],
    )
    if len(stage_command) > 4:
        subprocess.run(stage_command, check=True, cwd=ROOT, env=env)
    train_command = build_train_command(
        config_path=config_path,
        launch_settings=launch_settings,
        torchrun_bin=args.torchrun_bin,
    )
    target_gpu_indices = parse_visible_gpu_indices(env.get('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '')))
    target_pci_bus_ids = [
        str(row.get('pci_bus_id', '') or '')
        for row in visible_gpu_inventory.get('selected', [])
        if str(row.get('pci_bus_id', '') or '')
    ]
    sample_interval_seconds = float(preflight_cfg.get('gpu_sample_interval_seconds', 2.0) or 2.0)
    sample_interval_seconds = max(sample_interval_seconds, 0.5)
    combined_rss_limit_kib = kib_from_gib(preflight_cfg.get('max_combined_train_worker_rss_gib', 0))
    single_rss_limit_kib = kib_from_gib(preflight_cfg.get('max_single_train_worker_rss_gib', 0))

    started_at = utc_now_iso()
    started_clock = time.perf_counter()
    gpu_samples = []
    max_worker_rss_kib = 0
    max_combined_worker_rss_kib = 0
    rss_guard_abort = None

    proc = subprocess.Popen(
        train_command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        while proc.poll() is None:
            gpu_state = sample_gpu_state(
                target_gpu_indices=target_gpu_indices,
                target_pci_bus_ids=target_pci_bus_ids,
            )
            if gpu_state is not None:
                gpu_samples.append({
                    'elapsed_seconds': time.perf_counter() - started_clock,
                    'gpus': gpu_state,
                })
            rss_state = sample_train_worker_rss_kib()
            max_worker_rss_kib = max(max_worker_rss_kib, int(rss_state['rss_max_kib']))
            max_combined_worker_rss_kib = max(max_combined_worker_rss_kib, int(rss_state['rss_sum_kib']))
            if rss_guard_abort is None and combined_rss_limit_kib > 0 and rss_state['rss_sum_kib'] > combined_rss_limit_kib:
                rss_guard_abort = {
                    'reason': (
                        'combined_train_worker_rss_above_guard='
                        f"{rss_state['rss_sum_kib'] / (1024 ** 2):.1f}>"
                        f"{combined_rss_limit_kib / (1024 ** 2):.1f}GiB"
                    ),
                    'rss_sum_kib': int(rss_state['rss_sum_kib']),
                    'rss_max_kib': int(rss_state['rss_max_kib']),
                }
                return_code = terminate_process_group(proc)
                break
            if rss_guard_abort is None and single_rss_limit_kib > 0 and rss_state['rss_max_kib'] > single_rss_limit_kib:
                rss_guard_abort = {
                    'reason': (
                        'single_train_worker_rss_above_guard='
                        f"{rss_state['rss_max_kib'] / (1024 ** 2):.1f}>"
                        f"{single_rss_limit_kib / (1024 ** 2):.1f}GiB"
                    ),
                    'rss_sum_kib': int(rss_state['rss_sum_kib']),
                    'rss_max_kib': int(rss_state['rss_max_kib']),
                }
                return_code = terminate_process_group(proc)
                break
            time.sleep(sample_interval_seconds)
    finally:
        if proc.poll() is None:
            return_code = proc.wait()
        else:
            return_code = proc.returncode

    gpu_state = sample_gpu_state(
        target_gpu_indices=target_gpu_indices,
        target_pci_bus_ids=target_pci_bus_ids,
    )
    if gpu_state is not None:
        gpu_samples.append({
            'elapsed_seconds': time.perf_counter() - started_clock,
            'gpus': gpu_state,
        })

    metrics_events = load_metrics_events(metrics_jsonl_path)
    finished_at = utc_now_iso()
    summary = summarize_preflight(
        config_path=str(config_path),
        config_fingerprint_value=config_fingerprint_value,
        started_at=started_at,
        finished_at=finished_at,
        command=train_command,
        return_code=return_code,
        metrics_events=metrics_events,
        gpu_samples=gpu_samples,
        target_gpu_indices=target_gpu_indices,
        target_pci_bus_ids=target_pci_bus_ids,
        preflight_cfg=preflight_cfg,
        max_worker_rss_kib=max_worker_rss_kib,
        max_combined_worker_rss_kib=max_combined_worker_rss_kib,
        rss_guard_abort=rss_guard_abort,
    )
    summary['visible_cuda_devices'] = visible_gpu_inventory
    summary['stage_command'] = stage_command if len(stage_command) > 4 else []
    write_summary(summary_json_path, summary)

    if rss_guard_abort is not None:
        raise SystemExit(3)
    if return_code != 0:
        raise SystemExit(return_code)
    if not summary['gate']['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
