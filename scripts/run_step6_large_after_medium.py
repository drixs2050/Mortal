#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import (  # noqa: E402
    build_train_command,
    launcher_payload,
    subprocess_env,
    validate_torch_visible_launch_gpus,
    utc_now_iso,
)


OOM_PATTERNS = (
    'out of memory',
    'cuda out of memory',
    'cuda error: out of memory',
    'cudnn_status_alloc_failed',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Wait for the Step 6 medium campaign to finish, run the dual-A100 '
            '8192-per-GPU preflight, then launch either the high-batch or fallback '
            'large campaign depending on whether the preflight OOMs.'
        ),
    )
    parser.add_argument(
        '--wait-config',
        default='configs/step6_bc_medium_full8dan.toml',
        help='Completed campaign required before the large DDP gate proceeds.',
    )
    parser.add_argument(
        '--preflight-config',
        default='configs/step6_bc_large_preflight_full8dan_r5.toml',
        help='Short raw-baseline 8192-per-GPU preflight config.',
    )
    parser.add_argument(
        '--high-batch-config',
        default='configs/step6_bc_large_bounded_full8dan_8192_r5.toml',
        help='Current raw-baseline large campaign to use if the preflight succeeds.',
    )
    parser.add_argument(
        '--fallback-config',
        default='configs/step6_bc_large_bounded_full8dan_r5.toml',
        help='Current raw-baseline fallback campaign to use if the preflight OOMs.',
    )
    parser.add_argument(
        '--cuda-visible-devices',
        default='0,1',
        help='Physical GPUs to expose to the dual-A100 runs.',
    )
    parser.add_argument(
        '--poll-seconds',
        type=int,
        default=60,
        help='Polling interval while waiting for the medium campaign to finish.',
    )
    parser.add_argument(
        '--max-wait-seconds',
        type=int,
        default=0,
        help='Optional wait timeout; 0 means wait indefinitely.',
    )
    parser.add_argument(
        '--torchrun-bin',
        default='torchrun',
        help='Torchrun executable to use for the preflight train command.',
    )
    parser.add_argument(
        '--python-bin',
        default=sys.executable,
        help='Python executable to use for the selected large campaign.',
    )
    return parser.parse_args()


def write_gate_summary(summary_path: Path, payload: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def process_list_contains_config(ps_output: str, config_basename: str) -> bool:
    for line in ps_output.splitlines():
        if 'scripts/run_bc_campaign.py' in line and config_basename in line:
            return True
    return False


def active_campaign_for_config(config_basename: str) -> bool:
    result = subprocess.run(
        ['ps', '-eo', 'cmd'],
        capture_output=True,
        text=True,
        check=True,
    )
    return process_list_contains_config(result.stdout, config_basename)


def load_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def log_indicates_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding='utf-8', errors='replace').lower()
    return any(pattern in text for pattern in OOM_PATTERNS)


def run_logged(*, command: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        log_file.write('COMMAND: ' + ' '.join(command) + '\n')
        log_file.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def wait_for_completed_campaign(
    *,
    config_path: Path,
    summary_path: Path,
    poll_seconds: int,
    max_wait_seconds: int,
) -> dict:
    started_at = time.monotonic()
    while active_campaign_for_config(config_path.name):
        if max_wait_seconds > 0 and (time.monotonic() - started_at) >= max_wait_seconds:
            raise TimeoutError(
                f'timed out waiting for {config_path.name} to finish after {max_wait_seconds} seconds'
            )
        time.sleep(poll_seconds)

    if not summary_path.exists():
        raise FileNotFoundError(
            f'campaign finished running but no summary JSON was found for {config_path.name}: {summary_path}'
        )

    summary = load_json(summary_path)
    if summary.get('status') != 'completed':
        raise RuntimeError(
            f'{config_path.name} finished with status={summary.get("status")!r}; '
            'refusing to auto-launch the large DDP gate'
        )
    return summary


def large_campaign_command(*, config_path: Path, python_bin: str, torchrun_bin: str) -> list[str]:
    return [
        python_bin,
        'scripts/run_bc_campaign.py',
        '--config',
        os.fspath(config_path),
        '--python-bin',
        python_bin,
        '--torchrun-bin',
        torchrun_bin,
    ]


def main():
    args = parse_args()

    wait_config_path, _, wait_launch_settings, _ = launcher_payload(args.wait_config)
    preflight_config_path, _, preflight_launch_settings, _ = launcher_payload(args.preflight_config)
    high_batch_config_path, _, _, _ = launcher_payload(args.high_batch_config)
    fallback_config_path, _, _, _ = launcher_payload(args.fallback_config)

    summary_path = ROOT / 'artifacts' / 'reports' / 'step6_large_gate_summary.json'
    log_dir = ROOT / 'artifacts' / 'logs' / 'step6'
    preflight_log = log_dir / 'step6_large_preflight_full8dan.log'
    high_batch_log = log_dir / 'step6_large_bounded_full8dan_8192.log'
    fallback_log = log_dir / 'step6_large_bounded_full8dan_fallback4096.log'

    summary = {
        'started_at': utc_now_iso(),
        'status': 'waiting_for_medium',
        'wait_config': os.fspath(wait_config_path),
        'wait_summary_path': os.fspath(wait_launch_settings['campaign_summary_json']),
        'preflight_config': os.fspath(preflight_config_path),
        'high_batch_config': os.fspath(high_batch_config_path),
        'fallback_config': os.fspath(fallback_config_path),
        'cuda_visible_devices': args.cuda_visible_devices,
        'logs': {
            'preflight': os.fspath(preflight_log),
            'high_batch': os.fspath(high_batch_log),
            'fallback': os.fspath(fallback_log),
        },
    }
    write_gate_summary(summary_path, summary)

    medium_summary = wait_for_completed_campaign(
        config_path=wait_config_path,
        summary_path=wait_launch_settings['campaign_summary_json'],
        poll_seconds=max(args.poll_seconds, 1),
        max_wait_seconds=max(args.max_wait_seconds, 0),
    )
    summary['status'] = 'running_preflight'
    summary['medium_summary'] = medium_summary
    summary['medium_finished_at'] = utc_now_iso()
    write_gate_summary(summary_path, summary)

    preflight_env = subprocess_env(preflight_config_path)
    preflight_env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    validate_torch_visible_launch_gpus(
        env=preflight_env,
        python_bin=args.python_bin,
        expected_count=preflight_launch_settings['nproc_per_node'],
        required_name_substring='A100',
    )
    preflight_command = build_train_command(
        config_path=preflight_config_path,
        launch_settings=preflight_launch_settings,
        torchrun_bin=args.torchrun_bin,
    )
    preflight_return_code = run_logged(
        command=preflight_command,
        env=preflight_env,
        log_path=preflight_log,
    )
    preflight_oom = log_indicates_oom(preflight_log)

    summary['preflight'] = {
        'config_path': os.fspath(preflight_config_path),
        'command': preflight_command,
        'return_code': preflight_return_code,
        'oom_detected': preflight_oom,
        'finished_at': utc_now_iso(),
    }

    if preflight_return_code == 0:
        selected_config_path = high_batch_config_path
        selected_log_path = high_batch_log
        summary['selection_reason'] = 'preflight_completed_without_oom'
    elif preflight_oom:
        selected_config_path = fallback_config_path
        selected_log_path = fallback_log
        summary['selection_reason'] = 'preflight_oom_fallback_to_4096_per_gpu'
    else:
        summary['status'] = 'failed'
        summary['failed_stage'] = 'preflight'
        summary['finished_at'] = utc_now_iso()
        write_gate_summary(summary_path, summary)
        raise SystemExit(preflight_return_code or 1)

    summary['status'] = 'running_large_campaign'
    summary['selected_config'] = os.fspath(selected_config_path)
    write_gate_summary(summary_path, summary)

    selected_env = os.environ.copy()
    selected_env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    validate_torch_visible_launch_gpus(
        env=selected_env,
        python_bin=args.python_bin,
        expected_count=2,
        required_name_substring='A100',
    )
    campaign_command = large_campaign_command(
        config_path=selected_config_path,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
    )
    campaign_return_code = run_logged(
        command=campaign_command,
        env=selected_env,
        log_path=selected_log_path,
    )

    summary['large_campaign'] = {
        'config_path': os.fspath(selected_config_path),
        'command': campaign_command,
        'return_code': campaign_return_code,
        'finished_at': utc_now_iso(),
    }
    summary['status'] = 'completed' if campaign_return_code == 0 else 'failed'
    if campaign_return_code != 0:
        summary['failed_stage'] = 'large_campaign'
    summary['finished_at'] = utc_now_iso()
    write_gate_summary(summary_path, summary)

    if campaign_return_code != 0:
        raise SystemExit(campaign_return_code)


if __name__ == '__main__':
    main()
