from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import toml

from bc_ram_cache import runtime_cache_enabled
from bc_runtime import config_fingerprint


ROOT = Path(__file__).resolve().parents[1]
TORCH_VISIBLE_GPU_QUERY = """
import json
import torch

rows = []
count = torch.cuda.device_count() if torch.cuda.is_available() else 0
for i in range(count):
    props = torch.cuda.get_device_properties(i)
    rows.append({
        'cuda_index': i,
        'name': torch.cuda.get_device_name(i),
        'total_memory': int(getattr(props, 'total_memory', 0)),
        'pci_bus_id': str(getattr(props, 'pci_bus_id', '') or ''),
    })

print(json.dumps(rows))
""".strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def expand_config_path(config_path: str | os.PathLike[str]) -> Path:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def expand_runtime_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def required_runtime_path(*, raw_path: str, field_name: str) -> Path:
    value = str(raw_path or '').strip()
    if not value:
        raise ValueError(f'{field_name} is required')
    return expand_runtime_path(value)


def load_full_config(config_path: str | os.PathLike[str]) -> tuple[Path, dict]:
    resolved = expand_config_path(config_path)
    with resolved.open(encoding='utf-8') as f:
        return resolved, toml.load(f)


def resolve_launch_settings(full_config: dict) -> dict:
    bc_cfg = full_config.get('bc') or {}
    control_cfg = bc_cfg.get('control') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    launch_cfg = bc_cfg.get('launch') or {}

    nproc_per_node = int(launch_cfg.get('nproc_per_node', 1))
    master_port = int(launch_cfg.get('master_port', 29500))
    if nproc_per_node <= 0:
        raise ValueError('bc.launch.nproc_per_node must be positive')
    if master_port <= 0:
        raise ValueError('bc.launch.master_port must be positive')

    eval_device = str(launch_cfg.get('eval_device') or control_cfg.get('device', 'cpu'))
    final_val_json = str(launch_cfg.get('final_val_json') or '').strip()
    final_test_json = str(launch_cfg.get('final_test_json') or '').strip()
    campaign_summary_json = str(launch_cfg.get('campaign_summary_json') or '').strip()
    if not final_val_json:
        raise ValueError('bc.launch.final_val_json is required')
    if not final_test_json:
        raise ValueError('bc.launch.final_test_json is required')
    if not campaign_summary_json:
        raise ValueError('bc.launch.campaign_summary_json is required')

    return {
        'nproc_per_node': nproc_per_node,
        'master_port': master_port,
        'eval_device': eval_device,
        'final_val_json': expand_runtime_path(final_val_json),
        'final_test_json': expand_runtime_path(final_test_json),
        'campaign_summary_json': expand_runtime_path(campaign_summary_json),
        'state_file': required_runtime_path(
            raw_path=control_cfg.get('state_file', ''),
            field_name='bc.control.state_file',
        ),
        'best_state_file': required_runtime_path(
            raw_path=control_cfg.get('best_state_file', ''),
            field_name='bc.control.best_state_file',
        ),
        'train_list': required_runtime_path(
            raw_path=dataset_cfg.get('train_list', ''),
            field_name='bc.dataset.train_list',
        ),
        'val_list': required_runtime_path(
            raw_path=dataset_cfg.get('val_list', ''),
            field_name='bc.dataset.val_list',
        ),
        'test_list': required_runtime_path(
            raw_path=dataset_cfg.get('test_list', ''),
            field_name='bc.dataset.test_list',
        ),
        'path_cache': required_runtime_path(
            raw_path=dataset_cfg.get('path_cache', ''),
            field_name='bc.dataset.path_cache',
        ),
        'step_count_summary': required_runtime_path(
            raw_path=dataset_cfg.get('step_count_summary', ''),
            field_name='bc.dataset.step_count_summary',
        ),
        'actor_filter_index': (
            required_runtime_path(
                raw_path=dataset_cfg.get('actor_filter_index', ''),
                field_name='bc.dataset.actor_filter_index',
            )
            if dataset_cfg.get('min_actor_dan') is not None
            else None
        ),
    }


def required_input_paths(launch_settings: dict) -> dict[str, Path]:
    required = {
        'train_list': launch_settings['train_list'],
        'val_list': launch_settings['val_list'],
        'test_list': launch_settings['test_list'],
        'path_cache': launch_settings['path_cache'],
    }
    if launch_settings.get('actor_filter_index') is not None:
        required['actor_filter_index'] = launch_settings['actor_filter_index']
    return required


def missing_input_paths(launch_settings: dict) -> dict[str, str]:
    missing = {}
    for name, filename in required_input_paths(launch_settings).items():
        if not filename.exists():
            missing[name] = str(filename)
    return missing


def ensure_output_dirs(launch_settings: dict) -> None:
    for key in (
        'state_file',
        'best_state_file',
        'final_val_json',
        'final_test_json',
        'campaign_summary_json',
    ):
        launch_settings[key].parent.mkdir(parents=True, exist_ok=True)


def build_train_command(
    *,
    config_path: str | os.PathLike[str],
    launch_settings: dict,
    torchrun_bin: str = 'torchrun',
) -> list[str]:
    return [
        torchrun_bin,
        '--standalone',
        '--nproc_per_node',
        str(launch_settings['nproc_per_node']),
        '--master-port',
        str(launch_settings['master_port']),
        'mortal/train_bc.py',
    ]


def build_stage_command(
    *,
    config_path: str | os.PathLike[str],
    full_config: dict,
    python_bin: str = sys.executable,
    splits: list[str] | None = None,
    force: bool = False,
) -> list[str]:
    if runtime_cache_enabled(full_config):
        return []
    stage_cfg = ((full_config.get('bc') or {}).get('stage') or {})
    if not bool(stage_cfg.get('enabled', False)):
        return []
    requested_splits = splits or stage_cfg.get('required_splits') or ['train', 'val', 'test']
    normalized_splits = []
    for split in requested_splits:
        split_name = str(split).strip()
        if split_name not in ('train', 'val', 'test'):
            raise ValueError(f'unsupported bc.stage split: {split_name}')
        if split_name not in normalized_splits:
            normalized_splits.append(split_name)
    command = [
        python_bin,
        'scripts/stage_bc_tensor_shards.py',
        '--config',
        os.fspath(config_path),
    ]
    for split in normalized_splits:
        command.extend(['--split', split])
    if force:
        command.append('--force')
    return command


def build_eval_command(
    *,
    checkpoint: str | os.PathLike[str],
    split: str,
    output_json: str | os.PathLike[str],
    eval_device: str,
    python_bin: str = sys.executable,
) -> list[str]:
    return [
        python_bin,
        'mortal/eval_bc.py',
        '--checkpoint',
        os.fspath(checkpoint),
        '--split',
        split,
        '--device',
        eval_device,
        '--max-batches',
        '0',
        '--output-json',
        os.fspath(output_json),
    ]


def make_campaign_summary(
    *,
    config_path: str,
    config_fingerprint_value: str,
    started_at: str,
    finished_at: str,
    status: str,
    checkpoint_paths: dict[str, str],
    report_paths: dict[str, str],
    commands: dict[str, list[str]],
    failed_stage: str = '',
    return_code: int | None = None,
    error: str = '',
) -> dict:
    summary = {
        'config_path': config_path,
        'config_fingerprint': config_fingerprint_value,
        'started_at': started_at,
        'finished_at': finished_at,
        'status': status,
        'checkpoint_paths': checkpoint_paths,
        'report_paths': report_paths,
        'commands': commands,
    }
    if failed_stage:
        summary['failed_stage'] = failed_stage
    if return_code is not None:
        summary['return_code'] = return_code
    if error:
        summary['error'] = error
    return summary


def write_summary(summary_path: str | os.PathLike[str], summary: dict) -> None:
    output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def subprocess_env(config_path: str | os.PathLike[str]) -> dict[str, str]:
    env = os.environ.copy()
    env['MORTAL_CFG'] = os.fspath(config_path)
    return env


def query_torch_visible_gpu_inventory(
    *,
    env: dict[str, str] | None = None,
    python_bin: str = sys.executable,
) -> list[dict]:
    result = subprocess.run(
        [python_bin, '-c', TORCH_VISIBLE_GPU_QUERY],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    payload = result.stdout.strip() or '[]'
    inventory = json.loads(payload)
    if not isinstance(inventory, list):
        raise ValueError('torch-visible GPU inventory query returned a non-list payload')
    return inventory


def validate_torch_visible_launch_gpus(
    *,
    env: dict[str, str] | None = None,
    python_bin: str = sys.executable,
    expected_count: int,
    required_name_substring: str = '',
) -> dict:
    inventory = query_torch_visible_gpu_inventory(env=env, python_bin=python_bin)
    if len(inventory) < expected_count:
        raise ValueError(
            f'launch requires {expected_count} torch-visible CUDA device(s), '
            f'but only found {len(inventory)}'
        )
    selected = inventory[:expected_count]
    required = str(required_name_substring or '').strip().lower()
    if required:
        mismatched = [
            row for row in selected
            if required not in str(row.get('name', '')).lower()
        ]
        if mismatched:
            details = ', '.join(
                f"cuda:{row.get('cuda_index')}={row.get('name', 'unknown')}"
                for row in selected
            )
            raise ValueError(
                f'launch selected non-{required_name_substring} GPU(s): {details}'
            )
    return {
        'visible': inventory,
        'selected': selected,
    }


def run_command(*, command: list[str], env: dict[str, str], stage: str) -> None:
    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        env=env,
    )


def launcher_payload(config_path: str | os.PathLike[str]) -> tuple[Path, dict, dict, str]:
    resolved_config_path, full_config = load_full_config(config_path)
    launch_settings = resolve_launch_settings(full_config)
    return (
        resolved_config_path,
        full_config,
        launch_settings,
        config_fingerprint(full_config),
    )
