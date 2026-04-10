from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import toml
import torch

from bc_campaign import load_full_config
from bc_dataset import load_path_cache, normalize_file_list, save_path_cache


ROOT = Path(__file__).resolve().parents[1]
PHASE_ORDER = ('phase_a', 'phase_b', 'phase_c', 'phase_d')
GPU_ADVISORY_FAILURE_PREFIXES = (
    'steady_gpu_ratio_below_gate=',
    'no_post_warmup_gpu_samples',
)


def _filtered_trimmed_lines(lines) -> list[str]:
    return [line.strip() for line in lines if line.strip()]


def _load_path_list(filename: str, root_dir: str = '') -> list[str]:
    with open(filename, encoding='utf-8') as f:
        paths = _filtered_trimmed_lines(f)
    if root_dir:
        return [
            path if Path(path).is_absolute() else str((Path(root_dir) / path).resolve())
            for path in paths
        ]
    return paths


def configured_split_lists(full_config: dict, *, splits: list[str] | None = None) -> dict[str, list[str]]:
    bc_cfg = full_config.get('bc') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    root_dir = dataset_cfg.get('root_dir', '')
    path_cache = str(dataset_cfg.get('path_cache', '') or '').strip()
    resolved_splits = splits or ['train', 'val', 'test']

    if path_cache and Path(path_cache).exists():
        expected_sources = {
            split_name: dataset_cfg.get(f'{split_name}_list', '')
            for split_name in ('train', 'val', 'test')
            if split_name in resolved_splits and dataset_cfg.get(f'{split_name}_list', '')
        }
        return load_path_cache(
            path_cache,
            expected_splits=resolved_splits,
            expected_sources=expected_sources,
        )

    split_lists = {}
    for split_name in resolved_splits:
        list_path = str(dataset_cfg.get(f'{split_name}_list', '') or '').strip()
        if not list_path:
            raise ValueError(f'bc.dataset.{split_name}_list is required to prepare the Step 6 subset ladder')
        split_lists[split_name] = normalize_file_list(
            _load_path_list(list_path, root_dir),
            desc=f'PATHS-{split_name.upper()}',
        )
    return split_lists


def deterministic_round_robin_sample(items: list[str], sample_size: int) -> list[str]:
    total = len(items)
    if sample_size <= 0:
        raise ValueError('sample_size must be positive')
    if sample_size >= total:
        return list(items)
    return [
        items[(idx * total) // sample_size]
        for idx in range(sample_size)
    ]


def write_path_list(output_path: str | Path, file_list: list[str]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(file_list) + '\n', encoding='utf-8')
    return output


def prepare_subset_artifacts(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    subset_size: int,
) -> dict:
    resolved_config_path, full_config = load_full_config(config_path)
    split_lists = configured_split_lists(full_config, splits=['train', 'val', 'test'])
    subset_train = deterministic_round_robin_sample(split_lists['train'], subset_size)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    subset_train_list = write_path_list(output_root / 'train_subset.txt', subset_train)
    subset_val_list = write_path_list(output_root / 'val_full.txt', split_lists['val'])
    subset_test_list = write_path_list(output_root / 'test_full.txt', split_lists['test'])
    subset_path_cache = output_root / 'path_cache_subset.pth'
    save_path_cache(
        str(subset_path_cache),
        split_lists={
            'train': subset_train,
            'val': split_lists['val'],
            'test': split_lists['test'],
        },
        source_files={
            'train': str(subset_train_list),
            'val': str(subset_val_list),
            'test': str(subset_test_list),
        },
    )

    metadata = {
        'source_config_path': str(resolved_config_path),
        'subset_size_requested': int(subset_size),
        'train_file_count': len(subset_train),
        'val_file_count': len(split_lists['val']),
        'test_file_count': len(split_lists['test']),
        'train_list': str(subset_train_list),
        'val_list': str(subset_val_list),
        'test_list': str(subset_test_list),
        'path_cache': str(subset_path_cache),
    }
    metadata_path = output_root / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    metadata['metadata_path'] = str(metadata_path)
    return metadata


def validate_phase_window(*, start_at: str, stop_after: str) -> tuple[int, int]:
    if start_at not in PHASE_ORDER:
        raise ValueError(f'unsupported start_at phase: {start_at}')
    if stop_after not in PHASE_ORDER:
        raise ValueError(f'unsupported stop_after phase: {stop_after}')
    start_index = PHASE_ORDER.index(start_at)
    stop_index = PHASE_ORDER.index(stop_after)
    if start_index > stop_index:
        raise ValueError(
            f'invalid phase window: start_at={start_at} comes after stop_after={stop_after}'
        )
    return start_index, stop_index


def phase_in_window(*, phase: str, start_at: str, stop_after: str) -> bool:
    start_index, stop_index = validate_phase_window(start_at=start_at, stop_after=stop_after)
    phase_index = PHASE_ORDER.index(phase)
    return start_index <= phase_index <= stop_index


def deep_update(base: dict, updates: dict) -> dict:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def write_config(output_path: str | Path, config_payload: dict) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml.dumps(config_payload), encoding='utf-8')
    return output


def phase_runtime_overrides(*, run_root: str | Path, experiment_name: str) -> dict:
    base = Path(run_root) / experiment_name
    return {
        'bc': {
            'control': {
                'tensorboard_dir': str(base / 'tensorboard'),
                'state_file': str(base / 'state.pth'),
                'best_state_file': str(base / 'best.pth'),
                'metrics_jsonl': str(base / 'metrics.jsonl'),
            },
            'launch': {
                'final_val_json': str(base / 'final_val.json'),
                'final_test_json': str(base / 'final_test.json'),
                'campaign_summary_json': str(base / 'campaign_summary.json'),
            },
            'dataset': {
                'file_index': str(base / 'file_index.pth'),
            },
            'preflight': {
                'summary_json': str(base / 'preflight_summary.json'),
            },
            'wandb': {
                'enabled': False,
                'name': experiment_name,
                'group': 'step6_experiment_ladder',
            },
        },
    }


def loader_knob_overrides(
    *,
    knobs: dict,
    cache_root: str | Path | None = None,
    required_splits: list[str] | None = None,
) -> dict:
    return {
        'bc': {
            'dataset': {
                'device_prefetch_batches': int(knobs.get('device_prefetch_batches', 2)),
                'device_prefetch_startup_batches': 1,
            },
            'stage': {
                'enabled': False,
            },
            'runtime_cache': {
                'enabled': True,
                'mode': 'prepared_ram',
                'node_ram_budget_gib': int(knobs.get('node_ram_budget_gib', 160)),
                'node_pinned_budget_gib': int(knobs.get('node_pinned_budget_gib', 8)),
                'node_inflight_budget_gib': int(knobs.get('node_inflight_budget_gib', 8)),
                'raw_lru_budget_gib': int(knobs.get('raw_lru_budget_gib', 0)),
                'low_watermark': float(knobs.get('low_watermark', 0.65)),
                'high_watermark': float(knobs.get('high_watermark', 0.85)),
                'target_chunk_gib': float(knobs.get('target_chunk_gib', 2)),
                'max_chunk_gib': float(knobs.get('max_chunk_gib', 4)),
                'startup_ready_chunks': int(knobs.get('startup_ready_chunks', 2)),
                'producer_threads': int(knobs.get('producer_threads', 1)),
                'decode_threads': int(knobs.get('decode_threads', 4)),
                'max_inflight_chunk_builders': int(knobs.get('max_inflight_chunk_builders', 1)),
                'min_files_per_chunk': int(knobs.get('min_files_per_chunk', 16)),
                'max_files_per_chunk': int(knobs.get('max_files_per_chunk', 96)),
                'eval_node_ram_budget_gib': int(knobs.get('eval_node_ram_budget_gib', 16)),
                'eval_target_chunk_gib': float(knobs.get('eval_target_chunk_gib', 1)),
                'eval_decode_threads': int(knobs.get('eval_decode_threads', 2)),
            },
        },
    }


def raw_threaded_queue_overrides(
    *,
    cpu_ready_batches: int,
    cpu_ready_bytes_gib: float = 0.0,
) -> dict:
    return {
        'bc': {
            'dataset': {
                'num_workers': 0,
                'eval_num_workers': 0,
                'prefetch_chunks': 1,
                'eval_prefetch_chunks': 1,
                'prebatched': False,
                'cpu_batch_pipe_backend': 'thread',
                'cpu_ready_batches': int(cpu_ready_batches),
                'cpu_ready_bytes_gib': float(cpu_ready_bytes_gib),
                'device_prefetch_batches': 2,
                'device_prefetch_startup_batches': 1,
                'pin_memory': True,
                'handoff_pin_memory': False,
                'raw_source_backend': 'files',
                'loader_mode': 'baseline',
                'persistent_workers': False,
                'prefetch_factor': 2,
                'in_order': True,
            },
            'stage': {
                'enabled': False,
            },
            'runtime_cache': {
                'enabled': False,
            },
        },
    }


def raw_source_backend_overrides(
    *,
    backend: str,
    raw_pack_path: str = '',
    raw_pack_index_path: str = '',
) -> dict:
    return {
        'bc': {
            'dataset': {
                'raw_source_backend': str(backend),
                'raw_pack_path': str(raw_pack_path),
                'raw_pack_index_path': str(raw_pack_index_path),
            },
        },
    }


def phase3_worker_overrides(
    *,
    num_workers: int,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    in_order: bool = True,
    multiprocessing_context: str = 'spawn',
) -> dict:
    return {
        'bc': {
            'dataset': {
                'num_workers': int(num_workers),
                'cpu_batch_pipe_backend': 'thread',
                'persistent_workers': bool(persistent_workers),
                'prefetch_factor': int(prefetch_factor),
                'in_order': bool(in_order),
                'multiprocessing_context': str(multiprocessing_context),
                'raw_source_backend': 'files',
                'loader_mode': 'baseline',
            },
            'stage': {
                'enabled': False,
            },
            'runtime_cache': {
                'enabled': False,
            },
        },
    }


def phase4_preassembled_overrides(
    *,
    loader_block_target_samples: int = 65536,
) -> dict:
    return {
        'bc': {
            'dataset': {
                'num_workers': 0,
                'eval_num_workers': 0,
                'cpu_batch_pipe_backend': 'thread',
                'raw_source_backend': 'files',
                'loader_mode': 'preassembled_batches',
                'loader_block_target_samples': int(loader_block_target_samples),
            },
            'stage': {
                'enabled': False,
            },
            'runtime_cache': {
                'enabled': False,
            },
        },
    }


def load_json(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def latest_train_live_metrics(metrics_jsonl_path: str | Path, *, max_windows: int = 3) -> dict | None:
    metrics_path = Path(metrics_jsonl_path)
    if not metrics_path.exists():
        return None
    events = []
    with metrics_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get('event') == 'train_live':
                events.append(payload)
    if not events:
        return None
    recent = events[-max_windows:]
    return {
        'samples_per_second': sum(
            float(event['runtime_metrics']['samples_per_second'])
            for event in recent
        ) / len(recent),
        'steps_per_second': sum(
            float(event['runtime_metrics']['steps_per_second'])
            for event in recent
        ) / len(recent),
        'window_count': len(recent),
    }


def average_wait_fraction_after_step(
    metrics_jsonl_path: str | Path,
    *,
    min_step: int = 125,
) -> float | None:
    metrics_path = Path(metrics_jsonl_path)
    if not metrics_path.exists():
        return None
    values = []
    with metrics_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get('event') not in ('train_live', 'save_window'):
                continue
            if int(payload.get('step', 0) or 0) < int(min_step):
                continue
            loader_metrics = payload.get('loader_metrics') or {}
            if 'wait_fraction' not in loader_metrics:
                continue
            values.append(float(loader_metrics['wait_fraction']))
    if not values:
        return None
    return sum(values) / len(values)


def checkpoint_best_perf(checkpoint_path: str | Path) -> dict | None:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return None
    payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
    return payload.get('best_perf')


def startup_seconds_from_preflight(summary: dict) -> float | None:
    startup = summary.get('startup') or {}
    value = startup.get('startup_seconds')
    if value is None:
        return None
    return float(value)


def summarize_preflight_row(
    *,
    phase: str,
    name: str,
    knobs: dict,
    stage_summary: dict | None,
    preflight_summary: dict,
    backend: str = 'prepared_ram',
) -> dict:
    sustained = preflight_summary.get('sustained_metrics') or {}
    completed = preflight_summary.get('completed_window_metrics') or {}
    reported = completed or sustained
    steady_gpu = preflight_summary.get('steady_gpu') or {}
    startup = preflight_summary.get('startup') or {}
    loader_snapshot = startup.get('loader_snapshot') or {}
    return {
        'phase': phase,
        'name': name,
        'backend': backend,
        'target_chunk_gib': knobs.get('target_chunk_gib'),
        'decode_threads': knobs.get('decode_threads'),
        'max_inflight_chunk_builders': knobs.get('max_inflight_chunk_builders'),
        'startup_ready_chunks': knobs.get('startup_ready_chunks'),
        'raw_lru_budget_gib': knobs.get('raw_lru_budget_gib'),
        'device_prefetch_batches': knobs.get('device_prefetch_batches'),
        'cpu_ready_batches_target': knobs.get('cpu_ready_batches'),
        'cpu_ready_bytes_target_gib': knobs.get('cpu_ready_bytes_gib'),
        'startup_seconds': startup_seconds_from_preflight(preflight_summary),
        'completed_step': int(preflight_summary.get('completed_step', 0) or 0),
        'measurement_source': 'completed_window' if completed else 'sustained_post_warmup',
        'samples_per_second': reported.get('samples_per_second'),
        'steps_per_second': reported.get('steps_per_second'),
        'loader_wait_fraction': reported.get('wait_fraction'),
        'cpu_pipe_wait_fraction': reported.get('cpu_pipe_wait_fraction'),
        'cpu_pipe_wait_seconds': reported.get('cpu_pipe_wait_seconds'),
        'cpu_ready_batches': reported.get('cpu_ready_batches'),
        'cpu_ready_bytes_gib': reported.get('cpu_ready_bytes_gib'),
        'cpu_producer_blocked_put_fraction': reported.get('cpu_producer_blocked_put_fraction'),
        'gate_samples_per_second': sustained.get('samples_per_second'),
        'gate_loader_wait_fraction': sustained.get('wait_fraction'),
        'steady_gpu_ratio': steady_gpu.get('pass_ratio'),
        'ready_bytes_gib': reported.get('ready_bytes_gib'),
        'inflight_bytes_gib': reported.get('inflight_bytes_gib'),
        'pinned_batch_bytes_gib': reported.get('pinned_batch_bytes_gib'),
        'prefill_complete': reported.get('prefill_complete'),
        'producer_blocked_reason': reported.get('producer_blocked_reason'),
        'raw_read_fraction': reported.get('raw_read_fraction'),
        'collate_or_assemble_fraction': reported.get('collate_or_assemble_fraction'),
        'chunk_read_fraction': reported.get('chunk_read_fraction'),
        'chunk_decompress_fraction': reported.get('chunk_decompress_fraction'),
        'chunk_parse_fraction': reported.get('chunk_parse_fraction'),
        'chunk_assemble_fraction': reported.get('chunk_assemble_fraction'),
        'peak_combined_rss_gib': float(preflight_summary.get('max_combined_train_worker_rss_kib', 0)) / (1024 ** 2),
        'peak_single_rss_gib': float(preflight_summary.get('max_train_worker_rss_kib', 0)) / (1024 ** 2),
        'file_count': int(loader_snapshot.get('discovered_files', 0)),
        'submitted_files': int(loader_snapshot.get('submitted_files', 0)),
        'ready_chunks': int(loader_snapshot.get('ready_chunks', 0)),
        'status': preflight_summary.get('status'),
        'gate_passed': bool((preflight_summary.get('gate') or {}).get('passed', False)),
        'fail_reasons': list((preflight_summary.get('gate') or {}).get('reasons') or []),
        'preflight_return_code': int(preflight_summary.get('return_code', 0) or 0),
    }


def summarize_phase1_queue_row(
    *,
    name: str,
    knobs: dict,
    preflight_summary: dict,
) -> dict:
    return summarize_preflight_row(
        phase='phase1',
        name=name,
        knobs=knobs,
        stage_summary=None,
        preflight_summary=preflight_summary,
        backend='raw_threaded',
    )


def non_advisory_fail_reasons(row: dict) -> list[str]:
    reasons = list(row.get('fail_reasons') or [])
    return [
        reason
        for reason in reasons
        if not any(reason.startswith(prefix) for prefix in GPU_ADVISORY_FAILURE_PREFIXES)
    ]


def row_gate_samples_per_second(row: dict) -> float | None:
    value = row.get('gate_samples_per_second', row.get('samples_per_second'))
    if value is None:
        return None
    return float(value)


def row_gate_loader_wait_fraction(row: dict) -> float | None:
    value = row.get('gate_loader_wait_fraction', row.get('loader_wait_fraction'))
    if value is None:
        return None
    return float(value)


def row_meets_loader_decision_gate(row: dict) -> bool:
    return (
        int(row.get('preflight_return_code', 0) or 0) == 0
        and not non_advisory_fail_reasons(row)
        and row_gate_samples_per_second(row) is not None
        and float(row_gate_samples_per_second(row)) >= 5900.0
        and row_gate_loader_wait_fraction(row) is not None
        and float(row_gate_loader_wait_fraction(row)) <= 0.15
        and row.get('startup_seconds') is not None
        and float(row['startup_seconds']) <= 15.0
    )


def row_meets_loader_gate(row: dict) -> bool:
    return (
        bool(row.get('gate_passed'))
        and row_gate_samples_per_second(row) is not None
        and float(row_gate_samples_per_second(row)) >= 5900.0
        and row_gate_loader_wait_fraction(row) is not None
        and float(row_gate_loader_wait_fraction(row)) <= 0.15
        and row.get('steady_gpu_ratio') is not None
        and float(row['steady_gpu_ratio']) >= 0.70
        and row.get('startup_seconds') is not None
        and float(row['startup_seconds']) <= 15.0
    )


def should_run_shard_size_comparison(row: dict) -> bool:
    if not row_meets_loader_decision_gate(row):
        return True
    startup_seconds = float(row.get('startup_seconds') or 0.0)
    samples_per_second = float(row.get('samples_per_second') or 0.0)
    return startup_seconds > 10.0 or samples_per_second < 7000.0


def should_run_thread_comparison(row: dict) -> bool:
    if row_meets_loader_decision_gate(row):
        return False
    wait_fraction = float(row.get('loader_wait_fraction') or 0.0)
    steady_gpu_ratio = float(row.get('steady_gpu_ratio') or 0.0)
    return wait_fraction > 0.10 or steady_gpu_ratio < 0.85


def should_run_device_prefetch_comparison(row: dict) -> bool:
    if row_meets_loader_decision_gate(row):
        return False
    wait_fraction = float(row.get('loader_wait_fraction') or 1.0)
    steady_gpu_ratio = float(row.get('steady_gpu_ratio') or 0.0)
    return wait_fraction <= 0.15 and steady_gpu_ratio < 0.85


def candidate_improves(
    *,
    baseline: dict,
    candidate: dict,
    min_relative_gain: float,
    max_rss_growth_ratio: float | None = None,
    max_startup_growth_seconds: float | None = None,
) -> bool:
    if not row_meets_loader_decision_gate(candidate):
        return False
    baseline_sps = float(baseline.get('samples_per_second') or 0.0)
    candidate_sps = float(candidate.get('samples_per_second') or 0.0)
    if max_rss_growth_ratio is not None:
        baseline_rss = float(baseline.get('peak_combined_rss_gib') or 0.0)
        candidate_rss = float(candidate.get('peak_combined_rss_gib') or 0.0)
        if baseline_rss > 0 and candidate_rss > baseline_rss * (1.0 + max_rss_growth_ratio):
            return False
    if max_startup_growth_seconds is not None:
        baseline_startup = float(baseline.get('startup_seconds') or 0.0)
        candidate_startup = float(candidate.get('startup_seconds') or 0.0)
        if baseline_startup > 0 and candidate_startup > baseline_startup + max_startup_growth_seconds:
            return False
    if baseline_sps <= 0:
        return candidate_sps > 0
    return candidate_sps >= baseline_sps * (1.0 + min_relative_gain)


def phase1_candidate_beats_control(
    *,
    control: dict,
    candidate: dict,
) -> bool:
    if int(candidate.get('preflight_return_code', 0) or 0) != 0:
        return False
    control_sps = float(control.get('samples_per_second') or 0.0)
    candidate_sps = float(candidate.get('samples_per_second') or 0.0)
    control_wait = float(
        control.get('late_window_wait_fraction', control.get('loader_wait_fraction')) or 0.0
    )
    candidate_wait = float(
        candidate.get('late_window_wait_fraction', candidate.get('loader_wait_fraction')) or 0.0
    )
    if control_sps <= 0:
        return candidate_sps > 0.0
    if candidate_sps >= control_sps * 1.03:
        return True
    return (control_wait - candidate_wait) >= 0.03 and candidate_sps >= control_sps * 0.99


def phase2_candidate_beats_control(
    *,
    control: dict,
    candidate: dict,
) -> bool:
    if int(candidate.get('preflight_return_code', 0) or 0) != 0:
        return False
    control_sps = float(control.get('samples_per_second') or 0.0)
    candidate_sps = float(candidate.get('samples_per_second') or 0.0)
    control_raw_read = float(control.get('raw_read_fraction') or 0.0)
    candidate_raw_read = float(candidate.get('raw_read_fraction') or 0.0)
    if control_sps > 0.0 and candidate_sps >= control_sps * 1.03:
        return True
    return (control_raw_read - candidate_raw_read) >= 0.10


def phase5_worker_overrides(
    *,
    num_workers: int,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    cpu_ready_batches: int = 4,
    multiprocessing_context: str = 'spawn',
) -> dict:
    return {
        'bc': {
            'dataset': {
                'num_workers': int(num_workers),
                'cpu_batch_pipe_backend': 'thread',
                'persistent_workers': bool(persistent_workers),
                'prefetch_factor': int(prefetch_factor),
                'in_order': True,
                'multiprocessing_context': str(multiprocessing_context),
                'raw_source_backend': 'files',
                'loader_mode': 'baseline',
                'cpu_ready_batches': int(cpu_ready_batches),
                'prefetch_chunks': 1,
                'pin_memory': True,
                'handoff_pin_memory': False,
                'device_prefetch_batches': 2,
                'device_prefetch_startup_batches': 1,
            },
            'stage': {
                'enabled': False,
            },
            'runtime_cache': {
                'enabled': False,
            },
        },
    }


def phase5_candidate_beats_control(
    *,
    control: dict,
    candidate: dict,
    max_rss_gib: float = 150.0,
) -> bool:
    if int(candidate.get('preflight_return_code', 0) or 0) != 0:
        return False
    candidate_rss = float(candidate.get('peak_combined_rss_gib') or 0.0)
    if candidate_rss > max_rss_gib:
        return False
    control_sps = float(control.get('samples_per_second') or 0.0)
    candidate_sps = float(candidate.get('samples_per_second') or 0.0)
    control_wait = float(
        control.get('late_window_wait_fraction', control.get('loader_wait_fraction')) or 0.0
    )
    candidate_wait = float(
        candidate.get('late_window_wait_fraction', candidate.get('loader_wait_fraction')) or 0.0
    )
    if control_sps <= 0:
        return candidate_sps > 0.0
    if candidate_sps >= control_sps * 1.03:
        return True
    return (control_wait - candidate_wait) >= 0.05 and candidate_sps >= control_sps * 0.99


def phase4_candidate_beats_control(
    *,
    control: dict,
    candidate: dict,
) -> bool:
    if int(candidate.get('preflight_return_code', 0) or 0) != 0:
        return False
    control_sps = float(control.get('samples_per_second') or 0.0)
    candidate_sps = float(candidate.get('samples_per_second') or 0.0)
    control_collate = float(control.get('collate_or_assemble_fraction') or 0.0)
    candidate_collate = float(candidate.get('collate_or_assemble_fraction') or 0.0)
    if control_sps > 0.0 and candidate_sps >= control_sps * 1.05:
        return True
    return (control_collate - candidate_collate) >= 0.10 and candidate_sps >= control_sps * 0.99


def select_phase1_batch_count_winner(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('rows must not be empty')

    def sort_key(row: dict) -> tuple:
        return (
            1 if int(row.get('preflight_return_code', 0) or 0) == 0 else 0,
            float(row.get('samples_per_second') or 0.0),
            -float(row.get('loader_wait_fraction') or 0.0),
            -float(row.get('steady_gpu_ratio') or 0.0),
            -float(row.get('startup_seconds') or 0.0),
        )

    return max(rows, key=sort_key)


def render_markdown_table(rows: list[dict], *, columns: list[tuple[str, str]]) -> str:
    if not rows:
        return '_No rows_'
    header = '| ' + ' | '.join(label for _, label in columns) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    body = []
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key, '')
            if isinstance(value, float):
                values.append(f'{value:.4f}')
            elif isinstance(value, list):
                values.append(', '.join(str(item) for item in value))
            else:
                values.append(str(value))
        body.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join([header, divider, *body]) + '\n'
