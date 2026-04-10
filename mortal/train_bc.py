import collections
import inspect
import json
import math
import os
import queue
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch.nn.parallel import DistributedDataParallel
from wandb_utils import default_wandb_run_name, maybe_init_wandb_run
from bc_runtime import (
    broadcast_object,
    config_fingerprint,
    destroy_distributed_context,
    distributed_barrier,
    effective_global_batch,
    init_distributed_context,
    resolve_distributed_context,
    seed_everything,
    shard_file_list_round_robin,
    stored_config_fingerprint,
)

ACTION_CATEGORY_NAMES = (
    'discard',
    'riichi',
    'chi',
    'pon',
    'kan',
    'agari',
    'ryukyoku',
    'pass',
)


def atomic_torch_save(obj, output_path) -> None:
    output_path = os.fspath(output_path)
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(output_path)}.',
        suffix='.tmp',
        dir=output_dir,
    )
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def save_rolling_checkpoint(state, checkpoint_dir: str, steps: int, keep_recent: int) -> None:
    """Save a numbered checkpoint and prune old ones."""
    if not checkpoint_dir:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f'step_{steps:08d}.pth')
    atomic_torch_save(state, ckpt_path)
    # Prune: keep only the most recent `keep_recent` checkpoints
    existing = sorted(
        (f for f in os.listdir(checkpoint_dir) if f.startswith('step_') and f.endswith('.pth')),
    )
    to_remove = existing[:-keep_recent] if len(existing) > keep_recent else []
    for fname in to_remove:
        try:
            os.unlink(os.path.join(checkpoint_dir, fname))
        except OSError:
            pass


def save_stage_checkpoint(state, stage_save_dir: str, steps: int) -> None:
    """Save a stage checkpoint (kept forever for Tenhou play-testing)."""
    if not stage_save_dir:
        return
    os.makedirs(stage_save_dir, exist_ok=True)
    stage_path = os.path.join(stage_save_dir, f'stage_step_{steps:08d}.pth')
    atomic_torch_save(state, stage_path)


def normalize_best_perf(best_perf: dict | None, best_eval_split: str) -> dict:
    best_perf = dict(best_perf or {})
    legacy_accuracy = best_perf.get('val_accuracy', 0.0)
    legacy_nll = best_perf.get('val_nll', float('inf'))
    return {
        'split': best_perf.get('split', best_eval_split),
        'accuracy': best_perf.get('accuracy', legacy_accuracy),
        'nll': best_perf.get('nll', legacy_nll),
        'steps': best_perf.get('steps', 0),
    }


def is_better_eval_result(metrics: dict, best_perf: dict) -> bool:
    return (
        metrics['accuracy'] > best_perf['accuracy']
        or (
            metrics['accuracy'] == best_perf['accuracy']
            and metrics['nll'] < best_perf['nll']
        )
    )


def throughput_metrics(*, sample_count: int, step_count: int, elapsed_seconds: float) -> dict:
    elapsed = max(elapsed_seconds, 1e-9)
    return {
        'elapsed_seconds': elapsed,
        'steps_per_second': step_count / elapsed,
        'samples_per_second': sample_count / elapsed,
    }


def empty_window_observability() -> dict:
    return {
        'fw_bw_opt_seconds': 0.0,
        'ddp_sync_wait_seconds': 0.0,
        'save_checkpoint_wait_seconds': 0.0,
        'step_time_seconds_total': 0.0,
        'step_count': 0,
        'cpu_ready_batches_min': None,
        'cpu_ready_batches_max': None,
        'cpu_ready_batches_sum': 0.0,
        'cpu_ready_batches_samples': 0,
        'loader_ready_chunks_min': None,
        'loader_ready_chunks_max': None,
        'loader_ready_chunks_sum': 0.0,
        'loader_ready_chunks_samples': 0,
        'device_prefetch_depth_min': None,
        'device_prefetch_depth_max': None,
        'device_prefetch_depth_sum': 0.0,
        'device_prefetch_depth_samples': 0,
    }


def observe_window_depth(observability: dict, *, prefix: str, value: int | float) -> None:
    numeric = float(value)
    min_key = f'{prefix}_min'
    max_key = f'{prefix}_max'
    sum_key = f'{prefix}_sum'
    samples_key = f'{prefix}_samples'
    current_min = observability.get(min_key)
    current_max = observability.get(max_key)
    observability[min_key] = numeric if current_min is None else min(float(current_min), numeric)
    observability[max_key] = numeric if current_max is None else max(float(current_max), numeric)
    observability[sum_key] = float(observability.get(sum_key, 0.0)) + numeric
    observability[samples_key] = int(observability.get(samples_key, 0)) + 1


def observe_window_queue_depths(
    observability: dict,
    *,
    loader_snapshot: dict | None,
    queue_snapshot: dict | None,
) -> None:
    loader_snapshot = loader_snapshot or {}
    queue_snapshot = queue_snapshot or {}
    observe_window_depth(
        observability,
        prefix='cpu_ready_batches',
        value=max(int(loader_snapshot.get('cpu_ready_batches', 0)), 0),
    )
    observe_window_depth(
        observability,
        prefix='loader_ready_chunks',
        value=max(int(loader_snapshot.get('ready_chunks', 0)), 0),
    )
    observe_window_depth(
        observability,
        prefix='device_prefetch_depth',
        value=max(int(queue_snapshot.get('gpu_prefetch_depth', 0)), 0),
    )


def summarize_window_depths(observability: dict) -> dict:
    payload = {}
    for prefix in ('cpu_ready_batches', 'loader_ready_chunks', 'device_prefetch_depth'):
        samples = max(int(observability.get(f'{prefix}_samples', 0)), 0)
        total = float(observability.get(f'{prefix}_sum', 0.0))
        min_value = observability.get(f'{prefix}_min')
        max_value = observability.get(f'{prefix}_max')
        payload[f'{prefix}_min'] = 0.0 if min_value is None else float(min_value)
        payload[f'{prefix}_max'] = 0.0 if max_value is None else float(max_value)
        payload[f'{prefix}_avg'] = total / samples if samples > 0 else 0.0
        payload[f'{prefix}_samples'] = samples
    return payload


def loader_metrics_delta(previous: dict, current: dict) -> dict:
    return {
        'chunk_count': max(int(current.get('chunk_count_total', 0)) - int(previous.get('chunk_count_total', 0)), 0),
        'chunk_files': max(int(current.get('chunk_files_total', 0)) - int(previous.get('chunk_files_total', 0)), 0),
        'chunk_samples': max(int(current.get('chunk_samples_total', 0)) - int(previous.get('chunk_samples_total', 0)), 0),
        'chunk_bytes': max(int(current.get('chunk_bytes_total', 0)) - int(previous.get('chunk_bytes_total', 0)), 0),
        'chunk_build_seconds': max(
            float(current.get('chunk_build_seconds_total', 0.0)) - float(previous.get('chunk_build_seconds_total', 0.0)),
            0.0,
        ),
        'chunk_read_seconds': max(
            float(current.get('chunk_read_seconds_total', 0.0)) - float(previous.get('chunk_read_seconds_total', 0.0)),
            0.0,
        ),
        'chunk_decompress_seconds': max(
            float(current.get('chunk_decompress_seconds_total', 0.0)) - float(previous.get('chunk_decompress_seconds_total', 0.0)),
            0.0,
        ),
        'chunk_parse_seconds': max(
            float(current.get('chunk_parse_seconds_total', 0.0)) - float(previous.get('chunk_parse_seconds_total', 0.0)),
            0.0,
        ),
        'chunk_rust_convert_seconds': max(
            float(current.get('chunk_rust_convert_seconds_total', 0.0))
            - float(previous.get('chunk_rust_convert_seconds_total', 0.0)),
            0.0,
        ),
        'chunk_sample_materialize_seconds': max(
            float(current.get('chunk_sample_materialize_seconds_total', 0.0))
            - float(previous.get('chunk_sample_materialize_seconds_total', 0.0)),
            0.0,
        ),
        'chunk_assemble_seconds': max(
            float(current.get('chunk_assemble_seconds_total', 0.0)) - float(previous.get('chunk_assemble_seconds_total', 0.0)),
            0.0,
        ),
        'collate_seconds': max(
            float(current.get('collate_seconds_total', 0.0)) - float(previous.get('collate_seconds_total', 0.0)),
            0.0,
        ),
        'queued_bytes': max(int(current.get('queued_bytes', 0)), 0),
        'max_queued_bytes': max(int(current.get('max_queued_bytes', 0)) - int(previous.get('max_queued_bytes', 0)), 0),
        'ready_chunks': max(int(current.get('ready_chunks', 0)), 0),
        'ready_bytes': max(int(current.get('ready_bytes', 0)), 0),
        'max_ready_bytes': max(int(current.get('max_ready_bytes', 0)) - int(previous.get('max_ready_bytes', 0)), 0),
        'inflight_bytes': max(int(current.get('inflight_bytes', 0)), 0),
        'max_inflight_bytes': max(int(current.get('max_inflight_bytes', 0)) - int(previous.get('max_inflight_bytes', 0)), 0),
        'pinned_batch_bytes': max(int(current.get('pinned_batch_bytes', 0)), 0),
        'max_pinned_batch_bytes': max(int(current.get('max_pinned_batch_bytes', 0)) - int(previous.get('max_pinned_batch_bytes', 0)), 0),
        'raw_lru_bytes': max(int(current.get('raw_lru_bytes', 0)), 0),
        'max_raw_lru_bytes': max(int(current.get('max_raw_lru_bytes', 0)) - int(previous.get('max_raw_lru_bytes', 0)), 0),
        'budget_bytes': max(int(current.get('budget_bytes', 0)), 0),
        'discovered_files': max(int(current.get('discovered_files', 0)), 0),
        'submitted_files': max(int(current.get('submitted_files', 0)), 0),
        'prefill_complete': bool(current.get('prefill_complete', False)),
        'producer_blocked_reason': str(current.get('producer_blocked_reason', '')),
        'last_chunk_files': max(int(current.get('last_chunk_files', 0)), 0),
        'last_chunk_samples': max(int(current.get('last_chunk_samples', 0)), 0),
        'last_chunk_bytes': max(int(current.get('last_chunk_bytes', 0)), 0),
        'last_chunk_build_seconds': max(float(current.get('last_chunk_build_seconds', 0.0)), 0.0),
        'last_chunk_read_seconds': max(float(current.get('last_chunk_read_seconds', 0.0)), 0.0),
        'last_chunk_decompress_seconds': max(float(current.get('last_chunk_decompress_seconds', 0.0)), 0.0),
        'last_chunk_parse_seconds': max(float(current.get('last_chunk_parse_seconds', 0.0)), 0.0),
        'last_chunk_rust_convert_seconds': max(float(current.get('last_chunk_rust_convert_seconds', 0.0)), 0.0),
        'last_chunk_sample_materialize_seconds': max(
            float(current.get('last_chunk_sample_materialize_seconds', 0.0)),
            0.0,
        ),
        'last_chunk_assemble_seconds': max(float(current.get('last_chunk_assemble_seconds', 0.0)), 0.0),
        'cpu_ready_batches': max(int(current.get('cpu_ready_batches', 0)), 0),
        'max_cpu_ready_batches': max(
            int(current.get('max_cpu_ready_batches', 0)) - int(previous.get('max_cpu_ready_batches', 0)),
            0,
        ),
        'cpu_ready_bytes': max(int(current.get('cpu_ready_bytes', 0)), 0),
        'max_cpu_ready_bytes': max(
            int(current.get('max_cpu_ready_bytes', 0)) - int(previous.get('max_cpu_ready_bytes', 0)),
            0,
        ),
        'cpu_produced_batches': max(
            int(current.get('cpu_produced_batches_total', 0)) - int(previous.get('cpu_produced_batches_total', 0)),
            0,
        ),
        'cpu_produced_samples': max(
            int(current.get('cpu_produced_samples_total', 0)) - int(previous.get('cpu_produced_samples_total', 0)),
            0,
        ),
        'cpu_blocked_put_seconds': max(
            float(current.get('cpu_blocked_put_seconds_total', 0.0))
            - float(previous.get('cpu_blocked_put_seconds_total', 0.0)),
            0.0,
        ),
        'cpu_consumer_wait_seconds': max(
            float(current.get('cpu_consumer_wait_seconds_total', 0.0))
            - float(previous.get('cpu_consumer_wait_seconds_total', 0.0)),
            0.0,
        ),
    }


def cpu_pipe_wait_delta(previous: dict, current: dict) -> float:
    return max(
        float(current.get('cpu_consumer_wait_seconds_total', 0.0))
        - float(previous.get('cpu_consumer_wait_seconds_total', 0.0)),
        0.0,
    )


def loader_window_metrics(
    *,
    previous_snapshot: dict,
    current_snapshot: dict,
    wait_seconds: float,
    elapsed_seconds: float,
    cpu_pipe_wait_seconds_override: float | None = None,
) -> dict:
    delta = loader_metrics_delta(previous_snapshot, current_snapshot)
    elapsed = max(float(elapsed_seconds), 1e-9)
    delta['wait_seconds'] = max(float(wait_seconds), 0.0)
    delta['wait_fraction'] = min(max(delta['wait_seconds'] / elapsed, 0.0), 1.0)
    delta['cpu_pipe_wait_seconds_local'] = max(float(delta['cpu_consumer_wait_seconds']), 0.0)
    delta['cpu_pipe_wait_seconds'] = max(
        float(cpu_pipe_wait_seconds_override)
        if cpu_pipe_wait_seconds_override is not None
        else delta['cpu_pipe_wait_seconds_local'],
        0.0,
    )
    delta['cpu_pipe_empty_wait_seconds_local'] = delta['cpu_pipe_wait_seconds_local']
    delta['cpu_pipe_empty_wait_seconds'] = delta['cpu_pipe_wait_seconds']
    delta['cpu_pipe_wait_fraction_local'] = min(max(delta['cpu_pipe_wait_seconds_local'] / elapsed, 0.0), 1.0)
    delta['cpu_pipe_wait_fraction'] = min(max(delta['cpu_pipe_wait_seconds'] / elapsed, 0.0), 1.0)
    delta['cpu_pipe_empty_wait_fraction_local'] = delta['cpu_pipe_wait_fraction_local']
    delta['cpu_pipe_empty_wait_fraction'] = delta['cpu_pipe_wait_fraction']
    delta['device_prefetch_wait_seconds'] = max(delta['wait_seconds'] - delta['cpu_pipe_wait_seconds_local'], 0.0)
    delta['device_prefetch_wait_fraction'] = min(max(delta['device_prefetch_wait_seconds'] / elapsed, 0.0), 1.0)
    delta['cpu_producer_blocked_put_fraction'] = min(max(delta['cpu_blocked_put_seconds'] / elapsed, 0.0), 1.0)
    delta['queued_bytes_gib'] = delta['queued_bytes'] / (1024 ** 3)
    delta['max_queued_bytes_gib'] = delta['max_queued_bytes'] / (1024 ** 3)
    delta['ready_bytes_gib'] = delta['ready_bytes'] / (1024 ** 3)
    delta['max_ready_bytes_gib'] = delta['max_ready_bytes'] / (1024 ** 3)
    delta['inflight_bytes_gib'] = delta['inflight_bytes'] / (1024 ** 3)
    delta['max_inflight_bytes_gib'] = delta['max_inflight_bytes'] / (1024 ** 3)
    delta['pinned_batch_bytes_gib'] = delta['pinned_batch_bytes'] / (1024 ** 3)
    delta['max_pinned_batch_bytes_gib'] = delta['max_pinned_batch_bytes'] / (1024 ** 3)
    delta['raw_lru_bytes_gib'] = delta['raw_lru_bytes'] / (1024 ** 3)
    delta['max_raw_lru_bytes_gib'] = delta['max_raw_lru_bytes'] / (1024 ** 3)
    delta['budget_bytes_gib'] = delta['budget_bytes'] / (1024 ** 3)
    delta['chunk_bytes_gib'] = delta['chunk_bytes'] / (1024 ** 3)
    delta['cpu_ready_bytes_gib'] = delta['cpu_ready_bytes'] / (1024 ** 3)
    delta['max_cpu_ready_bytes_gib'] = delta['max_cpu_ready_bytes'] / (1024 ** 3)
    delta['avg_chunk_build_seconds'] = (
        delta['chunk_build_seconds'] / delta['chunk_count']
        if delta['chunk_count'] > 0
        else 0.0
    )
    delta['avg_chunk_parse_seconds'] = (
        delta['chunk_parse_seconds'] / delta['chunk_count']
        if delta['chunk_count'] > 0
        else 0.0
    )
    delta['avg_chunk_read_seconds'] = (
        delta['chunk_read_seconds'] / delta['chunk_count']
        if delta['chunk_count'] > 0
        else 0.0
    )
    delta['avg_chunk_decompress_seconds'] = (
        delta['chunk_decompress_seconds'] / delta['chunk_count']
        if delta['chunk_count'] > 0
        else 0.0
    )
    delta['avg_chunk_rust_convert_seconds'] = (
        delta['chunk_rust_convert_seconds'] / delta['chunk_count']
        if delta['chunk_count'] > 0
        else 0.0
    )
    delta['avg_chunk_sample_materialize_seconds'] = (
        delta['chunk_sample_materialize_seconds'] / delta['chunk_count']
        if delta['chunk_count'] > 0
        else 0.0
    )
    delta['avg_chunk_assemble_seconds'] = (
        delta['chunk_assemble_seconds'] / delta['chunk_count']
        if delta['chunk_count'] > 0
        else 0.0
    )
    delta['raw_read_seconds'] = delta['chunk_read_seconds']
    delta['raw_read_fraction'] = min(max(delta['raw_read_seconds'] / elapsed, 0.0), 1.0)
    delta['rust_convert_seconds'] = delta['chunk_rust_convert_seconds']
    delta['rust_convert_fraction'] = min(max(delta['rust_convert_seconds'] / elapsed, 0.0), 1.0)
    delta['sample_materialize_seconds'] = delta['chunk_sample_materialize_seconds']
    delta['sample_materialize_fraction'] = min(max(delta['sample_materialize_seconds'] / elapsed, 0.0), 1.0)
    delta['collate_or_assemble_seconds'] = delta['collate_seconds'] + delta['chunk_assemble_seconds']
    delta['collate_or_assemble_fraction'] = min(max(delta['collate_or_assemble_seconds'] / elapsed, 0.0), 1.0)
    build_component_seconds = max(
        delta['chunk_read_seconds']
        + delta['chunk_decompress_seconds']
        + delta['chunk_parse_seconds']
        + delta['chunk_assemble_seconds'],
        0.0,
    )
    if build_component_seconds > 0:
        delta['chunk_read_fraction'] = delta['chunk_read_seconds'] / build_component_seconds
        delta['chunk_decompress_fraction'] = delta['chunk_decompress_seconds'] / build_component_seconds
        delta['chunk_parse_fraction'] = delta['chunk_parse_seconds'] / build_component_seconds
        delta['chunk_assemble_fraction'] = delta['chunk_assemble_seconds'] / build_component_seconds
    else:
        delta['chunk_read_fraction'] = 0.0
        delta['chunk_decompress_fraction'] = 0.0
        delta['chunk_parse_fraction'] = 0.0
        delta['chunk_assemble_fraction'] = 0.0
    return delta


def merge_window_observability(
    *,
    runtime_metrics: dict,
    loader_metrics: dict,
    observability: dict,
    rank_step_time_ms_max: float,
    rank_step_time_ms_min: float,
) -> None:
    elapsed = max(float(runtime_metrics.get('elapsed_seconds', 0.0)), 1e-9)
    depth_metrics = summarize_window_depths(observability)
    loader_metrics.update(depth_metrics)
    runtime_metrics['fw_bw_opt_seconds'] = max(float(observability.get('fw_bw_opt_seconds', 0.0)), 0.0)
    runtime_metrics['fw_bw_opt_fraction'] = min(max(runtime_metrics['fw_bw_opt_seconds'] / elapsed, 0.0), 1.0)
    runtime_metrics['ddp_sync_wait_seconds'] = max(float(observability.get('ddp_sync_wait_seconds', 0.0)), 0.0)
    runtime_metrics['ddp_sync_wait_fraction'] = min(
        max(runtime_metrics['ddp_sync_wait_seconds'] / elapsed, 0.0),
        1.0,
    )
    runtime_metrics['save_checkpoint_wait_seconds'] = max(
        float(observability.get('save_checkpoint_wait_seconds', 0.0)),
        0.0,
    )
    runtime_metrics['save_checkpoint_wait_fraction'] = min(
        max(runtime_metrics['save_checkpoint_wait_seconds'] / elapsed, 0.0),
        1.0,
    )
    step_count = max(int(observability.get('step_count', 0)), 0)
    runtime_metrics['rank_step_time_ms'] = (
        float(observability.get('step_time_seconds_total', 0.0)) * 1000.0 / step_count
        if step_count > 0
        else 0.0
    )
    runtime_metrics['rank_step_time_ms_max_minus_min'] = max(
        float(rank_step_time_ms_max) - float(rank_step_time_ms_min),
        0.0,
    )


def move_batch_to_device(batch, *, device: torch.device, oracle: bool):
    if oracle:
        obs, invisible_obs, actions, masks = batch
        if device.type == 'cuda':
            invisible_obs = invisible_obs.to(dtype=torch.float32, device=device, non_blocking=True)
        else:
            invisible_obs = invisible_obs.to(dtype=torch.float32, device=device)
    else:
        obs, actions, masks = batch
        invisible_obs = None
    if device.type == 'cuda':
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
        masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
    else:
        obs = obs.to(dtype=torch.float32, device=device)
        actions = actions.to(dtype=torch.int64, device=device)
        masks = masks.to(dtype=torch.bool, device=device)
    if oracle:
        return obs, invisible_obs, actions, masks
    return obs, actions, masks


def batch_nsamples(batch) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.dim() > 0 else 1
    if isinstance(batch, (tuple, list)):
        for item in batch:
            n = batch_nsamples(item)
            if n > 0:
                return n
    return 0


def pin_batch_memory(batch):
    if isinstance(batch, torch.Tensor):
        if batch.device.type == 'cpu' and not batch.is_pinned():
            return batch.pin_memory()
        return batch
    if isinstance(batch, tuple):
        return tuple(pin_batch_memory(item) for item in batch)
    if isinstance(batch, list):
        return [pin_batch_memory(item) for item in batch]
    return batch


def allocate_staging_batch(batch, *, pin_memory: bool):
    if isinstance(batch, torch.Tensor):
        return torch.empty_like(batch, device='cpu', pin_memory=pin_memory)
    if isinstance(batch, tuple):
        return tuple(allocate_staging_batch(item, pin_memory=pin_memory) for item in batch)
    if isinstance(batch, list):
        return [allocate_staging_batch(item, pin_memory=pin_memory) for item in batch]
    return batch


def batch_fits_staging_slot(slot, batch) -> bool:
    if isinstance(batch, torch.Tensor):
        if not isinstance(slot, torch.Tensor):
            return False
        if slot.dtype != batch.dtype:
            return False
        if slot.dim() != batch.dim():
            return False
        if slot.shape == batch.shape:
            return True
        if batch.dim() == 0:
            return False
        return slot.shape[0] >= batch.shape[0] and slot.shape[1:] == batch.shape[1:]
    if isinstance(batch, tuple):
        return isinstance(slot, tuple) and len(slot) == len(batch) and all(
            batch_fits_staging_slot(slot_item, batch_item)
            for slot_item, batch_item in zip(slot, batch)
        )
    if isinstance(batch, list):
        return isinstance(slot, list) and len(slot) == len(batch) and all(
            batch_fits_staging_slot(slot_item, batch_item)
            for slot_item, batch_item in zip(slot, batch)
        )
    return slot == batch


def stage_batch_into_slot(slot, batch):
    if isinstance(batch, torch.Tensor):
        if slot.shape == batch.shape:
            target = slot
        elif batch.dim() > 0 and slot.shape[0] >= batch.shape[0] and slot.shape[1:] == batch.shape[1:]:
            target = slot.narrow(0, 0, batch.shape[0])
        else:
            raise ValueError(
                f'batch shape {tuple(batch.shape)} does not fit staging slot shape {tuple(slot.shape)}'
            )
        target.copy_(batch)
        return target
    if isinstance(batch, tuple):
        return tuple(stage_batch_into_slot(slot_item, batch_item) for slot_item, batch_item in zip(slot, batch))
    if isinstance(batch, list):
        return [stage_batch_into_slot(slot_item, batch_item) for slot_item, batch_item in zip(slot, batch)]
    return batch


def record_batch_stream(batch, stream) -> None:
    if isinstance(batch, torch.Tensor):
        if batch.device.type == 'cuda':
            batch.record_stream(stream)
        return
    if isinstance(batch, (tuple, list)):
        for item in batch:
            record_batch_stream(item, stream)


def batch_nbytes(batch) -> int:
    if isinstance(batch, torch.Tensor):
        return batch.element_size() * batch.numel()
    if isinstance(batch, (tuple, list)):
        return sum(batch_nbytes(item) for item in batch)
    return 0


_HANDOFF_END = object()


@dataclass
class PageableReadyBatch:
    batch: Any
    batch_idx: int
    nsamples: int
    nbytes: int
    t_cpu_ready_s: float


@dataclass
class PinnedReadyBatch:
    slot_id: int | None
    batch: Any
    batch_idx: int
    nsamples: int
    nbytes: int
    t_stage_start_s: float
    t_stage_done_s: float


@dataclass
class PendingSlotRelease:
    slot_id: int | None
    nbytes: int
    copy_end_event: Any
    copy_start_event: Any | None = None


@dataclass
class GpuReadyBatch:
    batch_idx: int
    gpu_batch: Any
    nbytes: int
    copy_end_event: Any
    copy_start_event: Any | None = None


class HandoffStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.cpu_ready_wait_s = 0.0
        self.stage_free_slot_wait_s = 0.0
        self.stage_copy_s = 0.0
        self.pinned_ready_wait_s = 0.0
        self.h2d_submit_s = 0.0
        self.copy_ready_on_pop = 0
        self.copy_not_ready_on_pop = 0
        self.h2d_copy_ms_total = 0.0
        self.h2d_copy_count = 0
        self.stage_batches = 0
        self.reclaimed_slots = 0
        self.copy_wait_sync_s = 0.0
        self.slot_shape_mismatch_batches = 0

    def add(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, getattr(self, key) + value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'cpu_ready_wait_s': self.cpu_ready_wait_s,
                'stage_free_slot_wait_s': self.stage_free_slot_wait_s,
                'stage_copy_s': self.stage_copy_s,
                'pinned_ready_wait_s': self.pinned_ready_wait_s,
                'h2d_submit_s': self.h2d_submit_s,
                'copy_ready_on_pop': self.copy_ready_on_pop,
                'copy_not_ready_on_pop': self.copy_not_ready_on_pop,
                'h2d_copy_ms_total': self.h2d_copy_ms_total,
                'h2d_copy_count': self.h2d_copy_count,
                'stage_batches': self.stage_batches,
                'reclaimed_slots': self.reclaimed_slots,
                'copy_wait_sync_s': self.copy_wait_sync_s,
                'slot_shape_mismatch_batches': self.slot_shape_mismatch_batches,
            }


class HandoffFillBlocked(RuntimeError):
    pass


def host_memory_stats_snapshot(*, enabled: bool, device: torch.device) -> dict | None:
    if not enabled or device.type != 'cuda':
        return None
    host_memory_stats = getattr(torch.cuda.memory, 'host_memory_stats', None)
    if not callable(host_memory_stats):
        return None
    try:
        return dict(host_memory_stats())
    except Exception:
        return None


def diff_host_mem(cur: dict | None, prev: dict | None) -> dict:
    if not cur or not prev:
        return {
            'host_num_alloc_delta': 0,
            'host_num_free_delta': 0,
            'host_alloc_time_us_delta': 0,
            'host_free_time_us_delta': 0,
            'host_active_bytes_cur': 0,
            'host_allocated_bytes_cur': 0,
        }
    return {
        'host_num_alloc_delta': int(cur.get('num_host_alloc', 0)) - int(prev.get('num_host_alloc', 0)),
        'host_num_free_delta': int(cur.get('num_host_free', 0)) - int(prev.get('num_host_free', 0)),
        'host_alloc_time_us_delta': int(cur.get('host_alloc_time.total', 0)) - int(prev.get('host_alloc_time.total', 0)),
        'host_free_time_us_delta': int(cur.get('host_free_time.total', 0)) - int(prev.get('host_free_time.total', 0)),
        'host_active_bytes_cur': int(cur.get('active_bytes.current', 0)),
        'host_allocated_bytes_cur': int(cur.get('allocated_bytes.current', 0)),
    }


def handoff_window_metrics(*, previous_snapshot: dict, current_snapshot: dict, elapsed_seconds: float) -> dict:
    elapsed = max(float(elapsed_seconds), 1e-9)

    def delta(key: str, *, cast=float):
        return max(
            cast(current_snapshot.get(key, 0)) - cast(previous_snapshot.get(key, 0)),
            cast(0),
        )

    copy_ready = delta('copy_ready_on_pop_total', cast=int)
    copy_not_ready = delta('copy_not_ready_on_pop_total', cast=int)
    h2d_copy_count = delta('h2d_copy_count_total', cast=int)
    h2d_copy_ms_total = delta('h2d_copy_ms_total', cast=float)
    return {
        'cpu_ready_wait_seconds': delta('cpu_ready_wait_s_total', cast=float),
        'cpu_ready_wait_fraction': min(max(delta('cpu_ready_wait_s_total', cast=float) / elapsed, 0.0), 1.0),
        'stage_free_slot_wait_seconds': delta('stage_free_slot_wait_s_total', cast=float),
        'stage_free_slot_wait_fraction': min(max(delta('stage_free_slot_wait_s_total', cast=float) / elapsed, 0.0), 1.0),
        'stage_copy_seconds': delta('stage_copy_s_total', cast=float),
        'stage_copy_fraction': min(max(delta('stage_copy_s_total', cast=float) / elapsed, 0.0), 1.0),
        'pinned_ready_wait_seconds': delta('pinned_ready_wait_s_total', cast=float),
        'pinned_ready_wait_fraction': min(max(delta('pinned_ready_wait_s_total', cast=float) / elapsed, 0.0), 1.0),
        'h2d_submit_seconds': delta('h2d_submit_s_total', cast=float),
        'h2d_submit_fraction': min(max(delta('h2d_submit_s_total', cast=float) / elapsed, 0.0), 1.0),
        'copy_ready_on_pop_fraction': (
            copy_ready / max(copy_ready + copy_not_ready, 1)
        ),
        'h2d_copy_ms_avg': h2d_copy_ms_total / max(h2d_copy_count, 1),
        'gpu_prefetch_depth': max(int(current_snapshot.get('gpu_prefetch_depth', 0)), 0),
        'free_handoff_slots_approx': max(int(current_snapshot.get('free_handoff_slots_approx', 0)), 0),
        'pinned_ready_q_approx': max(int(current_snapshot.get('pinned_ready_q_approx', 0)), 0),
        'copy_wait_sync_seconds': delta('copy_wait_sync_s_total', cast=float),
        'copy_wait_sync_fraction': min(max(delta('copy_wait_sync_s_total', cast=float) / elapsed, 0.0), 1.0),
        'handoff_stage_batches': delta('stage_batches_total', cast=int),
        'handoff_reclaimed_slots': delta('reclaimed_slots_total', cast=int),
        'handoff_slot_shape_mismatch_batches': delta('slot_shape_mismatch_batches_total', cast=int),
        **diff_host_mem(
            current_snapshot.get('host_memory_stats'),
            previous_snapshot.get('host_memory_stats'),
        ),
    }


class PinnedStageWorker:
    def __init__(
        self,
        *,
        next_pageable,
        slots,
        free_slot_q: queue.Queue,
        pinned_ready_q: queue.Queue,
        stop_evt: threading.Event,
        stats: HandoffStats,
        on_staged=None,
        poll_timeout_seconds: float = 0.1,
    ):
        self._next_pageable = next_pageable
        self._slots = slots
        self._free_slot_q = free_slot_q
        self._pinned_ready_q = pinned_ready_q
        self._stop_evt = stop_evt
        self._stats = stats
        self._on_staged = on_staged
        self._poll_timeout_seconds = max(float(poll_timeout_seconds), 1e-3)
        self.exc = None
        self._thread = threading.Thread(
            target=self._run,
            name='bc-pinned-stage',
            daemon=False,
        )

    def start(self):
        self._thread.start()
        return self

    def _acquire_free_slot(self) -> int:
        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            try:
                slot_id = self._free_slot_q.get(timeout=self._poll_timeout_seconds)
            except queue.Empty:
                continue
            self._stats.add(stage_free_slot_wait_s=time.perf_counter() - t0)
            return int(slot_id)
        raise StopIteration

    def _run(self):
        try:
            while not self._stop_evt.is_set():
                try:
                    ready = self._next_pageable()
                except StopIteration:
                    break
                slot_id = self._acquire_free_slot()
                t0 = time.perf_counter()
                stage_batch_into_slot(self._slots[slot_id], ready.batch)
                t1 = time.perf_counter()
                self._stats.add(stage_copy_s=(t1 - t0), stage_batches=1)
                item = PinnedReadyBatch(
                    slot_id=slot_id,
                    batch=self._slots[slot_id],
                    batch_idx=ready.batch_idx,
                    nsamples=ready.nsamples,
                    nbytes=ready.nbytes,
                    t_stage_start_s=t0,
                    t_stage_done_s=t1,
                )
                while not self._stop_evt.is_set():
                    try:
                        self._pinned_ready_q.put(item, timeout=self._poll_timeout_seconds)
                        if callable(self._on_staged):
                            self._on_staged(item)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            self.exc = exc
        finally:
            while True:
                try:
                    self._pinned_ready_q.put(_HANDOFF_END, timeout=self._poll_timeout_seconds)
                    break
                except queue.Full:
                    if self._stop_evt.is_set():
                        continue

    def close(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=10.0)


class DeviceBatchPrefetcher:
    def __init__(
        self,
        loader_iter=None,
        *,
        device: torch.device,
        oracle: bool,
        loader_stats=None,
        queue_depth: int = 1,
        startup_queue_depth: int | None = None,
        pin_handoff_batches: bool = False,
        handoff_stage_backend: str = 'inline',
        handoff_ring_slots: int = 2,
        handoff_measure_copy_wait_sync: bool = False,
        handoff_log_host_mem: bool = True,
    ):
        self.loader_iter = loader_iter
        self.device = device
        self.oracle = oracle
        self.loader_stats = loader_stats
        self.queue_depth = max(int(queue_depth or 1), 1)
        requested_startup_depth = self.queue_depth if startup_queue_depth is None else int(startup_queue_depth or 1)
        self.startup_queue_depth = min(max(requested_startup_depth, 1), self.queue_depth)
        self.pin_handoff_batches = bool(pin_handoff_batches) and device.type == 'cuda'
        self.handoff_stage_backend = str(handoff_stage_backend or 'inline')
        self.handoff_ring_slots = max(int(handoff_ring_slots or 2), 1)
        self.handoff_measure_copy_wait_sync = bool(handoff_measure_copy_wait_sync)
        self.handoff_log_host_mem = bool(handoff_log_host_mem)
        self.pinned_slot_count = self.handoff_ring_slots if self.pin_handoff_batches else 0
        self.stream = None
        if device.type == 'cuda':
            self.stream = torch.cuda.Stream(device=device)
        self.ready = deque()
        self.gpu_ready = collections.deque()
        self.slot_release = collections.deque()
        self._pinned_slots = []
        self._free_pinned_slots = deque()
        self.free_slot_q = None
        self.pinned_ready_q = None
        self.stage_worker = None
        self.stop_evt = threading.Event()
        self.handoff_stats = HandoffStats()
        self._handoff_lock = threading.Lock()
        self._occupied_slot_bytes = {}
        self._ephemeral_pinned_bytes = 0
        self._pinned_ready_batches = 0
        self._pinned_ready_bytes = 0
        self._next_pageable_batch_idx = 0
        self._pageable_source = None
        self._inline_seed_item = None
        self.debug_copy_wait_sync_s = 0.0
        self._host_mem_prev = None
        self.exhausted = False
        self._closed = False
        if loader_iter is not None:
            self._bind_pageable_source(loader_iter)
            self._fill_ready_queue(target_depth=self.startup_queue_depth)

    def _bind_pageable_source(self, source) -> None:
        self._pageable_source = source
        self.loader_iter = source

    def start(self, cpu_pipe):
        if self._pageable_source is not None:
            return self
        self._bind_pageable_source(cpu_pipe)
        if self.pin_handoff_batches and self.handoff_stage_backend == 'thread' and self.stream is not None:
            first = self._get_next_pageable()
            self._init_handoff_ring_from_first_batch(first)
            self.stage_worker = PinnedStageWorker(
                next_pageable=self._get_next_pageable,
                slots=self._pinned_slots,
                free_slot_q=self.free_slot_q,
                pinned_ready_q=self.pinned_ready_q,
                stop_evt=self.stop_evt,
                stats=self.handoff_stats,
                on_staged=self._on_pinned_ready_added,
            ).start()
            self._inline_seed_item = first
        self._fill_ready_queue(target_depth=self.startup_queue_depth)
        return self

    def _wrap_pageable_batch(self, batch, *, ready_time: float) -> PageableReadyBatch:
        item = PageableReadyBatch(
            batch=batch,
            batch_idx=self._next_pageable_batch_idx,
            nsamples=batch_nsamples(batch),
            nbytes=batch_nbytes(batch),
            t_cpu_ready_s=ready_time,
        )
        self._next_pageable_batch_idx += 1
        return item

    def _get_next_pageable(self) -> PageableReadyBatch:
        if self._inline_seed_item is not None:
            item = self._inline_seed_item
            self._inline_seed_item = None
            return item
        if self._pageable_source is None:
            raise StopIteration
        wait_started_at = time.perf_counter()
        batch = next(self._pageable_source)
        wait_seconds = time.perf_counter() - wait_started_at
        record_wait = getattr(self._pageable_source, 'record_consumer_wait', None)
        if callable(record_wait):
            record_wait(wait_seconds)
        self.handoff_stats.add(cpu_ready_wait_s=wait_seconds)
        return self._wrap_pageable_batch(batch, ready_time=time.perf_counter())

    def _initialize_pinned_slots(self, batch) -> None:
        self._pinned_slots = [
            allocate_staging_batch(batch, pin_memory=True)
            for _ in range(self.pinned_slot_count)
        ]
        self._free_pinned_slots = deque(range(len(self._pinned_slots)))

    def _init_handoff_ring_from_first_batch(self, first_ready: PageableReadyBatch) -> None:
        if not self.pin_handoff_batches or self.stream is None:
            return
        self._pinned_slots = [
            allocate_staging_batch(first_ready.batch, pin_memory=True)
            for _ in range(self.handoff_ring_slots)
        ]
        self._free_pinned_slots = deque(range(len(self._pinned_slots)))
        if self.handoff_stage_backend == 'thread':
            self.free_slot_q = queue.Queue(maxsize=len(self._pinned_slots))
            self.pinned_ready_q = queue.Queue(maxsize=len(self._pinned_slots))
            for slot_id in range(len(self._pinned_slots)):
                self.free_slot_q.put(slot_id)
        else:
            self.free_slot_q = None
            self.pinned_ready_q = None

    def _current_pinned_batch_bytes(self) -> int:
        with self._handoff_lock:
            return max(sum(self._occupied_slot_bytes.values()) + self._ephemeral_pinned_bytes, 0)

    def _refresh_loader_stats_queue_state(self) -> None:
        if self.loader_stats is None:
            return
        snapshot = self.loader_stats.snapshot()
        pinned_batch_bytes = self._current_pinned_batch_bytes()
        base_queued_bytes = max(
            int(snapshot.get('queued_bytes', 0)) - int(snapshot.get('pinned_batch_bytes', 0)),
            0,
        )
        self.loader_stats.update_queue_state(
            queued_bytes=base_queued_bytes + pinned_batch_bytes,
            ready_chunks=max(int(snapshot.get('ready_chunks', 0)), 0),
            pinned_batch_bytes=pinned_batch_bytes,
        )

    def _on_pinned_ready_added(self, item: PinnedReadyBatch) -> None:
        with self._handoff_lock:
            if item.slot_id is not None:
                self._occupied_slot_bytes[item.slot_id] = int(item.nbytes)
            else:
                self._ephemeral_pinned_bytes += int(item.nbytes)
            self._pinned_ready_batches += 1
            self._pinned_ready_bytes += int(item.nbytes)
        self._refresh_loader_stats_queue_state()

    def _on_pinned_ready_removed(self, item: PinnedReadyBatch) -> None:
        with self._handoff_lock:
            self._pinned_ready_batches = max(self._pinned_ready_batches - 1, 0)
            self._pinned_ready_bytes = max(self._pinned_ready_bytes - int(item.nbytes), 0)
        self._refresh_loader_stats_queue_state()

    def _on_slot_reclaimed(self, release: PendingSlotRelease) -> None:
        with self._handoff_lock:
            if release.slot_id is not None:
                self._occupied_slot_bytes.pop(release.slot_id, None)
            else:
                self._ephemeral_pinned_bytes = max(self._ephemeral_pinned_bytes - int(release.nbytes), 0)
        self._refresh_loader_stats_queue_state()

    def _bypass_ring_batch(self, ready: PageableReadyBatch) -> PinnedReadyBatch:
        pinned_batch = pin_batch_memory(ready.batch)
        self.handoff_stats.add(slot_shape_mismatch_batches=1)
        item = PinnedReadyBatch(
            slot_id=None,
            batch=pinned_batch,
            batch_idx=ready.batch_idx,
            nsamples=ready.nsamples,
            nbytes=ready.nbytes,
            t_stage_start_s=time.perf_counter(),
            t_stage_done_s=time.perf_counter(),
        )
        self._on_pinned_ready_added(item)
        return item

    def _get_next_pinned_ready(self):
        if not self.pin_handoff_batches or self.stream is None:
            ready = self._get_next_pageable()
            return PinnedReadyBatch(
                slot_id=None,
                batch=ready.batch,
                batch_idx=ready.batch_idx,
                nsamples=ready.nsamples,
                nbytes=ready.nbytes,
                t_stage_start_s=ready.t_cpu_ready_s,
                t_stage_done_s=ready.t_cpu_ready_s,
            )
        if self.handoff_stage_backend == 'thread':
            while not self.stop_evt.is_set():
                t0 = time.perf_counter()
                try:
                    item = self.pinned_ready_q.get(timeout=0.1)
                except queue.Empty:
                    self._reclaim_completed_slots()
                    continue
                self.handoff_stats.add(pinned_ready_wait_s=time.perf_counter() - t0)
                if item is _HANDOFF_END:
                    if self.stage_worker is not None and self.stage_worker.exc is not None:
                        raise self.stage_worker.exc
                    raise StopIteration
                self._on_pinned_ready_removed(item)
                return item
            raise StopIteration

        if not self._pinned_slots:
            ready = self._get_next_pageable()
            self._init_handoff_ring_from_first_batch(ready)
            if not self._pinned_slots:
                return self._bypass_ring_batch(ready)
        self._reclaim_completed_slots()
        if not self._free_pinned_slots:
            raise HandoffFillBlocked
        ready = self._get_next_pageable()
        if not batch_fits_staging_slot(self._pinned_slots[0], ready.batch):
            self.handoff_stats.add(slot_shape_mismatch_batches=1)
            return self._bypass_ring_batch(ready)
        slot_idx = self._free_pinned_slots.popleft()
        t0 = time.perf_counter()
        staged_batch = stage_batch_into_slot(self._pinned_slots[slot_idx], ready.batch)
        t1 = time.perf_counter()
        self.handoff_stats.add(stage_copy_s=(t1 - t0), stage_batches=1)
        item = PinnedReadyBatch(
            slot_id=slot_idx,
            batch=staged_batch,
            batch_idx=ready.batch_idx,
            nsamples=ready.nsamples,
            nbytes=ready.nbytes,
            t_stage_start_s=t0,
            t_stage_done_s=t1,
        )
        self._on_pinned_ready_added(item)
        return item

    def _reclaim_completed_slots(self) -> None:
        while self.slot_release and self.slot_release[0].copy_end_event.query():
            release = self.slot_release.popleft()
            if release.copy_start_event is not None:
                try:
                    copy_ms = float(release.copy_start_event.elapsed_time(release.copy_end_event))
                except Exception:
                    copy_ms = 0.0
                self.handoff_stats.add(h2d_copy_ms_total=copy_ms, h2d_copy_count=1)
            if release.slot_id is not None:
                if self.handoff_stage_backend == 'thread' and self.free_slot_q is not None:
                    try:
                        self.free_slot_q.put_nowait(release.slot_id)
                    except queue.Full:
                        pass
                else:
                    self._free_pinned_slots.append(release.slot_id)
            self.handoff_stats.add(reclaimed_slots=1)
            self._on_slot_reclaimed(release)

    def _launch_one_h2d(self, pinned: PinnedReadyBatch) -> None:
        submit_started_at = time.perf_counter()
        if self.stream is None:
            with torch.profiler.record_function('bc.h2d_submit'):
                self.gpu_ready.append(
                    GpuReadyBatch(
                        batch_idx=pinned.batch_idx,
                        gpu_batch=move_batch_to_device(pinned.batch, device=self.device, oracle=self.oracle),
                        nbytes=pinned.nbytes,
                        copy_end_event=None,
                        copy_start_event=None,
                    )
                )
            self.handoff_stats.add(h2d_submit_s=time.perf_counter() - submit_started_at)
            return
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        with torch.profiler.record_function('bc.h2d_submit'):
            with torch.cuda.stream(self.stream):
                copy_start.record(self.stream)
                gpu_batch = move_batch_to_device(pinned.batch, device=self.device, oracle=self.oracle)
                copy_end.record(self.stream)
        self.gpu_ready.append(
            GpuReadyBatch(
                batch_idx=pinned.batch_idx,
                gpu_batch=gpu_batch,
                nbytes=pinned.nbytes,
                copy_end_event=copy_end,
                copy_start_event=copy_start,
            )
        )
        self.handoff_stats.add(h2d_submit_s=time.perf_counter() - submit_started_at)
        self.slot_release.append(
            PendingSlotRelease(
                slot_id=pinned.slot_id,
                nbytes=pinned.nbytes,
                copy_end_event=copy_end,
                copy_start_event=copy_start,
            )
        )
        self._refresh_loader_stats_queue_state()

    def _fill_ready_queue(self, *, target_depth: int | None = None):
        desired_depth = self.queue_depth if target_depth is None else max(int(target_depth), 1)
        while not self.exhausted and len(self.gpu_ready) < desired_depth:
            self._reclaim_completed_slots()
            try:
                pinned = self._get_next_pinned_ready()
            except HandoffFillBlocked:
                break
            except StopIteration:
                self.exhausted = True
                break
            self._launch_one_h2d(pinned)
        self.ready = self.gpu_ready
        self._refresh_loader_stats_queue_state()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.gpu_ready:
            self._fill_ready_queue(target_depth=self.queue_depth)
            while (
                not self.gpu_ready
                and not self.exhausted
                and self.pin_handoff_batches
                and self.handoff_stage_backend == 'inline'
                and self.slot_release
            ):
                self.slot_release[0].copy_end_event.synchronize()
                self._reclaim_completed_slots()
                self._fill_ready_queue(target_depth=self.queue_depth)
            if not self.gpu_ready:
                raise StopIteration
        item = self.gpu_ready.popleft()
        if self.stream is not None:
            ready_now = item.copy_end_event.query()
            if ready_now:
                self.handoff_stats.add(copy_ready_on_pop=1)
            else:
                self.handoff_stats.add(copy_not_ready_on_pop=1)
            if self.handoff_measure_copy_wait_sync:
                t0 = time.perf_counter()
                item.copy_end_event.synchronize()
                waited = time.perf_counter() - t0
                self.debug_copy_wait_sync_s += waited
                self.handoff_stats.add(copy_wait_sync_s=waited)
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_event(item.copy_end_event)
            record_batch_stream(item.gpu_batch, current_stream)
        else:
            self.handoff_stats.add(copy_ready_on_pop=1)
        self._fill_ready_queue(target_depth=self.queue_depth)
        self.ready = self.gpu_ready
        self._refresh_loader_stats_queue_state()
        return item.gpu_batch

    def snapshot_handoff_state(self) -> dict:
        stats = self.handoff_stats.snapshot()
        current_host = host_memory_stats_snapshot(enabled=self.handoff_log_host_mem, device=self.device)
        if current_host is not None:
            self._host_mem_prev = current_host
        with self._handoff_lock:
            pinned_ready_batches = int(self._pinned_ready_batches)
            pinned_ready_bytes = int(self._pinned_ready_bytes)
        snapshot = {
            'cpu_ready_wait_s_total': float(stats['cpu_ready_wait_s']),
            'stage_free_slot_wait_s_total': float(stats['stage_free_slot_wait_s']),
            'stage_copy_s_total': float(stats['stage_copy_s']),
            'pinned_ready_wait_s_total': float(stats['pinned_ready_wait_s']),
            'h2d_submit_s_total': float(stats['h2d_submit_s']),
            'copy_ready_on_pop_total': int(stats['copy_ready_on_pop']),
            'copy_not_ready_on_pop_total': int(stats['copy_not_ready_on_pop']),
            'h2d_copy_ms_total': float(stats['h2d_copy_ms_total']),
            'h2d_copy_count_total': int(stats['h2d_copy_count']),
            'stage_batches_total': int(stats['stage_batches']),
            'reclaimed_slots_total': int(stats['reclaimed_slots']),
            'copy_wait_sync_s_total': float(stats['copy_wait_sync_s']),
            'slot_shape_mismatch_batches_total': int(stats['slot_shape_mismatch_batches']),
            'free_handoff_slots_approx': (
                int(self.free_slot_q.qsize()) if self.free_slot_q is not None else len(self._free_pinned_slots)
            ),
            'pinned_ready_q_approx': (
                int(self.pinned_ready_q.qsize()) if self.pinned_ready_q is not None else pinned_ready_batches
            ),
            'gpu_prefetch_depth': len(self.gpu_ready),
            'pinned_ready_batches': pinned_ready_batches,
            'pinned_ready_bytes': pinned_ready_bytes,
            'pinned_batch_bytes': self._current_pinned_batch_bytes(),
            'host_memory_stats': current_host,
        }
        self._refresh_loader_stats_queue_state()
        return snapshot

    def queue_depth_snapshot(self) -> dict:
        with self._handoff_lock:
            pinned_ready_batches = int(self._pinned_ready_batches)
            pinned_ready_bytes = int(self._pinned_ready_bytes)
        return {
            'gpu_prefetch_depth': len(self.gpu_ready),
            'pinned_ready_batches': pinned_ready_batches,
            'pinned_ready_bytes': pinned_ready_bytes,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_evt.set()
        if self.stage_worker is not None:
            close_source = getattr(self._pageable_source, 'close', None)
            if callable(close_source):
                close_source()
            self.stage_worker.close()
        if self.stream is not None:
            self.stream.synchronize()
        self.gpu_ready.clear()
        self.ready = self.gpu_ready
        self.slot_release.clear()
        close_fn = getattr(self.loader_iter, 'close', None)
        if callable(close_fn) and self.loader_iter is not self._pageable_source:
            close_fn()
        self._refresh_loader_stats_queue_state()


def append_metrics_event(metrics_jsonl_path: str, payload: dict) -> None:
    if not metrics_jsonl_path:
        return
    output_dir = os.path.dirname(metrics_jsonl_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(metrics_jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def preflight_windows_stable(*, windows: list[dict], required_windows: int, tolerance: float) -> bool:
    if required_windows <= 0:
        return False
    if len(windows) < required_windows:
        return False
    recent = windows[-required_windows:]
    for key in ('samples_per_second', 'wait_fraction'):
        values = [float(window[key]) for window in recent]
        baseline = sum(values) / len(values)
        scale = max(abs(baseline), 1e-9)
        if any(abs(value - baseline) / scale > tolerance for value in values):
            return False
    return True


def resolve_amp_dtype(control_cfg: dict) -> torch.dtype:
    raw = str(control_cfg.get('amp_dtype', 'float16')).lower()
    if raw in ('float16', 'fp16', 'half'):
        return torch.float16
    if raw in ('bfloat16', 'bf16'):
        return torch.bfloat16
    raise ValueError(f'unsupported bc.control.amp_dtype: {raw}')


def adamw_supports_fused() -> bool:
    return 'fused' in inspect.signature(torch.optim.AdamW).parameters


def resolve_fused_optimizer_enabled(*, optim_cfg: dict, device: torch.device) -> bool:
    return (
        bool(optim_cfg.get('enable_fused_optimizer', False))
        and device.type == 'cuda'
        and adamw_supports_fused()
    )


def grad_scaler_enabled(*, enable_amp: bool, amp_dtype: torch.dtype, device: torch.device) -> bool:
    return enable_amp and device.type == 'cuda' and amp_dtype == torch.float16


def apply_cuda_precision_settings(*, control_cfg: dict, device: torch.device) -> bool:
    enable_tf32 = bool(control_cfg.get('enable_tf32', False))
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high' if enable_tf32 else 'highest')
    return enable_tf32


def resolve_scheduler_config(*, optim_cfg: dict, max_steps: int) -> dict | None:
    scheduler_cfg = dict(optim_cfg.get('scheduler') or {})
    if not scheduler_cfg:
        return None

    peak = scheduler_cfg.get('peak', optim_cfg['lr'])
    final = scheduler_cfg.get('final', peak)
    schedule_max_steps = scheduler_cfg.get('max_steps', 0) or max_steps
    init = scheduler_cfg.get('init', 1e-8)
    offset = scheduler_cfg.get('offset', 0)
    epoch_size = scheduler_cfg.get('epoch_size', 0)
    warm_up_ratio = scheduler_cfg.get('warm_up_ratio')
    warm_up_fraction = scheduler_cfg.get('warm_up_fraction')
    warm_up_steps = scheduler_cfg.get('warm_up_steps')

    if schedule_max_steps <= 0:
        raise ValueError(
            'bc.optim.scheduler.max_steps must be positive, or bc.control.max_steps must be positive '
            'when the BC scheduler is enabled'
        )
    if warm_up_ratio is not None and warm_up_fraction is not None:
        raise ValueError('use only one of bc.optim.scheduler.warm_up_ratio or warm_up_fraction')
    if warm_up_steps is not None and (warm_up_ratio is not None or warm_up_fraction is not None):
        raise ValueError('use either bc.optim.scheduler.warm_up_steps or warm_up_ratio/warm_up_fraction, not both')
    if warm_up_ratio is None:
        warm_up_ratio = warm_up_fraction
    if warm_up_ratio is not None:
        if not 0 <= warm_up_ratio <= 1:
            raise ValueError('bc.optim.scheduler.warm_up_ratio must be between 0 and 1')
        warm_up_steps = math.ceil(schedule_max_steps * warm_up_ratio) if warm_up_ratio > 0 else 0
    if warm_up_steps is None:
        warm_up_steps = 0
    if warm_up_steps < 0:
        raise ValueError('bc.optim.scheduler.warm_up_steps must be non-negative')
    if warm_up_steps > schedule_max_steps:
        raise ValueError('bc.optim.scheduler.warm_up_steps must be <= bc.optim.scheduler.max_steps')

    return {
        'peak': peak,
        'final': final,
        'warm_up_steps': warm_up_steps,
        'warm_up_ratio': warm_up_ratio,
        'max_steps': schedule_max_steps,
        'init': init,
        'offset': offset,
        'epoch_size': epoch_size,
    }


def training_run_plan(*, steps: int, max_steps: int, save_every: int, best_eval_every: int) -> dict:
    if max_steps <= 0:
        return {
            'remaining_steps': None,
            'next_save_step': None,
            'next_best_eval_step': None,
            'remaining_save_windows': None,
            'remaining_best_evals': None,
        }

    remaining_steps = max(max_steps - steps, 0)
    next_best_eval_step = None
    remaining_best_evals = None
    if remaining_steps == 0:
        return {
            'remaining_steps': 0,
            'next_save_step': None,
            'next_best_eval_step': None,
            'remaining_save_windows': 0,
            'remaining_best_evals': 0 if best_eval_every > 0 else None,
        }

    next_save_step = min(((steps // save_every) + 1) * save_every, max_steps)
    if best_eval_every > 0:
        next_best_eval_step = min(((steps // best_eval_every) + 1) * best_eval_every, max_steps)
        remaining_best_evals = (remaining_steps + best_eval_every - 1) // best_eval_every
    return {
        'remaining_steps': remaining_steps,
        'next_save_step': next_save_step,
        'next_best_eval_step': next_best_eval_step,
        'remaining_save_windows': (remaining_steps + save_every - 1) // save_every,
        'remaining_best_evals': remaining_best_evals,
    }


def resolve_best_eval_every(*, control_cfg: dict, save_every: int) -> int:
    raw_value = control_cfg.get('best_eval_every')
    if raw_value is None:
        return save_every
    return int(raw_value)


def resolve_required_stage_splits(
    *,
    validation_enabled: bool,
    best_eval_every: int,
    best_eval_split: str,
) -> list[str]:
    splits = ['train']
    if validation_enabled:
        splits.append('val')
    if best_eval_every > 0 and best_eval_split not in splits:
        splits.append(best_eval_split)
    return splits


def device_memory_metrics(device: torch.device) -> dict:
    if device.type != 'cuda':
        return {}
    return {
        'max_allocated_gib': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        'max_reserved_gib': torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        'current_allocated_gib': torch.cuda.memory_allocated(device) / (1024 ** 3),
        'current_reserved_gib': torch.cuda.memory_reserved(device) / (1024 ** 3),
    }


def action_categories(actions: torch.Tensor) -> torch.Tensor:
    cats = torch.full_like(actions, -1)
    cats[actions <= 36] = 0
    cats[actions == 37] = 1
    cats[(38 <= actions) & (actions <= 40)] = 2
    cats[actions == 41] = 3
    cats[actions == 42] = 4
    cats[actions == 43] = 5
    cats[actions == 44] = 6
    cats[actions == 45] = 7
    if (cats < 0).any():
        unknown = actions[cats < 0].unique(sorted=True).tolist()
        raise ValueError(f'unknown action labels: {unknown}')
    return cats


def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~masks, -torch.inf)


def top_k_hits(logits: torch.Tensor, actions: torch.Tensor, top_k: int) -> torch.Tensor:
    k = min(top_k, logits.shape[-1])
    return logits.topk(k, dim=-1).indices.eq(actions.unsqueeze(-1)).any(dim=-1)


def empty_metric_sums() -> dict:
    return {
        'count': 0,
        'nll_sum': 0.0,
        'correct_sum': 0.0,
        'topk_sum': 0.0,
        'legal_sum': 0.0,
        'category_correct': [0.0] * len(ACTION_CATEGORY_NAMES),
        'category_total': [0.0] * len(ACTION_CATEGORY_NAMES),
    }


def update_metric_sums(
    sums: dict,
    *,
    loss: torch.Tensor,
    masked_pred: torch.Tensor,
    raw_pred: torch.Tensor,
    masked_scores: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    top_k: int,
) -> None:
    batch_size = actions.shape[0]
    device = actions.device
    idx = torch.arange(batch_size, device=device)
    cats = action_categories(actions)
    correct = masked_pred.eq(actions)
    legal = masks[idx, raw_pred]
    topk = top_k_hits(masked_scores, actions, top_k)

    sums['count'] += batch_size
    sums['nll_sum'] += loss.item() * batch_size
    sums['correct_sum'] += correct.to(torch.float64).sum().item()
    sums['topk_sum'] += topk.to(torch.float64).sum().item()
    sums['legal_sum'] += legal.to(torch.float64).sum().item()

    for cat_idx in range(len(ACTION_CATEGORY_NAMES)):
        cat_mask = cats == cat_idx
        if not cat_mask.any():
            continue
        sums['category_total'][cat_idx] += cat_mask.to(torch.float64).sum().item()
        sums['category_correct'][cat_idx] += correct[cat_mask].to(torch.float64).sum().item()


def finalize_metric_sums(sums: dict) -> dict:
    count = sums['count']
    if count == 0:
        raise ValueError('cannot finalize empty metric sums')

    category_accuracy = {}
    for cat_idx, name in enumerate(ACTION_CATEGORY_NAMES):
        total = sums['category_total'][cat_idx]
        if total > 0:
            category_accuracy[name] = sums['category_correct'][cat_idx] / total

    return {
        'count': count,
        'nll': sums['nll_sum'] / count,
        'accuracy': sums['correct_sum'] / count,
        'topk_accuracy': sums['topk_sum'] / count,
        'legal_rate': sums['legal_sum'] / count,
        'category_accuracy': category_accuracy,
    }


def extract_policy_features(brain_out):
    if isinstance(brain_out, tuple):
        return brain_out[0]
    return brain_out


def wandb_scalar_payload(
    *,
    top_k: int,
    train_metrics: dict,
    val_metrics: dict | None,
    runtime_metrics: dict,
    loader_metrics: dict | None,
    memory_metrics: dict,
    lr: float,
    steps: int,
    best_eval_split: str,
    best_eval_metrics: dict | None = None,
) -> dict:
    payload = {
        # train/ — is the model learning?
        'train/nll': train_metrics['nll'],
        'train/accuracy': train_metrics['accuracy'],
        f'train/top{top_k}_accuracy': train_metrics['topk_accuracy'],
        'train/legal_rate': train_metrics['legal_rate'],
        'train/lr': lr,
        # perf/ — is the GPU being used efficiently?
        'perf/samples_per_second': runtime_metrics['samples_per_second'],
        'perf/steps_per_second': runtime_metrics['steps_per_second'],
        'perf/fw_bw_opt_fraction': runtime_metrics.get('fw_bw_opt_fraction', 0.0),
    }
    for name, value in train_metrics.get('category_accuracy', {}).items():
        payload[f'train/acc_{name}'] = value
    # val/ — is it generalizing?
    if val_metrics is not None:
        payload['val/nll'] = val_metrics['nll']
        payload['val/accuracy'] = val_metrics['accuracy']
        payload[f'val/top{top_k}_accuracy'] = val_metrics['topk_accuracy']
        payload['val/legal_rate'] = val_metrics['legal_rate']
        for name, value in val_metrics.get('category_accuracy', {}).items():
            payload[f'val/acc_{name}'] = value
    if best_eval_metrics is not None:
        payload['val/best_accuracy'] = best_eval_metrics['accuracy']
        payload['val/best_nll'] = best_eval_metrics['nll']
    # perf/ memory
    if memory_metrics:
        payload['perf/gpu_mem_allocated_gib'] = memory_metrics.get('max_allocated_gib', 0.0)
        payload['perf/gpu_mem_reserved_gib'] = memory_metrics.get('max_reserved_gib', 0.0)
    # perf/ + pipeline/ — data pipeline health
    if loader_metrics is not None:
        payload['perf/loader_wait_fraction'] = loader_metrics['wait_fraction']
        payload['pipeline/cpu_pipe_wait'] = loader_metrics['cpu_pipe_wait_fraction']
        payload['pipeline/device_prefetch_wait'] = loader_metrics.get('device_prefetch_wait_fraction', 0.0)
        payload['pipeline/copy_ready_on_pop'] = loader_metrics.get('copy_ready_on_pop_fraction', 0.0)
        payload['pipeline/h2d_copy_ms'] = loader_metrics.get('h2d_copy_ms_avg', 0.0)
        payload['pipeline/cpu_ready_batches_avg'] = loader_metrics.get('cpu_ready_batches_avg', 0.0)
        payload['pipeline/chunk_build_seconds'] = loader_metrics.get('avg_chunk_build_seconds', 0.0)
    return payload


def wandb_train_only_payload(
    *,
    top_k: int,
    train_metrics: dict,
    runtime_metrics: dict,
    loader_metrics: dict | None,
    lr: float,
    steps: int,
) -> dict:
    """Higher-frequency live/ metrics logged between save windows."""
    payload = {
        'live/nll': train_metrics['nll'],
        'live/accuracy': train_metrics['accuracy'],
        'live/samples_per_second': runtime_metrics['samples_per_second'],
    }
    if loader_metrics is not None:
        payload['live/loader_wait_fraction'] = loader_metrics['wait_fraction']
    return payload


def current_learning_rate(optimizer) -> float:
    return float(optimizer.param_groups[0]['lr'])


def autocast_context_kwargs(*, device: torch.device, enable_amp: bool, amp_dtype: torch.dtype) -> dict:
    kwargs = {
        'device_type': device.type,
        'enabled': enable_amp,
    }
    if device.type == 'cuda':
        kwargs['dtype'] = amp_dtype
    return kwargs


def metric_sums_to_tensor(sums: dict, *, device: torch.device) -> torch.Tensor:
    values = [
        float(sums['count']),
        float(sums['nll_sum']),
        float(sums['correct_sum']),
        float(sums['topk_sum']),
        float(sums['legal_sum']),
        *[float(value) for value in sums['category_correct']],
        *[float(value) for value in sums['category_total']],
    ]
    return torch.tensor(values, dtype=torch.float64, device=device)


def tensor_to_metric_sums(payload: torch.Tensor) -> dict:
    names_len = len(ACTION_CATEGORY_NAMES)
    values = payload.tolist()
    return {
        'count': int(round(values[0])),
        'nll_sum': values[1],
        'correct_sum': values[2],
        'topk_sum': values[3],
        'legal_sum': values[4],
        'category_correct': values[5:5 + names_len],
        'category_total': values[5 + names_len:5 + names_len * 2],
    }


def synchronize_metric_sums(sums: dict, *, dist_ctx, device: torch.device) -> dict:
    if not dist_ctx.enabled:
        return dict(sums)
    payload = metric_sums_to_tensor(sums, device=device)
    torch.distributed.all_reduce(payload)
    return tensor_to_metric_sums(payload)


def reduce_max_scalar(value: float, *, dist_ctx, device: torch.device) -> float:
    if not dist_ctx.enabled:
        return float(value)
    payload = torch.tensor(float(value), dtype=torch.float64, device=device)
    torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.MAX)
    return float(payload.item())


def reduce_min_scalar(value: float, *, dist_ctx, device: torch.device) -> float:
    if not dist_ctx.enabled:
        return float(value)
    payload = torch.tensor(float(value), dtype=torch.float64, device=device)
    torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.MIN)
    return float(payload.item())


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def dqn_policy_outputs(dqn, phi: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked_scores = dqn(phi, masks)
    raw_logits = unwrap_model(dqn).action_logits(phi)
    return raw_logits, masked_scores


def validate_resume_fingerprint(*, state: dict, current_fingerprint: str) -> None:
    checkpoint_fingerprint = stored_config_fingerprint(state)
    if not checkpoint_fingerprint:
        raise ValueError(
            'existing BC checkpoint has no config fingerprint and no saved config; '
            'resume is unsafe, so start from a fresh state file'
        )
    if checkpoint_fingerprint != current_fingerprint:
        raise ValueError(
            'existing BC checkpoint config fingerprint does not match the active config; '
            'resume is unsafe, so use a fresh state file'
        )


def train():
    import prelude

    from contextlib import ExitStack
    import logging
    import time
    from os import path
    from glob import glob
    from datetime import datetime
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import parameter_count, filtered_trimmed_lines, load_path_list, tqdm
    from bc_dataset import (
        load_path_cache,
        normalize_file_list,
        resolve_actor_filter_map,
        save_path_cache,
    )
    from bc_ram_cache import (
        resolve_runtime_cache_settings,
        runtime_cache_enabled,
        runtime_cache_split_settings,
    )
    from bc_stage import (
        StagedShardIterableDataset,
        resolve_stage_settings,
        stage_enabled,
        stage_manifest_paths,
        stage_preload_budget_bytes,
    )
    from dataloader import (
        AsyncCpuBatchPipe,
        SyncCpuBatchPipe,
        build_action_file_dataloader,
        resolve_prefetch_budget_bytes,
    )
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN
    from config import config

    cfg = config['bc']
    control_cfg = cfg['control']
    dataset_cfg = cfg['dataset']
    runtime_cache_cfg = resolve_runtime_cache_settings(config)
    use_runtime_cache = runtime_cache_enabled(config)
    stage_cfg = resolve_stage_settings(config)
    use_staged_cache = stage_enabled(config) and not use_runtime_cache
    optim_cfg = cfg['optim']
    distributed_cfg = cfg.get('distributed', {})
    resnet_cfg = cfg.get('resnet', config['resnet'])
    wandb_cfg = cfg.get('wandb', {})

    root_control_cfg = config.get('control', {})
    version = control_cfg.get('version', root_control_cfg.get('version', 4))
    batch_size = control_cfg['batch_size']
    grad_accum_steps = control_cfg.get('grad_accum_steps', 1)
    save_every = control_cfg['save_every']
    val_steps = control_cfg['val_steps']
    max_steps = control_cfg.get('max_steps', 0)
    max_runtime_seconds = float(control_cfg.get('max_runtime_seconds', 0) or 0)
    top_k = control_cfg.get('top_k', 3)
    seed = int(control_cfg.get('seed', 0))
    train_log_every = control_cfg.get('train_log_every', save_every) or save_every
    val_log_every = int(control_cfg.get('val_log_every', 0) or 0)
    val_batch_size = control_cfg.get('val_batch_size', 0) or batch_size
    best_eval_split = control_cfg.get('best_eval_split', 'val')
    best_eval_every = resolve_best_eval_every(control_cfg=control_cfg, save_every=save_every)
    best_eval_batch_size = control_cfg.get('best_eval_batch_size', 0) or batch_size
    best_eval_max_batches = control_cfg.get('best_eval_max_batches', 0)
    current_config_fingerprint = config_fingerprint(config)
    metrics_jsonl = str(control_cfg.get('metrics_jsonl', '') or '')
    profiler_enabled = bool(control_cfg.get('profiler_enabled', False))
    profiler_wait_steps = max(int(control_cfg.get('profiler_wait_steps', train_log_every) or train_log_every), 0)
    profiler_warmup_steps = max(int(control_cfg.get('profiler_warmup_steps', 4) or 4), 0)
    profiler_active_steps = max(int(control_cfg.get('profiler_active_steps', 8) or 8), 1)
    profiler_repeat = max(int(control_cfg.get('profiler_repeat', 1) or 1), 1)
    profiler_record_shapes = bool(control_cfg.get('profiler_record_shapes', True))
    profiler_profile_memory = bool(control_cfg.get('profiler_profile_memory', False))
    profiler_with_stack = bool(control_cfg.get('profiler_with_stack', False))
    profiler_output_dir = str(control_cfg.get('profiler_output_dir', '') or '')
    dist_ctx = init_distributed_context(
        resolve_distributed_context(
            control_device=control_cfg['device'],
            distributed_cfg=distributed_cfg,
        )
    )
    device = dist_ctx.device
    if 'profiler_profile_memory' not in control_cfg:
        profiler_profile_memory = device.type == 'cuda'
    torch.backends.cudnn.benchmark = control_cfg['enable_cudnn_benchmark']
    enable_amp = control_cfg['enable_amp']
    amp_dtype = resolve_amp_dtype(control_cfg)
    enable_tf32 = apply_cuda_precision_settings(control_cfg=control_cfg, device=device)
    enable_compile = control_cfg['enable_compile']
    freeze_bn = control_cfg.get('freeze_bn', False)

    oracle = dataset_cfg.get('oracle', False)
    file_batch_size = dataset_cfg['file_batch_size']
    num_workers = dataset_cfg.get('num_workers', 0)
    eval_num_workers = dataset_cfg.get('eval_num_workers', 0)
    prefetch_chunks = dataset_cfg.get('prefetch_chunks', 0)
    eval_prefetch_chunks = dataset_cfg.get('eval_prefetch_chunks', prefetch_chunks)
    prefetch_strategy = str(dataset_cfg.get('prefetch_strategy', 'static_chunks') or 'static_chunks')
    prefetch_ram_budget_gib = float(dataset_cfg.get('prefetch_ram_budget_gib', 0) or 0)
    eval_prefetch_ram_budget_gib = float(
        dataset_cfg.get('eval_prefetch_ram_budget_gib', prefetch_ram_budget_gib) or 0
    )
    prefetch_target_chunk_gib = float(dataset_cfg.get('prefetch_target_chunk_gib', 0) or 0)
    prefetch_low_watermark = float(dataset_cfg.get('prefetch_low_watermark', 0.35))
    prefetch_high_watermark = float(dataset_cfg.get('prefetch_high_watermark', 0.85))
    prefetch_threads = int(dataset_cfg.get('prefetch_threads', 1) or 1)
    prebatched = bool(dataset_cfg.get('prebatched', False))
    prebatch_layout = str(dataset_cfg.get('prebatch_layout', 'chunk') or 'chunk')
    prebatch_shuffle_mode = str(dataset_cfg.get('prebatch_shuffle_mode', 'sample') or 'sample')
    prebatch_spill_across_chunks = bool(dataset_cfg.get('prebatch_spill_across_chunks', False))
    prefetch_out_of_order = bool(dataset_cfg.get('prefetch_out_of_order', False))
    prefetch_startup_file_batch_size = int(dataset_cfg.get('prefetch_startup_file_batch_size', 0) or 0)
    eval_prefetch_out_of_order = bool(
        dataset_cfg.get('eval_prefetch_out_of_order', prefetch_out_of_order)
    )
    eval_prefetch_startup_file_batch_size = int(
        dataset_cfg.get('eval_prefetch_startup_file_batch_size', prefetch_startup_file_batch_size) or 0
    )
    device_prefetch_batches = int(dataset_cfg.get('device_prefetch_batches', 2) or 2)
    device_prefetch_startup_batches = int(
        dataset_cfg.get('device_prefetch_startup_batches', min(device_prefetch_batches, 1)) or 1
    )
    eval_device_prefetch_batches = int(
        dataset_cfg.get('eval_device_prefetch_batches', device_prefetch_batches) or device_prefetch_batches
    )
    eval_device_prefetch_startup_batches = int(
        dataset_cfg.get('eval_device_prefetch_startup_batches', min(eval_device_prefetch_batches, 1)) or 1
    )
    pin_memory = dataset_cfg.get('pin_memory', device.type == 'cuda')
    eval_pin_memory = dataset_cfg.get('eval_pin_memory', pin_memory)
    handoff_pin_memory = bool(dataset_cfg.get('handoff_pin_memory', False))
    eval_handoff_pin_memory = bool(dataset_cfg.get('eval_handoff_pin_memory', handoff_pin_memory))
    handoff_stage_backend = str(dataset_cfg.get('handoff_stage_backend', 'inline') or 'inline')
    handoff_ring_slots = int(dataset_cfg.get('handoff_ring_slots', 2) or 2)
    handoff_measure_copy_wait_sync = bool(dataset_cfg.get('handoff_measure_copy_wait_sync', False))
    handoff_log_host_mem = bool(dataset_cfg.get('handoff_log_host_mem', True))
    cpu_batch_pipe_backend = str(dataset_cfg.get('cpu_batch_pipe_backend', 'sync') or 'sync')
    cpu_ready_batches = int(dataset_cfg.get('cpu_ready_batches', 4) or 4)
    cpu_ready_bytes_gib = float(dataset_cfg.get('cpu_ready_bytes_gib', 0) or 0)
    cpu_pipe_poll_timeout_seconds = float(dataset_cfg.get('cpu_pipe_poll_timeout_seconds', 0.1) or 0.1)
    raw_source_backend = str(dataset_cfg.get('raw_source_backend', 'files') or 'files')
    raw_pack_path = str(dataset_cfg.get('raw_pack_path', '') or '')
    raw_pack_index_path = str(dataset_cfg.get('raw_pack_index_path', '') or '')
    persistent_workers = bool(dataset_cfg.get('persistent_workers', False))
    raw_prefetch_factor = dataset_cfg.get('prefetch_factor')
    prefetch_factor = None if raw_prefetch_factor in (None, '') else int(raw_prefetch_factor)
    in_order = bool(dataset_cfg.get('in_order', True))
    loader_mode = str(dataset_cfg.get('loader_mode', 'baseline') or 'baseline')
    loader_block_target_samples = int(dataset_cfg.get('loader_block_target_samples', 65536) or 65536)
    loader_block_workers = int(dataset_cfg.get('loader_block_workers', 4) or 4)
    loader_block_slots = int(dataset_cfg.get('loader_block_slots', 8) or 8)
    loader_block_queue_timeout_s = float(dataset_cfg.get('loader_block_queue_timeout_s', 0.1) or 0.1)
    multiprocessing_context = dataset_cfg.get('multiprocessing_context', 'spawn')
    root_dir = dataset_cfg.get('root_dir', '')
    num_epochs = dataset_cfg.get('num_epochs', 1)
    step_count_summary_path = str(dataset_cfg.get('step_count_summary', '') or '')
    enable_augmentation = dataset_cfg.get('enable_augmentation', False)
    augmented_first = dataset_cfg.get('augmented_first', False)
    trust_seed = dataset_cfg.get('trust_seed', False)
    always_include_kan_select = dataset_cfg.get('always_include_kan_select', True)
    train_prefetch_budget_bytes = resolve_prefetch_budget_bytes(
        gib=prefetch_ram_budget_gib,
        world_size=dist_ctx.world_size,
    )
    cpu_ready_bytes_limit = max(int(cpu_ready_bytes_gib * (1024 ** 3)), 0)
    eval_prefetch_budget_bytes = resolve_prefetch_budget_bytes(
        gib=eval_prefetch_ram_budget_gib,
        world_size=1,
    )
    prefetch_target_chunk_bytes = max(int(prefetch_target_chunk_gib * (1024 ** 3)), 0)
    runtime_train_cache = (
        runtime_cache_split_settings(config, split_name='train', world_size=dist_ctx.world_size)
        if use_runtime_cache
        else {}
    )
    runtime_eval_cache = (
        runtime_cache_split_settings(config, split_name='val', world_size=1)
        if use_runtime_cache
        else {}
    )
    stage_preload_low_watermark = float(stage_cfg.get('preload_low_watermark', 0.65))
    stage_preload_high_watermark = float(stage_cfg.get('preload_high_watermark', 0.90))
    stage_preload_threads = int(stage_cfg.get('preload_threads', 4) or 4)
    train_stage_preload_budget_bytes = (
        stage_preload_budget_bytes(full_config=config, world_size=dist_ctx.world_size)
        if use_staged_cache
        else 0
    )
    eval_stage_preload_budget_bytes = (
        stage_preload_budget_bytes(full_config=config, world_size=1)
        if use_staged_cache
        else 0
    )
    preflight_cfg = dict(cfg.get('preflight') or {})
    preflight_enabled = bool(preflight_cfg.get('enabled', False))
    preflight_min_runtime_seconds = float(preflight_cfg.get('min_runtime_seconds', 0) or 0)
    preflight_min_steps_before_stop = int(
        preflight_cfg.get('min_steps_before_stop', 200 if preflight_enabled else 0) or 0
    )
    preflight_required_windows = int(preflight_cfg.get('required_stable_windows', 2) or 2)
    preflight_stability_tolerance = float(preflight_cfg.get('stability_tolerance', 0.05) or 0.05)
    validation_enabled = val_steps > 0
    best_eval_enabled = best_eval_every > 0
    cpu_affinity_profile = str(control_cfg.get('cpu_affinity_profile', 'none') or 'none')
    worker_torch_num_threads = int(control_cfg.get('worker_torch_num_threads', 1) or 1)
    worker_omp_num_threads = int(control_cfg.get('worker_omp_num_threads', 1) or 1)

    lr = optim_cfg['lr']
    eps = optim_cfg.get('eps', 1e-8)
    betas = tuple(optim_cfg.get('betas', (0.9, 0.999)))
    weight_decay = optim_cfg.get('weight_decay', 0.0)
    max_grad_norm = optim_cfg.get('max_grad_norm', 0.0)
    fused_optimizer_enabled = resolve_fused_optimizer_enabled(optim_cfg=optim_cfg, device=device)

    # Derive max_steps from epoch count + dataset stats when max_steps=0
    if max_steps <= 0 and step_count_summary_path and path.exists(step_count_summary_path):
        _pre_global_batch = effective_global_batch(
            batch_size=batch_size,
            world_size=dist_ctx.world_size,
            grad_accum_steps=grad_accum_steps,
        )
        with open(step_count_summary_path) as _f:
            _pre_stats = json.load(_f)
        _pre_train_steps = int(_pre_stats.get('splits', {}).get('train', {}).get('step_count', 0))
        if _pre_train_steps > 0 and _pre_global_batch > 0:
            _epoch_steps = _pre_train_steps // _pre_global_batch
            if _epoch_steps > 0 and num_epochs > 0:
                max_steps = _epoch_steps * num_epochs
                logging.info(
                    'auto max_steps: %s (%s epochs × %s steps/epoch, %s total samples, global_batch=%s)',
                    f'{max_steps:,}', num_epochs, f'{_epoch_steps:,}',
                    f'{_pre_train_steps:,}', f'{_pre_global_batch:,}',
                )

    scheduler_cfg = resolve_scheduler_config(optim_cfg=optim_cfg, max_steps=max_steps)

    # torch.compile + DDP: supported since PyTorch 2.0+; requires clean GPU state
    if grad_accum_steps <= 0:
        raise ValueError('bc.control.grad_accum_steps must be positive')
    if max_runtime_seconds < 0:
        raise ValueError('bc.control.max_runtime_seconds must be non-negative')
    if dist_ctx.enabled and batch_size <= 0:
        raise ValueError('bc.control.batch_size must be positive for distributed training')

    if save_every <= 0:
        raise ValueError('bc.control.save_every must be positive')
    if val_steps < 0:
        raise ValueError('bc.control.val_steps must be non-negative')
    assert top_k > 0
    if train_log_every <= 0:
        raise ValueError('bc.control.train_log_every must be positive')
    if val_log_every < 0:
        raise ValueError('bc.control.val_log_every must be non-negative')
    if val_log_every > 0 and val_log_every % train_log_every != 0:
        raise ValueError('bc.control.val_log_every must be a multiple of bc.control.train_log_every')
    if best_eval_every < 0:
        raise ValueError('bc.control.best_eval_every must be non-negative')
    if best_eval_every > 0 and best_eval_every % save_every != 0:
        raise ValueError('bc.control.best_eval_every must be a multiple of bc.control.save_every')
    if best_eval_split not in ('train', 'val', 'test'):
        raise ValueError('bc.control.best_eval_split must be one of train/val/test')
    if cpu_batch_pipe_backend not in ('sync', 'thread'):
        raise ValueError("bc.dataset.cpu_batch_pipe_backend must be 'sync' or 'thread'")
    if handoff_stage_backend not in ('inline', 'thread'):
        raise ValueError("bc.dataset.handoff_stage_backend must be 'inline' or 'thread'")
    if handoff_ring_slots <= 0:
        raise ValueError('bc.dataset.handoff_ring_slots must be positive')
    if cpu_ready_batches <= 0:
        raise ValueError('bc.dataset.cpu_ready_batches must be positive')
    if cpu_ready_bytes_gib < 0:
        raise ValueError('bc.dataset.cpu_ready_bytes_gib must be non-negative')
    if cpu_pipe_poll_timeout_seconds <= 0:
        raise ValueError('bc.dataset.cpu_pipe_poll_timeout_seconds must be positive')
    if raw_source_backend not in ('files', 'raw_pack'):
        raise ValueError("bc.dataset.raw_source_backend must be 'files' or 'raw_pack'")
    if raw_source_backend == 'raw_pack':
        if not raw_pack_path:
            raise ValueError('bc.dataset.raw_pack_path is required when raw_source_backend=raw_pack')
        if not raw_pack_index_path:
            raise ValueError(
                'bc.dataset.raw_pack_index_path is required when raw_source_backend=raw_pack'
            )
        if not Path(raw_pack_path).exists():
            raise ValueError(f'bc.dataset.raw_pack_path does not exist: {raw_pack_path}')
        if not Path(raw_pack_index_path).exists():
            raise ValueError(f'bc.dataset.raw_pack_index_path does not exist: {raw_pack_index_path}')
    if loader_mode not in ('baseline', 'preassembled_batches', 'batched_blocks'):
        raise ValueError(
            "bc.dataset.loader_mode must be 'baseline', 'preassembled_batches', or 'batched_blocks'"
        )
    if loader_mode == 'batched_blocks':
        raise ValueError(
            "bc.dataset.loader_mode='batched_blocks' is reserved for the Step 6 Phase 5 rewrite "
            'and is not implemented yet'
        )
    if loader_mode == 'preassembled_batches':
        if prebatched:
            raise ValueError(
                "bc.dataset.loader_mode='preassembled_batches' cannot be combined with prebatched=true"
            )
        if use_staged_cache:
            raise ValueError(
                "bc.dataset.loader_mode='preassembled_batches' cannot be combined with bc.stage.enabled=true"
            )
        if use_runtime_cache:
            raise ValueError(
                "bc.dataset.loader_mode='preassembled_batches' cannot be combined with "
                'bc.runtime_cache.enabled=true'
            )
        if num_workers != 0 or eval_num_workers != 0:
            raise ValueError(
                "bc.dataset.loader_mode='preassembled_batches' currently requires "
                'num_workers=0 and eval_num_workers=0'
            )
    if prefetch_factor is not None and prefetch_factor <= 0:
        raise ValueError('bc.dataset.prefetch_factor must be positive when set')
    if loader_block_target_samples <= 0:
        raise ValueError('bc.dataset.loader_block_target_samples must be positive')
    if loader_block_workers <= 0:
        raise ValueError('bc.dataset.loader_block_workers must be positive')
    if loader_block_slots <= 0:
        raise ValueError('bc.dataset.loader_block_slots must be positive')
    if loader_block_queue_timeout_s <= 0:
        raise ValueError('bc.dataset.loader_block_queue_timeout_s must be positive')
    if cpu_affinity_profile not in ('none', 'dual_rank_physical_split_v1'):
        raise ValueError(
            "bc.control.cpu_affinity_profile must be 'none' or 'dual_rank_physical_split_v1'"
        )
    if cpu_affinity_profile != 'none':
        raise ValueError(
            f"bc.control.cpu_affinity_profile={cpu_affinity_profile!r} is reserved for the Step 6 Phase 6 "
            'affinity work and is not implemented yet'
        )
    if worker_torch_num_threads <= 0:
        raise ValueError('bc.control.worker_torch_num_threads must be positive')
    if worker_omp_num_threads <= 0:
        raise ValueError('bc.control.worker_omp_num_threads must be positive')
    if preflight_enabled and validation_enabled:
        logging.warning('preflight is enabled but bc.control.val_steps > 0; validation windows will still run')
    effective_seed = seed_everything(seed, rank=dist_ctx.rank)
    if dist_ctx.enabled and not dist_ctx.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)
    global_batch_size = effective_global_batch(
        batch_size=batch_size,
        world_size=dist_ctx.world_size,
        grad_accum_steps=grad_accum_steps,
    )

    # Load dataset statistics for epoch progress tracking
    dataset_stats = None
    total_train_samples = 0
    epoch_steps = 0
    if step_count_summary_path and path.exists(step_count_summary_path):
        with open(step_count_summary_path) as f:
            dataset_stats = json.load(f)
        train_split_stats = dataset_stats.get('splits', {}).get('train', {})
        total_train_samples = int(train_split_stats.get('step_count', 0))
        if total_train_samples > 0 and global_batch_size > 0:
            epoch_steps = total_train_samples // global_batch_size

    mortal = Brain(version=version, is_oracle=oracle, **resnet_cfg).to(device)
    dqn = DQN(version=version, hidden_dim=mortal.hidden_dim).to(device)
    if enable_compile:
        if dist_ctx.enabled:
            torch._dynamo.config.optimize_ddp = True
        mortal.compile()
        dqn.compile()
    mortal.freeze_bn(freeze_bn)
    base_mortal = mortal
    base_dqn = dqn

    logging.info(f'version: {version}')
    logging.info(f'oracle: {oracle}')
    logging.info(f'mortal params: {parameter_count(base_mortal):,}')
    logging.info(f'dqn params: {parameter_count(base_dqn):,}')
    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')
    logging.info(
        'distributed: enabled=%s backend=%s world_size=%s rank=%s local_rank=%s',
        dist_ctx.enabled,
        dist_ctx.backend,
        dist_ctx.world_size,
        dist_ctx.rank,
        dist_ctx.local_rank,
    )
    logging.info(
        'precision: amp=%s amp_dtype=%s grad_scaler=%s tf32=%s compile=%s fused_adamw=%s',
        enable_amp,
        str(amp_dtype).replace('torch.', ''),
        grad_scaler_enabled(enable_amp=enable_amp, amp_dtype=amp_dtype, device=device),
        enable_tf32,
        enable_compile,
        fused_optimizer_enabled,
    )
    logging.info(
        'batching: per_rank_batch=%s world_size=%s grad_accum_steps=%s effective_global_batch=%s seed=%s effective_seed=%s',
        f'{batch_size:,}',
        dist_ctx.world_size,
        grad_accum_steps,
        f'{global_batch_size:,}',
        seed,
        effective_seed,
    )
    logging.info(
        'loader: file_batch_size=%s num_workers=%s eval_num_workers=%s prefetch_strategy=%s prefetch_chunks=%s '
        'eval_prefetch_chunks=%s prefetch_budget_gib=%s eval_prefetch_budget_gib=%s target_chunk_gib=%s '
        'prefetch_threads=%s prebatched=%s prebatch_layout=%s prebatch_shuffle_mode=%s '
        'prebatch_spill_across_chunks=%s '
        'startup_file_batch_size=%s eval_startup_file_batch_size=%s '
        'prefetch_out_of_order=%s eval_prefetch_out_of_order=%s '
        'pin_memory=%s eval_pin_memory=%s handoff_pin_memory=%s eval_handoff_pin_memory=%s '
        'handoff_stage_backend=%s handoff_ring_slots=%s handoff_measure_copy_wait_sync=%s handoff_log_host_mem=%s '
        'cpu_batch_pipe_backend=%s cpu_ready_batches=%s cpu_ready_bytes_gib=%.2f cpu_pipe_poll_timeout_s=%.3f '
        'raw_source_backend=%s loader_mode=%s persistent_workers=%s prefetch_factor=%s in_order=%s '
        'device_prefetch_batches=%s '
        'device_prefetch_startup_batches=%s eval_device_prefetch_batches=%s '
        'eval_device_prefetch_startup_batches=%s',
        runtime_train_cache.get('max_files_per_chunk', file_batch_size) if use_runtime_cache else file_batch_size,
        num_workers,
        eval_num_workers,
        runtime_cache_cfg.get('mode', 'prepared_ram') if use_runtime_cache else prefetch_strategy,
        0 if use_runtime_cache else prefetch_chunks,
        0 if use_runtime_cache else eval_prefetch_chunks,
        runtime_train_cache.get('data_budget_bytes', 0) / (1024 ** 3) if use_runtime_cache else prefetch_ram_budget_gib,
        runtime_eval_cache.get('data_budget_bytes', 0) / (1024 ** 3) if use_runtime_cache else eval_prefetch_ram_budget_gib,
        runtime_train_cache.get('target_chunk_bytes', 0) / (1024 ** 3) if use_runtime_cache else prefetch_target_chunk_gib,
        runtime_train_cache.get('max_inflight_chunk_builders', 1) if use_runtime_cache else prefetch_threads,
        prebatched,
        prebatch_layout,
        prebatch_shuffle_mode,
        prebatch_spill_across_chunks,
        runtime_train_cache.get('min_files_per_chunk', prefetch_startup_file_batch_size) if use_runtime_cache else prefetch_startup_file_batch_size,
        runtime_eval_cache.get('min_files_per_chunk', eval_prefetch_startup_file_batch_size) if use_runtime_cache else eval_prefetch_startup_file_batch_size,
        prefetch_out_of_order,
        eval_prefetch_out_of_order,
        pin_memory,
        eval_pin_memory,
        handoff_pin_memory,
        eval_handoff_pin_memory,
        handoff_stage_backend,
        handoff_ring_slots,
        handoff_measure_copy_wait_sync,
        handoff_log_host_mem,
        cpu_batch_pipe_backend,
        cpu_ready_batches,
        cpu_ready_bytes_gib,
        cpu_pipe_poll_timeout_seconds,
        raw_source_backend,
        loader_mode,
        persistent_workers,
        prefetch_factor,
        in_order,
        device_prefetch_batches,
        device_prefetch_startup_batches,
        eval_device_prefetch_batches,
        eval_device_prefetch_startup_batches,
    )
    if use_runtime_cache and stage_enabled(config):
        logging.warning('bc.runtime_cache.enabled=true takes precedence over bc.stage.enabled=true; disk staging is disabled')
    if use_runtime_cache:
        logging.info(
            'runtime cache: mode=%s node_ram_budget_gib=%s node_pinned_budget_gib=%s node_inflight_budget_gib=%s '
            'raw_lru_budget_gib=%s low_watermark=%.2f high_watermark=%.2f target_chunk_gib=%.2f max_chunk_gib=%.2f '
            'decode_threads=%s producer_threads=%s startup_ready_chunks=%s max_inflight_chunk_builders=%s '
            'min_files_per_chunk=%s max_files_per_chunk=%s '
            'train_ready_budget_gib=%.2f train_inflight_budget_gib=%.2f eval_ready_budget_gib=%.2f',
            runtime_cache_cfg['mode'],
            runtime_cache_cfg['node_ram_budget_gib'],
            runtime_cache_cfg['node_pinned_budget_gib'],
            runtime_cache_cfg['node_inflight_budget_gib'],
            runtime_cache_cfg['raw_lru_budget_gib'],
            float(runtime_cache_cfg['low_watermark']),
            float(runtime_cache_cfg['high_watermark']),
            float(runtime_cache_cfg['target_chunk_gib']),
            float(runtime_cache_cfg['max_chunk_gib']),
            int(runtime_cache_cfg['decode_threads']),
            1,
            int(runtime_cache_cfg['startup_ready_chunks']),
            int(runtime_cache_cfg['max_inflight_chunk_builders']),
            int(runtime_cache_cfg['min_files_per_chunk']),
            int(runtime_cache_cfg['max_files_per_chunk']),
            float(runtime_train_cache.get('ready_budget_bytes', 0)) / (1024 ** 3),
            float(runtime_train_cache.get('inflight_budget_bytes', 0)) / (1024 ** 3),
            float(runtime_eval_cache.get('ready_budget_bytes', 0)) / (1024 ** 3),
        )
        if int(runtime_cache_cfg.get('producer_threads', 1) or 1) != 1:
            logging.info(
                'runtime cache: producer_threads=%s requested, clamped to 1 coordinator thread in v1',
                int(runtime_cache_cfg.get('producer_threads', 1) or 1),
            )
    if use_staged_cache:
        logging.info(
            'stage cache: backend=%s cache_root=%s preload_ram_budget_gib=%s preload_low_watermark=%.2f '
            'preload_high_watermark=%.2f preload_threads=%s required_splits=%s',
            stage_cfg['backend'],
            stage_cfg['cache_root'],
            stage_cfg['preload_ram_budget_gib'],
            stage_preload_low_watermark,
            stage_preload_high_watermark,
            stage_preload_threads,
            ','.join(stage_cfg['required_splits']),
        )

    decay_params = []
    no_decay_params = []
    for model in (base_mortal, base_dqn):
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    optimizer_kwargs = {
        'lr': 1.0 if scheduler_cfg is not None else lr,
        'weight_decay': 0.0,
        'betas': betas,
        'eps': eps,
    }
    if fused_optimizer_enabled:
        optimizer_kwargs['fused'] = True
    optimizer = optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params},
        ],
        **optimizer_kwargs,
    )
    scheduler = None
    if scheduler_cfg is not None:
        scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **scheduler_cfg)
    scaler = GradScaler(
        device.type,
        enabled=grad_scaler_enabled(enable_amp=enable_amp, amp_dtype=amp_dtype, device=device),
    )

    state_file = control_cfg['state_file']
    best_state_file = control_cfg['best_state_file']
    full_eval_best_state_file = str(control_cfg.get('full_eval_best_state_file', '') or '')
    checkpoint_dir = str(control_cfg.get('checkpoint_dir', '') or '')
    checkpoint_keep_recent = int(control_cfg.get('checkpoint_keep_recent', 10) or 10)
    stage_save_dir = str(control_cfg.get('stage_save_dir', '') or '')
    best_perf = normalize_best_perf(None, best_eval_split)
    best_full_eval_perf = normalize_best_perf(None, best_eval_split)
    steps = 0
    state = None
    runtime_seconds_total = 0.0
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        validate_resume_fingerprint(
            state=state,
            current_fingerprint=current_config_fingerprint,
        )
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        base_mortal.load_state_dict(state['mortal'])
        base_dqn.load_state_dict(state['current_dqn'])
        optimizer.load_state_dict(state['optimizer'])
        if 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        steps = state.get('steps', 0)
        best_perf = normalize_best_perf(state.get('best_perf', {}), best_eval_split)
        best_full_eval_perf = normalize_best_perf(state.get('best_full_eval_perf', {}), best_eval_split)
        runtime_seconds_total = float(state.get('runtime_seconds_total', 0.0) or 0.0)
        if scheduler is not None:
            if 'scheduler' not in state:
                raise ValueError(
                    'bc.optim.scheduler is enabled, but the checkpoint has no scheduler state. '
                    'Use a fresh state file or disable the scheduler for this resume.'
                )
            scheduler.load_state_dict(state['scheduler'])
    if dist_ctx.enabled:
        ddp_kwargs = {
            'broadcast_buffers': bool(distributed_cfg.get('broadcast_buffers', False)),
            'find_unused_parameters': bool(distributed_cfg.get('find_unused_parameters', False)),
            'gradient_as_bucket_view': bool(distributed_cfg.get('gradient_as_bucket_view', True)),
        }
        if device.type == 'cuda':
            ddp_kwargs['device_ids'] = [dist_ctx.local_rank]
            ddp_kwargs['output_device'] = dist_ctx.local_rank
        if 'static_graph' in inspect.signature(DistributedDataParallel).parameters:
            ddp_kwargs['static_graph'] = bool(distributed_cfg.get('static_graph', False))
        mortal = DistributedDataParallel(base_mortal, **ddp_kwargs)
        dqn = DistributedDataParallel(base_dqn, **ddp_kwargs)

    run_plan = training_run_plan(
        steps=steps,
        max_steps=max_steps,
        save_every=save_every,
        best_eval_every=best_eval_every,
    )
    if scheduler_cfg is not None:
        logging.info(
            'scheduler: linear-warmup-cosine peak=%s final=%s warm_up_steps=%s max_steps=%s init=%s',
            scheduler_cfg['peak'],
            scheduler_cfg['final'],
            scheduler_cfg['warm_up_steps'],
            scheduler_cfg['max_steps'],
            scheduler_cfg['init'],
        )
    else:
        logging.info('scheduler: disabled (constant learning rate)')
    if max_steps > 0:
        logging.info(
            'run plan: current_step=%s target_max_steps=%s remaining_steps=%s save_every=%s sampled_val_every=%s val_log_every=%s '
            'best_eval_every=%s train_log_every=%s next_save_step=%s next_best_eval_step=%s remaining_save_windows=%s '
            'remaining_best_evals=%s',
            f'{steps:,}',
            f'{max_steps:,}',
            f"{run_plan['remaining_steps']:,}",
            f'{save_every:,}',
            f'{save_every:,}',
            f'{val_log_every:,}' if val_log_every > 0 else 'disabled',
            f'{best_eval_every:,}',
            f'{train_log_every:,}',
            f"{run_plan['next_save_step']:,}" if run_plan['next_save_step'] is not None else 'n/a',
            f"{run_plan['next_best_eval_step']:,}" if run_plan['next_best_eval_step'] is not None else 'n/a',
            f"{run_plan['remaining_save_windows']:,}",
            f"{run_plan['remaining_best_evals']:,}" if run_plan['remaining_best_evals'] is not None else 'n/a',
        )
    else:
        logging.info(
            'run plan: current_step=%s target_max_steps=unbounded save_every=%s sampled_val_every=%s val_log_every=%s '
            'best_eval_every=%s train_log_every=%s runtime_cap_seconds=%s',
            f'{steps:,}',
            f'{save_every:,}',
            f'{save_every:,}',
            f'{val_log_every:,}' if val_log_every > 0 else 'disabled',
            f'{best_eval_every:,}',
            f'{train_log_every:,}',
            f'{max_runtime_seconds:.1f}' if max_runtime_seconds > 0 else 'unbounded',
        )

    if dataset_stats is not None:
        train_split_stats = dataset_stats['splits'].get('train', {})
        logging.info(
            'dataset: source=%s total_train_samples=%s trajectories=%s files=%s '
            'epoch_steps=%s avg_steps_per_trajectory=%.1f avg_steps_per_file=%.1f',
            step_count_summary_path,
            f'{total_train_samples:,}',
            f"{int(train_split_stats.get('trajectory_count', 0)):,}",
            f"{int(train_split_stats.get('file_count', 0)):,}",
            f'{epoch_steps:,}' if epoch_steps > 0 else 'n/a',
            float(train_split_stats.get('avg_steps_per_trajectory', 0)),
            float(train_split_stats.get('avg_steps_per_file', 0)),
        )
        for split_name in ('val', 'test'):
            split_data = dataset_stats['splits'].get(split_name, {})
            if split_data:
                logging.info(
                    'dataset (%s): samples=%s trajectories=%s files=%s',
                    split_name,
                    f"{int(split_data.get('step_count', 0)):,}",
                    f"{int(split_data.get('trajectory_count', 0)):,}",
                    f"{int(split_data.get('file_count', 0)):,}",
                )
    elif step_count_summary_path:
        logging.warning('dataset stats: %s not found, epoch progress tracking disabled', step_count_summary_path)

    logging.info(
        'checkpointing: save_every=%s keep_recent=%s checkpoint_dir=%s '
        'val_best=%s full_eval_best=%s stage_dir=%s stage_every=%s',
        save_every,
        checkpoint_keep_recent,
        checkpoint_dir or '(disabled)',
        best_state_file,
        full_eval_best_state_file or '(disabled)',
        stage_save_dir or '(disabled)',
        f'{epoch_steps} (1 epoch)' if epoch_steps > 0 and stage_save_dir else '(disabled)',
    )

    def load_name_filters(raw_paths):
        names = set()
        for filename in raw_paths:
            with open(filename, encoding='utf-8') as f:
                names.update(filtered_trimmed_lines(f))
        return sorted(names)

    player_names = load_name_filters(dataset_cfg.get('player_names_files', []))
    excludes = load_name_filters(dataset_cfg.get('exclude_names_files', []))

    train_list = dataset_cfg.get('train_list', '')
    val_list = dataset_cfg.get('val_list', '')
    test_list = dataset_cfg.get('test_list', '')
    train_globs = dataset_cfg.get('train_globs', [])
    val_globs = dataset_cfg.get('val_globs', [])
    file_index = dataset_cfg.get('file_index', '')
    path_cache = dataset_cfg.get('path_cache', '')
    cached_split_lists = {}
    lists_are_normalized = False
    if train_list or val_list:
        if not train_list or not val_list:
            raise ValueError('bc.dataset.train_list and bc.dataset.val_list must be set together')
        if path_cache and path.exists(path_cache):
            requested_cache_splits = ['train', 'val'] + (['test'] if test_list else [])
            cached_split_lists = load_path_cache(
                path_cache,
                expected_splits=requested_cache_splits,
                expected_sources={
                    'train': train_list,
                    'val': val_list,
                    **({'test': test_list} if test_list else {}),
                },
            )
            train_file_list = cached_split_lists['train']
            val_file_list = cached_split_lists['val']
            lists_are_normalized = True
            logging.info(f'loaded normalized path cache from {path_cache}')
        else:
            train_file_list = load_path_list(train_list, root_dir)
            val_file_list = load_path_list(val_list, root_dir)
            logging.info(f'loaded train list from {train_list}')
            logging.info(f'loaded val list from {val_list}')
    else:
        if not file_index:
            raise ValueError('bc.dataset.file_index is required when train/val file lists are not provided')
        if path.exists(file_index):
            index = torch.load(file_index, weights_only=True)
            train_file_list = index['train_file_list']
            val_file_list = index['val_file_list']
        else:
            logging.info('building file index...')
            train_file_list = []
            val_file_list = []
            for pat in train_globs:
                train_file_list.extend(glob(pat, recursive=True))
            for pat in val_globs:
                val_file_list.extend(glob(pat, recursive=True))
            train_file_list.sort(reverse=True)
            val_file_list.sort(reverse=True)
            atomic_torch_save(
                {
                    'train_file_list': train_file_list,
                    'val_file_list': val_file_list,
                },
                file_index,
            )

    def load_split_file_list(split_name):
        if split_name in cached_split_lists:
            return cached_split_lists[split_name]
        if split_name == 'train':
            return train_file_list
        if split_name == 'val':
            return val_file_list
        if split_name == 'test':
            test_globs = dataset_cfg.get('test_globs', [])
            if test_list:
                logging.info(f'loaded test list from {test_list}')
                return load_path_list(test_list, root_dir)
            test_file_list = []
            for pat in test_globs:
                test_file_list.extend(glob(pat, recursive=True))
            test_file_list.sort(reverse=True)
            if test_file_list:
                return test_file_list
            raise ValueError('bc.dataset.test_list or bc.dataset.test_globs is required for best_eval_split=test')
        raise ValueError(f'unexpected split name: {split_name}')

    if not lists_are_normalized:
        train_file_list = normalize_file_list(train_file_list, desc='PATHS-TRAIN')
        val_file_list = normalize_file_list(val_file_list, desc='PATHS-VAL')
        cached_split_lists['train'] = train_file_list
        cached_split_lists['val'] = val_file_list
        if path_cache:
            if test_list:
                cached_split_lists['test'] = normalize_file_list(
                    load_path_list(test_list, root_dir),
                    desc='PATHS-TEST',
                )
            save_path_cache(
                path_cache,
                split_lists=cached_split_lists,
                source_files={
                    'train': train_list,
                    'val': val_list,
                    **({'test': test_list} if test_list else {}),
                },
            )
            logging.info(f'saved normalized path cache to {path_cache}')
    if not best_eval_enabled:
        if best_eval_split == 'train':
            best_eval_file_list = train_file_list
        elif best_eval_split == 'val':
            best_eval_file_list = val_file_list
        else:
            best_eval_file_list = []
    elif best_eval_split == 'train':
        best_eval_file_list = train_file_list
    elif best_eval_split == 'val':
        best_eval_file_list = val_file_list
    else:
        loaded_best_eval_files = load_split_file_list(best_eval_split)
        if best_eval_split in cached_split_lists:
            best_eval_file_list = loaded_best_eval_files
        else:
            best_eval_file_list = normalize_file_list(
                loaded_best_eval_files,
                desc=f'PATHS-{best_eval_split.upper()}',
            )

    stage_manifest_map = {}
    if use_staged_cache:
        required_stage_splits = resolve_required_stage_splits(
            validation_enabled=validation_enabled,
            best_eval_every=best_eval_every,
            best_eval_split=best_eval_split,
        )
        stage_manifest_map = stage_manifest_paths(
            config,
            splits=required_stage_splits,
        )
        missing_stage_manifests = {
            split: manifest_path
            for split, manifest_path in stage_manifest_map.items()
            if not manifest_path.exists()
        }
        if missing_stage_manifests:
            missing_desc = ', '.join(
                f'{split}={manifest_path}'
                for split, manifest_path in missing_stage_manifests.items()
            )
            raise FileNotFoundError(
                'missing staged BC shard manifest(s); run scripts/stage_bc_tensor_shards.py first: '
                f'{missing_desc}'
            )

    local_train_file_list = shard_file_list_round_robin(
        train_file_list,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
    )
    if not local_train_file_list:
        raise ValueError(
            f'local train shard for rank {dist_ctx.rank} is empty; '
            f'train_file_count={len(train_file_list)} world_size={dist_ctx.world_size}'
        )

    logging.info(f'train file list size: {len(train_file_list):,}')
    logging.info(f'local train shard size: {len(local_train_file_list):,}')
    logging.info(f'val file list size: {len(val_file_list):,}')
    logging.info(f'best eval split: {best_eval_split} ({len(best_eval_file_list):,} files)')

    actor_filter_map = None
    min_actor_dan = dataset_cfg.get('min_actor_dan')
    actor_filter_manifest = dataset_cfg.get('actor_filter_manifest', '')
    actor_filter_index = dataset_cfg.get('actor_filter_index', '')
    if min_actor_dan is not None and not use_staged_cache:
        actor_filter_map, actor_filter_summary = resolve_actor_filter_map(
            file_lists=[train_file_list, val_file_list, best_eval_file_list],
            min_actor_dan=min_actor_dan,
            actor_filter_manifest=actor_filter_manifest,
            actor_filter_index=actor_filter_index,
            inputs_are_normalized=True,
        )
        logging.info(
            'actor dan filter enabled: source=%s min_actor_dan=%s matched_files=%s eligible_files=%s filtered_out_files=%s',
            actor_filter_summary.get('source', 'unknown'),
            actor_filter_summary['min_actor_dan'],
            f"{actor_filter_summary['matched_row_count']:,}",
            f"{actor_filter_summary['eligible_file_count']:,}",
            f"{actor_filter_summary['filtered_out_file_count']:,}",
        )
        local_eligible_train_files = sum(
            1 for filename in local_train_file_list
            if actor_filter_map.get(filename)
        )
        logging.info(
            'local eligible train files after actor filter: %s / %s',
            f'{local_eligible_train_files:,}',
            f'{len(local_train_file_list):,}',
        )
    elif min_actor_dan is not None and use_staged_cache:
        logging.info(
            'actor dan filter is embedded in the staged cache: min_actor_dan=%s manifest=%s index=%s',
            min_actor_dan,
            actor_filter_manifest or 'n/a',
            actor_filter_index or 'n/a',
        )

    def make_loader(
        file_list,
        *,
        split_name,
        batch_size_override,
        num_epochs_override,
        enable_augmentation_override,
        augmented_first_override,
        cycle,
        shuffle,
        num_workers_override,
        pin_memory_override,
    ):
        if use_staged_cache and split_name in stage_manifest_map:
            if num_workers_override > 0 and dist_ctx.is_main_process:
                logging.warning(
                    'stage loader ignores num_workers=%s and always uses in-process iteration',
                    num_workers_override,
                )
            data = StagedShardIterableDataset(
                manifest_path=stage_manifest_map[split_name],
                batch_size=batch_size_override,
                shuffle=shuffle,
                cycle=cycle,
                num_epochs=num_epochs_override,
                preload_budget_bytes=(
                    train_stage_preload_budget_bytes if split_name == 'train'
                    else eval_stage_preload_budget_bytes
                ),
                preload_low_watermark=stage_preload_low_watermark,
                preload_high_watermark=stage_preload_high_watermark,
                preload_threads=stage_preload_threads,
                rank=dist_ctx.rank if split_name == 'train' else 0,
                world_size=dist_ctx.world_size if split_name == 'train' else 1,
            )
            loader = DataLoader(
                dataset=data,
                batch_size=None,
                drop_last=False,
                num_workers=0,
                pin_memory=pin_memory_override,
            )
            loader.loader_stats = data.loader_stats
            return loader
        active_runtime_cache = runtime_train_cache if use_runtime_cache and cycle else runtime_eval_cache if use_runtime_cache else {}
        active_file_batch_size = (
            int(active_runtime_cache.get('max_files_per_chunk', file_batch_size))
            if use_runtime_cache
            else file_batch_size
        )
        active_prefetch_strategy = runtime_cache_cfg['mode'] if use_runtime_cache else prefetch_strategy
        active_prefetch_budget_bytes = (
            int(active_runtime_cache.get('data_budget_bytes', 0))
            if use_runtime_cache
            else (eval_prefetch_budget_bytes if not cycle else train_prefetch_budget_bytes)
        )
        active_prefetch_target_chunk_bytes = (
            int(active_runtime_cache.get('target_chunk_bytes', 0))
            if use_runtime_cache
            else prefetch_target_chunk_bytes
        )
        active_prefetch_low_watermark = (
            float(active_runtime_cache.get('low_watermark', prefetch_low_watermark))
            if use_runtime_cache
            else prefetch_low_watermark
        )
        active_prefetch_high_watermark = (
            float(active_runtime_cache.get('high_watermark', prefetch_high_watermark))
            if use_runtime_cache
            else prefetch_high_watermark
        )
        active_prefetch_threads = (
            int(active_runtime_cache.get('max_inflight_chunk_builders', 1))
            if use_runtime_cache
            else prefetch_threads
        )
        active_decode_threads = (
            int(active_runtime_cache.get('decode_threads', 1))
            if use_runtime_cache
            else 1
        )
        active_prefetch_startup_file_batch_size = (
            int(active_runtime_cache.get('min_files_per_chunk', prefetch_startup_file_batch_size))
            if use_runtime_cache
            else (eval_prefetch_startup_file_batch_size if not cycle else prefetch_startup_file_batch_size)
        )
        active_prefetch_startup_ready_chunks = (
            int(active_runtime_cache.get('startup_ready_chunks', 1))
            if use_runtime_cache
            else 1
        )
        active_prefetch_inflight_budget_bytes = (
            int(active_runtime_cache.get('inflight_budget_bytes', 0))
            if use_runtime_cache
            else 0
        )
        active_prefetch_ready_budget_bytes = (
            int(active_runtime_cache.get('ready_budget_bytes', 0))
            if use_runtime_cache
            else 0
        )
        active_prefetch_max_inflight_chunks = (
            int(active_runtime_cache.get('max_inflight_chunk_builders', 1))
            if use_runtime_cache
            else max(int(prefetch_threads or 1), 1)
        )
        active_prefetch_min_file_batch_size = (
            int(active_runtime_cache.get('min_files_per_chunk', 1))
            if use_runtime_cache
            else 1
        )
        active_prefetch_raw_lru_budget_bytes = (
            int(active_runtime_cache.get('raw_lru_budget_bytes', 0))
            if use_runtime_cache
            else 0
        )
        loader, _loader_stats = build_action_file_dataloader(
            version=version,
            file_list=file_list,
            oracle=oracle,
            file_batch_size=active_file_batch_size,
            player_names=player_names or None,
            excludes=excludes or None,
            num_epochs=num_epochs_override,
            enable_augmentation=enable_augmentation_override,
            augmented_first=augmented_first_override,
            trust_seed=trust_seed,
            always_include_kan_select=always_include_kan_select,
            cycle=cycle,
            shuffle=shuffle,
            allowed_player_ids_by_path=actor_filter_map,
            prefetch_chunks=0 if use_runtime_cache else (eval_prefetch_chunks if not cycle else prefetch_chunks),
            prefetch_strategy=active_prefetch_strategy,
            prefetch_budget_bytes=active_prefetch_budget_bytes,
            prefetch_target_chunk_bytes=active_prefetch_target_chunk_bytes,
            prefetch_low_watermark=active_prefetch_low_watermark,
            prefetch_high_watermark=active_prefetch_high_watermark,
            prefetch_threads=active_prefetch_threads,
            decode_threads=active_decode_threads,
            batch_size=batch_size_override,
            prebatched=prebatched,
            prebatch_layout=prebatch_layout,
            prebatch_shuffle_mode=prebatch_shuffle_mode,
            prebatch_spill_across_chunks=prebatch_spill_across_chunks,
            prefetch_out_of_order=eval_prefetch_out_of_order if not cycle else prefetch_out_of_order,
            prefetch_startup_file_batch_size=active_prefetch_startup_file_batch_size,
            prefetch_startup_ready_chunks=active_prefetch_startup_ready_chunks,
            prefetch_inflight_budget_bytes=active_prefetch_inflight_budget_bytes,
            prefetch_ready_budget_bytes=active_prefetch_ready_budget_bytes,
            prefetch_max_inflight_chunks=active_prefetch_max_inflight_chunks,
            prefetch_min_file_batch_size=active_prefetch_min_file_batch_size,
            prefetch_raw_lru_budget_bytes=active_prefetch_raw_lru_budget_bytes,
            num_workers=num_workers_override,
            pin_memory=pin_memory_override,
            multiprocessing_context=multiprocessing_context,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            in_order=in_order,
            raw_source_backend=raw_source_backend,
            raw_pack_path=raw_pack_path,
            raw_pack_index_path=raw_pack_index_path,
            loader_mode=loader_mode,
            loader_block_target_samples=loader_block_target_samples,
        )
        return loader

    def build_cpu_batch_pipe(loader_obj):
        loader_stats = getattr(loader_obj, 'loader_stats', None)
        if cpu_batch_pipe_backend == 'sync':
            return SyncCpuBatchPipe(
                lambda: iter(loader_obj),
                loader_stats=loader_stats,
            ).start()
        if getattr(loader_obj, 'num_workers', 0) > 0:
            logging.info(
                'cpu_batch_pipe_backend=thread with num_workers=%d: '
                'the producer thread will iterate the multi-worker DataLoader. '
                'Batch stream order will differ from num_workers=0 (acceptable for BC training).',
                getattr(loader_obj, 'num_workers', 0),
            )
        return AsyncCpuBatchPipe(
            make_iter=lambda: iter(loader_obj),
            max_ready_batches=cpu_ready_batches,
            max_ready_bytes=cpu_ready_bytes_limit,
            loader_stats=loader_stats,
            poll_timeout_seconds=cpu_pipe_poll_timeout_seconds,
        ).start()

    writer = None
    wandb_run = None
    train_loader = None
    train_cpu_pipe = None
    profiler = None

    def forward_batch(batch):
        if oracle:
            obs, invisible_obs, actions, masks = batch
        else:
            obs, actions, masks = batch
            invisible_obs = None
        with torch.autocast(**autocast_context_kwargs(device=device, enable_amp=enable_amp, amp_dtype=amp_dtype)):
            brain_out = mortal(obs, invisible_obs)
            phi = extract_policy_features(brain_out)
            raw_logits, masked_scores = dqn_policy_outputs(dqn, phi, masks)
            loss = nn.functional.cross_entropy(masked_scores, actions)
        return loss, raw_logits, masked_scores, actions, masks

    def evaluate_split(file_list, *, max_batches, desc, shuffle, batch_size_override):
        stats = empty_metric_sums()
        eval_mortal = unwrap_model(mortal)
        eval_dqn = unwrap_model(dqn)
        eval_mortal.eval()
        eval_dqn.eval()
        with torch.inference_mode():
            loader = make_loader(
                file_list,
                split_name=desc.split('-', 1)[-1].lower() if desc.startswith('BEST-') else desc.lower(),
                batch_size_override=batch_size_override,
                num_epochs_override=1,
                enable_augmentation_override=False,
                augmented_first_override=False,
                cycle=False,
                shuffle=shuffle,
                num_workers_override=eval_num_workers,
                pin_memory_override=eval_pin_memory,
            )
            if dist_ctx.is_main_process:
                logging.info(
                    'loader priming: building initial %s batches on %s '
                    '(startup_queue_depth=%s full_queue_depth=%s strategy=%s startup_files=%s target_chunk_gib=%.2f '
                    'runtime_cache_enabled=%s stage_enabled=%s)',
                    desc,
                    device,
                    eval_device_prefetch_startup_batches,
                    eval_device_prefetch_batches,
                    runtime_cache_cfg['mode'] if use_runtime_cache else prefetch_strategy,
                    runtime_eval_cache.get('min_files_per_chunk', eval_prefetch_startup_file_batch_size) if use_runtime_cache else eval_prefetch_startup_file_batch_size,
                    runtime_eval_cache.get('target_chunk_bytes', 0) / (1024 ** 3) if use_runtime_cache else prefetch_target_chunk_gib,
                    use_runtime_cache,
                    use_staged_cache,
                )
            eval_loader_prime_started_at = time.perf_counter()
            pb = tqdm(
                total=max_batches if max_batches > 0 else None,
                desc=desc,
                disable=not dist_ctx.is_main_process,
            )
            eval_iter = DeviceBatchPrefetcher(
                iter(loader),
                device=device,
                oracle=oracle,
                queue_depth=eval_device_prefetch_batches,
                startup_queue_depth=eval_device_prefetch_startup_batches,
                pin_handoff_batches=eval_handoff_pin_memory,
            )
            try:
                if dist_ctx.is_main_process:
                    eval_loader_stats = getattr(loader, 'loader_stats', None)
                    eval_loader_snapshot = (
                        eval_loader_stats.snapshot()
                        if eval_loader_stats is not None
                        else {}
                    )
                    logging.info(
                        'loader priming: %s batches ready in %.2fs queued_gib=%.2f ready_gib=%.2f inflight_gib=%.2f '
                        'ready_chunks=%s discovered_files=%s submitted_files=%s last_chunk_files=%s last_chunk_samples=%s',
                        desc,
                        time.perf_counter() - eval_loader_prime_started_at,
                        float(eval_loader_snapshot.get('queued_bytes', 0)) / (1024 ** 3),
                        float(eval_loader_snapshot.get('ready_bytes', 0)) / (1024 ** 3),
                        float(eval_loader_snapshot.get('inflight_bytes', 0)) / (1024 ** 3),
                        int(eval_loader_snapshot.get('ready_chunks', 0)),
                        int(eval_loader_snapshot.get('discovered_files', 0)),
                        int(eval_loader_snapshot.get('submitted_files', 0)),
                        int(eval_loader_snapshot.get('last_chunk_files', 0)),
                        int(eval_loader_snapshot.get('last_chunk_samples', 0)),
                    )
                for idx, batch in enumerate(eval_iter):
                    if max_batches > 0 and idx == max_batches:
                        break
                    if oracle:
                        obs, invisible_obs, actions, masks = batch
                    else:
                        obs, actions, masks = batch
                        invisible_obs = None
                    with torch.autocast(
                        **autocast_context_kwargs(device=device, enable_amp=enable_amp, amp_dtype=amp_dtype)
                    ):
                        brain_out = eval_mortal(obs, invisible_obs)
                        phi = extract_policy_features(brain_out)
                        raw_logits, masked_scores = dqn_policy_outputs(eval_dqn, phi, masks)
                        loss = nn.functional.cross_entropy(masked_scores, actions)
                    masked_pred = masked_scores.argmax(dim=-1)
                    raw_pred = raw_logits.argmax(dim=-1)
                    update_metric_sums(
                        stats,
                        loss=loss,
                        masked_pred=masked_pred,
                        raw_pred=raw_pred,
                        masked_scores=masked_scores,
                        actions=actions,
                        masks=masks,
                        top_k=top_k,
                    )
                    pb.update(1)
            finally:
                eval_iter.close()
            pb.close()
        eval_mortal.train()
        eval_dqn.train()
        return finalize_metric_sums(stats)

    def build_state(
        *,
        train_metrics,
        val_metrics,
        runtime_metrics,
        loader_metrics,
        memory_metrics,
        best_eval_metrics=None,
    ):
        payload = {
            'mortal': unwrap_model(mortal).state_dict(),
            'current_dqn': unwrap_model(dqn).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            'timestamp': datetime.now().timestamp(),
            'best_perf': best_perf,
            'best_full_eval_perf': best_full_eval_perf,
            'config': config,
            'config_fingerprint': current_config_fingerprint,
            'trainer': 'behavior_cloning',
            'seed': seed,
            'effective_seed': effective_seed,
            'grad_accum_steps': grad_accum_steps,
            'world_size': dist_ctx.world_size,
            'effective_global_batch': global_batch_size,
            'runtime_seconds_total': runtime_metrics['runtime_seconds_total'],
            'metrics': {
                'train': train_metrics,
                'runtime': runtime_metrics,
                'loader': loader_metrics,
                'memory': memory_metrics,
            },
        }
        if val_metrics is not None:
            payload['metrics']['sample_val'] = val_metrics
        if best_eval_metrics is not None:
            payload['metrics']['best_eval'] = best_eval_metrics
        if scheduler is not None:
            payload['scheduler'] = scheduler.state_dict()
        if wandb_run is not None:
            payload['wandb_run_id'] = wandb_run.id
        return payload

    try:
        if dist_ctx.is_main_process:
            writer = SummaryWriter(control_cfg['tensorboard_dir'])
            wandb_run = maybe_init_wandb_run(
                full_config=config,
                wandb_cfg=wandb_cfg,
                fallback_name=default_wandb_run_name(),
                job_type='train',
                run_id=(state or {}).get('wandb_run_id', ''),
            )
            if wandb_run is not None:
                # Only store non-config summary keys that are useful for
                # comparing runs at a glance.  Everything else is already
                # in wandb.config via full_config.
                wandb_run.summary['mortal_params'] = parameter_count(base_mortal)
                wandb_run.summary['dqn_params'] = parameter_count(base_dqn)
                if epoch_steps > 0:
                    wandb_run.summary['epoch_total_samples'] = total_train_samples
                    wandb_run.summary['epoch_total_steps'] = epoch_steps

        if profiler_enabled:
            if not profiler_output_dir:
                profiler_output_dir = os.path.join(control_cfg['tensorboard_dir'], 'torch_profiler')
            os.makedirs(profiler_output_dir, exist_ok=True)
            profiler_activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == 'cuda':
                profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(
                activities=profiler_activities,
                schedule=torch.profiler.schedule(
                    wait=profiler_wait_steps,
                    warmup=profiler_warmup_steps,
                    active=profiler_active_steps,
                    repeat=profiler_repeat,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profiler_output_dir,
                    worker_name=f'rank{dist_ctx.rank}',
                ),
                record_shapes=profiler_record_shapes,
                profile_memory=profiler_profile_memory,
                with_stack=profiler_with_stack,
            )
            profiler.__enter__()
            if dist_ctx.is_main_process:
                logging.info(
                    'torch profiler enabled: output_dir=%s wait=%s warmup=%s active=%s repeat=%s record_shapes=%s profile_memory=%s with_stack=%s',
                    profiler_output_dir,
                    profiler_wait_steps,
                    profiler_warmup_steps,
                    profiler_active_steps,
                    profiler_repeat,
                    profiler_record_shapes,
                    profiler_profile_memory,
                    profiler_with_stack,
                )

        train_loader_obj = make_loader(
            local_train_file_list,
            split_name='train',
            batch_size_override=batch_size,
            num_epochs_override=num_epochs,
            enable_augmentation_override=enable_augmentation,
            augmented_first_override=augmented_first,
            cycle=True,
            shuffle=True,
            num_workers_override=num_workers,
            pin_memory_override=pin_memory,
        )
        train_loader_stats = getattr(train_loader_obj, 'loader_stats', None)
        train_cpu_pipe = build_cpu_batch_pipe(train_loader_obj)
        if dist_ctx.is_main_process:
            logging.info(
                'loader priming: building initial train batches on %s '
                '(startup_queue_depth=%s full_queue_depth=%s strategy=%s startup_files=%s startup_ready_chunks=%s target_chunk_gib=%.2f '
                'cpu_batch_pipe_backend=%s cpu_ready_batches=%s cpu_ready_bytes_gib=%.2f '
                'runtime_cache_enabled=%s stage_enabled=%s)',
                device,
                device_prefetch_startup_batches,
                device_prefetch_batches,
                runtime_cache_cfg['mode'] if use_runtime_cache else prefetch_strategy,
                runtime_train_cache.get('min_files_per_chunk', prefetch_startup_file_batch_size) if use_runtime_cache else prefetch_startup_file_batch_size,
                runtime_train_cache.get('startup_ready_chunks', 1) if use_runtime_cache else 1,
                runtime_train_cache.get('target_chunk_bytes', 0) / (1024 ** 3) if use_runtime_cache else prefetch_target_chunk_gib,
                cpu_batch_pipe_backend,
                cpu_ready_batches,
                cpu_ready_bytes_gib,
                use_runtime_cache,
                use_staged_cache,
            )
        train_loader_prime_started_at = time.perf_counter()
        train_loader = DeviceBatchPrefetcher(
            device=device,
            oracle=oracle,
            loader_stats=train_loader_stats,
            queue_depth=device_prefetch_batches,
            startup_queue_depth=device_prefetch_startup_batches,
            pin_handoff_batches=handoff_pin_memory,
            handoff_stage_backend=handoff_stage_backend,
            handoff_ring_slots=handoff_ring_slots,
            handoff_measure_copy_wait_sync=handoff_measure_copy_wait_sync,
            handoff_log_host_mem=handoff_log_host_mem,
        )
        train_loader.start(train_cpu_pipe)
        train_loader_prime_seconds = time.perf_counter() - train_loader_prime_started_at
        train_handoff_prime_snapshot = train_loader.snapshot_handoff_state()
        if dist_ctx.is_main_process:
            train_loader_prime_snapshot = (
                train_loader_stats.snapshot()
                if train_loader_stats is not None
                else {}
            )
            logging.info(
                'loader priming: train batches ready in %.2fs queued_gib=%.2f ready_gib=%.2f inflight_gib=%.2f '
                'pinned_gib=%.2f cpu_ready_batches=%s cpu_ready_gib=%.2f '
                'ready_chunks=%s prefill_complete=%s blocked=%s discovered_files=%s submitted_files=%s '
                'last_chunk_files=%s last_chunk_samples=%s',
                train_loader_prime_seconds,
                float(train_loader_prime_snapshot.get('queued_bytes', 0)) / (1024 ** 3),
                float(train_loader_prime_snapshot.get('ready_bytes', 0)) / (1024 ** 3),
                float(train_loader_prime_snapshot.get('inflight_bytes', 0)) / (1024 ** 3),
                float(train_loader_prime_snapshot.get('pinned_batch_bytes', 0)) / (1024 ** 3),
                int(train_loader_prime_snapshot.get('cpu_ready_batches', 0)),
                float(train_loader_prime_snapshot.get('cpu_ready_bytes', 0)) / (1024 ** 3),
                int(train_loader_prime_snapshot.get('ready_chunks', 0)),
                bool(train_loader_prime_snapshot.get('prefill_complete', False)),
                str(train_loader_prime_snapshot.get('producer_blocked_reason', '')),
                int(train_loader_prime_snapshot.get('discovered_files', 0)),
                int(train_loader_prime_snapshot.get('submitted_files', 0)),
                int(train_loader_prime_snapshot.get('last_chunk_files', 0)),
                int(train_loader_prime_snapshot.get('last_chunk_samples', 0)),
            )
            append_metrics_event(
                metrics_jsonl,
                {
                    'event': 'loader_priming',
                    'split': 'train',
                    'startup_seconds': train_loader_prime_seconds,
                    'loader_snapshot': train_loader_prime_snapshot,
                    'handoff_snapshot': train_handoff_prime_snapshot,
                    'runtime_cache_enabled': use_runtime_cache,
                    'stage_enabled': use_staged_cache,
                },
            )

        if max_steps > 0 and steps >= max_steps:
            logging.info(f'max_steps={max_steps} already reached at step {steps}, nothing to do')
            distributed_barrier(dist_ctx)
            return

        train_stats = empty_metric_sums()
        train_live_stats = empty_metric_sums()
        last_saved_steps = steps
        window_start_time = time.perf_counter()
        train_live_window_start = window_start_time
        runtime_started_at = time.perf_counter()
        window_wait_seconds = 0.0
        train_live_wait_seconds = 0.0
        window_observability = empty_window_observability()
        train_live_observability = empty_window_observability()
        train_live_last_steps = steps
        window_loader_snapshot = train_loader_stats.snapshot() if train_loader_stats is not None else {}
        window_handoff_snapshot = dict(train_handoff_prime_snapshot)
        train_live_loader_snapshot = dict(window_loader_snapshot)
        train_live_handoff_snapshot = dict(window_handoff_snapshot)
        preflight_windows = deque(maxlen=max(preflight_required_windows, 8))
        preflight_should_stop = False
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        progress_total = save_every if max_steps <= 0 else min(save_every, max(max_steps - steps, 0))
        if epoch_steps > 0:
            current_epoch = steps // epoch_steps + 1
            within_epoch_pct = (steps % epoch_steps) / epoch_steps * 100
            epoch_desc = f'TRAIN [ep{current_epoch} {within_epoch_pct:.0f}%]'
        else:
            epoch_desc = 'TRAIN'
        pb = tqdm(
            total=max(progress_total, 1),
            desc=epoch_desc,
            disable=dist_ctx.enabled and not dist_ctx.is_main_process,
        )
        try:
            while True:
                step_started_at = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                fw_bw_opt_started_at = time.perf_counter()
                with torch.profiler.record_function('bc.forward_backward_optimizer'):
                    for micro_step in range(grad_accum_steps):
                        batch_wait_started = time.perf_counter()
                        batch = next(train_loader)
                        batch_wait_seconds = time.perf_counter() - batch_wait_started
                        window_wait_seconds += batch_wait_seconds
                        train_live_wait_seconds += batch_wait_seconds
                        sync_grads = (micro_step + 1) == grad_accum_steps or not dist_ctx.enabled
                        with ExitStack() as stack:
                            if not sync_grads:
                                stack.enter_context(mortal.no_sync())
                                stack.enter_context(dqn.no_sync())
                            loss, raw_logits, masked_scores, actions, masks = forward_batch(batch)
                            scaler.scale(loss / grad_accum_steps).backward()

                        with torch.inference_mode():
                            masked_pred = masked_scores.argmax(dim=-1)
                            raw_pred = raw_logits.argmax(dim=-1)
                            update_metric_sums(
                                train_stats,
                                loss=loss,
                                masked_pred=masked_pred,
                                raw_pred=raw_pred,
                                masked_scores=masked_scores,
                                actions=actions,
                                masks=masks,
                                top_k=top_k,
                            )
                            update_metric_sums(
                                train_live_stats,
                                loss=loss,
                                masked_pred=masked_pred,
                                raw_pred=raw_pred,
                                masked_scores=masked_scores,
                                actions=actions,
                                masks=masks,
                                top_k=top_k,
                            )

                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(
                            list(unwrap_model(mortal).parameters()) + list(unwrap_model(dqn).parameters()),
                            max_grad_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                fw_bw_opt_seconds = time.perf_counter() - fw_bw_opt_started_at
                window_observability['fw_bw_opt_seconds'] += fw_bw_opt_seconds
                train_live_observability['fw_bw_opt_seconds'] += fw_bw_opt_seconds

                steps += 1
                pb.update(1)

                reached_step_limit = max_steps > 0 and steps >= max_steps
                reached_runtime_limit = (
                    max_runtime_seconds > 0
                    and (runtime_seconds_total + (time.perf_counter() - runtime_started_at)) >= max_runtime_seconds
                )
                if dist_ctx.enabled:
                    stop_flags = torch.tensor(
                        [int(reached_step_limit), int(reached_runtime_limit), int(preflight_should_stop)],
                        dtype=torch.int32,
                        device=device,
                    )
                    ddp_wait_started_at = time.perf_counter()
                    with torch.profiler.record_function('bc.ddp_collective'):
                        torch.distributed.all_reduce(stop_flags, op=torch.distributed.ReduceOp.MAX)
                    ddp_wait_seconds = time.perf_counter() - ddp_wait_started_at
                    window_observability['ddp_sync_wait_seconds'] += ddp_wait_seconds
                    train_live_observability['ddp_sync_wait_seconds'] += ddp_wait_seconds
                    reached_step_limit = bool(stop_flags[0].item())
                    reached_runtime_limit = bool(stop_flags[1].item())
                    preflight_should_stop = bool(stop_flags[2].item())
                step_loader_snapshot = (
                    train_loader_stats.snapshot()
                    if train_loader_stats is not None
                    else {}
                )
                step_queue_snapshot = train_loader.queue_depth_snapshot()
                observe_window_queue_depths(
                    window_observability,
                    loader_snapshot=step_loader_snapshot,
                    queue_snapshot=step_queue_snapshot,
                )
                observe_window_queue_depths(
                    train_live_observability,
                    loader_snapshot=step_loader_snapshot,
                    queue_snapshot=step_queue_snapshot,
                )
                step_elapsed_seconds = time.perf_counter() - step_started_at
                window_observability['step_time_seconds_total'] += step_elapsed_seconds
                window_observability['step_count'] += 1
                train_live_observability['step_time_seconds_total'] += step_elapsed_seconds
                train_live_observability['step_count'] += 1
                if profiler is not None:
                    profiler.step()
                reached_end = reached_step_limit or reached_runtime_limit or preflight_should_stop
                should_save = steps % save_every == 0
                should_log_train_live = (
                    steps % train_log_every == 0
                    and not should_save
                    and not reached_end
                )
                should_run_live_val = (
                    should_log_train_live
                    and val_log_every > 0
                    and steps % val_log_every == 0
                    and validation_enabled
                )
                if should_log_train_live:
                    ddp_wait_started_at = time.perf_counter()
                    with torch.profiler.record_function('bc.ddp_collective'):
                        synced_train_live_stats = synchronize_metric_sums(
                            train_live_stats,
                            dist_ctx=dist_ctx,
                            device=device,
                        )
                    train_live_observability['ddp_sync_wait_seconds'] += time.perf_counter() - ddp_wait_started_at
                    live_runtime_seconds_total = runtime_seconds_total + (time.perf_counter() - runtime_started_at)
                    live_runtime_metrics = throughput_metrics(
                        sample_count=synced_train_live_stats['count'],
                        step_count=steps - train_live_last_steps,
                        elapsed_seconds=time.perf_counter() - train_live_window_start,
                    )
                    live_runtime_metrics['runtime_seconds_total'] = live_runtime_seconds_total
                    live_runtime_metrics['effective_global_batch'] = global_batch_size
                    current_handoff_snapshot = train_loader.snapshot_handoff_state()
                    current_loader_snapshot = (
                        train_loader_stats.snapshot()
                        if train_loader_stats is not None
                        else {}
                    )
                    live_rank_step_time_ms_local = (
                        train_live_observability['step_time_seconds_total']
                        * 1000.0
                        / max(int(train_live_observability['step_count']), 1)
                    )
                    ddp_wait_started_at = time.perf_counter()
                    with torch.profiler.record_function('bc.ddp_collective'):
                        cpu_pipe_wait_seconds = reduce_max_scalar(
                            cpu_pipe_wait_delta(train_live_loader_snapshot, current_loader_snapshot),
                            dist_ctx=dist_ctx,
                            device=device,
                        )
                        live_rank_step_time_ms_max = reduce_max_scalar(
                            live_rank_step_time_ms_local,
                            dist_ctx=dist_ctx,
                            device=device,
                        )
                        live_rank_step_time_ms_min = reduce_min_scalar(
                            live_rank_step_time_ms_local,
                            dist_ctx=dist_ctx,
                            device=device,
                        )
                    train_live_observability['ddp_sync_wait_seconds'] += time.perf_counter() - ddp_wait_started_at
                    live_loader_metrics = loader_window_metrics(
                        previous_snapshot=train_live_loader_snapshot,
                        current_snapshot=current_loader_snapshot,
                        wait_seconds=train_live_wait_seconds,
                        elapsed_seconds=live_runtime_metrics['elapsed_seconds'],
                        cpu_pipe_wait_seconds_override=cpu_pipe_wait_seconds,
                    )
                    live_loader_metrics.update(
                        handoff_window_metrics(
                            previous_snapshot=train_live_handoff_snapshot,
                            current_snapshot=current_handoff_snapshot,
                            elapsed_seconds=live_runtime_metrics['elapsed_seconds'],
                        )
                    )
                    merge_window_observability(
                        runtime_metrics=live_runtime_metrics,
                        loader_metrics=live_loader_metrics,
                        observability=train_live_observability,
                        rank_step_time_ms_max=live_rank_step_time_ms_max,
                        rank_step_time_ms_min=live_rank_step_time_ms_min,
                    )
                    live_val_metrics = None
                    if should_run_live_val:
                        if dist_ctx.is_main_process:
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            live_val_metrics = evaluate_split(
                                val_file_list,
                                max_batches=val_steps,
                                desc='VAL',
                                shuffle=True,
                                batch_size_override=val_batch_size,
                            )
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            # Update val-best checkpoint if improved
                            if live_val_metrics is not None and is_better_eval_result(live_val_metrics, best_perf):
                                best_perf = {
                                    'split': best_eval_split,
                                    'accuracy': live_val_metrics['accuracy'],
                                    'nll': live_val_metrics['nll'],
                                    'steps': steps,
                                }
                                live_best_state = build_state(
                                    train_metrics=finalize_metric_sums(synced_train_live_stats),
                                    val_metrics=live_val_metrics,
                                    runtime_metrics=live_runtime_metrics,
                                    loader_metrics=live_loader_metrics,
                                    memory_metrics=device_memory_metrics(device),
                                )
                                atomic_torch_save(live_best_state, best_state_file)
                                logging.info(
                                    'new val best: acc %.6f nll %.6f step %s -> %s',
                                    best_perf['accuracy'],
                                    best_perf['nll'],
                                    f'{steps:,}',
                                    best_state_file,
                                )
                        distributed_barrier(dist_ctx)
                    if wandb_run is not None and synced_train_live_stats['count'] > 0:
                        train_live_metrics = finalize_metric_sums(synced_train_live_stats)
                        live_lr = current_learning_rate(optimizer)
                        live_memory_metrics = device_memory_metrics(device)
                        live_wandb_payload = wandb_scalar_payload(
                            top_k=top_k,
                            train_metrics=train_live_metrics,
                            val_metrics=live_val_metrics,
                            runtime_metrics=live_runtime_metrics,
                            loader_metrics=live_loader_metrics,
                            memory_metrics=live_memory_metrics,
                            lr=live_lr,
                            steps=steps,
                            best_eval_split=best_eval_split,
                            best_eval_metrics=None,
                        )
                        if epoch_steps > 0:
                            live_wandb_payload['epoch/current'] = steps // epoch_steps + 1
                            live_wandb_payload['epoch/progress'] = steps / epoch_steps
                            live_wandb_payload['epoch/remaining_steps'] = max(epoch_steps * num_epochs - steps, 0)
                        wandb_run.log(live_wandb_payload, step=steps)
                    if dist_ctx.is_main_process and synced_train_live_stats['count'] > 0:
                        train_live_metrics = finalize_metric_sums(synced_train_live_stats)
                        append_metrics_event(
                            metrics_jsonl,
                            {
                                'event': 'train_live',
                                'step': steps,
                                'runtime_seconds_total': live_runtime_seconds_total,
                                'train_metrics': train_live_metrics,
                                'runtime_metrics': live_runtime_metrics,
                                'loader_metrics': live_loader_metrics,
                                'val_metrics': live_val_metrics,
                            },
                        )
                        if preflight_enabled and live_runtime_seconds_total >= preflight_min_runtime_seconds:
                            preflight_windows.append({
                                'samples_per_second': live_runtime_metrics['samples_per_second'],
                                'wait_fraction': live_loader_metrics['wait_fraction'],
                                'step': steps,
                                'runtime_seconds_total': live_runtime_seconds_total,
                            })
                            if (
                                steps >= preflight_min_steps_before_stop
                                and preflight_windows_stable(
                                windows=list(preflight_windows),
                                required_windows=preflight_required_windows,
                                tolerance=preflight_stability_tolerance,
                                )
                            ):
                                preflight_should_stop = True
                    train_live_handoff_snapshot = dict(current_handoff_snapshot)
                    if dist_ctx.enabled:
                        preflight_flag = torch.tensor(
                            int(preflight_should_stop if dist_ctx.is_main_process else 0),
                            dtype=torch.int32,
                            device=device,
                        )
                        ddp_wait_started_at = time.perf_counter()
                        with torch.profiler.record_function('bc.ddp_collective'):
                            torch.distributed.all_reduce(preflight_flag, op=torch.distributed.ReduceOp.MAX)
                        train_live_observability['ddp_sync_wait_seconds'] += time.perf_counter() - ddp_wait_started_at
                        preflight_should_stop = bool(preflight_flag.item())
                    if preflight_should_stop:
                        reached_end = True
                    train_live_stats = empty_metric_sums()
                    train_live_wait_seconds = 0.0
                    train_live_observability = empty_window_observability()
                    train_live_window_start = time.perf_counter()
                    train_live_last_steps = steps
                    train_live_loader_snapshot = current_loader_snapshot
                    train_live_handoff_snapshot = dict(current_handoff_snapshot)
                if not should_save and not reached_end:
                    continue

                ddp_wait_started_at = time.perf_counter()
                with torch.profiler.record_function('bc.ddp_collective'):
                    distributed_barrier(dist_ctx)
                window_observability['ddp_sync_wait_seconds'] += time.perf_counter() - ddp_wait_started_at
                pb.close()
                ddp_wait_started_at = time.perf_counter()
                with torch.profiler.record_function('bc.ddp_collective'):
                    synced_train_stats = synchronize_metric_sums(
                        train_stats,
                        dist_ctx=dist_ctx,
                        device=device,
                    )
                window_observability['ddp_sync_wait_seconds'] += time.perf_counter() - ddp_wait_started_at
                current_handoff_snapshot = train_loader.snapshot_handoff_state()
                current_loader_snapshot = (
                    train_loader_stats.snapshot()
                    if train_loader_stats is not None
                    else {}
                )
                rank_step_time_ms_local = (
                    window_observability['step_time_seconds_total']
                    * 1000.0
                    / max(int(window_observability['step_count']), 1)
                )
                # Keep collective ordering identical across ranks on save windows.
                ddp_wait_started_at = time.perf_counter()
                with torch.profiler.record_function('bc.ddp_collective'):
                    cpu_pipe_wait_seconds = reduce_max_scalar(
                        cpu_pipe_wait_delta(window_loader_snapshot, current_loader_snapshot),
                        dist_ctx=dist_ctx,
                        device=device,
                    )
                    rank_step_time_ms_max = reduce_max_scalar(
                        rank_step_time_ms_local,
                        dist_ctx=dist_ctx,
                        device=device,
                    )
                    rank_step_time_ms_min = reduce_min_scalar(
                        rank_step_time_ms_local,
                        dist_ctx=dist_ctx,
                        device=device,
                    )
                window_observability['ddp_sync_wait_seconds'] += time.perf_counter() - ddp_wait_started_at
                if dist_ctx.is_main_process:
                    train_metrics = finalize_metric_sums(synced_train_stats)
                    current_runtime_seconds_total = runtime_seconds_total + (time.perf_counter() - runtime_started_at)
                    runtime_metrics = throughput_metrics(
                        sample_count=train_metrics['count'],
                        step_count=steps - last_saved_steps,
                        elapsed_seconds=time.perf_counter() - window_start_time,
                    )
                    runtime_metrics['runtime_seconds_total'] = current_runtime_seconds_total
                    runtime_metrics['effective_global_batch'] = global_batch_size
                    loader_runtime = loader_window_metrics(
                        previous_snapshot=window_loader_snapshot,
                        current_snapshot=current_loader_snapshot,
                        wait_seconds=window_wait_seconds,
                        elapsed_seconds=runtime_metrics['elapsed_seconds'],
                        cpu_pipe_wait_seconds_override=cpu_pipe_wait_seconds,
                    )
                    loader_runtime.update(
                        handoff_window_metrics(
                            previous_snapshot=window_handoff_snapshot,
                            current_snapshot=current_handoff_snapshot,
                            elapsed_seconds=runtime_metrics['elapsed_seconds'],
                        )
                    )
                    merge_window_observability(
                        runtime_metrics=runtime_metrics,
                        loader_metrics=loader_runtime,
                        observability=window_observability,
                        rank_step_time_ms_max=rank_step_time_ms_max,
                        rank_step_time_ms_min=rank_step_time_ms_min,
                    )
                    writer.add_scalar('runtime/save_checkpoint_wait_seconds', runtime_metrics.get('save_checkpoint_wait_seconds', 0.0), steps)
                    writer.add_scalar('runtime/save_checkpoint_wait_fraction', runtime_metrics.get('save_checkpoint_wait_fraction', 0.0), steps)
                    runtime_metrics['loader_wait_seconds'] = loader_runtime['wait_seconds']
                    runtime_metrics['loader_wait_fraction'] = loader_runtime['wait_fraction']
                    memory_metrics = device_memory_metrics(device)
                    logged_lr = current_learning_rate(optimizer)
                    val_metrics = None
                    if validation_enabled:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        val_metrics = evaluate_split(
                            val_file_list,
                            max_batches=val_steps,
                            desc='VAL',
                            shuffle=True,
                            batch_size_override=val_batch_size,
                        )
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()

                    if val_metrics is not None:
                        writer.add_scalars('loss', {
                            'train': train_metrics['nll'],
                            'val': val_metrics['nll'],
                        }, steps)
                        writer.add_scalars('accuracy', {
                            'train': train_metrics['accuracy'],
                            'val': val_metrics['accuracy'],
                        }, steps)
                        writer.add_scalars('topk_accuracy', {
                            'train': train_metrics['topk_accuracy'],
                            'val': val_metrics['topk_accuracy'],
                        }, steps)
                        writer.add_scalars('legal_rate', {
                            'train': train_metrics['legal_rate'],
                            'val': val_metrics['legal_rate'],
                        }, steps)
                    else:
                        writer.add_scalar('loss/train', train_metrics['nll'], steps)
                        writer.add_scalar('accuracy/train', train_metrics['accuracy'], steps)
                        writer.add_scalar(f'topk_accuracy/train_top{top_k}', train_metrics['topk_accuracy'], steps)
                        writer.add_scalar('legal_rate/train', train_metrics['legal_rate'], steps)
                    writer.add_scalar('hparam/lr', logged_lr, steps)
                    writer.add_scalar('throughput/steps_per_second', runtime_metrics['steps_per_second'], steps)
                    writer.add_scalar('throughput/samples_per_second', runtime_metrics['samples_per_second'], steps)
                    writer.add_scalar('throughput/window_seconds', runtime_metrics['elapsed_seconds'], steps)
                    writer.add_scalar('throughput/effective_global_batch', global_batch_size, steps)
                    writer.add_scalar('runtime/runtime_seconds_total', current_runtime_seconds_total, steps)
                    if epoch_steps > 0:
                        writer.add_scalar('epoch/current', steps // epoch_steps + 1, steps)
                        writer.add_scalar('epoch/progress', steps / epoch_steps, steps)
                        writer.add_scalar('epoch/remaining_steps', max(epoch_steps * num_epochs - steps, 0), steps)
                    writer.add_scalar('runtime/fw_bw_opt_seconds', runtime_metrics.get('fw_bw_opt_seconds', 0.0), steps)
                    writer.add_scalar('runtime/fw_bw_opt_fraction', runtime_metrics.get('fw_bw_opt_fraction', 0.0), steps)
                    writer.add_scalar('runtime/ddp_sync_wait_seconds', runtime_metrics.get('ddp_sync_wait_seconds', 0.0), steps)
                    writer.add_scalar('runtime/ddp_sync_wait_fraction', runtime_metrics.get('ddp_sync_wait_fraction', 0.0), steps)
                    writer.add_scalar('runtime/save_checkpoint_wait_seconds', runtime_metrics.get('save_checkpoint_wait_seconds', 0.0), steps)
                    writer.add_scalar('runtime/save_checkpoint_wait_fraction', runtime_metrics.get('save_checkpoint_wait_fraction', 0.0), steps)
                    writer.add_scalar('runtime/rank_step_time_ms', runtime_metrics.get('rank_step_time_ms', 0.0), steps)
                    writer.add_scalar('runtime/rank_step_time_ms_max_minus_min', runtime_metrics.get('rank_step_time_ms_max_minus_min', 0.0), steps)
                    writer.add_scalar('loader/wait_seconds', loader_runtime['wait_seconds'], steps)
                    writer.add_scalar('loader/wait_fraction', loader_runtime['wait_fraction'], steps)
                    writer.add_scalar('loader/cpu_pipe_wait_seconds', loader_runtime['cpu_pipe_wait_seconds'], steps)
                    writer.add_scalar('loader/cpu_pipe_wait_fraction', loader_runtime['cpu_pipe_wait_fraction'], steps)
                    writer.add_scalar('loader/cpu_pipe_empty_wait_seconds', loader_runtime.get('cpu_pipe_empty_wait_seconds', 0.0), steps)
                    writer.add_scalar('loader/cpu_pipe_empty_wait_fraction', loader_runtime.get('cpu_pipe_empty_wait_fraction', 0.0), steps)
                    writer.add_scalar('loader/device_prefetch_wait_seconds', loader_runtime.get('device_prefetch_wait_seconds', 0.0), steps)
                    writer.add_scalar('loader/device_prefetch_wait_fraction', loader_runtime.get('device_prefetch_wait_fraction', 0.0), steps)
                    writer.add_scalar('loader/cpu_ready_wait_seconds', loader_runtime.get('cpu_ready_wait_seconds', 0.0), steps)
                    writer.add_scalar('loader/cpu_ready_wait_fraction', loader_runtime.get('cpu_ready_wait_fraction', 0.0), steps)
                    writer.add_scalar('loader/stage_free_slot_wait_seconds', loader_runtime.get('stage_free_slot_wait_seconds', 0.0), steps)
                    writer.add_scalar('loader/stage_free_slot_wait_fraction', loader_runtime.get('stage_free_slot_wait_fraction', 0.0), steps)
                    writer.add_scalar('loader/stage_copy_seconds', loader_runtime.get('stage_copy_seconds', 0.0), steps)
                    writer.add_scalar('loader/stage_copy_fraction', loader_runtime.get('stage_copy_fraction', 0.0), steps)
                    writer.add_scalar('loader/pinned_ready_wait_seconds', loader_runtime.get('pinned_ready_wait_seconds', 0.0), steps)
                    writer.add_scalar('loader/pinned_ready_wait_fraction', loader_runtime.get('pinned_ready_wait_fraction', 0.0), steps)
                    writer.add_scalar('loader/h2d_submit_seconds', loader_runtime.get('h2d_submit_seconds', 0.0), steps)
                    writer.add_scalar('loader/h2d_submit_fraction', loader_runtime.get('h2d_submit_fraction', 0.0), steps)
                    writer.add_scalar('loader/copy_ready_on_pop_fraction', loader_runtime.get('copy_ready_on_pop_fraction', 0.0), steps)
                    writer.add_scalar('loader/h2d_copy_ms_avg', loader_runtime.get('h2d_copy_ms_avg', 0.0), steps)
                    writer.add_scalar('loader/gpu_prefetch_depth', loader_runtime.get('gpu_prefetch_depth', 0), steps)
                    writer.add_scalar('loader/cpu_ready_batches_min', loader_runtime.get('cpu_ready_batches_min', 0.0), steps)
                    writer.add_scalar('loader/cpu_ready_batches_avg', loader_runtime.get('cpu_ready_batches_avg', 0.0), steps)
                    writer.add_scalar('loader/cpu_ready_batches_max', loader_runtime.get('cpu_ready_batches_max', 0.0), steps)
                    writer.add_scalar('loader/loader_ready_chunks_min', loader_runtime.get('loader_ready_chunks_min', 0.0), steps)
                    writer.add_scalar('loader/loader_ready_chunks_avg', loader_runtime.get('loader_ready_chunks_avg', 0.0), steps)
                    writer.add_scalar('loader/loader_ready_chunks_max', loader_runtime.get('loader_ready_chunks_max', 0.0), steps)
                    writer.add_scalar('loader/device_prefetch_depth_min', loader_runtime.get('device_prefetch_depth_min', 0.0), steps)
                    writer.add_scalar('loader/device_prefetch_depth_avg', loader_runtime.get('device_prefetch_depth_avg', 0.0), steps)
                    writer.add_scalar('loader/device_prefetch_depth_max', loader_runtime.get('device_prefetch_depth_max', 0.0), steps)
                    writer.add_scalar('loader/free_handoff_slots_approx', loader_runtime.get('free_handoff_slots_approx', 0), steps)
                    writer.add_scalar('loader/pinned_ready_q_approx', loader_runtime.get('pinned_ready_q_approx', 0), steps)
                    writer.add_scalar('loader/host_num_alloc_delta', loader_runtime.get('host_num_alloc_delta', 0), steps)
                    writer.add_scalar('loader/host_num_free_delta', loader_runtime.get('host_num_free_delta', 0), steps)
                    writer.add_scalar('loader/host_alloc_time_us_delta', loader_runtime.get('host_alloc_time_us_delta', 0), steps)
                    writer.add_scalar('loader/host_free_time_us_delta', loader_runtime.get('host_free_time_us_delta', 0), steps)
                    writer.add_scalar('loader/host_active_bytes_cur', loader_runtime.get('host_active_bytes_cur', 0), steps)
                    writer.add_scalar('loader/host_allocated_bytes_cur', loader_runtime.get('host_allocated_bytes_cur', 0), steps)
                    writer.add_scalar('loader/cpu_ready_batches', loader_runtime['cpu_ready_batches'], steps)
                    writer.add_scalar('loader/cpu_ready_bytes_gib', loader_runtime['cpu_ready_bytes_gib'], steps)
                    writer.add_scalar('loader/queued_bytes_gib', loader_runtime['queued_bytes_gib'], steps)
                    writer.add_scalar('loader/max_queued_bytes_gib', loader_runtime['max_queued_bytes_gib'], steps)
                    writer.add_scalar('loader/ready_chunks', loader_runtime['ready_chunks'], steps)
                    writer.add_scalar('loader/chunk_count', loader_runtime['chunk_count'], steps)
                    writer.add_scalar('loader/chunk_files', loader_runtime['chunk_files'], steps)
                    writer.add_scalar('loader/chunk_samples', loader_runtime['chunk_samples'], steps)
                    writer.add_scalar('loader/chunk_bytes_gib', loader_runtime['chunk_bytes_gib'], steps)
                    writer.add_scalar('loader/chunk_build_seconds', loader_runtime['chunk_build_seconds'], steps)
                    writer.add_scalar('loader/avg_chunk_build_seconds', loader_runtime['avg_chunk_build_seconds'], steps)
                    writer.add_scalar('loader/raw_read_seconds', loader_runtime.get('raw_read_seconds', 0.0), steps)
                    writer.add_scalar('loader/raw_read_fraction', loader_runtime.get('raw_read_fraction', 0.0), steps)
                    writer.add_scalar('loader/rust_convert_seconds', loader_runtime.get('rust_convert_seconds', 0.0), steps)
                    writer.add_scalar('loader/rust_convert_fraction', loader_runtime.get('rust_convert_fraction', 0.0), steps)
                    writer.add_scalar('loader/sample_materialize_seconds', loader_runtime.get('sample_materialize_seconds', 0.0), steps)
                    writer.add_scalar('loader/sample_materialize_fraction', loader_runtime.get('sample_materialize_fraction', 0.0), steps)
                    writer.add_scalar('loader/collate_or_assemble_seconds', loader_runtime.get('collate_or_assemble_seconds', 0.0), steps)
                    writer.add_scalar('loader/collate_or_assemble_fraction', loader_runtime.get('collate_or_assemble_fraction', 0.0), steps)
                    for name, value in memory_metrics.items():
                        writer.add_scalar(f'memory/{name}', value, steps)
                    if val_metrics is not None:
                        for name, value in val_metrics['category_accuracy'].items():
                            writer.add_scalar(f'val_accuracy/{name}', value, steps)

                    state = build_state(
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        runtime_metrics=runtime_metrics,
                        loader_metrics=loader_runtime,
                        memory_metrics=memory_metrics,
                    )
                    save_started_at = time.perf_counter()
                    with torch.profiler.record_function('bc.checkpoint_save'):
                        atomic_torch_save(state, state_file)
                        save_rolling_checkpoint(state, checkpoint_dir, steps, checkpoint_keep_recent)
                    window_observability['save_checkpoint_wait_seconds'] += time.perf_counter() - save_started_at

                    # Val-triggered best checkpoint (from sampled val)
                    if val_metrics is not None and is_better_eval_result(val_metrics, best_perf):
                        best_perf = {
                            'split': best_eval_split,
                            'accuracy': val_metrics['accuracy'],
                            'nll': val_metrics['nll'],
                            'steps': steps,
                        }
                        state['best_perf'] = best_perf
                        save_started_at = time.perf_counter()
                        with torch.profiler.record_function('bc.checkpoint_save'):
                            atomic_torch_save(state, state_file)
                            atomic_torch_save(state, best_state_file)
                        window_observability['save_checkpoint_wait_seconds'] += time.perf_counter() - save_started_at
                        logging.info(
                            'new val best: acc %.6f nll %.6f step %s -> %s',
                            best_perf['accuracy'],
                            best_perf['nll'],
                            f'{steps:,}',
                            best_state_file,
                        )

                    # Full eval (deterministic, less frequent)
                    should_run_best_eval = (
                        validation_enabled
                        and best_eval_every > 0
                        and ((steps % best_eval_every == 0) or reached_end)
                    )
                    best_eval_metrics = None
                    if should_run_best_eval:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        best_eval_metrics = evaluate_split(
                            best_eval_file_list,
                            max_batches=best_eval_max_batches,
                            desc=f'FULL-{best_eval_split.upper()}',
                            shuffle=False,
                            batch_size_override=best_eval_batch_size,
                        )
                        state = build_state(
                            train_metrics=train_metrics,
                            val_metrics=val_metrics,
                            runtime_metrics=runtime_metrics,
                            loader_metrics=loader_runtime,
                            memory_metrics=memory_metrics,
                            best_eval_metrics=best_eval_metrics,
                        )
                        writer.add_scalars(f'best_eval/{best_eval_split}/loss', {
                            'nll': best_eval_metrics['nll'],
                        }, steps)
                        writer.add_scalars(f'best_eval/{best_eval_split}/accuracy', {
                            'top1': best_eval_metrics['accuracy'],
                            f'top{top_k}': best_eval_metrics['topk_accuracy'],
                        }, steps)
                        writer.add_scalar(f'best_eval/{best_eval_split}/legal_rate', best_eval_metrics['legal_rate'], steps)
                        for name, value in best_eval_metrics['category_accuracy'].items():
                            writer.add_scalar(f'best_eval/{best_eval_split}/category_accuracy/{name}', value, steps)
                        save_started_at = time.perf_counter()
                        with torch.profiler.record_function('bc.checkpoint_save'):
                            atomic_torch_save(state, state_file)
                        window_observability['save_checkpoint_wait_seconds'] += time.perf_counter() - save_started_at

                        # Full eval best checkpoint
                        if is_better_eval_result(best_eval_metrics, best_full_eval_perf):
                            best_full_eval_perf = {
                                'split': best_eval_split,
                                'accuracy': best_eval_metrics['accuracy'],
                                'nll': best_eval_metrics['nll'],
                                'steps': steps,
                            }
                            state['best_full_eval_perf'] = best_full_eval_perf
                            save_started_at = time.perf_counter()
                            with torch.profiler.record_function('bc.checkpoint_save'):
                                atomic_torch_save(state, state_file)
                                if full_eval_best_state_file:
                                    atomic_torch_save(state, full_eval_best_state_file)
                            window_observability['save_checkpoint_wait_seconds'] += time.perf_counter() - save_started_at
                            logging.info(
                                'new full eval best: acc %.6f nll %.6f step %s -> %s',
                                best_full_eval_perf['accuracy'],
                                best_full_eval_perf['nll'],
                                f'{steps:,}',
                                full_eval_best_state_file or state_file,
                            )

                    # Stage save at epoch boundaries (check if an epoch boundary
                    # was crossed within this save window, since epoch_steps may
                    # not be a multiple of save_every)
                    if epoch_steps > 0 and stage_save_dir and (steps // epoch_steps) > (last_saved_steps // epoch_steps):
                        save_started_at = time.perf_counter()
                        with torch.profiler.record_function('bc.checkpoint_save'):
                            save_stage_checkpoint(state, stage_save_dir, steps)
                        window_observability['save_checkpoint_wait_seconds'] += time.perf_counter() - save_started_at
                        logging.info('stage save: step %s -> %s', f'{steps:,}', stage_save_dir)

                    merge_window_observability(
                        runtime_metrics=runtime_metrics,
                        loader_metrics=loader_runtime,
                        observability=window_observability,
                        rank_step_time_ms_max=rank_step_time_ms_max,
                        rank_step_time_ms_min=rank_step_time_ms_min,
                    )

                    if wandb_run is not None:
                        wandb_payload = wandb_scalar_payload(
                            top_k=top_k,
                            train_metrics=train_metrics,
                            val_metrics=val_metrics,
                            runtime_metrics=runtime_metrics,
                            loader_metrics=loader_runtime,
                            memory_metrics=memory_metrics,
                            lr=logged_lr,
                            steps=steps,
                            best_eval_split=best_eval_split,
                            best_eval_metrics=best_eval_metrics,
                        )
                        if epoch_steps > 0:
                            wandb_payload['epoch/current'] = steps // epoch_steps + 1
                            wandb_payload['epoch/progress'] = steps / epoch_steps
                            wandb_payload['epoch/remaining_steps'] = max(epoch_steps * num_epochs - steps, 0)
                        wandb_run.log(wandb_payload, step=steps)
                        wandb_run.summary['last_step'] = steps
                        wandb_run.summary['runtime_seconds_total'] = current_runtime_seconds_total
                    append_metrics_event(
                        metrics_jsonl if dist_ctx.is_main_process else '',
                        {
                            'event': 'save_window',
                            'step': steps,
                            'stop_reason': (
                                'runtime_limit'
                                if reached_runtime_limit
                                else 'max_steps'
                                if reached_step_limit
                                else 'preflight_stable'
                                if preflight_should_stop
                                else ''
                            ),
                            'runtime_seconds_total': current_runtime_seconds_total,
                            'train_metrics': train_metrics,
                            'val_metrics': val_metrics,
                            'runtime_metrics': runtime_metrics,
                            'loader_metrics': loader_runtime,
                            'memory_metrics': memory_metrics,
                            'best_eval_metrics': best_eval_metrics,
                        },
                    )
                    train_live_stats = empty_metric_sums()

                    save_started_at = time.perf_counter()
                    with torch.profiler.record_function('bc.checkpoint_save'):
                        writer.flush()
                    window_observability['save_checkpoint_wait_seconds'] += time.perf_counter() - save_started_at
                    merge_window_observability(
                        runtime_metrics=runtime_metrics,
                        loader_metrics=loader_runtime,
                        observability=window_observability,
                        rank_step_time_ms_max=rank_step_time_ms_max,
                        rank_step_time_ms_min=rank_step_time_ms_min,
                    )

                    if epoch_steps > 0:
                        current_epoch = steps // epoch_steps + 1
                        within_epoch_pct = (steps % epoch_steps) / epoch_steps * 100
                        total_epoch_pct = steps / (epoch_steps * num_epochs) * 100 if num_epochs > 0 else 0
                        epoch_info = f'ep{current_epoch}/{num_epochs} {within_epoch_pct:.1f}%'
                        epoch_remaining = f'{max(epoch_steps * num_epochs - steps, 0):,}'
                    else:
                        epoch_info = 'n/a'
                        epoch_remaining = 'n/a'
                    log_parts = [
                        f"steps={steps:,}",
                        f"epoch={epoch_info}",
                        f"remaining={epoch_remaining}",
                        f"train_nll={train_metrics['nll']:.6f}",
                        f"train_acc={train_metrics['accuracy']:.6f}",
                        f"train_top{top_k}={train_metrics['topk_accuracy']:.6f}",
                        f"steps_per_s={runtime_metrics['steps_per_second']:.3f}",
                        f"samples_per_s={runtime_metrics['samples_per_second']:.1f}",
                        f"window_s={runtime_metrics['elapsed_seconds']:.2f}",
                        f"runtime_s={current_runtime_seconds_total:.1f}",
                        f"loader_wait={loader_runtime['wait_fraction']:.3f}",
                        f"cpu_pipe_wait={loader_runtime['cpu_pipe_wait_fraction']:.3f}",
                        f"device_wait={loader_runtime.get('device_prefetch_wait_fraction', 0.0):.3f}",
                        f"fwbw={runtime_metrics.get('fw_bw_opt_fraction', 0.0):.3f}",
                        f"ddp_sync={runtime_metrics.get('ddp_sync_wait_fraction', 0.0):.3f}",
                        f"pinned_ready_wait={loader_runtime.get('pinned_ready_wait_fraction', 0.0):.3f}",
                        f"copy_ready={loader_runtime.get('copy_ready_on_pop_fraction', 0.0):.3f}",
                        f"cpu_ready_batches={loader_runtime['cpu_ready_batches']}",
                        f"queued_gib={loader_runtime['queued_bytes_gib']:.2f}",
                    ]
                    if val_metrics is not None:
                        log_parts.extend([
                            f"val_nll={val_metrics['nll']:.6f}",
                            f"val_acc={val_metrics['accuracy']:.6f}",
                            f"val_top{top_k}={val_metrics['topk_accuracy']:.6f}",
                            f"val_legal={val_metrics['legal_rate']:.6f}",
                        ])
                    if memory_metrics:
                        log_parts.append(f"mem_alloc_gib={memory_metrics['max_allocated_gib']:.2f}")
                        log_parts.append(f"mem_resv_gib={memory_metrics['max_reserved_gib']:.2f}")
                    if best_eval_metrics is not None:
                        log_parts.append(
                            f"best_{best_eval_split}_acc={best_eval_metrics['accuracy']:.6f}"
                        )
                        log_parts.append(
                            f"best_{best_eval_split}_nll={best_eval_metrics['nll']:.6f}"
                        )
                    if reached_runtime_limit:
                        log_parts.append('stop_reason=runtime_limit')
                    elif reached_step_limit:
                        log_parts.append('stop_reason=max_steps')
                    elif preflight_should_stop:
                        log_parts.append('stop_reason=preflight_stable')
                    logging.info(' '.join(log_parts))

                    if best_eval_metrics is not None and is_better_eval_result(best_eval_metrics, best_perf):
                        best_perf = {
                            'split': best_eval_split,
                            'accuracy': best_eval_metrics['accuracy'],
                            'nll': best_eval_metrics['nll'],
                            'steps': steps,
                        }
                        state['best_perf'] = best_perf
                        save_started_at = time.perf_counter()
                        with torch.profiler.record_function('bc.checkpoint_save'):
                            atomic_torch_save(state, state_file)
                        window_observability['save_checkpoint_wait_seconds'] += time.perf_counter() - save_started_at
                        logging.info(
                            'new best checkpoint on %s: acc %.6f nll %.6f step %s -> %s',
                            best_eval_split,
                            best_perf['accuracy'],
                            best_perf['nll'],
                            f'{steps:,}',
                            best_state_file,
                        )
                        save_started_at = time.perf_counter()
                        with torch.profiler.record_function('bc.checkpoint_save'):
                            atomic_torch_save(state, best_state_file)
                        window_observability['save_checkpoint_wait_seconds'] += time.perf_counter() - save_started_at
                        merge_window_observability(
                            runtime_metrics=runtime_metrics,
                            loader_metrics=loader_runtime,
                            observability=window_observability,
                            rank_step_time_ms_max=rank_step_time_ms_max,
                            rank_step_time_ms_min=rank_step_time_ms_min,
                        )
                        if wandb_run is not None:
                            wandb_run.summary['best_eval_split'] = best_perf['split']
                            wandb_run.summary['best_eval_accuracy'] = best_perf['accuracy']
                            wandb_run.summary['best_eval_nll'] = best_perf['nll']
                            wandb_run.summary['best_eval_steps'] = best_perf['steps']
                            wandb_run.summary['best_state_file'] = best_state_file
                    runtime_seconds_total = current_runtime_seconds_total
                runtime_seconds_total = broadcast_object(dist_ctx, runtime_seconds_total, src=0)
                best_perf = broadcast_object(dist_ctx, best_perf, src=0)
                best_full_eval_perf = broadcast_object(dist_ctx, best_full_eval_perf, src=0)
                distributed_barrier(dist_ctx)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                train_stats = empty_metric_sums()
                train_live_stats = empty_metric_sums()
                last_saved_steps = steps
                window_start_time = time.perf_counter()
                train_live_window_start = window_start_time
                runtime_started_at = time.perf_counter()
                window_wait_seconds = 0.0
                train_live_wait_seconds = 0.0
                window_observability = empty_window_observability()
                train_live_observability = empty_window_observability()
                train_live_last_steps = steps
                window_loader_snapshot = (
                    train_loader_stats.snapshot()
                    if train_loader_stats is not None
                    else {}
                )
                window_handoff_snapshot = train_loader.snapshot_handoff_state()
                train_live_loader_snapshot = dict(window_loader_snapshot)
                train_live_handoff_snapshot = dict(window_handoff_snapshot)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                if reached_end:
                    break

                remaining = save_every if max_steps <= 0 else min(save_every, max_steps - steps)
                if epoch_steps > 0:
                    current_epoch = steps // epoch_steps + 1
                    within_epoch_pct = (steps % epoch_steps) / epoch_steps * 100
                    epoch_desc = f'TRAIN [ep{current_epoch} {within_epoch_pct:.0f}%]'
                else:
                    epoch_desc = 'TRAIN'
                pb = tqdm(
                    total=max(remaining, 1),
                    desc=epoch_desc,
                    disable=dist_ctx.enabled and not dist_ctx.is_main_process,
                )
        finally:
            pb.close()

        if steps != last_saved_steps:
            logging.info('stopping without checkpoint would lose progress; this should not happen')
    finally:
        if train_loader is not None:
            train_loader.close()
        if train_cpu_pipe is not None:
            train_cpu_pipe.close()
        if profiler is not None:
            profiler.__exit__(None, None, None)
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()
        destroy_distributed_context(dist_ctx)


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
