from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    world_size: int
    rank: int
    local_rank: int
    backend: str
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


_DISTRIBUTED_PERF_ONLY_KEYS = frozenset({
    'static_graph',
    'gradient_as_bucket_view',
})

_DATASET_LOADER_ONLY_KEYS = frozenset({
    'file_batch_size',
    'num_workers',
    'eval_num_workers',
    'persistent_workers',
    'prefetch_factor',
    'multiprocessing_context',
    'pin_memory',
    'cpu_batch_pipe_backend',
    'cpu_ready_batches',
    'cpu_ready_bytes_gib',
    'cpu_pipe_poll_timeout_s',
    'device_prefetch_batches',
    'device_prefetch_startup_batches',
    'eval_device_prefetch_batches',
    'eval_device_prefetch_startup_batches',
    'eval_pin_memory',
    'handoff_pin_memory',
    'eval_handoff_pin_memory',
    'handoff_stage_backend',
    'handoff_ring_slots',
    'handoff_measure_copy_wait_sync',
    'handoff_log_host_mem',
    'raw_source_backend',
    'loader_mode',
    'prefetch_strategy',
    'prefetch_chunks',
    'eval_prefetch_chunks',
    'prefetch_budget_gib',
    'eval_prefetch_budget_gib',
    'target_chunk_gib',
    'prefetch_threads',
    'prebatched',
    'prebatch_layout',
    'prebatch_shuffle_mode',
    'prebatch_spill_across_chunks',
    'startup_file_batch_size',
    'eval_startup_file_batch_size',
    'prefetch_out_of_order',
    'eval_prefetch_out_of_order',
    'in_order',
})


def effective_bc_config(full_config: dict) -> dict:
    bc_cfg = full_config.get('bc', {})
    resnet_cfg = bc_cfg.get('resnet', full_config.get('resnet', {}))
    training_bc_cfg = {
        key: value
        for key, value in bc_cfg.items()
        if key not in ('launch', 'wandb')
    }
    # Strip loader-only keys from dataset — they affect performance but not
    # training results, so changing them should not block resume.
    if 'dataset' in training_bc_cfg and isinstance(training_bc_cfg['dataset'], dict):
        training_bc_cfg['dataset'] = {
            k: v
            for k, v in training_bc_cfg['dataset'].items()
            if k not in _DATASET_LOADER_ONLY_KEYS
        }
    # Strip performance-only keys from distributed — static_graph and
    # gradient_as_bucket_view affect DDP communication strategy but not
    # the mathematical training results.
    if 'distributed' in training_bc_cfg and isinstance(training_bc_cfg['distributed'], dict):
        training_bc_cfg['distributed'] = {
            k: v
            for k, v in training_bc_cfg['distributed'].items()
            if k not in _DISTRIBUTED_PERF_ONLY_KEYS
        }
    training_bc_cfg['resnet'] = resnet_cfg
    return {
        'resnet': resnet_cfg,
        'bc': training_bc_cfg,
    }


def config_fingerprint(full_config: dict) -> str:
    payload = json.dumps(
        effective_bc_config(full_config),
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def stored_config_fingerprint(state: dict) -> str:
    # Always recompute from saved config so both sides use the same
    # effective_bc_config filtering (e.g. loader-only key exclusion).
    saved_config = state.get('config')
    if saved_config:
        return config_fingerprint(saved_config)
    if 'config_fingerprint' in state:
        return str(state['config_fingerprint'])
    return ''


def effective_global_batch(
    *,
    batch_size: int,
    world_size: int,
    grad_accum_steps: int,
) -> int:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if world_size <= 0:
        raise ValueError('world_size must be positive')
    if grad_accum_steps <= 0:
        raise ValueError('grad_accum_steps must be positive')
    return batch_size * world_size * grad_accum_steps


def shard_file_list_round_robin(
    file_list: list[str],
    *,
    rank: int,
    world_size: int,
) -> list[str]:
    if world_size <= 0:
        raise ValueError('world_size must be positive')
    if rank < 0 or rank >= world_size:
        raise ValueError(f'rank must be in [0, {world_size}), got {rank}')
    if world_size == 1:
        return list(file_list)
    return list(file_list[rank::world_size])


def resolve_distributed_context(
    *,
    control_device: str,
    distributed_cfg: dict | None = None,
    env: dict[str, str] | None = None,
    cuda_available: bool | None = None,
) -> DistributedContext:
    distributed_cfg = distributed_cfg or {}
    env = env or os.environ
    world_size = int(env.get('WORLD_SIZE', '1') or '1')
    rank = int(env.get('RANK', '0') or '0')
    local_rank = int(env.get('LOCAL_RANK', str(rank)) or str(rank))
    enabled = world_size > 1

    raw_device = str(control_device).strip() or 'cpu'
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()

    if enabled and raw_device.startswith('cuda') and cuda_available:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(raw_device)

    default_backend = 'nccl' if device.type == 'cuda' else 'gloo'
    backend = str(distributed_cfg.get('backend', default_backend))
    return DistributedContext(
        enabled=enabled,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend=backend,
        device=device,
    )


def init_distributed_context(ctx: DistributedContext) -> DistributedContext:
    if not ctx.enabled:
        return ctx
    if ctx.device.type == 'cuda':
        torch.cuda.set_device(ctx.device)
    if not dist.is_initialized():
        dist.init_process_group(backend=ctx.backend)
    return ctx


def destroy_distributed_context(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(ctx: DistributedContext) -> None:
    if ctx.enabled:
        dist.barrier()


def broadcast_object(ctx: DistributedContext, value, *, src: int = 0):
    if not ctx.enabled:
        return value
    payload = [value if ctx.rank == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def seed_everything(seed: int, *, rank: int = 0) -> int:
    if seed < 0:
        raise ValueError('seed must be non-negative')
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2**32))
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed
