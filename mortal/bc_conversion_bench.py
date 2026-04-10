from __future__ import annotations

import math
import random
import time
from glob import glob
from os import path
from pathlib import Path

import torch
from torch import nn, optim
from torch.amp import GradScaler
from torch.utils.data._utils.collate import default_collate

from bc_campaign import load_full_config
from bc_dataset import load_path_cache, normalize_file_list, resolve_actor_filter_map
from bc_runtime import seed_everything, shard_file_list_round_robin
from common import filtered_trimmed_lines
from dataloader import (
    ActionBatchBuffer,
    ActionChunkBuffer,
    ActionFileDatasetsIter,
    buffer_sample_count,
    buffer_size_bytes,
    build_action_file_dataloader,
)
from model import Brain, DQN
from train_bc import (
    DeviceBatchPrefetcher,
    autocast_context_kwargs,
    apply_cuda_precision_settings,
    batch_nbytes,
    dqn_policy_outputs,
    extract_policy_features,
    grad_scaler_enabled,
    loader_metrics_delta,
    resolve_amp_dtype,
    resolve_fused_optimizer_enabled,
)


def split_sources(dataset_cfg: dict, split: str) -> tuple[str, list[str]]:
    if split == 'train':
        return str(dataset_cfg.get('train_list', '') or ''), dataset_cfg.get('train_globs', [])
    if split == 'val':
        return str(dataset_cfg.get('val_list', '') or ''), dataset_cfg.get('val_globs', [])
    if split == 'test':
        return str(dataset_cfg.get('test_list', '') or ''), dataset_cfg.get('test_globs', [])
    raise ValueError(f'unexpected split: {split}')


def load_path_list(filename: str, root_dir: str = '') -> list[str]:
    with open(filename, encoding='utf-8') as f:
        raw_paths = list(filtered_trimmed_lines(f))
    if root_dir:
        root_dir = path.abspath(root_dir)
        return [
            p if path.isabs(p) else path.join(root_dir, p)
            for p in raw_paths
        ]
    return raw_paths


def load_name_filters(raw_paths: list[str]) -> list[str]:
    names = set()
    for filename in raw_paths:
        with open(filename, encoding='utf-8') as f:
            names.update(filtered_trimmed_lines(f))
    return sorted(names)


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


def select_benchmark_files(
    file_list: list[str],
    *,
    sample_size: int,
    sample_strategy: str,
) -> list[str]:
    if sample_size <= 0 or sample_size >= len(file_list):
        return list(file_list)
    if sample_strategy == 'round_robin':
        return deterministic_round_robin_sample(file_list, sample_size)
    if sample_strategy == 'head':
        return list(file_list[:sample_size])
    raise ValueError(f'unsupported sample_strategy: {sample_strategy}')


def resolve_split_file_list(full_config: dict, *, split: str) -> list[str]:
    dataset_cfg = ((full_config.get('bc') or {}).get('dataset') or {})
    root_dir = str(dataset_cfg.get('root_dir', '') or '')
    path_cache = str(dataset_cfg.get('path_cache', '') or '').strip()
    list_file, globs = split_sources(dataset_cfg, split)

    if path_cache and path.exists(path_cache) and list_file:
        cached_split_lists = load_path_cache(
            path_cache,
            expected_splits=[split],
            expected_sources={split: list_file},
        )
        return cached_split_lists[split]

    if list_file:
        return normalize_file_list(load_path_list(list_file, root_dir), desc=f'PATHS-{split.upper()}')

    if globs:
        file_list = []
        for pattern in globs:
            file_list.extend(glob(pattern, recursive=True))
        file_list.sort(reverse=True)
        return normalize_file_list(file_list, desc=f'PATHS-{split.upper()}')

    raise ValueError(f'bc.dataset has no source configured for split={split}')


def resolve_benchmark_actor_filter(
    full_config: dict,
    *,
    file_list: list[str],
) -> tuple[dict[str, tuple[int, ...]] | None, dict | None]:
    dataset_cfg = ((full_config.get('bc') or {}).get('dataset') or {})
    min_actor_dan = dataset_cfg.get('min_actor_dan')
    if min_actor_dan is None:
        return None, None
    actor_filter_map, summary = resolve_actor_filter_map(
        file_lists=[file_list],
        min_actor_dan=int(min_actor_dan),
        actor_filter_manifest=str(dataset_cfg.get('actor_filter_manifest', '') or ''),
        actor_filter_index=str(dataset_cfg.get('actor_filter_index', '') or ''),
        inputs_are_normalized=True,
    )
    return actor_filter_map, summary


def chunk_file_list(file_list: list[str], *, file_batch_size: int) -> list[list[str]]:
    return [
        file_list[start_idx:start_idx + file_batch_size]
        for start_idx in range(0, len(file_list), file_batch_size)
    ]


def _split_runtime_knobs(dataset_cfg: dict, *, split: str) -> dict:
    is_train = split == 'train'
    return {
        'num_workers': int(dataset_cfg.get('num_workers' if is_train else 'eval_num_workers', 0)),
        'prefetch_chunks': int(dataset_cfg.get('prefetch_chunks' if is_train else 'eval_prefetch_chunks', 0)),
        'prefetch_out_of_order': bool(dataset_cfg.get('prefetch_out_of_order' if is_train else 'eval_prefetch_out_of_order', False)),
        'device_prefetch_batches': int(dataset_cfg.get('device_prefetch_batches' if is_train else 'eval_device_prefetch_batches', 1)),
        'device_prefetch_startup_batches': int(
            dataset_cfg.get('device_prefetch_startup_batches' if is_train else 'eval_device_prefetch_startup_batches', 1)
        ),
        'pin_memory': bool(dataset_cfg.get('pin_memory' if is_train else 'eval_pin_memory', False)),
        'prefetch_startup_file_batch_size': int(
            dataset_cfg.get('prefetch_startup_file_batch_size' if is_train else 'eval_prefetch_startup_file_batch_size', 0)
        ),
        'shuffle': is_train,
    }


def _override_bool(value: bool, override: bool | None) -> bool:
    return value if override is None else bool(override)


def make_exact_benchmark_loader(
    full_config: dict,
    *,
    split: str,
    file_list: list[str],
    actor_filter_map: dict[str, tuple[int, ...]] | None,
    shuffle_override: bool | None = None,
    pin_memory_override: bool | None = None,
):
    bc_cfg = full_config.get('bc') or {}
    control_cfg = bc_cfg.get('control') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    stage_cfg = bc_cfg.get('stage') or {}
    runtime_cache_cfg = bc_cfg.get('runtime_cache') or {}
    if bool(stage_cfg.get('enabled', False)) or bool(runtime_cache_cfg.get('enabled', False)):
        raise ValueError(
            'run_conversion_vs_training_benchmark only supports the raw loader path '
            '(bc.stage.enabled=false and bc.runtime_cache.enabled=false)'
        )

    split_knobs = _split_runtime_knobs(dataset_cfg, split=split)
    split_knobs['shuffle'] = _override_bool(split_knobs['shuffle'], shuffle_override)
    split_knobs['pin_memory'] = _override_bool(split_knobs['pin_memory'], pin_memory_override)
    multiprocessing_context = str(dataset_cfg.get('multiprocessing_context', '') or '')
    batch_size = int(control_cfg.get('batch_size', 0) or 0)
    if batch_size <= 0:
        raise ValueError('bc.control.batch_size must be positive')

    player_names = load_name_filters(dataset_cfg.get('player_names_files', []))
    excludes = load_name_filters(dataset_cfg.get('exclude_names_files', []))

    loader, _loader_stats = build_action_file_dataloader(
        version=int(control_cfg.get('version', 4)),
        file_list=file_list,
        oracle=bool(dataset_cfg.get('oracle', False)),
        file_batch_size=int(dataset_cfg.get('file_batch_size', 20)),
        player_names=player_names or None,
        excludes=excludes or None,
        num_epochs=1,
        enable_augmentation=bool(dataset_cfg.get('enable_augmentation', False)),
        augmented_first=bool(dataset_cfg.get('augmented_first', False)),
        trust_seed=bool(dataset_cfg.get('trust_seed', False)),
        always_include_kan_select=bool(dataset_cfg.get('always_include_kan_select', True)),
        cycle=False,
        shuffle=bool(split_knobs['shuffle']),
        allowed_player_ids_by_path=actor_filter_map,
        prefetch_chunks=split_knobs['prefetch_chunks'],
        prefetch_strategy=str(dataset_cfg.get('prefetch_strategy', 'static_chunks') or 'static_chunks'),
        prefetch_budget_bytes=0,
        prefetch_target_chunk_bytes=0,
        prefetch_low_watermark=float(dataset_cfg.get('prefetch_low_watermark', 0.35)),
        prefetch_high_watermark=float(dataset_cfg.get('prefetch_high_watermark', 0.85)),
        prefetch_threads=int(dataset_cfg.get('prefetch_threads', 1)),
        decode_threads=1,
        batch_size=batch_size,
        prebatched=bool(dataset_cfg.get('prebatched', False)),
        prebatch_layout=str(dataset_cfg.get('prebatch_layout', 'chunk') or 'chunk'),
        prebatch_shuffle_mode=str(dataset_cfg.get('prebatch_shuffle_mode', 'sample') or 'sample'),
        prebatch_spill_across_chunks=bool(dataset_cfg.get('prebatch_spill_across_chunks', False)),
        prefetch_out_of_order=split_knobs['prefetch_out_of_order'],
        prefetch_startup_file_batch_size=split_knobs['prefetch_startup_file_batch_size'],
        num_workers=split_knobs['num_workers'],
        pin_memory=split_knobs['pin_memory'],
        multiprocessing_context=multiprocessing_context,
        loader_mode=str(dataset_cfg.get('loader_mode', 'baseline') or 'baseline'),
        loader_block_target_samples=int(dataset_cfg.get('loader_block_target_samples', 65536) or 65536),
    )
    return loader, {
        'batch_size': batch_size,
        'file_batch_size': int(dataset_cfg.get('file_batch_size', 20)),
        'num_workers': split_knobs['num_workers'],
        'pin_memory': split_knobs['pin_memory'],
        'handoff_pin_memory': bool(dataset_cfg.get('handoff_pin_memory' if split == 'train' else 'eval_handoff_pin_memory', False)),
        'prefetch_chunks': split_knobs['prefetch_chunks'],
        'prefetch_strategy': str(dataset_cfg.get('prefetch_strategy', 'static_chunks') or 'static_chunks'),
        'prebatched': bool(dataset_cfg.get('prebatched', False)),
        'prebatch_layout': str(dataset_cfg.get('prebatch_layout', 'chunk') or 'chunk'),
        'prebatch_shuffle_mode': str(dataset_cfg.get('prebatch_shuffle_mode', 'sample') or 'sample'),
        'prebatch_spill_across_chunks': bool(dataset_cfg.get('prebatch_spill_across_chunks', False)),
        'loader_mode': str(dataset_cfg.get('loader_mode', 'baseline') or 'baseline'),
        'loader_block_target_samples': int(dataset_cfg.get('loader_block_target_samples', 65536) or 65536),
        'device_prefetch_batches': split_knobs['device_prefetch_batches'],
        'device_prefetch_startup_batches': split_knobs['device_prefetch_startup_batches'],
        'prefetch_startup_file_batch_size': split_knobs['prefetch_startup_file_batch_size'],
        'shuffle': bool(split_knobs['shuffle']),
    }


def make_conversion_builder(
    full_config: dict,
    *,
    file_list: list[str],
    actor_filter_map: dict[str, tuple[int, ...]] | None,
    batch_size: int,
    shuffle_batches: bool,
    prebatched: bool = True,
    prebatch_spill_across_chunks: bool = False,
) -> ActionFileDatasetsIter:
    bc_cfg = full_config.get('bc') or {}
    control_cfg = bc_cfg.get('control') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}

    player_names = load_name_filters(dataset_cfg.get('player_names_files', []))
    excludes = load_name_filters(dataset_cfg.get('exclude_names_files', []))

    return ActionFileDatasetsIter(
        version=int(control_cfg.get('version', 4)),
        file_list=file_list,
        oracle=bool(dataset_cfg.get('oracle', False)),
        file_batch_size=int(dataset_cfg.get('file_batch_size', 20)),
        player_names=player_names or None,
        excludes=excludes or None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        trust_seed=bool(dataset_cfg.get('trust_seed', False)),
        always_include_kan_select=bool(dataset_cfg.get('always_include_kan_select', True)),
        cycle=False,
        shuffle=shuffle_batches,
        allowed_player_ids_by_path=actor_filter_map,
        prefetch_chunks=0,
        prefetch_strategy='static_chunks',
        prefetch_budget_bytes=0,
        prefetch_target_chunk_bytes=0,
        prefetch_threads=1,
        decode_threads=1,
        batch_size=batch_size,
        prebatched=prebatched,
        prebatch_layout=str(dataset_cfg.get('prebatch_layout', 'chunk') or 'chunk'),
        prebatch_shuffle_mode=str(dataset_cfg.get('prebatch_shuffle_mode', 'sample') or 'sample'),
        prebatch_spill_across_chunks=prebatch_spill_across_chunks,
        prefetch_out_of_order=False,
        prefetch_startup_file_batch_size=0,
    )


def benchmark_conversion(
    builder: ActionFileDatasetsIter,
    *,
    file_chunks: list[list[str]],
) -> tuple[list, dict]:
    chunks: list = []
    started_at = time.perf_counter()
    for chunk_files in file_chunks:
        chunk = builder.build_buffer_for_files(chunk_files, augmented=False)
        chunks.append(chunk)
    elapsed_seconds = time.perf_counter() - started_at
    snapshot = builder.loader_stats.snapshot()
    sample_count = sum(buffer_or_list_sample_count(chunk) for chunk in chunks)
    size_bytes = sum(buffer_or_list_size_bytes(chunk) for chunk in chunks)
    return chunks, {
        'chunk_count': len(chunks),
        'file_count': sum(len(chunk_files) for chunk_files in file_chunks),
        'sample_count': sample_count,
        'size_bytes': size_bytes,
        'elapsed_seconds': elapsed_seconds,
        'samples_per_second': sample_count / max(elapsed_seconds, 1e-9),
        'bytes_per_second_gib': (size_bytes / (1024 ** 3)) / max(elapsed_seconds, 1e-9),
        'loader_snapshot': snapshot,
        'read_seconds': float(snapshot.get('chunk_read_seconds_total', 0.0)),
        'decompress_seconds': float(snapshot.get('chunk_decompress_seconds_total', 0.0)),
        'parse_seconds': float(snapshot.get('chunk_parse_seconds_total', 0.0)),
        'assemble_seconds': float(snapshot.get('chunk_assemble_seconds_total', 0.0)),
    }


def maybe_pin_batch(batch, *, pin_memory: bool, device: torch.device):
    if not pin_memory or device.type != 'cuda':
        return batch
    return tuple(tensor.pin_memory() for tensor in batch)


def buffer_or_list_sample_count(buffer) -> int:
    if isinstance(buffer, list):
        return len(buffer)
    return int(buffer_sample_count(buffer))


def buffer_or_list_size_bytes(buffer) -> int:
    if isinstance(buffer, list):
        return sum(batch_nbytes(entry) for entry in buffer)
    return int(buffer_size_bytes(buffer))


def collate_sample_entries(entries: list) -> tuple[torch.Tensor, ...]:
    batch = default_collate(entries)
    if isinstance(batch, list):
        return tuple(batch)
    if isinstance(batch, tuple):
        return batch
    raise TypeError(f'unexpected collated batch type: {type(batch)!r}')


def benchmark_stream_batch_materialization(
    builder: ActionFileDatasetsIter,
    *,
    buffers: list,
    pin_memory: bool,
    device: torch.device,
) -> tuple[list[tuple[torch.Tensor, ...]], dict]:
    batches: list[tuple[torch.Tensor, ...]] = []
    pending_entries: list = []
    started_at = time.perf_counter()
    for buffer in buffers:
        if isinstance(buffer, ActionBatchBuffer):
            for batch in buffer.batches:
                batches.append(maybe_pin_batch(batch, pin_memory=pin_memory, device=device))
            continue
        if isinstance(buffer, ActionChunkBuffer):
            for batch in builder.iter_batches_from_buffer(buffer):
                batches.append(maybe_pin_batch(batch, pin_memory=pin_memory, device=device))
            continue
        buffer_size = len(buffer)
        if builder.shuffle:
            order = random.sample(range(buffer_size), buffer_size)
        else:
            order = range(buffer_size)
        for idx in order:
            pending_entries.append(buffer[idx])
            if len(pending_entries) < builder.batch_size:
                continue
            batch = collate_sample_entries(pending_entries)
            batches.append(maybe_pin_batch(batch, pin_memory=pin_memory, device=device))
            pending_entries = []
    if pending_entries:
        batch = collate_sample_entries(pending_entries)
        batches.append(maybe_pin_batch(batch, pin_memory=pin_memory, device=device))
    elapsed_seconds = time.perf_counter() - started_at
    sample_count = 0
    size_bytes = 0
    for batch in batches:
        if len(batch) == 4:
            _obs, _invisible_obs, actions, _masks = batch
        else:
            _obs, actions, _masks = batch
        sample_count += int(actions.shape[0])
        size_bytes += batch_nbytes(batch)
    return batches, {
        'batch_count': len(batches),
        'sample_count': sample_count,
        'size_bytes': size_bytes,
        'elapsed_seconds': elapsed_seconds,
        'samples_per_second': sample_count / max(elapsed_seconds, 1e-9),
        'bytes_per_second_gib': (size_bytes / (1024 ** 3)) / max(elapsed_seconds, 1e-9),
        'avg_batch_bytes_gib': (size_bytes / max(len(batches), 1)) / (1024 ** 3),
    }


def benchmark_batch_materialization(
    builder: ActionFileDatasetsIter,
    *,
    chunks: list[ActionChunkBuffer | ActionBatchBuffer],
    pin_memory: bool,
    device: torch.device,
) -> tuple[list[tuple[torch.Tensor, ...]], dict]:
    batches: list[tuple[torch.Tensor, ...]] = []
    started_at = time.perf_counter()
    for chunk in chunks:
        for batch in builder.iter_batches_from_buffer(chunk):
            batches.append(maybe_pin_batch(batch, pin_memory=pin_memory, device=device))
    elapsed_seconds = time.perf_counter() - started_at
    sample_count = 0
    size_bytes = 0
    for batch in batches:
        if len(batch) == 4:
            _obs, _invisible_obs, actions, _masks = batch
        else:
            _obs, actions, _masks = batch
        sample_count += int(actions.shape[0])
        size_bytes += batch_nbytes(batch)
    return batches, {
        'batch_count': len(batches),
        'sample_count': sample_count,
        'size_bytes': size_bytes,
        'elapsed_seconds': elapsed_seconds,
        'samples_per_second': sample_count / max(elapsed_seconds, 1e-9),
        'bytes_per_second_gib': (size_bytes / (1024 ** 3)) / max(elapsed_seconds, 1e-9),
        'avg_batch_bytes_gib': (size_bytes / max(len(batches), 1)) / (1024 ** 3),
    }


def benchmark_raw_loader_to_cpu_batches(
    loader,
    *,
    split: str,
    loader_config: dict,
    max_batches: int = 0,
) -> tuple[list[tuple[torch.Tensor, ...]], dict]:
    loader_stats = getattr(getattr(loader, 'dataset', None), 'loader_stats', None)
    if loader_stats is None:
        loader_stats = getattr(loader, 'loader_stats', None)
    previous_snapshot = loader_stats.snapshot() if loader_stats is not None else {}
    batches: list[tuple[torch.Tensor, ...]] = []
    started_at = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        batches.append(tuple(tensor.detach() for tensor in batch))
        if max_batches > 0 and batch_idx >= max_batches:
            break
    elapsed_seconds = time.perf_counter() - started_at
    current_snapshot = loader_stats.snapshot() if loader_stats is not None else {}

    sample_count = 0
    size_bytes = 0
    batch_sizes: list[int] = []
    for batch in batches:
        if len(batch) == 4:
            _obs, _invisible_obs, actions, _masks = batch
        else:
            _obs, actions, _masks = batch
        batch_sample_count = int(actions.shape[0])
        sample_count += batch_sample_count
        size_bytes += batch_nbytes(batch)
        batch_sizes.append(batch_sample_count)

    loader_delta = loader_metrics_delta(previous_snapshot, current_snapshot) if loader_stats is not None else {}
    return batches, {
        'split': split,
        'batch_count': len(batches),
        'sample_count': sample_count,
        'size_bytes': size_bytes,
        'elapsed_seconds': elapsed_seconds,
        'samples_per_second': sample_count / max(elapsed_seconds, 1e-9),
        'bytes_per_second_gib': (size_bytes / (1024 ** 3)) / max(elapsed_seconds, 1e-9),
        'avg_batch_bytes_gib': (size_bytes / max(len(batches), 1)) / (1024 ** 3),
        'min_batch_samples': min(batch_sizes) if batch_sizes else 0,
        'max_batch_samples': max(batch_sizes) if batch_sizes else 0,
        'avg_batch_samples': (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0.0,
        'loader_config': loader_config,
        'loader_snapshot_before': previous_snapshot,
        'loader_snapshot_after': current_snapshot,
        'loader_delta': loader_delta,
    }


def build_optimizer(
    *,
    mortal: Brain,
    dqn: DQN,
    optim_cfg: dict,
    fused_optimizer_enabled: bool,
):
    decay_params = []
    no_decay_params = []
    for model in (mortal, dqn):
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
        'lr': float(optim_cfg.get('lr', 1e-4)),
        'weight_decay': 0.0,
        'betas': tuple(optim_cfg.get('betas', [0.9, 0.999])),
        'eps': float(optim_cfg.get('eps', 1e-8)),
    }
    if fused_optimizer_enabled:
        optimizer_kwargs['fused'] = True
    return optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': float(optim_cfg.get('weight_decay', 0.0))},
            {'params': no_decay_params},
        ],
        **optimizer_kwargs,
    )


def benchmark_prepared_training(
    full_config: dict,
    *,
    batches: list[tuple[torch.Tensor, ...]],
    device: torch.device,
    queue_depth: int,
    startup_queue_depth: int,
    pin_handoff_batches: bool,
) -> dict:
    if not batches:
        raise ValueError('benchmark_prepared_training requires at least one batch')

    bc_cfg = full_config.get('bc') or {}
    control_cfg = bc_cfg.get('control') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    optim_cfg = bc_cfg.get('optim') or {}
    resnet_cfg = bc_cfg.get('resnet', full_config.get('resnet', {})) or {}

    oracle = bool(dataset_cfg.get('oracle', False))
    version = int(control_cfg.get('version', 4))
    seed = int(control_cfg.get('seed', 0))
    enable_amp = bool(control_cfg.get('enable_amp', False))
    amp_dtype = resolve_amp_dtype(control_cfg)
    freeze_bn = bool(control_cfg.get('freeze_bn', False))
    fused_optimizer_enabled = resolve_fused_optimizer_enabled(optim_cfg=optim_cfg, device=device)

    seed_everything(seed, rank=0)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = bool(control_cfg.get('enable_cudnn_benchmark', False))
    apply_cuda_precision_settings(control_cfg=control_cfg, device=device)

    mortal = Brain(version=version, is_oracle=oracle, **resnet_cfg).to(device)
    dqn = DQN(version=version, hidden_dim=mortal.hidden_dim).to(device)
    mortal.freeze_bn(freeze_bn)
    mortal.train()
    dqn.train()

    optimizer = build_optimizer(
        mortal=mortal,
        dqn=dqn,
        optim_cfg=optim_cfg,
        fused_optimizer_enabled=fused_optimizer_enabled,
    )
    scaler = GradScaler(
        device.type,
        enabled=grad_scaler_enabled(enable_amp=enable_amp, amp_dtype=amp_dtype, device=device),
    )
    max_grad_norm = float(optim_cfg.get('max_grad_norm', 0.0) or 0.0)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    prefetcher = DeviceBatchPrefetcher(
        iter(batches),
        device=device,
        oracle=oracle,
        queue_depth=queue_depth,
        startup_queue_depth=startup_queue_depth,
        pin_handoff_batches=pin_handoff_batches,
    )
    sample_count = 0
    batch_count = 0
    for batch in prefetcher:
        if oracle:
            obs, invisible_obs, actions, masks = batch
        else:
            obs, actions, masks = batch
            invisible_obs = None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast_context_kwargs(device=device, enable_amp=enable_amp, amp_dtype=amp_dtype)):
            brain_out = mortal(obs, invisible_obs)
            phi = extract_policy_features(brain_out)
            raw_logits, masked_scores = dqn_policy_outputs(dqn, phi, masks)
            del raw_logits
            loss = nn.functional.cross_entropy(masked_scores, actions)
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(mortal.parameters()) + list(dqn.parameters()),
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        sample_count += int(actions.shape[0])
        batch_count += 1
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        'batch_count': batch_count,
        'sample_count': sample_count,
        'elapsed_seconds': elapsed_seconds,
        'samples_per_second': sample_count / max(elapsed_seconds, 1e-9),
        'steps_per_second': batch_count / max(elapsed_seconds, 1e-9),
        'memory': {
            'max_allocated_gib': torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == 'cuda' else 0.0,
            'max_reserved_gib': torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device.type == 'cuda' else 0.0,
        },
    }


def diagnose_training_warmup(
    full_config: dict,
    *,
    batches: list[tuple[torch.Tensor, ...]],
    device: torch.device,
    queue_depth: int,
    startup_queue_depth: int,
    pin_handoff_batches: bool,
) -> dict:
    if not batches:
        return {
            'step_count': 0,
            'first_step_seconds': 0.0,
            'mean_remaining_step_seconds': 0.0,
            'median_remaining_step_seconds': 0.0,
            'post_warmup_samples_per_second': 0.0,
            'includes_prefetch_startup': True,
        }

    bc_cfg = full_config.get('bc') or {}
    control_cfg = bc_cfg.get('control') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    optim_cfg = bc_cfg.get('optim') or {}
    resnet_cfg = bc_cfg.get('resnet', full_config.get('resnet', {})) or {}

    oracle = bool(dataset_cfg.get('oracle', False))
    version = int(control_cfg.get('version', 4))
    seed = int(control_cfg.get('seed', 0))
    enable_amp = bool(control_cfg.get('enable_amp', False))
    amp_dtype = resolve_amp_dtype(control_cfg)
    freeze_bn = bool(control_cfg.get('freeze_bn', False))
    fused_optimizer_enabled = resolve_fused_optimizer_enabled(optim_cfg=optim_cfg, device=device)

    seed_everything(seed, rank=0)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = bool(control_cfg.get('enable_cudnn_benchmark', False))
    apply_cuda_precision_settings(control_cfg=control_cfg, device=device)

    mortal = Brain(version=version, is_oracle=oracle, **resnet_cfg).to(device)
    dqn = DQN(version=version, hidden_dim=mortal.hidden_dim).to(device)
    mortal.freeze_bn(freeze_bn)
    mortal.train()
    dqn.train()

    optimizer = build_optimizer(
        mortal=mortal,
        dqn=dqn,
        optim_cfg=optim_cfg,
        fused_optimizer_enabled=fused_optimizer_enabled,
    )
    scaler = GradScaler(
        device.type,
        enabled=grad_scaler_enabled(enable_amp=enable_amp, amp_dtype=amp_dtype, device=device),
    )
    max_grad_norm = float(optim_cfg.get('max_grad_norm', 0.0) or 0.0)

    prefetcher = DeviceBatchPrefetcher(
        iter(batches),
        device=device,
        oracle=oracle,
        queue_depth=queue_depth,
        startup_queue_depth=startup_queue_depth,
        pin_handoff_batches=pin_handoff_batches,
    )
    step_seconds: list[float] = []
    sample_counts: list[int] = []
    for batch in prefetcher:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        if oracle:
            obs, invisible_obs, actions, masks = batch
        else:
            obs, actions, masks = batch
            invisible_obs = None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast_context_kwargs(device=device, enable_amp=enable_amp, amp_dtype=amp_dtype)):
            brain_out = mortal(obs, invisible_obs)
            phi = extract_policy_features(brain_out)
            raw_logits, masked_scores = dqn_policy_outputs(dqn, phi, masks)
            del raw_logits
            loss = nn.functional.cross_entropy(masked_scores, actions)
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(mortal.parameters()) + list(dqn.parameters()),
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started_at)
        sample_counts.append(int(actions.shape[0]))

    remaining_seconds = step_seconds[1:]
    remaining_samples = sample_counts[1:]
    median_remaining = 0.0
    if remaining_seconds:
        sorted_seconds = sorted(remaining_seconds)
        mid = len(sorted_seconds) // 2
        if len(sorted_seconds) % 2 == 1:
            median_remaining = sorted_seconds[mid]
        else:
            median_remaining = 0.5 * (sorted_seconds[mid - 1] + sorted_seconds[mid])
    return {
        'step_count': len(step_seconds),
        'first_step_seconds': step_seconds[0] if step_seconds else 0.0,
        'mean_remaining_step_seconds': (
            sum(remaining_seconds) / len(remaining_seconds)
            if remaining_seconds
            else 0.0
        ),
        'median_remaining_step_seconds': median_remaining,
        'post_warmup_samples_per_second': (
            sum(remaining_samples) / max(sum(remaining_seconds), 1e-9)
            if remaining_seconds
            else 0.0
        ),
        'includes_prefetch_startup': True,
    }


def compare_pipeline_stages(
    *,
    conversion_summary: dict,
    materialization_summary: dict,
    training_summary: dict,
) -> dict:
    conversion_seconds = float(conversion_summary.get('elapsed_seconds', 0.0))
    materialization_seconds = float(materialization_summary.get('elapsed_seconds', 0.0))
    training_seconds = float(training_summary.get('elapsed_seconds', 0.0))
    conversion_and_batch_seconds = conversion_seconds + materialization_seconds

    conversion_sps = float(conversion_summary.get('samples_per_second', 0.0))
    conversion_and_batch_sps = (
        float(materialization_summary.get('sample_count', 0))
        / max(conversion_and_batch_seconds, 1e-9)
    )
    training_sps = float(training_summary.get('samples_per_second', 0.0))

    return {
        'conversion_only_vs_training_time_ratio': conversion_seconds / max(training_seconds, 1e-9),
        'conversion_plus_batching_vs_training_time_ratio': conversion_and_batch_seconds / max(training_seconds, 1e-9),
        'conversion_only_vs_training_sps_ratio': conversion_sps / max(training_sps, 1e-9),
        'conversion_plus_batching_vs_training_sps_ratio': conversion_and_batch_sps / max(training_sps, 1e-9),
        'can_hide_conversion_only': conversion_seconds <= training_seconds,
        'can_hide_conversion_plus_batching': conversion_and_batch_seconds <= training_seconds,
        'conversion_bottleneck': conversion_and_batch_seconds > training_seconds,
    }


def compare_producer_vs_training(
    *,
    producer_summary: dict,
    training_summary: dict,
    warmup_summary: dict | None = None,
) -> dict:
    producer_seconds = float(producer_summary.get('elapsed_seconds', 0.0))
    training_seconds = float(training_summary.get('elapsed_seconds', 0.0))
    producer_sps = float(producer_summary.get('samples_per_second', 0.0))
    training_sps = float(training_summary.get('samples_per_second', 0.0))
    comparison = {
        'producer_vs_training_time_ratio': producer_seconds / max(training_seconds, 1e-9),
        'producer_vs_training_sps_ratio': producer_sps / max(training_sps, 1e-9),
        'can_hide_producer': producer_seconds <= training_seconds,
        'producer_bottleneck': producer_seconds > training_seconds,
    }
    if warmup_summary:
        post_warmup_sps = float(warmup_summary.get('post_warmup_samples_per_second', 0.0))
        sample_count = float(training_summary.get('sample_count', 0.0))
        if post_warmup_sps > 0 and sample_count > 0:
            post_warmup_training_seconds = sample_count / post_warmup_sps
            comparison['post_warmup_training_seconds_estimate'] = post_warmup_training_seconds
            comparison['producer_vs_post_warmup_training_time_ratio'] = (
                producer_seconds / max(post_warmup_training_seconds, 1e-9)
            )
            comparison['producer_vs_post_warmup_training_sps_ratio'] = (
                producer_sps / max(post_warmup_sps, 1e-9)
            )
            comparison['can_hide_producer_vs_post_warmup_training'] = (
                producer_seconds <= post_warmup_training_seconds
            )
        else:
            comparison['post_warmup_training_seconds_estimate'] = 0.0
            comparison['producer_vs_post_warmup_training_time_ratio'] = 0.0
            comparison['producer_vs_post_warmup_training_sps_ratio'] = 0.0
            comparison['can_hide_producer_vs_post_warmup_training'] = False
    return comparison


def run_chunk_stage_breakdown(
    full_config: dict,
    *,
    split: str,
    file_list: list[str],
    actor_filter_map: dict[str, tuple[int, ...]] | None,
    device: torch.device,
    shuffle_override: bool | None = None,
    pin_memory_override: bool | None = None,
) -> dict:
    bc_cfg = full_config.get('bc') or {}
    control_cfg = bc_cfg.get('control') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}

    split_knobs = _split_runtime_knobs(dataset_cfg, split=split)
    shuffle_batches = _override_bool(split_knobs['shuffle'], shuffle_override)
    pin_memory = _override_bool(split_knobs['pin_memory'], pin_memory_override)
    handoff_pin_batches = bool(dataset_cfg.get('handoff_pin_memory' if split == 'train' else 'eval_handoff_pin_memory', False))
    batch_size = int(control_cfg.get('batch_size', 0) or 0)
    if batch_size <= 0:
        raise ValueError('bc.control.batch_size must be positive')

    builder = make_conversion_builder(
        full_config,
        file_list=file_list,
        actor_filter_map=actor_filter_map,
        batch_size=batch_size,
        shuffle_batches=shuffle_batches,
        prebatched=False,
    )
    file_chunks = chunk_file_list(
        file_list,
        file_batch_size=int(dataset_cfg.get('file_batch_size', 20)),
    )
    chunks, conversion_summary = benchmark_conversion(builder, file_chunks=file_chunks)
    batches, materialization_summary = benchmark_stream_batch_materialization(
        builder,
        buffers=chunks,
        pin_memory=pin_memory,
        device=device,
    )
    training_summary = benchmark_prepared_training(
        full_config,
        batches=batches,
        device=device,
        queue_depth=split_knobs['device_prefetch_batches'],
        startup_queue_depth=split_knobs['device_prefetch_startup_batches'],
        pin_handoff_batches=handoff_pin_batches,
    )
    warmup_summary = diagnose_training_warmup(
        full_config,
        batches=batches,
        device=device,
        queue_depth=split_knobs['device_prefetch_batches'],
        startup_queue_depth=split_knobs['device_prefetch_startup_batches'],
        pin_handoff_batches=handoff_pin_batches,
    )
    comparison = compare_pipeline_stages(
        conversion_summary=conversion_summary,
        materialization_summary=materialization_summary,
        training_summary=training_summary,
    )
    return {
        'kind': 'chunk_local_module_breakdown',
        'notes': [
            'conversion_only measures ActionFileDatasetsIter.build_buffer_for_files() over the selected raw file chunks',
            'batch_materialization measures sample-buffer collation into CPU train batches using the same batch_size and per-buffer shuffle behavior as the live non-prebatched loader path',
            'the batchifier carries pending samples across chunk boundaries so the breakdown matches the live sample stream more closely than chunk-local batching',
        ],
        'split': split,
        'selected_file_count': len(file_list),
        'chunk_count': len(file_chunks),
        'shuffle': bool(shuffle_batches),
        'pin_memory': bool(pin_memory),
        'handoff_pin_memory': handoff_pin_batches,
        'conversion_only': conversion_summary,
        'batch_materialization': materialization_summary,
        'training_on_materialized_batches': training_summary,
        'training_warmup_diagnostic': warmup_summary,
        'comparison': comparison,
    }


def run_conversion_vs_training_benchmark(
    *,
    config_path: str | Path,
    split: str,
    sample_size: int,
    sample_strategy: str,
    shard_world_size: int,
    shard_rank: int,
    device: str,
    max_batches: int = 0,
    shuffle_override: bool | None = None,
    pin_memory_override: bool | None = None,
) -> dict:
    resolved_config_path, full_config = load_full_config(config_path)
    bc_cfg = full_config.get('bc') or {}
    control_cfg = bc_cfg.get('control') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}

    device_obj = torch.device(device)
    split_knobs = _split_runtime_knobs(dataset_cfg, split=split)
    handoff_pin_batches = bool(dataset_cfg.get('handoff_pin_memory' if split == 'train' else 'eval_handoff_pin_memory', False))

    full_file_list = resolve_split_file_list(full_config, split=split)
    sharded_file_list = shard_file_list_round_robin(
        full_file_list,
        rank=shard_rank,
        world_size=shard_world_size,
    )
    selected_files = select_benchmark_files(
        sharded_file_list,
        sample_size=sample_size,
        sample_strategy=sample_strategy,
    )
    actor_filter_map, actor_filter_summary = resolve_benchmark_actor_filter(
        full_config,
        file_list=selected_files,
    )

    seed_everything(int(control_cfg.get('seed', 0) or 0), rank=shard_rank)

    loader, loader_config = make_exact_benchmark_loader(
        full_config,
        split=split,
        file_list=selected_files,
        actor_filter_map=actor_filter_map,
        shuffle_override=shuffle_override,
        pin_memory_override=pin_memory_override,
    )
    produced_batches, producer_summary = benchmark_raw_loader_to_cpu_batches(
        loader,
        split=split,
        loader_config=loader_config,
        max_batches=max_batches,
    )
    training_summary = benchmark_prepared_training(
        full_config,
        batches=produced_batches,
        device=device_obj,
        queue_depth=split_knobs['device_prefetch_batches'],
        startup_queue_depth=split_knobs['device_prefetch_startup_batches'],
        pin_handoff_batches=handoff_pin_batches,
    )
    warmup_summary = diagnose_training_warmup(
        full_config,
        batches=produced_batches,
        device=device_obj,
        queue_depth=split_knobs['device_prefetch_batches'],
        startup_queue_depth=split_knobs['device_prefetch_startup_batches'],
        pin_handoff_batches=handoff_pin_batches,
    )
    comparison = compare_producer_vs_training(
        producer_summary=producer_summary,
        training_summary=training_summary,
        warmup_summary=warmup_summary,
    )
    module_breakdown = run_chunk_stage_breakdown(
        full_config,
        split=split,
        file_list=selected_files,
        actor_filter_map=actor_filter_map,
        device=device_obj,
        shuffle_override=shuffle_override,
        pin_memory_override=pin_memory_override,
    )

    return {
        'config_path': str(resolved_config_path),
        'split': split,
        'device': str(device_obj),
        'sample_strategy': sample_strategy,
        'sample_size_requested': sample_size,
        'selected_file_count': len(selected_files),
        'full_split_file_count': len(full_file_list),
        'sharded_split_file_count': len(sharded_file_list),
        'shard_world_size': shard_world_size,
        'shard_rank': shard_rank,
        'batch_size': int(control_cfg.get('batch_size', 0) or 0),
        'file_batch_size': int(dataset_cfg.get('file_batch_size', 20)),
        'max_batches': int(max_batches or 0),
        'shuffle_override': shuffle_override,
        'pin_memory_override': pin_memory_override,
        'handoff_pin_memory': handoff_pin_batches,
        'actor_filter_summary': actor_filter_summary,
        'raw_to_cpu_batches': producer_summary,
        'training_on_produced_batches': training_summary,
        'training_warmup_diagnostic': warmup_summary,
        'comparison': comparison,
        'module_stage_breakdown': module_breakdown,
    }
