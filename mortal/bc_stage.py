from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from glob import glob
from os import path
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from bc_dataset import load_path_cache, normalize_file_list, normalize_file_path, resolve_actor_filter_map
from dataloader import ActionFileDatasetsIter, resolve_prefetch_budget_bytes


ROOT = Path(__file__).resolve().parents[1]
STAGE_MANIFEST_FORMAT = 'bc_stage_manifest_v1'
STAGE_SHARD_META_FORMAT = 'bc_stage_shard_v1'
DEFAULT_STAGE_SETTINGS = {
    'enabled': False,
    'backend': 'npy_shards',
    'cache_root': 'artifacts/cache/bc_stage',
    'format_version': 1,
    'target_shard_size_mib': 1024,
    'max_shard_size_mib': 2048,
    'max_stage_size_gib': 0,
    'preload_ram_budget_gib': 160,
    'preload_low_watermark': 0.65,
    'preload_high_watermark': 0.90,
    'preload_threads': 4,
    'reuse': 'if_valid',
    'required_splits': ['train', 'val', 'test'],
    'allow_zarr': True,
}
_ZARR_MISSING = object()
_ZARR_MODULE = None


def _filtered_trimmed_lines(lines) -> list[str]:
    return [line.strip() for line in lines if line.strip()]


def _load_path_list(filename: str, root_dir: str = '') -> list[str]:
    with open(filename, encoding='utf-8') as f:
        paths = _filtered_trimmed_lines(f)
    if root_dir:
        return [
            p if path.isabs(p) else path.join(root_dir, p)
            for p in paths
        ]
    return paths


def _load_name_filters(raw_paths: list[str]) -> list[str]:
    names = set()
    for filename in raw_paths:
        with open(filename, encoding='utf-8') as f:
            names.update(_filtered_trimmed_lines(f))
    return sorted(names)


def resolve_stage_settings(full_config: dict) -> dict:
    bc_cfg = full_config.get('bc') or {}
    stage_cfg = dict(DEFAULT_STAGE_SETTINGS)
    stage_cfg.update((bc_cfg.get('stage') or {}))
    stage_cfg['backend'] = str(stage_cfg.get('backend', 'npy_shards') or 'npy_shards')
    stage_cfg['required_splits'] = [
        str(split).strip()
        for split in (stage_cfg.get('required_splits') or [])
        if str(split).strip()
    ]
    if not stage_cfg['required_splits']:
        stage_cfg['required_splits'] = ['train', 'val', 'test']
    return stage_cfg


def stage_enabled(full_config: dict) -> bool:
    return bool(resolve_stage_settings(full_config).get('enabled', False))


def _optional_zarr():
    global _ZARR_MODULE
    if _ZARR_MODULE is _ZARR_MISSING:
        return None
    if _ZARR_MODULE is None:
        try:
            import zarr  # type: ignore
        except ImportError:
            _ZARR_MODULE = _ZARR_MISSING
            return None
        _ZARR_MODULE = zarr
    return _ZARR_MODULE


def stage_backend_available(backend: str) -> bool:
    if backend == 'npy_shards':
        return True
    if backend == 'zarr':
        return _optional_zarr() is not None
    return False


def validate_stage_backend(stage_cfg: dict) -> None:
    backend = stage_cfg['backend']
    if backend not in ('npy_shards', 'zarr'):
        raise ValueError(f"unsupported bc.stage.backend: {backend}")
    if backend == 'zarr' and not bool(stage_cfg.get('allow_zarr', True)):
        raise ValueError('bc.stage.backend=zarr requires bc.stage.allow_zarr=true')
    if backend == 'zarr' and not stage_backend_available('zarr'):
        raise ValueError(
            'bc.stage.backend=zarr requested, but the zarr package is not installed in this environment'
        )


def _split_sources(dataset_cfg: dict, split: str) -> tuple[str, list[str]]:
    if split == 'train':
        return dataset_cfg.get('train_list', ''), dataset_cfg.get('train_globs', [])
    if split == 'val':
        return dataset_cfg.get('val_list', ''), dataset_cfg.get('val_globs', [])
    if split == 'test':
        return dataset_cfg.get('test_list', ''), dataset_cfg.get('test_globs', [])
    raise ValueError(f'unexpected split: {split}')


def resolve_split_file_lists(full_config: dict, splits: list[str]) -> dict[str, list[str]]:
    bc_cfg = full_config.get('bc') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    root_dir = dataset_cfg.get('root_dir', '')
    path_cache = dataset_cfg.get('path_cache', '')

    if path_cache and path.exists(path_cache):
        expected_sources = {
            split_name: dataset_cfg.get(f'{split_name}_list', '')
            for split_name in ('train', 'val', 'test')
            if dataset_cfg.get(f'{split_name}_list', '')
        }
        return load_path_cache(
            path_cache,
            expected_splits=splits,
            expected_sources=expected_sources,
        )

    split_lists = {}
    for split in splits:
        list_file, globs_list = _split_sources(dataset_cfg, split)
        if list_file:
            file_list = _load_path_list(list_file, root_dir)
        elif globs_list:
            file_list = []
            for pattern in globs_list:
                file_list.extend(glob(pattern, recursive=True))
            file_list.sort(reverse=True)
        else:
            raise ValueError(f'bc.dataset has no configured {split} split list or globs')
        split_lists[split] = normalize_file_list(file_list, desc=f'PATHS-{split.upper()}')
    return split_lists


def stage_required_splits(full_config: dict, *, requested_splits: list[str] | None = None) -> list[str]:
    stage_cfg = resolve_stage_settings(full_config)
    splits = requested_splits or stage_cfg['required_splits']
    normalized = []
    for split in splits:
        split_name = str(split).strip()
        if split_name not in ('train', 'val', 'test'):
            raise ValueError(f'unsupported stage split: {split_name}')
        if split_name not in normalized:
            normalized.append(split_name)
    return normalized


def stage_fingerprint(
    full_config: dict,
    *,
    split_lists: dict[str, list[str]],
) -> str:
    bc_cfg = full_config.get('bc') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    control_cfg = bc_cfg.get('control') or {}
    stage_cfg = resolve_stage_settings(full_config)
    payload = {
        'stage_backend': stage_cfg['backend'],
        'stage_format_version': stage_cfg['format_version'],
        'target_shard_size_mib': stage_cfg.get('target_shard_size_mib'),
        'max_shard_size_mib': stage_cfg.get('max_shard_size_mib'),
        'max_stage_size_gib': stage_cfg.get('max_stage_size_gib'),
        'splits': split_lists,
        'version': control_cfg.get('version'),
        'oracle': dataset_cfg.get('oracle', False),
        'trust_seed': dataset_cfg.get('trust_seed', False),
        'always_include_kan_select': dataset_cfg.get('always_include_kan_select', True),
        'min_actor_dan': dataset_cfg.get('min_actor_dan'),
        'actor_filter_index': dataset_cfg.get('actor_filter_index', ''),
        'actor_filter_manifest': dataset_cfg.get('actor_filter_manifest', ''),
        'player_names': _load_name_filters(dataset_cfg.get('player_names_files', [])),
        'exclude_names': _load_name_filters(dataset_cfg.get('exclude_names_files', [])),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def stage_base_dir(full_config: dict, *, split_lists: dict[str, list[str]]) -> Path:
    stage_cfg = resolve_stage_settings(full_config)
    fingerprint = stage_fingerprint(full_config, split_lists=split_lists)
    cache_root = Path(stage_cfg['cache_root']).expanduser()
    if not cache_root.is_absolute():
        cache_root = (ROOT / cache_root).resolve()
    return cache_root / f"v{int(stage_cfg['format_version'])}" / fingerprint / stage_cfg['backend']


def stage_manifest_path(full_config: dict, *, split: str, split_lists: dict[str, list[str]]) -> Path:
    if split not in split_lists:
        raise KeyError(f'missing split list for stage manifest path: {split}')
    split_specific_lists = {
        split: split_lists[split],
    }
    return stage_base_dir(full_config, split_lists=split_specific_lists) / split / 'manifest.json'


def stage_manifest_paths(full_config: dict, *, splits: list[str] | None = None) -> dict[str, Path]:
    resolved_splits = stage_required_splits(full_config, requested_splits=splits)
    split_lists = resolve_split_file_lists(full_config, resolved_splits)
    return {
        split: stage_manifest_path(full_config, split=split, split_lists=split_lists)
        for split in resolved_splits
    }


def _atomic_write_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    tmp_path.write_text(text, encoding='utf-8')
    tmp_path.replace(output_path)


def _empty_numpy_buffers(*, oracle: bool) -> dict[str, np.ndarray]:
    buffers = {
        'obs': np.empty((0,), dtype=np.int16),
        'actions': np.empty((0,), dtype=np.int64),
        'masks': np.empty((0, 46), dtype=np.bool_),
    }
    if oracle:
        buffers['invisible_obs'] = np.empty((0,), dtype=np.int16)
    return buffers


def _concat_shard_arrays(parts: dict[str, list[np.ndarray]], *, oracle: bool) -> dict[str, np.ndarray]:
    buffers = {}
    if parts['obs']:
        buffers['obs'] = np.concatenate(parts['obs'], axis=0)
        buffers['actions'] = np.concatenate(parts['actions'], axis=0)
        buffers['masks'] = np.concatenate(parts['masks'], axis=0)
        if oracle:
            buffers['invisible_obs'] = np.concatenate(parts['invisible_obs'], axis=0)
    else:
        buffers = _empty_numpy_buffers(oracle=oracle)
    return buffers


def _buffers_nbytes(buffers: dict[str, np.ndarray]) -> int:
    return int(sum(int(array.nbytes) for array in buffers.values()))


def _write_npy_shard(shard_dir: Path, *, buffers: dict[str, np.ndarray], meta: dict) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for name, array in buffers.items():
        np.save(shard_dir / f'{name}.npy', array, allow_pickle=False)
    _atomic_write_text(
        shard_dir / 'meta.json',
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + '\n',
    )


def _bytes_per_sample(buffers: dict[str, np.ndarray]) -> float:
    sample_count = int(buffers['actions'].shape[0])
    if sample_count <= 0:
        return 0.0
    return _buffers_nbytes(buffers) / sample_count


def _write_zarr_shard(shard_dir: Path, *, buffers: dict[str, np.ndarray], meta: dict) -> None:
    zarr = _optional_zarr()
    if zarr is None:
        raise ValueError('zarr backend requested, but zarr is not installed')
    shard_dir.mkdir(parents=True, exist_ok=True)
    sample_count = int(buffers['actions'].shape[0])
    bytes_per_sample = max(_bytes_per_sample(buffers), 1.0)
    target_chunk_bytes = 256 * (1024 ** 2)
    chunk_len = max(1, min(sample_count, int(target_chunk_bytes / bytes_per_sample)))
    group = zarr.open_group(str(shard_dir), mode='w')
    for name, array in buffers.items():
        chunks = (chunk_len,) + tuple(array.shape[1:])
        group.create_dataset(
            name,
            data=array,
            shape=array.shape,
            dtype=array.dtype,
            chunks=chunks,
            overwrite=True,
        )
    if hasattr(zarr, 'consolidate_metadata'):
        zarr.consolidate_metadata(str(shard_dir))
    _atomic_write_text(
        shard_dir / 'meta.json',
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + '\n',
    )


def _load_npy_shard(shard_meta: dict) -> tuple[dict[str, np.ndarray], int]:
    shard_dir = Path(shard_meta['path'])
    buffers = {
        'obs': np.load(shard_dir / 'obs.npy', allow_pickle=False),
        'actions': np.load(shard_dir / 'actions.npy', allow_pickle=False),
        'masks': np.load(shard_dir / 'masks.npy', allow_pickle=False),
    }
    if shard_meta.get('oracle', False):
        buffers['invisible_obs'] = np.load(shard_dir / 'invisible_obs.npy', allow_pickle=False)
    size_bytes = _buffers_nbytes(buffers)
    return buffers, size_bytes


def _load_zarr_shard(shard_meta: dict) -> tuple[dict[str, np.ndarray], int]:
    zarr = _optional_zarr()
    if zarr is None:
        raise ValueError('zarr backend requested, but zarr is not installed')
    shard_dir = Path(shard_meta['path'])
    group = zarr.open_group(str(shard_dir), mode='r')
    buffers = {
        'obs': np.asarray(group['obs']),
        'actions': np.asarray(group['actions']),
        'masks': np.asarray(group['masks']),
    }
    if shard_meta.get('oracle', False):
        buffers['invisible_obs'] = np.asarray(group['invisible_obs'])
    size_bytes = _buffers_nbytes(buffers)
    return buffers, size_bytes


def load_stage_manifest(manifest_path: str | Path) -> dict:
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding='utf-8'))
    if payload.get('format') != STAGE_MANIFEST_FORMAT:
        raise ValueError(f'unsupported stage manifest format in {manifest_file}')
    if payload.get('backend') not in ('npy_shards', 'zarr'):
        raise ValueError(f'unsupported stage manifest backend in {manifest_file}')
    for shard in payload.get('shards') or []:
        shard_path = Path(shard['path'])
        if not shard_path.exists():
            raise FileNotFoundError(f'missing stage shard path from manifest: {shard_path}')
    return payload


def _stage_chunk_builder(
    *,
    version: int,
    oracle: bool,
    player_names: list[str],
    excludes: list[str],
    trust_seed: bool,
    always_include_kan_select: bool,
    allowed_player_ids_by_path: dict[str, tuple[int, ...]] | None,
) -> ActionFileDatasetsIter:
    return ActionFileDatasetsIter(
        version=version,
        file_list=[],
        oracle=oracle,
        file_batch_size=1,
        player_names=player_names or None,
        excludes=excludes or None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        trust_seed=trust_seed,
        always_include_kan_select=always_include_kan_select,
        cycle=False,
        shuffle=False,
        allowed_player_ids_by_path=allowed_player_ids_by_path,
        batch_size=1,
        prebatched=True,
    )


def _flush_stage_shard(
    *,
    split_dir: Path,
    backend: str,
    split: str,
    shard_index: int,
    oracle: bool,
    fingerprint: str,
    parts: dict[str, list[np.ndarray]],
    file_count: int,
) -> dict | None:
    if not parts['obs']:
        return None
    buffers = _concat_shard_arrays(parts, oracle=oracle)
    sample_count = int(buffers['actions'].shape[0])
    size_bytes = _buffers_nbytes(buffers)
    shard_id = f'shard-{shard_index:06d}'
    shard_dir = split_dir / shard_id
    meta = {
        'format': STAGE_SHARD_META_FORMAT,
        'backend': backend,
        'split': split,
        'oracle': oracle,
        'fingerprint': fingerprint,
        'shard_id': shard_id,
        'sample_count': sample_count,
        'file_count': file_count,
        'size_bytes': size_bytes,
        'arrays': {
            name: {
                'shape': list(array.shape),
                'dtype': str(array.dtype),
                'nbytes': int(array.nbytes),
            }
            for name, array in buffers.items()
        },
    }
    if backend == 'npy_shards':
        _write_npy_shard(shard_dir, buffers=buffers, meta=meta)
    else:
        _write_zarr_shard(shard_dir, buffers=buffers, meta=meta)
    logging.info(
        'stage shard ready split=%s shard=%s files=%s samples=%s size_gib=%.3f backend=%s',
        split,
        shard_id,
        file_count,
        sample_count,
        size_bytes / (1024 ** 3),
        backend,
    )
    return {
        'shard_id': shard_id,
        'path': str(shard_dir),
        'sample_count': sample_count,
        'file_count': file_count,
        'size_bytes': size_bytes,
        'oracle': oracle,
        'backend': backend,
    }


def build_stage_cache(
    full_config: dict,
    *,
    splits: list[str] | None = None,
    force: bool = False,
) -> dict[str, dict]:
    if not stage_enabled(full_config):
        return {}
    stage_cfg = resolve_stage_settings(full_config)
    validate_stage_backend(stage_cfg)
    resolved_splits = stage_required_splits(full_config, requested_splits=splits)
    split_lists = resolve_split_file_lists(full_config, resolved_splits)

    bc_cfg = full_config.get('bc') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    control_cfg = bc_cfg.get('control') or {}
    version = int(control_cfg.get('version', 4))
    oracle = bool(dataset_cfg.get('oracle', False))
    trust_seed = bool(dataset_cfg.get('trust_seed', False))
    always_include_kan_select = bool(dataset_cfg.get('always_include_kan_select', True))
    player_names = _load_name_filters(dataset_cfg.get('player_names_files', []))
    excludes = _load_name_filters(dataset_cfg.get('exclude_names_files', []))
    min_actor_dan = dataset_cfg.get('min_actor_dan')
    actor_filter_manifest = dataset_cfg.get('actor_filter_manifest', '')
    actor_filter_index = dataset_cfg.get('actor_filter_index', '')
    actor_filter_map = None
    actor_filter_summary = None
    if min_actor_dan is not None:
        actor_filter_map, actor_filter_summary = resolve_actor_filter_map(
            file_lists=[split_lists[split] for split in resolved_splits],
            min_actor_dan=min_actor_dan,
            actor_filter_manifest=actor_filter_manifest,
            actor_filter_index=actor_filter_index,
            inputs_are_normalized=True,
        )

    target_shard_bytes = max(int(float(stage_cfg['target_shard_size_mib']) * (1024 ** 2)), 1)
    max_shard_bytes = max(int(float(stage_cfg['max_shard_size_mib']) * (1024 ** 2)), target_shard_bytes)
    max_stage_size_gib = float(stage_cfg.get('max_stage_size_gib', 0) or 0)
    max_stage_size_bytes = max(int(max_stage_size_gib * (1024 ** 3)), 0)
    results = {}
    builder = _stage_chunk_builder(
        version=version,
        oracle=oracle,
        player_names=player_names,
        excludes=excludes,
        trust_seed=trust_seed,
        always_include_kan_select=always_include_kan_select,
        allowed_player_ids_by_path=actor_filter_map,
    )

    for split in resolved_splits:
        split_specific_lists = {
            split: split_lists[split],
        }
        split_dir = stage_base_dir(full_config, split_lists=split_specific_lists) / split
        fingerprint = stage_fingerprint(full_config, split_lists=split_specific_lists)
        manifest_file = split_dir / 'manifest.json'
        if manifest_file.exists() and not force:
            try:
                payload = load_stage_manifest(manifest_file)
                logging.info(
                    'reusing staged cache split=%s backend=%s shards=%s files=%s samples=%s size_gib=%.3f manifest=%s',
                    split,
                    payload.get('backend', stage_cfg['backend']),
                    payload.get('shard_count', 0),
                    sum(int(shard.get('file_count', 0)) for shard in (payload.get('shards') or [])),
                    payload.get('sample_count', 0),
                    float(payload.get('size_bytes', 0)) / (1024 ** 3),
                    manifest_file,
                )
                results[split] = payload
                continue
            except Exception:
                logging.warning('rebuilding invalid stage manifest at %s', manifest_file)
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        total_files = len(split_lists[split])
        split_started = time.perf_counter()
        progress_every = 250
        processed_files = 0
        processed_samples = 0
        logging.info(
            'starting staged cache build split=%s backend=%s files=%s target_shard_mib=%s max_shard_mib=%s max_stage_size_gib=%.3f output_dir=%s',
            split,
            stage_cfg['backend'],
            total_files,
            stage_cfg['target_shard_size_mib'],
            stage_cfg['max_shard_size_mib'],
            max_stage_size_gib,
            split_dir,
        )

        current_parts = {
            'obs': [],
            'actions': [],
            'masks': [],
            'invisible_obs': [] if oracle else [],
        }
        current_bytes = 0
        current_file_count = 0
        shard_index = 0
        shard_entries = []
        total_sample_count = 0
        total_size_bytes = 0
        staged_file_count = 0
        truncated = False
        truncate_reason = ''
        try:
            for filename in split_lists[split]:
                if max_stage_size_bytes > 0 and total_size_bytes >= max_stage_size_bytes:
                    truncated = True
                    truncate_reason = 'max_stage_size_gib'
                    logging.info(
                        'stage size cap reached split=%s staged_files=%s/%s staged_size_gib=%.3f limit_gib=%.3f; stopping cache build early',
                        split,
                        staged_file_count,
                        total_files,
                        total_size_bytes / (1024 ** 3),
                        max_stage_size_gib,
                    )
                    break
                chunk = builder.build_buffer_for_files([filename], augmented=False)
                processed_files += 1
                if chunk.sample_count <= 0:
                    if processed_files % progress_every == 0 or processed_files == total_files:
                        logging.info(
                            'stage progress split=%s files=%s/%s (%.1f%%) samples=%s pending_shard_files=%s pending_shard_gib=%.3f shards=%s elapsed_seconds=%.1f',
                            split,
                            processed_files,
                            total_files,
                            (processed_files / max(total_files, 1)) * 100.0,
                            processed_samples,
                            current_file_count,
                            current_bytes / (1024 ** 3),
                            len(shard_entries),
                            time.perf_counter() - split_started,
                        )
                    continue
                chunk_obs = chunk.obs.numpy()
                chunk_actions = chunk.actions.numpy()
                chunk_masks = chunk.masks.numpy()
                chunk_bytes = int(chunk.size_bytes)
                next_buffered_bytes = total_size_bytes + current_bytes + chunk_bytes
                if (
                    max_stage_size_bytes > 0
                    and total_size_bytes > 0
                    and next_buffered_bytes > max_stage_size_bytes
                ):
                    truncated = True
                    truncate_reason = 'max_stage_size_gib'
                    logging.info(
                        'stage size cap reached split=%s staged_files=%s/%s staged_size_gib=%.3f limit_gib=%.3f; stopping cache build early',
                        split,
                        staged_file_count,
                        total_files,
                        total_size_bytes / (1024 ** 3),
                        max_stage_size_gib,
                    )
                    break
                processed_samples += int(chunk.sample_count)
                if current_parts['obs'] and (
                    current_bytes >= target_shard_bytes
                    or current_bytes + chunk_bytes > max_shard_bytes
                ):
                    shard_entry = _flush_stage_shard(
                        split_dir=split_dir,
                        backend=stage_cfg['backend'],
                        split=split,
                        shard_index=shard_index,
                        oracle=oracle,
                        fingerprint=fingerprint,
                        parts=current_parts,
                        file_count=current_file_count,
                    )
                    if shard_entry is not None:
                        shard_entries.append(shard_entry)
                        total_sample_count += shard_entry['sample_count']
                        total_size_bytes += shard_entry['size_bytes']
                        shard_index += 1
                    current_parts = {
                        'obs': [],
                        'actions': [],
                        'masks': [],
                        'invisible_obs': [] if oracle else [],
                    }
                    current_bytes = 0
                    current_file_count = 0
                current_parts['obs'].append(chunk_obs)
                current_parts['actions'].append(chunk_actions)
                current_parts['masks'].append(chunk_masks)
                if oracle:
                    current_parts['invisible_obs'].append(chunk.invisible_obs.numpy())
                current_bytes += chunk_bytes
                current_file_count += 1
                staged_file_count += 1
                if max_stage_size_bytes > 0 and total_size_bytes + current_bytes >= max_stage_size_bytes:
                    truncated = True
                    truncate_reason = 'max_stage_size_gib'
                    logging.info(
                        'stage size cap reached split=%s staged_files=%s/%s buffered_size_gib=%.3f limit_gib=%.3f; finalizing current shard and stopping cache build early',
                        split,
                        staged_file_count,
                        total_files,
                        (total_size_bytes + current_bytes) / (1024 ** 3),
                        max_stage_size_gib,
                    )
                    break
                if processed_files % progress_every == 0 or processed_files == total_files:
                    logging.info(
                        'stage progress split=%s files=%s/%s (%.1f%%) samples=%s pending_shard_files=%s pending_shard_gib=%.3f shards=%s elapsed_seconds=%.1f',
                        split,
                        processed_files,
                        total_files,
                        (processed_files / max(total_files, 1)) * 100.0,
                        processed_samples,
                        current_file_count,
                        current_bytes / (1024 ** 3),
                        len(shard_entries),
                        time.perf_counter() - split_started,
                    )

            shard_entry = _flush_stage_shard(
                split_dir=split_dir,
                backend=stage_cfg['backend'],
                split=split,
                shard_index=shard_index,
                oracle=oracle,
                fingerprint=fingerprint,
                parts=current_parts,
                file_count=current_file_count,
            )
            if shard_entry is not None:
                shard_entries.append(shard_entry)
                total_sample_count += shard_entry['sample_count']
                total_size_bytes += shard_entry['size_bytes']
        except Exception:
            shutil.rmtree(split_dir, ignore_errors=True)
            raise

        payload = {
            'format': STAGE_MANIFEST_FORMAT,
            'backend': stage_cfg['backend'],
            'fingerprint': fingerprint,
            'split': split,
            'oracle': oracle,
            'version': version,
            'sample_count': total_sample_count,
            'size_bytes': total_size_bytes,
            'shard_count': len(shard_entries),
            'truncated': bool(truncated),
            'truncate_reason': truncate_reason,
            'source_file_count': total_files,
            'staged_file_count': sum(int(shard.get('file_count', 0)) for shard in shard_entries),
            'max_stage_size_gib': max_stage_size_gib,
            'required_splits': resolved_splits,
            'source': {
                'path_cache': dataset_cfg.get('path_cache', ''),
                'actor_filter_index': actor_filter_index,
                'actor_filter_manifest': actor_filter_manifest,
                'min_actor_dan': min_actor_dan,
            },
            'actor_filter_summary': actor_filter_summary,
            'shards': shard_entries,
        }
        _atomic_write_text(
            manifest_file,
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + '\n',
        )
        logging.info(
            'completed staged cache split=%s backend=%s shards=%s files=%s/%s samples=%s size_gib=%.3f truncated=%s manifest=%s elapsed_seconds=%.1f',
            split,
            stage_cfg['backend'],
            payload['shard_count'],
            payload['staged_file_count'],
            total_files,
            payload['sample_count'],
            payload['size_bytes'] / (1024 ** 3),
            payload['truncated'],
            manifest_file,
            time.perf_counter() - split_started,
        )
        results[split] = payload
    return results


def ensure_stage_cache(
    full_config: dict,
    *,
    splits: list[str] | None = None,
    force: bool = False,
) -> dict[str, dict]:
    return build_stage_cache(full_config, splits=splits, force=force)


class StageLoaderStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = {
            'queued_bytes': 0,
            'max_queued_bytes': 0,
            'ready_chunks': 0,
            'chunk_count_total': 0,
            'chunk_files_total': 0,
            'chunk_samples_total': 0,
            'chunk_bytes_total': 0,
            'chunk_build_seconds_total': 0.0,
            'last_chunk_files': 0,
            'last_chunk_samples': 0,
            'last_chunk_bytes': 0,
            'last_chunk_build_seconds': 0.0,
            'discovered_shards': 0,
            'loaded_shards': 0,
            'resident_bytes': 0,
            'resident_shards': 0,
            'inflight_bytes': 0,
            'inflight_shards': 0,
        }

    def set_discovered_shards(self, count: int) -> None:
        with self._lock:
            self._values['discovered_shards'] = int(count)

    def update_resident(
        self,
        *,
        resident_bytes: int,
        resident_shards: int,
        inflight_bytes: int = 0,
        inflight_shards: int = 0,
    ) -> None:
        with self._lock:
            self._values['queued_bytes'] = max(int(resident_bytes), 0)
            self._values['resident_bytes'] = max(int(resident_bytes), 0)
            self._values['ready_chunks'] = max(int(resident_shards), 0)
            self._values['resident_shards'] = max(int(resident_shards), 0)
            self._values['inflight_bytes'] = max(int(inflight_bytes), 0)
            self._values['inflight_shards'] = max(int(inflight_shards), 0)
            self._values['max_queued_bytes'] = max(
                self._values['max_queued_bytes'],
                self._values['queued_bytes'],
            )

    def record_loaded_shard(self, *, file_count: int, sample_count: int, size_bytes: int, load_seconds: float) -> None:
        with self._lock:
            self._values['loaded_shards'] += 1
            self._values['chunk_count_total'] += 1
            self._values['chunk_files_total'] += int(file_count)
            self._values['chunk_samples_total'] += int(sample_count)
            self._values['chunk_bytes_total'] += int(size_bytes)
            self._values['chunk_build_seconds_total'] += float(load_seconds)
            self._values['last_chunk_files'] = int(file_count)
            self._values['last_chunk_samples'] = int(sample_count)
            self._values['last_chunk_bytes'] = int(size_bytes)
            self._values['last_chunk_build_seconds'] = float(load_seconds)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._values)


@dataclass
class LoadedShard:
    meta: dict
    buffers: dict[str, np.ndarray]
    size_bytes: int


class StagePreloadManager:
    def __init__(
        self,
        *,
        shard_entries: list[dict],
        backend: str,
        preload_budget_bytes: int,
        low_watermark: float,
        high_watermark: float,
        preload_threads: int,
        loader_stats: StageLoaderStats,
        log_prefix: str,
    ):
        self.shard_entries = list(shard_entries)
        self.backend = backend
        self.preload_budget_bytes = max(int(preload_budget_bytes or 0), 0)
        self.low_watermark_bytes = int(self.preload_budget_bytes * float(low_watermark))
        self.high_watermark_bytes = int(self.preload_budget_bytes * float(high_watermark))
        self.preload_threads = max(int(preload_threads or 1), 1)
        if self.preload_budget_bytes <= 0:
            self.high_watermark_bytes = 0
            self.low_watermark_bytes = 0
        self.loader_stats = loader_stats
        self.log_prefix = log_prefix
        self._condition = threading.Condition()
        self._loaded: OrderedDict[int, LoadedShard] = OrderedDict()
        self._next_load_idx = 0
        self._resident_bytes = 0
        self._inflight_bytes = 0
        self._pending: dict = {}
        self._error: Exception | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=self.preload_threads,
            thread_name_prefix='bc-stage-preload-worker',
        )
        self._thread = threading.Thread(target=self._run, name='bc-stage-preload', daemon=True)
        self.loader_stats.set_discovered_shards(len(self.shard_entries))
        self._thread.start()

    def _load_shard(self, shard_meta: dict) -> tuple[dict[str, np.ndarray], int]:
        if self.backend == 'npy_shards':
            return _load_npy_shard(shard_meta)
        if self.backend == 'zarr':
            return _load_zarr_shard(shard_meta)
        raise ValueError(f'unsupported stage backend: {self.backend}')

    def _reserved_bytes_for_shard(self, shard_meta: dict) -> int:
        return max(int(shard_meta.get('size_bytes', 0) or 0), 1)

    def _update_stats_locked(self) -> None:
        self.loader_stats.update_resident(
            resident_bytes=self._resident_bytes,
            resident_shards=len(self._loaded),
            inflight_bytes=self._inflight_bytes,
            inflight_shards=len(self._pending),
        )

    def _can_submit_locked(self) -> bool:
        if self._closed or self._error is not None:
            return False
        if self._next_load_idx >= len(self.shard_entries):
            return False
        if len(self._pending) >= self.preload_threads:
            return False
        shard_meta = self.shard_entries[self._next_load_idx]
        reserved_bytes = self._reserved_bytes_for_shard(shard_meta)
        if self.preload_budget_bytes <= 0:
            return True
        accounted_bytes = self._resident_bytes + self._inflight_bytes
        if accounted_bytes >= self.high_watermark_bytes:
            return False
        if accounted_bytes + reserved_bytes > self.preload_budget_bytes and accounted_bytes > 0:
            return False
        return True

    def _submit_next_locked(self) -> None:
        load_idx = self._next_load_idx
        shard_meta = self.shard_entries[load_idx]
        reserved_bytes = self._reserved_bytes_for_shard(shard_meta)
        started_at = time.perf_counter()
        future = self._executor.submit(self._load_shard, shard_meta)
        self._pending[future] = (load_idx, shard_meta, reserved_bytes, started_at)
        self._next_load_idx += 1
        self._inflight_bytes += reserved_bytes
        self._update_stats_locked()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._can_submit_locked():
                    self._submit_next_locked()
                if self._closed:
                    return
                if self._error is not None:
                    return
                if not self._pending and self._next_load_idx >= len(self.shard_entries):
                    return
                pending_futures = list(self._pending.keys())

            if not pending_futures:
                with self._condition:
                    self._condition.wait(timeout=0.1)
                continue

            done, _ = wait(pending_futures, timeout=0.1, return_when=FIRST_COMPLETED)
            if not done:
                continue

            for future in done:
                try:
                    buffers, size_bytes = future.result()
                except Exception as exc:
                    with self._condition:
                        pending_item = self._pending.pop(future, None)
                        if pending_item is not None:
                            _load_idx, _shard_meta, reserved_bytes, _started_at = pending_item
                            self._inflight_bytes = max(self._inflight_bytes - reserved_bytes, 0)
                        self._error = exc
                        self._update_stats_locked()
                        self._condition.notify_all()
                    logging.exception('%s shard preload failed: %s: %s', self.log_prefix, type(exc).__name__, exc)
                    return

                with self._condition:
                    pending_item = self._pending.pop(future, None)
                    if pending_item is None:
                        continue
                    load_idx, shard_meta, reserved_bytes, started_at = pending_item
                    self._inflight_bytes = max(self._inflight_bytes - reserved_bytes, 0)
                    actual_size_bytes = max(int(size_bytes), 0)
                    while (
                        not self._closed
                        and self.preload_budget_bytes > 0
                        and self._resident_bytes > 0
                        and self._resident_bytes + self._inflight_bytes + actual_size_bytes > self.preload_budget_bytes
                    ):
                        self._condition.wait(timeout=0.1)
                    if self._closed:
                        return
                    self._loaded[load_idx] = LoadedShard(
                        meta=shard_meta,
                        buffers=buffers,
                        size_bytes=actual_size_bytes,
                    )
                    self._resident_bytes += actual_size_bytes
                    load_seconds = time.perf_counter() - started_at
                    self.loader_stats.record_loaded_shard(
                        file_count=int(shard_meta.get('file_count', 0)),
                        sample_count=int(shard_meta.get('sample_count', 0)),
                        size_bytes=actual_size_bytes,
                        load_seconds=load_seconds,
                    )
                    self._update_stats_locked()
                    logging.info(
                        '%s loaded shard %s/%s samples=%s resident_gib=%.2f inflight_gib=%.2f ready_shards=%s',
                        self.log_prefix,
                        load_idx + 1,
                        len(self.shard_entries),
                        f"{int(shard_meta.get('sample_count', 0)):,}",
                        self._resident_bytes / (1024 ** 3),
                        self._inflight_bytes / (1024 ** 3),
                        len(self._loaded),
                    )
                    self._condition.notify_all()

    def get(self, idx: int) -> LoadedShard:
        with self._condition:
            while idx not in self._loaded:
                if self._error is not None:
                    raise RuntimeError('stage preload failed') from self._error
                if self._closed:
                    raise StopIteration
                self._condition.wait(timeout=0.1)
            return self._loaded[idx]

    def release(self, idx: int) -> None:
        with self._condition:
            loaded = self._loaded.pop(idx, None)
            if loaded is None:
                return
            self._resident_bytes = max(self._resident_bytes - loaded.size_bytes, 0)
            self._update_stats_locked()
            if (
                self.preload_budget_bytes <= 0
                or self._resident_bytes <= self.low_watermark_bytes
            ):
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._loaded.clear()
            self._resident_bytes = 0
            self._inflight_bytes = 0
            self._pending.clear()
            self.loader_stats.update_resident(
                resident_bytes=0,
                resident_shards=0,
                inflight_bytes=0,
                inflight_shards=0,
            )
            self._condition.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._thread.join(timeout=1.0)


class StagedShardIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        batch_size: int,
        shuffle: bool,
        cycle: bool,
        num_epochs: int,
        preload_budget_bytes: int,
        preload_low_watermark: float,
        preload_high_watermark: float,
        preload_threads: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.manifest = load_stage_manifest(self.manifest_path)
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self.cycle = bool(cycle)
        self.num_epochs = max(int(num_epochs or 1), 1)
        self.preload_budget_bytes = max(int(preload_budget_bytes or 0), 0)
        self.preload_low_watermark = float(preload_low_watermark)
        self.preload_high_watermark = float(preload_high_watermark)
        self.preload_threads = max(int(preload_threads or 1), 1)
        self.rank = max(int(rank or 0), 0)
        self.world_size = max(int(world_size or 1), 1)
        self.oracle = bool(self.manifest.get('oracle', False))
        self.backend = str(self.manifest.get('backend') or 'npy_shards')
        all_shards = list(self.manifest.get('shards') or [])
        self.shard_entries = all_shards[self.rank::self.world_size]
        self.loader_stats = StageLoaderStats()
        self.iterator = None

    def _iter_epoch(self):
        shard_entries = list(self.shard_entries)
        if self.shuffle:
            random.shuffle(shard_entries)
        manager = StagePreloadManager(
            shard_entries=shard_entries,
            backend=self.backend,
            preload_budget_bytes=self.preload_budget_bytes,
            low_watermark=self.preload_low_watermark,
            high_watermark=self.preload_high_watermark,
            preload_threads=self.preload_threads,
            loader_stats=self.loader_stats,
            log_prefix='stage preload:',
        )
        try:
            for idx, shard_meta in enumerate(shard_entries):
                loaded = manager.get(idx)
                buffers = loaded.buffers
                sample_count = int(buffers['actions'].shape[0])
                if sample_count <= 0:
                    manager.release(idx)
                    continue
                order = None
                if self.shuffle:
                    order = torch.randperm(sample_count)
                for start_idx in range(0, sample_count, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, sample_count)
                    if order is None:
                        obs_np = buffers['obs'][start_idx:end_idx]
                        actions_np = buffers['actions'][start_idx:end_idx]
                        masks_np = buffers['masks'][start_idx:end_idx]
                        invisible_np = (
                            buffers['invisible_obs'][start_idx:end_idx]
                            if self.oracle
                            else None
                        )
                    else:
                        batch_indices = order[start_idx:end_idx].numpy()
                        obs_np = buffers['obs'][batch_indices]
                        actions_np = buffers['actions'][batch_indices]
                        masks_np = buffers['masks'][batch_indices]
                        invisible_np = (
                            buffers['invisible_obs'][batch_indices]
                            if self.oracle
                            else None
                        )
                    obs = torch.from_numpy(obs_np)
                    actions = torch.from_numpy(actions_np)
                    masks = torch.from_numpy(masks_np)
                    if self.oracle:
                        invisible_obs = torch.from_numpy(invisible_np)
                        yield obs, invisible_obs, actions, masks
                    else:
                        yield obs, actions, masks
                manager.release(idx)
        finally:
            manager.close()

    def build_iter(self):
        while True:
            for _ in range(self.num_epochs):
                yield from self._iter_epoch()
            if not self.cycle:
                break

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def stage_preload_budget_bytes(*, full_config: dict, world_size: int) -> int:
    stage_cfg = resolve_stage_settings(full_config)
    return resolve_prefetch_budget_bytes(
        gib=stage_cfg.get('preload_ram_budget_gib', 0),
        world_size=world_size,
    )
