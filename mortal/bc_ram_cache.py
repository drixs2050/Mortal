from __future__ import annotations

from copy import deepcopy


GIB = 1024 ** 3

DEFAULT_RUNTIME_CACHE_SETTINGS = {
    'enabled': False,
    'mode': 'prepared_ram',
    'node_ram_budget_gib': 160,
    'node_pinned_budget_gib': 8,
    'node_inflight_budget_gib': 8,
    'raw_lru_budget_gib': 0,
    'low_watermark': 0.65,
    'high_watermark': 0.85,
    'target_chunk_gib': 2,
    'max_chunk_gib': 4,
    'startup_ready_chunks': 2,
    'producer_threads': 1,
    'decode_threads': 4,
    'max_inflight_chunk_builders': 1,
    'min_files_per_chunk': 16,
    'max_files_per_chunk': 96,
    'eval_node_ram_budget_gib': 16,
    'eval_target_chunk_gib': 1,
    'eval_decode_threads': 2,
}


def resolve_runtime_cache_settings(full_config: dict) -> dict:
    bc_cfg = full_config.get('bc') or {}
    runtime_cfg = deepcopy(DEFAULT_RUNTIME_CACHE_SETTINGS)
    runtime_cfg.update((bc_cfg.get('runtime_cache') or {}))
    runtime_cfg['enabled'] = bool(runtime_cfg.get('enabled', False))
    runtime_cfg['mode'] = str(runtime_cfg.get('mode', 'prepared_ram') or 'prepared_ram')
    return runtime_cfg


def runtime_cache_enabled(full_config: dict) -> bool:
    runtime_cfg = resolve_runtime_cache_settings(full_config)
    return bool(runtime_cfg.get('enabled', False))


def gib_to_bytes(raw_value: float | int | None) -> int:
    return max(int(float(raw_value or 0) * GIB), 0)


def divide_bytes(total_bytes: int, parts: int) -> int:
    if total_bytes <= 0:
        return 0
    return max(total_bytes // max(int(parts), 1), 1)


def runtime_cache_split_settings(
    full_config: dict,
    *,
    split_name: str,
    world_size: int,
) -> dict:
    runtime_cfg = resolve_runtime_cache_settings(full_config)
    if split_name == 'train':
        node_ram_budget_bytes = gib_to_bytes(runtime_cfg.get('node_ram_budget_gib'))
        node_pinned_budget_bytes = gib_to_bytes(runtime_cfg.get('node_pinned_budget_gib'))
        node_inflight_budget_bytes = gib_to_bytes(runtime_cfg.get('node_inflight_budget_gib'))
        node_raw_lru_budget_bytes = gib_to_bytes(runtime_cfg.get('raw_lru_budget_gib'))
        target_chunk_bytes = gib_to_bytes(runtime_cfg.get('target_chunk_gib'))
        max_chunk_bytes = gib_to_bytes(runtime_cfg.get('max_chunk_gib'))
        decode_threads = int(runtime_cfg.get('decode_threads', 4) or 4)
    else:
        node_ram_budget_bytes = gib_to_bytes(runtime_cfg.get('eval_node_ram_budget_gib'))
        node_pinned_budget_bytes = 0
        node_inflight_budget_bytes = 0
        node_raw_lru_budget_bytes = 0
        target_chunk_bytes = gib_to_bytes(runtime_cfg.get('eval_target_chunk_gib'))
        max_chunk_bytes = max(target_chunk_bytes, 0)
        decode_threads = int(runtime_cfg.get('eval_decode_threads', 2) or 2)

    node_data_budget_bytes = max(node_ram_budget_bytes - node_pinned_budget_bytes, 0)
    node_ready_budget_bytes = max(
        node_data_budget_bytes - node_inflight_budget_bytes - node_raw_lru_budget_bytes,
        0,
    )

    divisor = world_size if split_name == 'train' else 1
    return {
        'enabled': runtime_cfg['enabled'],
        'mode': runtime_cfg['mode'],
        'split_name': split_name,
        'world_size': divisor,
        'node_ram_budget_bytes': node_ram_budget_bytes,
        'node_pinned_budget_bytes': node_pinned_budget_bytes,
        'node_inflight_budget_bytes': node_inflight_budget_bytes,
        'node_raw_lru_budget_bytes': node_raw_lru_budget_bytes,
        'node_data_budget_bytes': node_data_budget_bytes,
        'node_ready_budget_bytes': node_ready_budget_bytes,
        'data_budget_bytes': divide_bytes(node_data_budget_bytes, divisor),
        'ready_budget_bytes': divide_bytes(node_ready_budget_bytes, divisor),
        'pinned_budget_bytes': divide_bytes(node_pinned_budget_bytes, divisor),
        'inflight_budget_bytes': divide_bytes(node_inflight_budget_bytes, divisor),
        'raw_lru_budget_bytes': divide_bytes(node_raw_lru_budget_bytes, divisor),
        'low_watermark': float(runtime_cfg.get('low_watermark', 0.65)),
        'high_watermark': float(runtime_cfg.get('high_watermark', 0.85)),
        'target_chunk_bytes': target_chunk_bytes,
        'max_chunk_bytes': max(max_chunk_bytes, target_chunk_bytes),
        'startup_ready_chunks': max(int(runtime_cfg.get('startup_ready_chunks', 2) or 2), 1),
        'producer_threads': 1,
        'decode_threads': max(decode_threads, 1),
        'max_inflight_chunk_builders': max(
            int(runtime_cfg.get('max_inflight_chunk_builders', 1) or 1),
            1,
        ),
        'min_files_per_chunk': max(int(runtime_cfg.get('min_files_per_chunk', 16) or 16), 1),
        'max_files_per_chunk': max(int(runtime_cfg.get('max_files_per_chunk', 96) or 96), 1),
    }
