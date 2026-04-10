#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'
os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import load_full_config, utc_now_iso  # noqa: E402
from bc_dataset import resolve_actor_filter_map  # noqa: E402
from bc_runtime import seed_everything, shard_file_list_round_robin  # noqa: E402
from dataloader import build_action_file_dataloader  # noqa: E402
from step6_experiments import configured_split_lists, render_markdown_table  # noqa: E402


DEFAULT_CONFIG = 'configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml'
DEFAULT_COLUMNS = [
    ('name', 'Experiment'),
    ('loader_mode', 'Loader Mode'),
    ('loader_block_target_samples', 'Block Target'),
    ('batch_limit', 'Batches'),
    ('semantic_match', 'Semantic Match'),
    ('control_batches_checked', 'Control Checked'),
    ('candidate_batches_checked', 'Candidate Checked'),
    ('first_mismatch_batch', 'First Mismatch'),
    ('control_hash', 'Control Hash'),
    ('candidate_hash', 'Candidate Hash'),
    ('status', 'Status'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run the Phase 4 preassembled-batch semantic probe against the confirmed '
            'Step 6 control path before any GPU preflight benchmarking.'
        ),
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Confirmed Step 6 control config.')
    parser.add_argument(
        '--loader-block-target-samples',
        nargs='*',
        type=int,
        default=[65536],
        help='Candidate SampleBlock target sizes to compare against the control.',
    )
    parser.add_argument(
        '--batch-limit',
        type=int,
        default=16,
        help='Number of rank-local train batches to compare.',
    )
    parser.add_argument(
        '--output-json',
        default='artifacts/reports/step6_phase4_semantic_probe.json',
        help='JSON report path.',
    )
    parser.add_argument(
        '--output-md',
        default='artifacts/reports/step6_phase4_semantic_probe.md',
        help='Markdown report path.',
    )
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def save_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def batch_digest(batch) -> str:
    digest = hashlib.sha256()

    def update(value) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b'tensor')
            digest.update(str(tensor.dtype).encode('utf-8'))
            digest.update(json.dumps(list(tensor.shape)).encode('utf-8'))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b'ndarray')
            digest.update(str(array.dtype).encode('utf-8'))
            digest.update(json.dumps(list(array.shape)).encode('utf-8'))
            digest.update(array.tobytes())
            return
        if isinstance(value, dict):
            digest.update(b'dict')
            for key in sorted(value.keys()):
                digest.update(str(key).encode('utf-8'))
                update(value[key])
            return
        if isinstance(value, (tuple, list)):
            digest.update(b'seq')
            digest.update(str(len(value)).encode('utf-8'))
            for item in value:
                update(item)
            return
        digest.update(repr(value).encode('utf-8'))

    update(batch)
    return digest.hexdigest()


def build_rank0_train_inputs(full_config: dict) -> tuple[list[str], dict | None]:
    bc_cfg = full_config.get('bc') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    launch_cfg = bc_cfg.get('launch') or {}
    train_file_list = configured_split_lists(full_config, splits=['train'])['train']
    world_size = int(launch_cfg.get('nproc_per_node', 1) or 1)
    local_train_file_list = shard_file_list_round_robin(
        train_file_list,
        rank=0,
        world_size=world_size,
    )
    min_actor_dan = dataset_cfg.get('min_actor_dan')
    if min_actor_dan is None:
        return local_train_file_list, None
    actor_filter_map, _summary = resolve_actor_filter_map(
        file_lists=[train_file_list],
        min_actor_dan=int(min_actor_dan),
        actor_filter_manifest=str(dataset_cfg.get('actor_filter_manifest', '') or ''),
        actor_filter_index=str(dataset_cfg.get('actor_filter_index', '') or ''),
        inputs_are_normalized=True,
    )
    return local_train_file_list, actor_filter_map


def collect_batch_hashes(
    *,
    full_config: dict,
    file_list: list[str],
    actor_filter_map: dict | None,
    batch_limit: int,
    loader_mode: str,
    loader_block_target_samples: int,
) -> list[str]:
    bc_cfg = full_config.get('bc') or {}
    control_cfg = bc_cfg.get('control') or {}
    dataset_cfg = bc_cfg.get('dataset') or {}
    seed_everything(int(control_cfg.get('seed', 0) or 0), rank=0)
    loader, _loader_stats = build_action_file_dataloader(
        version=int(control_cfg.get('version', 4) or 4),
        file_list=file_list,
        oracle=bool(dataset_cfg.get('oracle', False)),
        file_batch_size=int(dataset_cfg.get('file_batch_size', 48) or 48),
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=bool(dataset_cfg.get('augmented_first', False)),
        trust_seed=bool(dataset_cfg.get('trust_seed', False)),
        always_include_kan_select=bool(dataset_cfg.get('always_include_kan_select', True)),
        cycle=False,
        shuffle=True,
        allowed_player_ids_by_path=actor_filter_map,
        prefetch_chunks=int(dataset_cfg.get('prefetch_chunks', 1) or 1),
        prefetch_strategy=str(dataset_cfg.get('prefetch_strategy', 'static_chunks') or 'static_chunks'),
        prefetch_budget_bytes=0,
        prefetch_target_chunk_bytes=0,
        prefetch_low_watermark=float(dataset_cfg.get('prefetch_low_watermark', 0.35) or 0.35),
        prefetch_high_watermark=float(dataset_cfg.get('prefetch_high_watermark', 0.85) or 0.85),
        prefetch_threads=int(dataset_cfg.get('prefetch_threads', 1) or 1),
        decode_threads=1,
        batch_size=int(control_cfg.get('batch_size', 8192) or 8192),
        prebatched=False,
        prebatch_layout='chunk',
        prebatch_shuffle_mode='sample',
        prebatch_spill_across_chunks=False,
        prefetch_out_of_order=bool(dataset_cfg.get('prefetch_out_of_order', False)),
        prefetch_startup_file_batch_size=int(
            dataset_cfg.get('prefetch_startup_file_batch_size', 0) or 0
        ),
        num_workers=0,
        pin_memory=False,
        multiprocessing_context='spawn',
        persistent_workers=False,
        prefetch_factor=2,
        in_order=True,
        raw_source_backend='files',
        raw_pack_path='',
        raw_pack_index_path='',
        loader_mode=loader_mode,
        loader_block_target_samples=int(loader_block_target_samples),
    )
    iterator = iter(loader)
    digests: list[str] = []
    try:
        for _ in range(batch_limit):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            digests.append(batch_digest(batch))
    finally:
        shutdown = getattr(iterator, '_shutdown_workers', None)
        if callable(shutdown):
            shutdown()
        del iterator
        del loader
    return digests


def compare_hashes(control_hashes: list[str], candidate_hashes: list[str]) -> tuple[bool, int | None]:
    for idx, (control_hash, candidate_hash) in enumerate(zip(control_hashes, candidate_hashes), start=1):
        if control_hash != candidate_hash:
            return False, idx
    if len(control_hashes) != len(candidate_hashes):
        return False, min(len(control_hashes), len(candidate_hashes)) + 1
    return True, None


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    output_json = resolve_path(args.output_json)
    output_md = resolve_path(args.output_md)

    _resolved_config_path, full_config = load_full_config(config_path)
    local_train_file_list, actor_filter_map = build_rank0_train_inputs(full_config)

    control_hashes = collect_batch_hashes(
        full_config=full_config,
        file_list=local_train_file_list,
        actor_filter_map=actor_filter_map,
        batch_limit=args.batch_limit,
        loader_mode='baseline',
        loader_block_target_samples=65536,
    )

    rows = []
    for block_target_samples in args.loader_block_target_samples:
        candidate_hashes = collect_batch_hashes(
            full_config=full_config,
            file_list=local_train_file_list,
            actor_filter_map=actor_filter_map,
            batch_limit=args.batch_limit,
            loader_mode='preassembled_batches',
            loader_block_target_samples=int(block_target_samples),
        )
        semantic_match, first_mismatch_batch = compare_hashes(control_hashes, candidate_hashes)
        rows.append({
            'name': f'preassembled_{int(block_target_samples)}',
            'loader_mode': 'preassembled_batches',
            'loader_block_target_samples': int(block_target_samples),
            'batch_limit': int(args.batch_limit),
            'semantic_match': bool(semantic_match),
            'control_batches_checked': len(control_hashes),
            'candidate_batches_checked': len(candidate_hashes),
            'first_mismatch_batch': first_mismatch_batch,
            'control_hash': control_hashes[first_mismatch_batch - 1][:16] if first_mismatch_batch else control_hashes[0][:16] if control_hashes else '',
            'candidate_hash': candidate_hashes[first_mismatch_batch - 1][:16] if first_mismatch_batch and len(candidate_hashes) >= first_mismatch_batch else candidate_hashes[0][:16] if candidate_hashes else '',
            'status': 'semantic_match' if semantic_match else 'semantic_mismatch',
            'control_hashes': control_hashes,
            'candidate_hashes': candidate_hashes,
        })

    markdown = render_markdown_table(rows, columns=DEFAULT_COLUMNS)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown + '\n', encoding='utf-8')

    payload = {
        'created_at': utc_now_iso(),
        'source_config': str(config_path),
        'batch_limit': int(args.batch_limit),
        'rank': 0,
        'control_loader_mode': 'baseline',
        'control_hashes': control_hashes,
        'rows': rows,
        'columns': DEFAULT_COLUMNS,
        'markdown_table': markdown,
    }
    save_json(output_json, payload)


if __name__ == '__main__':
    main()
