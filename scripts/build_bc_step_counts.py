#!/usr/bin/env python

import json
import os
import sys
from argparse import ArgumentParser
from glob import glob
from os import path
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))


def parse_args():
    default_jobs = max(1, min(16, (os.cpu_count() or 1) // 8))
    parser = ArgumentParser(
        description='Build learnable-step counts for the active Step 5 BC dataset config.',
    )
    parser.add_argument(
        '--split',
        action='append',
        choices=('train', 'val', 'test'),
        default=[],
        help='Split to count. Repeat to limit the build; defaults to all configured splits.',
    )
    parser.add_argument(
        '--output',
        default='',
        help='Optional output JSON path. Defaults to bc.dataset.step_count_summary in the active config.',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=default_jobs,
        help=(
            'Concurrent worker processes for counting. '
            f'Defaults to {default_jobs} on this machine; set 1 to keep the counter single-process.'
        ),
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=128,
        help='How many files to submit per counting task.',
    )
    return parser.parse_args()


def split_sources(dataset_cfg: dict, split: str) -> tuple[str, list[str]]:
    if split == 'train':
        return dataset_cfg.get('train_list', ''), dataset_cfg.get('train_globs', [])
    if split == 'val':
        return dataset_cfg.get('val_list', ''), dataset_cfg.get('val_globs', [])
    if split == 'test':
        return dataset_cfg.get('test_list', ''), dataset_cfg.get('test_globs', [])
    raise ValueError(f'unexpected split: {split}')


def filtered_trimmed_lines(lines):
    return [line.strip() for line in lines if line.strip()]


def load_path_list(filename: str, root_dir: str = '') -> list[str]:
    with open(filename, encoding='utf-8') as f:
        paths = filtered_trimmed_lines(f)
    if root_dir:
        return [
            p if path.isabs(p) else path.join(root_dir, p)
            for p in paths
        ]
    return paths


def resolve_split_file_list(dataset_cfg: dict, split: str, root_dir: str) -> list[str]:
    list_file, globs = split_sources(dataset_cfg, split)
    if list_file:
        return load_path_list(list_file, root_dir)
    if globs:
        file_list = []
        for pattern in globs:
            file_list.extend(glob(pattern, recursive=True))
        return sorted(file_list, reverse=True)
    raise ValueError(
        f'bc.dataset has no configured {split} split. '
        f'Expected {split}_list or {split}_globs.'
    )


def configured_splits(dataset_cfg: dict) -> list[str]:
    splits = []
    for split in ('train', 'val', 'test'):
        list_file, globs = split_sources(dataset_cfg, split)
        if list_file or globs:
            splits.append(split)
    return splits


def main():
    args = parse_args()

    import prelude  # noqa: F401

    from config import config
    from bc_dataset import (
        load_path_cache,
        normalize_file_list,
        resolve_actor_filter_map,
    )
    from bc_step_counts import (
        build_step_count_summary,
        save_step_count_summary,
        step_count_config_summary,
    )

    bc_cfg = config['bc']
    control_cfg = bc_cfg['control']
    dataset_cfg = bc_cfg['dataset']
    version = control_cfg.get('version', 4)
    root_dir = dataset_cfg.get('root_dir', '')
    path_cache = dataset_cfg.get('path_cache', '')
    splits = args.split or configured_splits(dataset_cfg)
    if not splits:
        raise ValueError('bc.dataset has no configured splits to count')

    if path_cache and path.exists(path_cache):
        split_lists = load_path_cache(
            path_cache,
            expected_splits=splits,
            expected_sources={
                split_name: dataset_cfg.get(f'{split_name}_list', '')
                for split_name in ('train', 'val', 'test')
                if dataset_cfg.get(f'{split_name}_list', '')
            },
        )
    else:
        split_lists = {
            split_name: normalize_file_list(
                resolve_split_file_list(dataset_cfg, split_name, root_dir),
                desc=f'PATHS-{split_name.upper()}',
            )
            for split_name in splits
        }

    player_names = []
    exclude_names = []
    for filename in dataset_cfg.get('player_names_files', []):
        with open(filename, encoding='utf-8') as f:
            player_names.extend(filtered_trimmed_lines(f))
    for filename in dataset_cfg.get('exclude_names_files', []):
        with open(filename, encoding='utf-8') as f:
            exclude_names.extend(filtered_trimmed_lines(f))

    actor_filter_map = None
    min_actor_dan = dataset_cfg.get('min_actor_dan')
    actor_filter_manifest = dataset_cfg.get('actor_filter_manifest', '')
    actor_filter_index = dataset_cfg.get('actor_filter_index', '')
    if min_actor_dan is not None:
        actor_filter_map, _ = resolve_actor_filter_map(
            file_lists=list(split_lists.values()),
            min_actor_dan=min_actor_dan,
            actor_filter_manifest=actor_filter_manifest,
            actor_filter_index=actor_filter_index,
            inputs_are_normalized=True,
        )

    output = args.output or dataset_cfg.get('step_count_summary', '')
    if not output:
        raise ValueError(
            'provide --output or set bc.dataset.step_count_summary in the active config'
        )

    summary = build_step_count_summary(
        split_lists=split_lists,
        version=version,
        oracle=dataset_cfg.get('oracle', False),
        file_batch_size=dataset_cfg['file_batch_size'],
        player_names=sorted(set(player_names)) or None,
        excludes=sorted(set(exclude_names)) or None,
        trust_seed=dataset_cfg.get('trust_seed', False),
        always_include_kan_select=dataset_cfg.get('always_include_kan_select', True),
        actor_filter_map=actor_filter_map,
        batch_size_reference=control_cfg['batch_size'],
        jobs=args.jobs,
        chunk_size=args.chunk_size,
        config_summary=step_count_config_summary(
            path_cache=path_cache,
            actor_filter_index=actor_filter_index,
            actor_filter_manifest=actor_filter_manifest,
            min_actor_dan=min_actor_dan,
            version=version,
            oracle=dataset_cfg.get('oracle', False),
            trust_seed=dataset_cfg.get('trust_seed', False),
            always_include_kan_select=dataset_cfg.get('always_include_kan_select', True),
            file_batch_size=dataset_cfg['file_batch_size'],
            batch_size_reference=control_cfg['batch_size'],
            jobs=args.jobs,
            chunk_size=args.chunk_size,
        ),
    )

    save_step_count_summary(output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
