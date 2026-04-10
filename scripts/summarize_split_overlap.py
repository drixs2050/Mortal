#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

from progress_report import ProgressReporter, count_lines


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize player overlap across train/val/test split files.',
    )
    parser.add_argument('--manifest', required=True, help='Path to a JSONL normalized manifest.')
    parser.add_argument('--split-dir', required=True, help='Directory containing split text files.')
    parser.add_argument('--output', required=True, help='Where to write the overlap summary JSON.')
    parser.add_argument(
        '--raw-manifest',
        action='append',
        default=[],
        help='Optional raw snapshot manifest path. May be provided more than once.',
    )
    parser.add_argument(
        '--root-dir',
        default='',
        help='Optional root dir used when split files contain absolute paths.',
    )
    parser.add_argument(
        '--split-name',
        action='append',
        default=[],
        help='Split basename to include, without .txt. Defaults to train,val,test.',
    )
    return parser.parse_args()


def load_jsonl(filename: Path) -> list[dict]:
    rows = []
    progress = ProgressReporter(
        total=count_lines(filename),
        desc='OVERLAP-LOAD',
        unit='line',
    )
    with filename.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            progress.update(status=f'rows={len(rows)}')
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'failed to parse {filename}:{lineno}: {exc}') from exc
    progress.close(status=f'rows={len(rows)}')
    return rows


def load_split_lines(filename: Path) -> list[str]:
    with filename.open(encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def infer_raw_manifest_paths(rows: list[dict], explicit_paths: list[str]) -> list[Path]:
    if explicit_paths:
        return [Path(path) for path in explicit_paths]

    inferred = []
    seen = set()
    for row in rows:
        source = row.get('source')
        raw_snapshot_id = row.get('raw_snapshot_id')
        if not source or not raw_snapshot_id:
            continue
        key = (source, raw_snapshot_id)
        if key in seen:
            continue
        seen.add(key)
        inferred.append(ROOT / 'data' / 'manifests' / 'raw' / source / f'{raw_snapshot_id}.json')

    missing = [str(path) for path in inferred if not path.exists()]
    if missing:
        joined = ', '.join(missing[:5])
        raise ValueError(f'failed to infer raw manifests; missing: {joined}')
    return inferred


def build_manifest_indexes(rows: list[dict], root_dir: str) -> tuple[dict[str, dict], dict[str, dict]]:
    relative_index = {}
    absolute_index = {}
    root_path = Path(root_dir).resolve() if root_dir else None
    for row in rows:
        relative_path = row.get('relative_path')
        if not relative_path:
            continue
        relative_index[relative_path] = row
        if root_path is not None:
            absolute_index[str((root_path / relative_path).resolve())] = row
    return relative_index, absolute_index


def resolve_split_row(
    path_str: str,
    relative_index: dict[str, dict],
    absolute_index: dict[str, dict],
) -> dict | None:
    if path_str in relative_index:
        return relative_index[path_str]
    if path_str in absolute_index:
        return absolute_index[path_str]

    candidate = Path(path_str)
    if candidate.is_absolute():
        resolved = str(candidate.resolve())
        return absolute_index.get(resolved)
    return None


def hash_player_name(source: str, player_name: str) -> str:
    return hashlib.sha256(f'{source}:{player_name}'.encode('utf-8')).hexdigest()[:16]


def build_raw_player_index(raw_manifest_paths: list[Path]) -> dict[tuple[str, str], list[str]]:
    index = {}
    progress = ProgressReporter(
        total=len(raw_manifest_paths),
        desc='OVERLAP-RAW',
        unit='manifest',
    )
    for raw_manifest_path in raw_manifest_paths:
        payload = json.loads(raw_manifest_path.read_text(encoding='utf-8'))
        source = payload.get('source')
        for file_row in payload.get('files', []):
            source_game_id = file_row.get('source_game_id')
            if not source or not source_game_id:
                continue
            index[(source, source_game_id)] = list(file_row.get('player_names') or [])
        progress.update(status=f'indexed_games={len(index)}')
    progress.close(status=f'indexed_games={len(index)}')
    return index


def summarize_split_overlap(
    rows: list[dict],
    split_map: dict[str, list[str]],
    raw_player_index: dict[tuple[str, str], list[str]],
    root_dir: str = '',
) -> dict:
    relative_index, absolute_index = build_manifest_indexes(rows, root_dir)
    split_stats = {}
    split_player_hash_sets = {}
    split_game_sets = {}
    total_split_paths = sum(len(paths) for paths in split_map.values())
    progress = ProgressReporter(
        total=total_split_paths if total_split_paths > 0 else 1,
        desc='OVERLAP',
        unit='path',
    )
    resolved_rows = 0

    for split_name, split_paths in split_map.items():
        game_ids = set()
        player_hashes = set()
        unresolved_split_paths = []
        unresolved_source_game_ids = []
        rows_without_player_names = []
        total_player_slots = 0

        for path_str in split_paths:
            row = resolve_split_row(path_str, relative_index, absolute_index)
            if row is None:
                unresolved_split_paths.append(path_str)
                progress.update(status=f'split={split_name} resolved={resolved_rows}')
                continue

            resolved_rows += 1
            source = row.get('source')
            source_game_id = row.get('source_game_id')
            if source_game_id:
                game_ids.add(source_game_id)

            raw_names = raw_player_index.get((source, source_game_id))
            if raw_names is None:
                unresolved_source_game_ids.append(source_game_id or '<missing>')
                progress.update(status=f'split={split_name} resolved={resolved_rows}')
                continue
            if not raw_names:
                rows_without_player_names.append(source_game_id or '<missing>')
                progress.update(status=f'split={split_name} resolved={resolved_rows}')
                continue

            total_player_slots += len(raw_names)
            for player_name in raw_names:
                if player_name:
                    player_hashes.add(hash_player_name(source or 'unknown', player_name))
            progress.update(status=f'split={split_name} resolved={resolved_rows}')

        split_player_hash_sets[split_name] = player_hashes
        split_game_sets[split_name] = game_ids
        split_stats[split_name] = {
            'game_count': len(game_ids),
            'unique_player_hash_count': len(player_hashes),
            'total_player_slots': total_player_slots,
            'unresolved_split_path_count': len(unresolved_split_paths),
            'unresolved_source_game_id_count': len(unresolved_source_game_ids),
            'rows_without_player_names_count': len(rows_without_player_names),
            'unresolved_split_path_examples': unresolved_split_paths[:5],
            'unresolved_source_game_id_examples': unresolved_source_game_ids[:5],
            'rows_without_player_names_examples': rows_without_player_names[:5],
        }

    progress.close(status=f'resolved={resolved_rows}')

    pairwise_overlap = {}
    split_names = list(split_map.keys())
    for left_name, right_name in combinations(split_names, 2):
        left_players = split_player_hash_sets[left_name]
        right_players = split_player_hash_sets[right_name]
        shared_players = left_players & right_players
        shared_games = split_game_sets[left_name] & split_game_sets[right_name]

        pairwise_overlap[f'{left_name}_{right_name}'] = {
            'left_split': left_name,
            'right_split': right_name,
            'shared_unique_player_hash_count': len(shared_players),
            'left_unique_player_hash_count': len(left_players),
            'right_unique_player_hash_count': len(right_players),
            'left_shared_fraction': (
                len(shared_players) / len(left_players) if left_players else 0.0
            ),
            'right_shared_fraction': (
                len(shared_players) / len(right_players) if right_players else 0.0
            ),
            'shared_source_game_count': len(shared_games),
        }

    all_split_overlap = None
    if split_names:
        shared_all = set.intersection(*(split_player_hash_sets[name] for name in split_names))
        shared_games_all = set.intersection(*(split_game_sets[name] for name in split_names))
        all_split_overlap = {
            'split_names': split_names,
            'shared_unique_player_hash_count': len(shared_all),
            'shared_source_game_count': len(shared_games_all),
        }

    return {
        'player_identity_mode': 'raw_names_hashed_from_raw_manifest',
        'splits': split_stats,
        'pairwise_overlap': pairwise_overlap,
        'all_split_overlap': all_split_overlap,
    }


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    split_dir = Path(args.split_dir)
    output_path = Path(args.output)
    split_names = args.split_name or ['train', 'val', 'test']

    rows = load_jsonl(manifest_path)
    raw_manifest_paths = infer_raw_manifest_paths(rows, args.raw_manifest)
    raw_player_index = build_raw_player_index(raw_manifest_paths)

    split_map = {
        split_name: load_split_lines(split_dir / f'{split_name}.txt')
        for split_name in split_names
    }
    summary = summarize_split_overlap(
        rows=rows,
        split_map=split_map,
        raw_player_index=raw_player_index,
        root_dir=args.root_dir,
    )
    summary.update({
        'manifest': str(manifest_path),
        'split_dir': str(split_dir),
        'raw_manifests': [str(path) for path in raw_manifest_paths],
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
