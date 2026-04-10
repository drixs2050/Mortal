#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from progress_report import ProgressReporter


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from libriichi.dataset import GameplayLoader, Grp
from tenhou_xml import (
    build_normalized_manifest_row,
    xml_to_mjai_lines,
    official_json_to_mjai_lines,
    parse_tenhou_xml,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ingest a local Tenhou raw snapshot into normalized Mortal logs plus manifests.',
    )
    parser.add_argument(
        '--raw-snapshot-id',
        required=True,
        help='Snapshot id under data/raw/tenhou/, for example 2026-03-28_sample_a.',
    )
    parser.add_argument(
        '--dataset-id',
        default='tenhou_official_json_v0',
        help='Normalized dataset id written into success manifests.',
    )
    parser.add_argument(
        '--converter-version',
        default='tenhou-official-json-v0',
        help='Converter version string written into success manifests.',
    )
    parser.add_argument(
        '--raw-root',
        default='data/raw/tenhou',
        help='Root directory that contains per-snapshot raw Tenhou files.',
    )
    parser.add_argument(
        '--normalized-root',
        default='data/normalized/v1',
        help='Root directory where normalized logs will be written.',
    )
    parser.add_argument(
        '--manifest-root',
        default='data/manifests',
        help='Root directory for normalized/failure manifests.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Optional maximum number of official-json files to ingest from the snapshot.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Rewrite normalized outputs and manifests if they already exist.',
    )
    parser.add_argument(
        '--converter-source',
        choices=('auto', 'official_json', 'xml'),
        default='auto',
        help='Which Tenhou source to convert from. auto prefers saved replay JSON when present, otherwise XML.',
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_with_loaders(path: Path) -> dict:
    gameplay_loader = GameplayLoader(version=1, oracle=False)
    gameplay_data = gameplay_loader.load_gz_log_files([str(path)])
    if len(gameplay_data) != 1:
        raise ValueError(f'GameplayLoader returned {len(gameplay_data)} files for one input')
    if len(gameplay_data[0]) != 4:
        raise ValueError(f'GameplayLoader returned {len(gameplay_data[0])} player views instead of 4')

    gameplay_action_counts = []
    for game in gameplay_data[0]:
        gameplay_action_counts.append(len(game.take_actions()))

    grp_games = Grp.load_gz_log_files([str(path)])
    if len(grp_games) != 1:
        raise ValueError(f'Grp returned {len(grp_games)} games for one input')
    grp_game = grp_games[0]
    grp_feature = grp_game.take_feature()
    grp_ranks = grp_game.take_rank_by_player()
    grp_scores = grp_game.take_final_scores()

    return {
        'gameplay_file_count': len(gameplay_data),
        'gameplay_views': len(gameplay_data[0]),
        'gameplay_action_counts': gameplay_action_counts,
        'grp_game_count': len(grp_games),
        'grp_feature_shape': list(grp_feature.shape),
        'grp_rank_count': len(grp_ranks),
        'grp_final_score_count': len(grp_scores),
    }


def build_failure_row(*, raw_snapshot_id, run_id, source_game_id, category, message):
    return {
        'source': 'tenhou',
        'source_game_id': source_game_id,
        'raw_snapshot_id': raw_snapshot_id,
        'error_category': category,
        'error_message': str(message),
        'run_id': run_id,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')))
            f.write('\n')


def write_gz_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')))
            f.write('\n')


def relative_normalized_path(parsed: dict, normalized_root: Path) -> Path:
    return (
        normalized_root
        / 'source=tenhou'
        / f"year={parsed['year']:04d}"
        / f"month={parsed['month']:02d}"
        / f"{parsed['source_game_id']}.json.gz"
    )


def collect_source_game_ids(raw_dir: Path, converter_source: str) -> list[str]:
    xml_ids = {
        path.name.removesuffix('.xml')
        for path in raw_dir.glob('*.xml')
    }
    json_ids = {
        path.name.removesuffix('.mjlog2json.json')
        for path in raw_dir.glob('*.mjlog2json.json')
    }

    if converter_source == 'xml':
        return sorted(xml_ids)
    if converter_source == 'official_json':
        return sorted(json_ids)
    return sorted(xml_ids | json_ids)


def main():
    args = parse_args()

    raw_root = ROOT / args.raw_root
    normalized_root = ROOT / args.normalized_root
    manifest_root = ROOT / args.manifest_root
    raw_dir = raw_root / args.raw_snapshot_id
    if not raw_dir.exists():
        raise SystemExit(f'raw snapshot not found: {raw_dir}')

    success_manifest_path = manifest_root / 'normalized' / 'v1' / f'{args.dataset_id}.jsonl'
    failure_manifest_path = manifest_root / 'failures' / 'tenhou' / f'{args.dataset_id}__{args.raw_snapshot_id}.jsonl'
    summary_path = manifest_root / 'normalized' / 'v1' / f'{args.dataset_id}.summary.json'
    run_id = f'{args.dataset_id}__{args.raw_snapshot_id}'

    if not args.overwrite:
        for path in (success_manifest_path, failure_manifest_path, summary_path):
            if path.exists():
                raise SystemExit(f'output already exists, re-run with --overwrite: {path}')

    source_game_ids = collect_source_game_ids(raw_dir, args.converter_source)
    if args.limit > 0:
        source_game_ids = source_game_ids[:args.limit]

    success_rows = []
    failure_rows = []
    failures_by_category = Counter()
    progress = ProgressReporter(
        total=len(source_game_ids),
        desc='INGEST',
        unit='game',
    )

    for source_game_id in source_game_ids:
        official_json_path = raw_dir / f'{source_game_id}.mjlog2json.json'
        xml_path = raw_dir / f'{source_game_id}.xml'
        has_xml = xml_path.exists()
        has_official_json = official_json_path.exists()

        if args.converter_source == 'xml' and not has_xml:
            category = 'missing_xml'
            failure_rows.append(build_failure_row(
                raw_snapshot_id=args.raw_snapshot_id,
                run_id=run_id,
                source_game_id=source_game_id,
                category=category,
                message=f'missing XML source file: {xml_path.name}',
            ))
            failures_by_category[category] += 1
            progress.update(status=f'accepted={len(success_rows)} rejected={len(failure_rows)}')
            continue

        if args.converter_source == 'official_json' and not has_official_json:
            category = 'missing_official_json'
            failure_rows.append(build_failure_row(
                raw_snapshot_id=args.raw_snapshot_id,
                run_id=run_id,
                source_game_id=source_game_id,
                category=category,
                message=f'missing replay JSON source file: {official_json_path.name}',
            ))
            failures_by_category[category] += 1
            progress.update(status=f'accepted={len(success_rows)} rejected={len(failure_rows)}')
            continue

        if not has_xml:
            category = 'missing_xml'
            failure_rows.append(build_failure_row(
                raw_snapshot_id=args.raw_snapshot_id,
                run_id=run_id,
                source_game_id=source_game_id,
                category=category,
                message=f'missing XML companion file: {xml_path.name}',
            ))
            failures_by_category[category] += 1
            progress.update(status=f'accepted={len(success_rows)} rejected={len(failure_rows)}')
            continue

        try:
            parsed = parse_tenhou_xml(
                xml_path,
                official_json_filename=official_json_path if has_official_json else None,
                include_round_events=False,
            )
            relative_path = relative_normalized_path(parsed, Path(args.normalized_root))
            output_path = ROOT / relative_path
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f'normalized output already exists: {output_path}')

            if args.converter_source == 'xml':
                events = xml_to_mjai_lines(xml_path)
                conversion_source = 'xml'
            elif args.converter_source == 'official_json':
                events = official_json_to_mjai_lines(official_json_path)
                conversion_source = 'official_json'
            elif has_official_json:
                events = official_json_to_mjai_lines(official_json_path)
                conversion_source = 'official_json'
            else:
                events = xml_to_mjai_lines(xml_path)
                conversion_source = 'xml'
            write_gz_jsonl(output_path, events)
            loader_validation = validate_with_loaders(output_path)

            row = build_normalized_manifest_row(
                parsed,
                raw_snapshot_id=args.raw_snapshot_id,
                relative_path=str(relative_path),
                dataset_id=args.dataset_id,
                converter_version=args.converter_version,
                validation_status='loader_validated',
            )
            row['event_count'] = len(events)
            row['source_event_count'] = parsed['summary']['event_count']
            row['byte_size'] = output_path.stat().st_size
            row['file_sha256'] = compute_sha256(output_path)
            row['loader_validation'] = loader_validation
            row['conversion_source'] = conversion_source
            success_rows.append(row)
        except Exception as exc:
            category = exc.__class__.__name__
            failure_rows.append(build_failure_row(
                raw_snapshot_id=args.raw_snapshot_id,
                run_id=run_id,
                source_game_id=source_game_id,
                category=category,
                message=exc,
            ))
            failures_by_category[category] += 1
        progress.update(status=f'accepted={len(success_rows)} rejected={len(failure_rows)}')

    progress.close(status=f'accepted={len(success_rows)} rejected={len(failure_rows)}')

    write_jsonl(success_manifest_path, success_rows)
    write_jsonl(failure_manifest_path, failure_rows)

    summary = {
        'dataset_id': args.dataset_id,
        'run_id': run_id,
        'raw_snapshot_id': args.raw_snapshot_id,
        'candidate_count': len(source_game_ids),
        'accepted_count': len(success_rows),
        'rejected_count': len(failure_rows),
        'failures_by_category': dict(sorted(failures_by_category.items())),
        'success_manifest': display_path(success_manifest_path),
        'failure_manifest': display_path(failure_manifest_path),
    }
    ensure_parent(summary_path)
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
