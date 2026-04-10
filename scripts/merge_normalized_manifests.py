#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path

from progress_report import ProgressReporter, count_lines


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LoadedManifestRow:
    manifest_path: Path
    row_number: int
    row: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge multiple normalized dataset manifests into one merged release manifest.',
    )
    parser.add_argument(
        '--manifest',
        action='append',
        default=[],
        help='Path to a normalized JSONL manifest. Repeat as needed.',
    )
    parser.add_argument(
        '--manifest-glob',
        action='append',
        default=[],
        help='Glob pattern for normalized JSONL manifests, relative to the repo root or absolute. Repeat as needed.',
    )
    parser.add_argument(
        '--dataset-id',
        required=True,
        help='Merged dataset id to write into every output row.',
    )
    parser.add_argument(
        '--output-manifest',
        required=True,
        help='Where to write the merged JSONL manifest.',
    )
    parser.add_argument(
        '--output-summary',
        required=True,
        help='Where to write the merge summary JSON.',
    )
    parser.add_argument(
        '--on-duplicate',
        choices=('error', 'keep-first'),
        default='error',
        help='How to handle duplicate source games across input manifests.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output files if they already exist.',
    )
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')))
            f.write('\n')


def expand_manifest_glob(raw_pattern: str) -> list[Path]:
    pattern_path = Path(raw_pattern).expanduser()
    if not pattern_path.is_absolute():
        pattern_path = ROOT / pattern_path
    matches = [Path(match).resolve() for match in sorted(glob(str(pattern_path)))]
    if not matches:
        raise ValueError(f'glob matched no manifests: {raw_pattern}')
    return matches


def collect_manifest_paths(raw_paths: list[str], raw_globs: list[str]) -> list[Path]:
    manifest_paths: list[Path] = []
    seen = set()

    for raw_path in raw_paths:
        path = resolve_path(raw_path)
        if path in seen:
            continue
        seen.add(path)
        manifest_paths.append(path)

    for raw_glob in raw_globs:
        for path in expand_manifest_glob(raw_glob):
            if path in seen:
                continue
            seen.add(path)
            manifest_paths.append(path)

    if not manifest_paths:
        raise ValueError('provide at least one --manifest or --manifest-glob')

    missing = [display_path(path) for path in manifest_paths if not path.exists()]
    if missing:
        joined = ', '.join(missing[:5])
        raise ValueError(f'input manifests not found: {joined}')

    return manifest_paths


def load_manifest_rows(manifest_paths: list[Path]) -> list[LoadedManifestRow]:
    total_lines = sum(count_lines(path) for path in manifest_paths)
    progress = ProgressReporter(
        total=total_lines if total_lines > 0 else 1,
        desc='MERGE-LOAD',
        unit='line',
    )
    loaded_rows: list[LoadedManifestRow] = []
    for manifest_path in manifest_paths:
        with manifest_path.open(encoding='utf-8') as f:
            for row_number, line in enumerate(f, start=1):
                line = line.strip()
                progress.update(status=f'manifest={manifest_path.name} rows={len(loaded_rows)}')
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'failed to parse {manifest_path}:{row_number}: {exc}') from exc
                loaded_rows.append(LoadedManifestRow(
                    manifest_path=manifest_path,
                    row_number=row_number,
                    row=row,
                ))
    progress.close(status=f'rows={len(loaded_rows)}')
    return loaded_rows


def row_merge_key(row: dict) -> tuple[str, str]:
    source = row.get('source')
    source_game_id = row.get('source_game_id')
    if source and source_game_id:
        return str(source), str(source_game_id)

    duplicate_group = row.get('duplicate_group')
    if duplicate_group:
        return 'duplicate_group', str(duplicate_group)

    relative_path = row.get('relative_path')
    if relative_path:
        return 'relative_path', str(relative_path)

    raise ValueError(
        'manifest row is missing merge identity fields; expected source/source_game_id, '
        f'duplicate_group, or relative_path: {row}'
    )


def merged_row_copy(row: dict, dataset_id: str) -> dict:
    out = dict(row)
    original_dataset_id = out.get('dataset_id')
    if original_dataset_id is not None and 'batch_dataset_id' not in out:
        out['batch_dataset_id'] = original_dataset_id
    out['dataset_id'] = dataset_id
    return out


def row_sort_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get('game_date') or ''),
        str(row.get('source_game_id') or ''),
        str(row.get('raw_snapshot_id') or ''),
        str(row.get('relative_path') or ''),
    )


def summarize_rows(rows: list[dict]) -> dict:
    game_dates = sorted(str(row.get('game_date')) for row in rows if row.get('game_date'))
    return {
        'row_count': len(rows),
        'dataset_ids': sorted({row.get('dataset_id') for row in rows if row.get('dataset_id')}),
        'raw_snapshot_ids': sorted({row.get('raw_snapshot_id') for row in rows if row.get('raw_snapshot_id')}),
        'date_range': {
            'min': game_dates[0] if game_dates else None,
            'max': game_dates[-1] if game_dates else None,
        },
    }


def merge_loaded_rows(
    loaded_rows: list[LoadedManifestRow],
    *,
    dataset_id: str,
    on_duplicate: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    merged_rows: list[dict] = []
    duplicate_rows: list[dict] = []
    seen_keys: dict[tuple[str, str], LoadedManifestRow] = {}
    rows_by_manifest: dict[Path, list[dict]] = {}
    progress = ProgressReporter(
        total=len(loaded_rows) if len(loaded_rows) > 0 else 1,
        desc='MERGE',
        unit='row',
    )

    for loaded in loaded_rows:
        rows_by_manifest.setdefault(loaded.manifest_path, []).append(loaded.row)
        merge_key = row_merge_key(loaded.row)
        previous = seen_keys.get(merge_key)
        if previous is not None:
            duplicate_rows.append({
                'source': loaded.row.get('source'),
                'source_game_id': loaded.row.get('source_game_id'),
                'duplicate_key': ':'.join(merge_key),
                'first_manifest': display_path(previous.manifest_path),
                'first_row_number': previous.row_number,
                'duplicate_manifest': display_path(loaded.manifest_path),
                'duplicate_row_number': loaded.row_number,
                'first_raw_snapshot_id': previous.row.get('raw_snapshot_id'),
                'duplicate_raw_snapshot_id': loaded.row.get('raw_snapshot_id'),
            })
            progress.update(status=f'merged={len(merged_rows)} duplicates={len(duplicate_rows)}')
            continue

        seen_keys[merge_key] = loaded
        merged_rows.append(merged_row_copy(loaded.row, dataset_id))
        progress.update(status=f'merged={len(merged_rows)} duplicates={len(duplicate_rows)}')

    progress.close(status=f'merged={len(merged_rows)} duplicates={len(duplicate_rows)}')

    per_manifest_summary = []
    for manifest_path, rows in rows_by_manifest.items():
        summary = summarize_rows(rows)
        summary['manifest'] = display_path(manifest_path)
        per_manifest_summary.append(summary)
    per_manifest_summary.sort(key=lambda item: item['manifest'])

    if duplicate_rows and on_duplicate == 'error':
        examples = ', '.join(
            f"{row['duplicate_key']} ({row['first_manifest']}:{row['first_row_number']} vs "
            f"{row['duplicate_manifest']}:{row['duplicate_row_number']})"
            for row in duplicate_rows[:5]
        )
        raise ValueError(
            f'found {len(duplicate_rows)} duplicate source games across manifests: {examples}'
        )

    merged_rows.sort(key=row_sort_key)
    return merged_rows, duplicate_rows, per_manifest_summary


def build_merge_summary(
    *,
    dataset_id: str,
    output_manifest_path: Path,
    manifest_paths: list[Path],
    merged_rows: list[dict],
    duplicate_rows: list[dict],
    per_manifest_summary: list[dict],
    on_duplicate: str,
) -> dict:
    merged_game_dates = sorted(str(row.get('game_date')) for row in merged_rows if row.get('game_date'))
    return {
        'merged_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'dataset_id': dataset_id,
        'output_manifest': display_path(output_manifest_path),
        'input_manifest_count': len(manifest_paths),
        'input_manifests': [display_path(path) for path in manifest_paths],
        'input_row_count': sum(summary['row_count'] for summary in per_manifest_summary),
        'output_row_count': len(merged_rows),
        'duplicate_policy': on_duplicate,
        'duplicate_count': len(duplicate_rows),
        'duplicate_examples': duplicate_rows[:5],
        'input_dataset_ids': sorted({
            dataset
            for summary in per_manifest_summary
            for dataset in summary['dataset_ids']
        }),
        'batch_dataset_ids': sorted({
            row.get('batch_dataset_id')
            for row in merged_rows
            if row.get('batch_dataset_id')
        }),
        'raw_snapshot_ids': sorted({
            row.get('raw_snapshot_id')
            for row in merged_rows
            if row.get('raw_snapshot_id')
        }),
        'source_counts': dict(sorted(
            Counter(row.get('source') for row in merged_rows if row.get('source')).items()
        )),
        'date_range': {
            'min': merged_game_dates[0] if merged_game_dates else None,
            'max': merged_game_dates[-1] if merged_game_dates else None,
        },
        'input_manifests_summary': per_manifest_summary,
    }


def main():
    args = parse_args()
    manifest_paths = collect_manifest_paths(args.manifest, args.manifest_glob)
    output_manifest_path = resolve_path(args.output_manifest)
    output_summary_path = resolve_path(args.output_summary)

    if not args.overwrite:
        for output_path in (output_manifest_path, output_summary_path):
            if output_path.exists():
                raise SystemExit(f'output already exists, re-run with --overwrite: {output_path}')

    loaded_rows = load_manifest_rows(manifest_paths)
    merged_rows, duplicate_rows, per_manifest_summary = merge_loaded_rows(
        loaded_rows,
        dataset_id=args.dataset_id,
        on_duplicate=args.on_duplicate,
    )

    write_jsonl(output_manifest_path, merged_rows)

    summary = build_merge_summary(
        dataset_id=args.dataset_id,
        output_manifest_path=output_manifest_path,
        manifest_paths=manifest_paths,
        merged_rows=merged_rows,
        duplicate_rows=duplicate_rows,
        per_manifest_summary=per_manifest_summary,
        on_duplicate=args.on_duplicate,
    )
    ensure_parent(output_summary_path)
    output_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
