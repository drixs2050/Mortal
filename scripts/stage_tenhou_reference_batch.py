#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from progress_report import ProgressReporter

ROOT = Path(__file__).resolve().parents[1]

from tenhou_xml import parse_tenhou_xml


XML_URL_TEMPLATE = 'https://tenhou.net/0/log/?{source_game_id}'
DEFAULT_STAGE_JOBS = 16
DEFAULT_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0


@dataclass(frozen=True)
class StagedReplay:
    index: int
    file_rows: list[dict]
    total_bytes: int
    staged_xml_count: int
    staged_json_count: int


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage a small Tenhou raw snapshot from official XML fetches with optional local replay-JSON oracles.',
    )
    parser.add_argument(
        '--snapshot-id',
        required=True,
        help='Snapshot id to create under data/raw/tenhou/.',
    )
    parser.add_argument(
        '--refs-file',
        required=True,
        help='Text file containing one Tenhou source_game_id per line.',
    )
    parser.add_argument(
        '--source-json-dir',
        default='',
        help='Directory containing saved Tenhou replay JSON files named <source_game_id>.json or <source_game_id>.mjlog2json.json.',
    )
    parser.add_argument(
        '--output-root',
        default='data/raw/tenhou',
        help='Root directory for staged raw snapshots.',
    )
    parser.add_argument(
        '--manifest-root',
        default='data/manifests/raw/tenhou',
        help='Root directory for raw snapshot manifests.',
    )
    parser.add_argument(
        '--usage-status',
        default='mini-batch-validation-only',
        help='Usage status recorded in the raw manifest.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files in the snapshot directory and manifest.',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=DEFAULT_STAGE_JOBS,
        help='Number of concurrent replay downloads to stage.',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='Network timeout in seconds for replay downloads.',
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=DEFAULT_RETRIES,
        help='Number of retry attempts for transient replay download failures.',
    )
    parser.add_argument(
        '--retry-backoff-seconds',
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help='Base exponential backoff in seconds between replay download retries.',
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_refs(path: Path) -> list[str]:
    refs = []
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        refs.append(line)
    if not refs:
        raise ValueError(f'no source_game_id entries found in {path}')
    return refs


def find_json_source(source_json_dir: Path, source_game_id: str) -> Path:
    candidates = [
        source_json_dir / f'{source_game_id}.mjlog2json.json',
        source_json_dir / f'{source_game_id}.json',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'could not find replay JSON for {source_game_id} in {source_json_dir}'
    )


def fetch_xml(
    source_game_id: str,
    output_path: Path,
    *,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> str:
    url = XML_URL_TEMPLATE.format(source_game_id=source_game_id)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0',
                },
            )
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            break
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, HTTPError) and exc.code == 404:
                raise
            if attempt >= retries:
                raise
            time.sleep(retry_backoff_seconds * (2 ** attempt))
    else:
        raise RuntimeError('replay download retries exhausted') from last_exc

    if b'<mjloggm' not in payload[:256]:
        preview = payload[:120].decode('utf-8', errors='replace')
        raise ValueError(f'official XML fetch did not look like a replay log: {preview!r}')
    ensure_parent(output_path)
    output_path.write_bytes(payload)
    return url


def copy_json(source_path: Path, output_path: Path) -> None:
    ensure_parent(output_path)
    shutil.copy2(source_path, output_path)


def build_xml_file_row(parsed: dict, xml_path: Path) -> dict:
    return {
        'relative_path': str(xml_path.relative_to(ROOT)),
        'source_game_id': parsed['source_game_id'],
        'sha256': compute_sha256(xml_path),
        'byte_size': xml_path.stat().st_size,
        'lobby': str(parsed['lobby']),
        'type': str(parsed['go_type']),
        'player_names': parsed['players']['names'],
        'player_dan': parsed['players']['dan_ids'],
        'player_rate': parsed['players']['rates'],
        'player_sex': parsed['players']['sexes'],
    }


def build_json_file_row(parsed: dict, json_path: Path, source_json_path: Path) -> dict:
    return {
        'relative_path': str(json_path.relative_to(ROOT)),
        'source_game_id': parsed['source_game_id'],
        'sha256': compute_sha256(json_path),
        'byte_size': json_path.stat().st_size,
        'content_type': 'saved_mjlog2json_oracle',
        'source_json_origin': str(source_json_path),
        'notes': [
            'Saved replay JSON used as the current Step 3 conversion oracle.',
            'XML remains the primary raw source of truth.',
        ],
        'lobby': str(parsed['lobby']),
        'lobby_display': parsed['lobby_display'],
        'ranking_lobby': parsed['ranking_lobby'],
        'rule_display': parsed.get('official_rule_display'),
        'room': parsed.get('official_room_code'),
        'player_dan_label': parsed['players']['dan_labels_oracle'] or parsed['players']['dan_labels_inferred'],
    }


def build_manifest_notes(usage_status: str, has_source_json: bool) -> list[str]:
    notes = [
        'Tenhou replay snapshot staged from official replay fetches.',
        'XML files were fetched from the official Tenhou replay path.',
    ]

    normalized_usage = usage_status.lower()
    if 'smoke' in normalized_usage:
        notes.append(
            'This snapshot is intended for bounded pipeline validation before larger corpus runs.',
        )
    elif 'corpus-expansion' in normalized_usage:
        notes.append(
            'This snapshot is intended for corpus expansion, ingestion, QA, and split generation.',
        )
    else:
        notes.append(
            'This snapshot records raw replay provenance for ingestion and QA.',
        )

    if has_source_json:
        notes.insert(
            1,
            'Replay JSON oracles were copied from a local reference set for validation parity checks.',
        )

    return notes


def stage_one_replay(
    *,
    index: int,
    source_game_id: str,
    output_dir: Path,
    source_json_dir: Path | None,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> StagedReplay:
    xml_output_path = output_dir / f'{source_game_id}.xml'
    fetch_xml(
        source_game_id,
        xml_output_path,
        timeout=timeout,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    source_json_path = None
    json_output_path = None
    if source_json_dir is not None:
        source_json_path = find_json_source(source_json_dir, source_game_id)
        json_output_path = output_dir / f'{source_game_id}.mjlog2json.json'
        copy_json(source_json_path, json_output_path)

    parsed = parse_tenhou_xml(
        xml_output_path,
        official_json_filename=json_output_path,
        include_round_events=False,
    )

    file_rows = []
    total_bytes = 0
    xml_row = build_xml_file_row(parsed, xml_output_path)
    file_rows.append(xml_row)
    total_bytes += xml_row['byte_size']

    staged_json_count = 0
    if json_output_path is not None and source_json_path is not None:
        json_row = build_json_file_row(parsed, json_output_path, source_json_path)
        file_rows.append(json_row)
        total_bytes += json_row['byte_size']
        staged_json_count = 1

    return StagedReplay(
        index=index,
        file_rows=file_rows,
        total_bytes=total_bytes,
        staged_xml_count=1,
        staged_json_count=staged_json_count,
    )


def main():
    args = parse_args()

    refs_file = ROOT / args.refs_file
    source_json_dir = None
    if args.source_json_dir:
        source_json_dir = Path(args.source_json_dir).expanduser().resolve()
    output_dir = ROOT / args.output_root / args.snapshot_id
    manifest_path = ROOT / args.manifest_root / f'{args.snapshot_id}.json'

    refs = load_refs(refs_file)
    if (output_dir.exists() and any(output_dir.iterdir())) and not args.overwrite:
        raise SystemExit(f'output directory is not empty, re-run with --overwrite: {output_dir}')
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f'manifest already exists, re-run with --overwrite: {manifest_path}')

    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    total_bytes = 0
    staged_xml_count = 0
    staged_json_count = 0
    progress = ProgressReporter(
        total=len(refs),
        desc='STAGE',
        unit='replay',
    )
    jobs = min(args.jobs, len(refs))
    ordered_results: list[StagedReplay | None] = [None] * len(refs)
    if jobs <= 1:
        for index, source_game_id in enumerate(refs):
            result = stage_one_replay(
                index=index,
                source_game_id=source_game_id,
                output_dir=output_dir,
                source_json_dir=source_json_dir,
                timeout=args.timeout,
                retries=args.retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
            ordered_results[index] = result
            staged_xml_count += result.staged_xml_count
            staged_json_count += result.staged_json_count
            total_bytes += result.total_bytes
            progress.update(
                status=(
                    f'xml={staged_xml_count} '
                    f'json={staged_json_count}'
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            pending = {}
            next_index = 0

            while next_index < len(refs) and len(pending) < jobs:
                future = executor.submit(
                    stage_one_replay,
                    index=next_index,
                    source_game_id=refs[next_index],
                    output_dir=output_dir,
                    source_json_dir=source_json_dir,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
                pending[future] = refs[next_index]
                next_index += 1

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    source_game_id = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        for pending_future in pending:
                            pending_future.cancel()
                        raise RuntimeError(f'failed staging {source_game_id}: {exc}') from exc

                    ordered_results[result.index] = result
                    staged_xml_count += result.staged_xml_count
                    staged_json_count += result.staged_json_count
                    total_bytes += result.total_bytes
                    progress.update(
                        status=(
                            f'xml={staged_xml_count} '
                            f'json={staged_json_count}'
                        ),
                    )

                    if next_index < len(refs):
                        new_future = executor.submit(
                            stage_one_replay,
                            index=next_index,
                            source_game_id=refs[next_index],
                            output_dir=output_dir,
                            source_json_dir=source_json_dir,
                            timeout=args.timeout,
                            retries=args.retries,
                            retry_backoff_seconds=args.retry_backoff_seconds,
                        )
                        pending[new_future] = refs[next_index]
                        next_index += 1

    for result in ordered_results:
        if result is None:
            raise RuntimeError('staging completed with a missing replay result')
        files.extend(result.file_rows)

    progress.close(
        status=(
            f'xml={staged_xml_count} '
            f'json={staged_json_count}'
        ),
    )

    notes = build_manifest_notes(
        usage_status=args.usage_status,
        has_source_json=source_json_dir is not None,
    )
    manifest = {
        'source': 'tenhou',
        'snapshot_id': args.snapshot_id,
        'acquired_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'acquired_by': 'codex',
        'official_access_pattern': XML_URL_TEMPLATE,
        'usage_status': args.usage_status,
        'selected_refs': refs,
        'notes': notes,
        'file_count': len(files),
        'total_bytes': total_bytes,
        'files': files,
    }
    ensure_parent(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    print(output_dir.relative_to(ROOT))
    print(manifest_path.relative_to(ROOT))


if __name__ == '__main__':
    main()
