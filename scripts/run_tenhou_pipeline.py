#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'


@dataclass(frozen=True)
class Step:
    name: str
    cmd: list[str]
    env_updates: dict[str, str]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the Tenhou archive fetch/select/stage/ingest pipeline as one command.',
    )
    parser.add_argument('--snapshot-id', required=True, help='Raw snapshot id to create.')
    parser.add_argument(
        '--dataset-id',
        default='',
        help='Normalized dataset id. Defaults to <snapshot-id>_v0.',
    )
    parser.add_argument(
        '--date',
        action='append',
        default=[],
        help='Specific archive date in YYYY-MM-DD format. Repeat as needed.',
    )
    parser.add_argument('--start-date', default='', help='Inclusive start date in YYYY-MM-DD format.')
    parser.add_argument('--end-date', default='', help='Inclusive end date in YYYY-MM-DD format.')
    parser.add_argument(
        '--archive-dir',
        default='',
        help='Directory for downloaded scc*.html.gz files. Defaults to /tmp/tenhou_<snapshot-id>.',
    )
    parser.add_argument(
        '--ruleset',
        action='append',
        default=[],
        help='Exact Tenhou archive ruleset label to keep. Repeat as needed.',
    )
    parser.add_argument(
        '--max-per-archive',
        type=int,
        default=0,
        help='Optional replay cap per archive during selection.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Optional global replay cap during selection.',
    )
    parser.add_argument(
        '--archive-jobs',
        type=int,
        default=8,
        help='Concurrent archive downloads during fetch.',
    )
    parser.add_argument(
        '--stage-jobs',
        type=int,
        default=16,
        help='Concurrent replay downloads during staging.',
    )
    parser.add_argument(
        '--download-timeout',
        type=float,
        default=30.0,
        help='Network timeout in seconds for fetch and stage downloads.',
    )
    parser.add_argument(
        '--download-retries',
        type=int,
        default=2,
        help='Retry attempts for transient fetch and stage download failures.',
    )
    parser.add_argument(
        '--retry-backoff-seconds',
        type=float,
        default=1.0,
        help='Base exponential backoff in seconds between download retries.',
    )
    parser.add_argument(
        '--archive-publish-lag-days',
        type=int,
        default=14,
        help='Treat recent archive 404s inside this lag window as not yet published instead of failed.',
    )
    parser.add_argument(
        '--year-archive-cache-dir',
        default='',
        help='Optional cache directory for old yearly Tenhou archive zip downloads.',
    )
    parser.add_argument(
        '--usage-status',
        default='pipeline-run',
        help='Usage status written into the raw snapshot manifest.',
    )
    parser.add_argument(
        '--converter-version',
        default='tenhou-xml-v0',
        help='Converter version string written into normalized manifests.',
    )
    parser.add_argument(
        '--converter-source',
        choices=('auto', 'official_json', 'xml'),
        default='xml',
        help='Which raw source to convert from during ingestion.',
    )
    parser.add_argument(
        '--with-release-artifacts',
        action='store_true',
        help='Also run QA summary, split generation, and overlap reporting after ingestion.',
    )
    parser.add_argument(
        '--stop-after',
        choices=('fetch', 'select', 'stage', 'ingest', 'release'),
        default='release',
        help='Stop after this pipeline phase. release means the end of the requested pipeline.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the commands that would run without executing them.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Pass --overwrite to the underlying steps that support it.',
    )
    return parser.parse_args()


def default_dataset_id(snapshot_id: str) -> str:
    return f'{snapshot_id}_v0'


def default_archive_dir(snapshot_id: str) -> str:
    return f'/tmp/tenhou_{snapshot_id}'


def rel_raw_manifest_path(snapshot_id: str, suffix: str) -> str:
    return f'data/manifests/raw/tenhou/{snapshot_id}{suffix}'


def rel_normalized_manifest_path(dataset_id: str, suffix: str) -> str:
    return f'data/manifests/normalized/v1/{dataset_id}{suffix}'


def rel_failure_manifest_path(dataset_id: str, snapshot_id: str) -> str:
    return f'data/manifests/failures/tenhou/{dataset_id}__{snapshot_id}.jsonl'


def rel_split_dir(dataset_id: str, name: str) -> str:
    return f'data/splits/v1/{dataset_id}/{name}'


def join_shell(cmd: list[str]) -> str:
    return ' '.join(shlex.quote(part) for part in cmd)


def mortal_env_updates() -> dict[str, str]:
    return {'PYTHONPATH': str(MORTAL_DIR)}


def append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def append_repeated_option(cmd: list[str], option: str, values: list[str]) -> None:
    for value in values:
        cmd.extend([option, value])


def build_steps(args) -> list[Step]:
    dataset_id = args.dataset_id or default_dataset_id(args.snapshot_id)
    archive_dir = args.archive_dir or default_archive_dir(args.snapshot_id)
    rulesets = args.ruleset or ['四鳳南喰赤－']

    archives_txt = rel_raw_manifest_path(args.snapshot_id, '.archives.txt')
    archives_json = rel_raw_manifest_path(args.snapshot_id, '.archives.json')
    refs_txt = rel_raw_manifest_path(args.snapshot_id, '.refs.txt')
    selection_json = rel_raw_manifest_path(args.snapshot_id, '.selection.json')

    normalized_manifest = rel_normalized_manifest_path(dataset_id, '.jsonl')
    normalized_qa = rel_normalized_manifest_path(dataset_id, '.qa_summary.json')
    hanchan_split_dir = rel_split_dir(dataset_id, 'phoenix_hanchan_7dan_all')
    all_split_dir = rel_split_dir(dataset_id, 'phoenix_all_7dan_all')
    overlap_json = f'{hanchan_split_dir}/player_overlap_summary.json'

    steps: list[Step] = []

    fetch_cmd = [
        sys.executable,
        'scripts/fetch_tenhou_scc_archives.py',
        '--output-dir',
        archive_dir,
        '--output-list',
        archives_txt,
        '--summary',
        archives_json,
        '--jobs',
        str(args.archive_jobs),
        '--timeout',
        str(args.download_timeout),
        '--retries',
        str(args.download_retries),
        '--retry-backoff-seconds',
        str(args.retry_backoff_seconds),
        '--publish-lag-days',
        str(args.archive_publish_lag_days),
    ]
    if args.year_archive_cache_dir:
        fetch_cmd.extend(['--year-archive-cache-dir', args.year_archive_cache_dir])
    if args.date:
        append_repeated_option(fetch_cmd, '--date', args.date)
    else:
        if args.start_date:
            fetch_cmd.extend(['--start-date', args.start_date])
        if args.end_date:
            fetch_cmd.extend(['--end-date', args.end_date])
    append_flag(fetch_cmd, '--overwrite', args.overwrite)
    steps.append(Step('fetch', fetch_cmd, {}))

    select_cmd = [
        sys.executable,
        'scripts/select_tenhou_scc_refs.py',
        '--archive-list',
        archives_txt,
        '--output-refs',
        refs_txt,
        '--output-summary',
        selection_json,
    ]
    append_repeated_option(select_cmd, '--ruleset', rulesets)
    if args.max_per_archive > 0:
        select_cmd.extend(['--max-per-archive', str(args.max_per_archive)])
    if args.limit > 0:
        select_cmd.extend(['--limit', str(args.limit)])
    steps.append(Step('select', select_cmd, {}))

    stage_cmd = [
        sys.executable,
        'scripts/stage_tenhou_reference_batch.py',
        '--snapshot-id',
        args.snapshot_id,
        '--refs-file',
        refs_txt,
        '--usage-status',
        args.usage_status,
        '--jobs',
        str(args.stage_jobs),
        '--timeout',
        str(args.download_timeout),
        '--retries',
        str(args.download_retries),
        '--retry-backoff-seconds',
        str(args.retry_backoff_seconds),
    ]
    append_flag(stage_cmd, '--overwrite', args.overwrite)
    steps.append(Step('stage', stage_cmd, mortal_env_updates()))

    ingest_cmd = [
        sys.executable,
        'scripts/ingest_tenhou_snapshot.py',
        '--raw-snapshot-id',
        args.snapshot_id,
        '--dataset-id',
        dataset_id,
        '--converter-version',
        args.converter_version,
        '--converter-source',
        args.converter_source,
    ]
    append_flag(ingest_cmd, '--overwrite', args.overwrite)
    steps.append(Step('ingest', ingest_cmd, mortal_env_updates()))

    if args.with_release_artifacts:
        steps.extend([
            Step(
                'qa',
                [
                    sys.executable,
                    'scripts/summarize_normalized_manifest.py',
                    '--manifest',
                    normalized_manifest,
                    '--output',
                    normalized_qa,
                ],
                {},
            ),
            Step(
                'split-all',
                [
                    sys.executable,
                    'scripts/build_dataset_splits.py',
                    '--manifest',
                    normalized_manifest,
                    '--output-dir',
                    all_split_dir,
                    '--source',
                    'tenhou',
                    '--room',
                    '鳳',
                    '--min-player-dan',
                    '16',
                    '--player-threshold-mode',
                    'all',
                ],
                {},
            ),
            Step(
                'split-hanchan',
                [
                    sys.executable,
                    'scripts/build_dataset_splits.py',
                    '--manifest',
                    normalized_manifest,
                    '--output-dir',
                    hanchan_split_dir,
                    '--source',
                    'tenhou',
                    '--room',
                    '鳳',
                    '--ruleset',
                    '鳳南喰赤',
                    '--go-type',
                    '169',
                    '--min-player-dan',
                    '16',
                    '--player-threshold-mode',
                    'all',
                ],
                {},
            ),
            Step(
                'overlap',
                [
                    sys.executable,
                    'scripts/summarize_split_overlap.py',
                    '--manifest',
                    normalized_manifest,
                    '--split-dir',
                    hanchan_split_dir,
                    '--output',
                    overlap_json,
                ],
                {},
            ),
        ])

    return steps


def selected_steps(steps: list[Step], stop_after: str) -> list[Step]:
    if stop_after == 'release':
        return steps

    out = []
    for step in steps:
        out.append(step)
        if step.name == stop_after:
            break
    return out


def run_step(step: Step, *, index: int, total: int, dry_run: bool) -> None:
    print(f'[{index}/{total}] {step.name.upper()}')
    print(join_shell(step.cmd))
    if dry_run:
        return

    env = os.environ.copy()
    env.update(step.env_updates)
    subprocess.run(
        step.cmd,
        cwd=ROOT,
        env=env,
        check=True,
    )


def main():
    args = parse_args()
    if not args.date and not (args.start_date and args.end_date):
        raise SystemExit('provide either repeated --date values or both --start-date and --end-date')
    if bool(args.start_date) != bool(args.end_date):
        raise SystemExit('--start-date and --end-date must be set together')
    if args.stop_after == 'release' and not args.with_release_artifacts:
        args.stop_after = 'ingest'

    steps = build_steps(args)
    planned_steps = selected_steps(steps, args.stop_after)

    for idx, step in enumerate(planned_steps, start=1):
        run_step(step, index=idx, total=len(planned_steps), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
