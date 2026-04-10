#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from progress_report import ProgressReporter


ROOT = Path(__file__).resolve().parents[1]


SCC_LINE_RE = re.compile(
    r'^(?P<start>\d{2}:\d{2}) \| '
    r'(?P<duration>\d{2}) \| '
    r'(?P<ruleset>[^|]+?) \| '
    r'<a href="http://tenhou\.net/0/\?log=(?P<source_game_id>[^"]+)">牌譜</a> \| '
    r'(?P<players>.+?)<br>$'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select Tenhou replay ids from official scc*.html.gz Phoenix raw archives.',
    )
    parser.add_argument(
        '--archive',
        action='append',
        default=[],
        help='Path to a local scc*.html.gz archive. Repeat for multiple files.',
    )
    parser.add_argument(
        '--archive-list',
        action='append',
        default=[],
        help='Path to a text file containing one archive path per line.',
    )
    parser.add_argument(
        '--ruleset',
        action='append',
        default=[],
        help='Exact ruleset label to keep. Repeat to allow multiple labels.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=32,
        help='Maximum number of replay ids to output.',
    )
    parser.add_argument(
        '--max-per-archive',
        type=int,
        default=0,
        help='Optional maximum number of selected replay ids per archive.',
    )
    parser.add_argument(
        '--output-refs',
        required=True,
        help='Output refs file relative to the repo root.',
    )
    parser.add_argument(
        '--output-summary',
        default='',
        help='Optional summary JSON relative to the repo root.',
    )
    return parser.parse_args()


def load_path_lines(path: Path) -> list[Path]:
    entries = []
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        entries.append(candidate)
    return entries


def collect_archive_paths(archive_args: list[str], archive_list_args: list[str]) -> list[Path]:
    archives = []
    seen = set()
    loaded_list_paths = []
    empty_list_paths = []

    for raw_path in archive_args:
        path = Path(raw_path).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            archives.append(path)

    for raw_list_path in archive_list_args:
        list_path = Path(raw_list_path).expanduser()
        if not list_path.is_absolute():
            list_path = (ROOT / list_path).resolve()
        else:
            list_path = list_path.resolve()
        loaded_list_paths.append(list_path)
        list_entries = load_path_lines(list_path)
        if not list_entries:
            empty_list_paths.append(list_path)
        for archive_path in list_entries:
            if archive_path not in seen:
                seen.add(archive_path)
                archives.append(archive_path)

    if not archives:
        if loaded_list_paths and len(empty_list_paths) == len(loaded_list_paths):
            joined = ', '.join(str(path) for path in empty_list_paths[:5])
            raise ValueError(
                'archive-list input file(s) contained no archive paths; '
                f'first examples: {joined}. Check the fetch summary and archive-list output.'
            )
        raise ValueError('provide at least one --archive or --archive-list input')

    return archives


def iter_archive_rows(archive_path: Path):
    with gzip.open(archive_path, 'rt', encoding='utf-8', errors='replace') as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip('\n')
            match = SCC_LINE_RE.match(line)
            if not match:
                continue
            row = match.groupdict()
            row['archive'] = str(archive_path)
            row['archive_name'] = archive_path.name
            row['lineno'] = lineno
            yield row


def write_refs(path: Path, refs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{ref}\n' for ref in refs), encoding='utf-8')


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    archives = collect_archive_paths(args.archive, args.archive_list)
    allowed_rulesets = set(args.ruleset or ['四鳳南喰赤－'])

    selected_rows = []
    seen = set()
    ruleset_counts = Counter()
    archive_counts = Counter()
    progress = ProgressReporter(
        total=len(archives),
        desc='SELECT',
        unit='archive',
    )

    for archive_path in archives:
        for row in iter_archive_rows(archive_path):
            if args.max_per_archive > 0 and archive_counts[archive_path.name] >= args.max_per_archive:
                break
            if row['ruleset'] not in allowed_rulesets:
                continue
            source_game_id = row['source_game_id']
            if source_game_id in seen:
                continue
            seen.add(source_game_id)
            selected_rows.append(row)
            ruleset_counts[row['ruleset']] += 1
            archive_counts[row['archive_name']] += 1
            if args.limit > 0 and len(selected_rows) >= args.limit:
                break
        progress.update(status=f'selected={len(selected_rows)}')
        if args.limit > 0 and len(selected_rows) >= args.limit:
            break

    progress.close(status=f'selected={len(selected_rows)}')

    refs = [row['source_game_id'] for row in selected_rows]
    output_refs_path = ROOT / args.output_refs
    write_refs(output_refs_path, refs)

    if args.output_summary:
        output_summary_path = ROOT / args.output_summary
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            'archive_count': len(archives),
            'archives': [str(path) for path in archives],
            'rulesets': sorted(allowed_rulesets),
            'selected_count': len(selected_rows),
            'ruleset_counts': dict(sorted(ruleset_counts.items())),
            'archive_counts': dict(sorted(archive_counts.items())),
            'selected_rows': selected_rows,
            'output_refs': display_path(output_refs_path),
        }
        output_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    print(display_path(output_refs_path))
    print(len(refs))


if __name__ == '__main__':
    main()
