#!/usr/bin/env python3

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

from progress_report import ProgressReporter, count_lines

TENHOU_GAME_ID_RE = re.compile(r'^(?P<ts>\d{10})gm-')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build train/val/test file lists from a normalized dataset manifest.',
    )
    parser.add_argument('--manifest', required=True, help='Path to a JSONL normalized manifest.')
    parser.add_argument('--output-dir', required=True, help='Directory to write split files.')
    parser.add_argument('--root-dir', default='', help='Optional root for writing absolute paths.')
    parser.add_argument(
        '--source',
        action='append',
        default=[],
        help='Restrict to one or more exact source values.',
    )
    parser.add_argument(
        '--lobby',
        action='append',
        default=[],
        help='Restrict to one or more exact lobby values.',
    )
    parser.add_argument(
        '--room',
        action='append',
        default=[],
        help='Restrict to one or more exact room values.',
    )
    parser.add_argument(
        '--ruleset',
        action='append',
        default=[],
        help='Restrict to one or more exact ruleset values.',
    )
    parser.add_argument(
        '--go-type',
        action='append',
        default=[],
        help='Restrict to one or more exact go_type values.',
    )
    parser.add_argument(
        '--min-player-dan',
        type=int,
        default=None,
        help='Require player_dan values to meet a minimum threshold.',
    )
    parser.add_argument(
        '--min-player-rate',
        type=float,
        default=None,
        help='Require player_rate values to meet a minimum threshold.',
    )
    parser.add_argument(
        '--player-threshold-mode',
        choices=('any', 'all'),
        default='any',
        help='Whether thresholds apply to any player or all players.',
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Fraction of accepted rows to place into train.',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Fraction of accepted rows to place into val.',
    )
    parser.add_argument(
        '--write-absolute',
        action='store_true',
        help='Write absolute paths instead of manifest-relative paths.',
    )
    return parser.parse_args()


def parse_jsonl(filename):
    rows = []
    progress = ProgressReporter(
        total=count_lines(filename),
        desc='SPLIT-LOAD',
        unit='line',
    )
    with open(filename, encoding='utf-8') as f:
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


def parse_datetime(row):
    game_date = row.get('game_date')
    if game_date:
        normalized = game_date.replace('Z', '+00:00')
        return datetime.fromisoformat(normalized)

    if row.get('source') == 'tenhou':
        source_game_id = row.get('source_game_id', '')
        match = TENHOU_GAME_ID_RE.match(source_game_id)
        if match:
            return datetime.strptime(match.group('ts'), '%Y%m%d%H')

    return None


def pass_threshold(values, threshold, mode):
    if threshold is None:
        return True
    if not isinstance(values, list) or not values:
        return False
    if mode == 'all':
        return all(v >= threshold for v in values)
    return any(v >= threshold for v in values)


def row_matches(row, args):
    if args.source and row.get('source') not in args.source:
        return False
    if args.lobby and str(row.get('lobby', '')) not in args.lobby:
        return False
    if args.room and str(row.get('room', '')) not in args.room:
        return False
    if args.ruleset and str(row.get('ruleset', '')) not in args.ruleset:
        return False
    if args.go_type and str(row.get('go_type', '')) not in args.go_type:
        return False
    if not pass_threshold(row.get('player_dan'), args.min_player_dan, args.player_threshold_mode):
        return False
    if not pass_threshold(row.get('player_rate'), args.min_player_rate, args.player_threshold_mode):
        return False
    return True


def materialize_path(relative_path, root_dir, write_absolute):
    if write_absolute:
        if not root_dir:
            raise ValueError('--write-absolute requires --root-dir')
        return str((Path(root_dir) / relative_path).resolve())
    return relative_path


def write_lines(filename, lines):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def split_rows(rows, train_ratio, val_ratio):
    total = len(rows)
    train_n = math.floor(total * train_ratio)
    val_n = math.floor(total * val_ratio)
    if train_n + val_n > total:
        raise ValueError('split ratios produce more rows than available')
    return (
        rows[:train_n],
        rows[train_n:train_n + val_n],
        rows[train_n + val_n:],
    )


def main():
    args = parse_args()
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio > 1:
        raise ValueError('train_ratio and val_ratio must be non-negative and sum to at most 1')

    rows = parse_jsonl(args.manifest)
    with_keys = []
    missing_sort_key = []
    accepted_count = 0
    progress = ProgressReporter(
        total=len(rows),
        desc='SPLIT',
        unit='row',
    )
    for row in rows:
        if not row_matches(row, args):
            progress.update(status=f'accepted={accepted_count} missing_dates={len(missing_sort_key)}')
            continue
        accepted_count += 1
        sort_key = parse_datetime(row)
        if sort_key is None:
            missing_sort_key.append(row.get('relative_path', '<missing>'))
            progress.update(status=f'accepted={accepted_count} missing_dates={len(missing_sort_key)}')
            continue
        with_keys.append((sort_key, row.get('source_game_id', ''), row['relative_path'], row))
        progress.update(status=f'accepted={accepted_count} missing_dates={len(missing_sort_key)}')
    progress.close(status=f'accepted={accepted_count} missing_dates={len(missing_sort_key)}')

    if missing_sort_key:
        joined = ', '.join(missing_sort_key[:5])
        raise ValueError(
            'manifest rows are missing sortable game dates; first examples: '
            f'{joined}'
        )

    with_keys.sort()
    sorted_rows = [row for _, _, _, row in with_keys]
    train_rows, val_rows, test_rows = split_rows(sorted_rows, args.train_ratio, args.val_ratio)

    def to_lines(items):
        return [
            materialize_path(row['relative_path'], args.root_dir, args.write_absolute)
            for row in items
        ]

    output_dir = Path(args.output_dir)
    write_lines(output_dir / 'train.txt', to_lines(train_rows))
    write_lines(output_dir / 'val.txt', to_lines(val_rows))
    write_lines(output_dir / 'test.txt', to_lines(test_rows))

    summary = {
        'manifest': args.manifest,
        'root_dir': args.root_dir,
        'write_absolute': args.write_absolute,
        'filters': {
            'source': args.source,
            'lobby': args.lobby,
            'room': args.room,
            'ruleset': args.ruleset,
            'go_type': args.go_type,
            'min_player_dan': args.min_player_dan,
            'min_player_rate': args.min_player_rate,
            'player_threshold_mode': args.player_threshold_mode,
        },
        'counts': {
            'manifest_rows': len(rows),
            'accepted_rows': len(sorted_rows),
            'train_rows': len(train_rows),
            'val_rows': len(val_rows),
            'test_rows': len(test_rows),
        },
    }
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write('\n')


if __name__ == '__main__':
    main()
