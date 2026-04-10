#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from progress_report import ProgressReporter, count_lines


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize a normalized dataset manifest into a compact QA report.',
    )
    parser.add_argument('--manifest', required=True, help='Path to a JSONL normalized manifest.')
    parser.add_argument('--output', required=True, help='Where to write the summary JSON.')
    return parser.parse_args()


def load_jsonl(filename: Path) -> list[dict]:
    rows = []
    progress = ProgressReporter(
        total=count_lines(filename),
        desc='QA-LOAD',
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


def counter_dict(values) -> dict:
    return dict(sorted(Counter(values).items()))


def summarize_numeric(values: list[float | int]) -> dict | None:
    if not values:
        return None
    return {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / len(values),
        'sum': sum(values),
    }


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    rows = load_jsonl(manifest_path)

    game_dates = sorted(row['game_date'] for row in rows if row.get('game_date'))
    rulesets = [row.get('ruleset') for row in rows if row.get('ruleset')]
    rooms = [row.get('room') for row in rows if row.get('room')]
    lobbies = [str(row.get('lobby')) for row in rows if row.get('lobby') is not None]
    go_types = [row.get('go_type') for row in rows if row.get('go_type') is not None]
    validation_statuses = [row.get('validation_status') for row in rows if row.get('validation_status')]
    years = [row.get('year') for row in rows if row.get('year') is not None]
    months = [f"{row['year']:04d}-{row['month']:02d}" for row in rows if row.get('year') is not None and row.get('month') is not None]

    all_dan_ids = []
    all_dan_labels = []
    all_rates = []
    kyoku_counts = []
    event_counts = []
    byte_sizes = []
    progress = ProgressReporter(
        total=len(rows),
        desc='QA',
        unit='row',
    )

    for row in rows:
        all_dan_ids.extend(row.get('player_dan') or [])
        all_dan_labels.extend(row.get('player_dan_label') or [])
        all_rates.extend(row.get('player_rate') or [])
        if row.get('kyoku_count') is not None:
            kyoku_counts.append(row['kyoku_count'])
        if row.get('event_count') is not None:
            event_counts.append(row['event_count'])
        if row.get('byte_size') is not None:
            byte_sizes.append(row['byte_size'])
        progress.update()
    progress.close(status=f'rows={len(rows)}')

    summary = {
        'manifest': str(manifest_path),
        'row_count': len(rows),
        'dataset_ids': sorted({row.get('dataset_id') for row in rows if row.get('dataset_id')}),
        'date_range': {
            'min': game_dates[0] if game_dates else None,
            'max': game_dates[-1] if game_dates else None,
        },
        'source_counts': counter_dict(row.get('source') for row in rows if row.get('source')),
        'ruleset_counts': counter_dict(rulesets),
        'room_counts': counter_dict(rooms),
        'lobby_counts': counter_dict(lobbies),
        'go_type_counts': counter_dict(go_types),
        'validation_status_counts': counter_dict(validation_statuses),
        'year_counts': counter_dict(years),
        'year_month_counts': counter_dict(months),
        'player_dan_id_counts': counter_dict(all_dan_ids),
        'player_dan_label_counts': counter_dict(all_dan_labels),
        'player_rate_summary': summarize_numeric(all_rates),
        'kyoku_count_summary': summarize_numeric(kyoku_counts),
        'event_count_summary': summarize_numeric(event_counts),
        'byte_size_summary': summarize_numeric(byte_sizes),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
