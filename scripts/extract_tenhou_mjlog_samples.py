#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

SAMPLE_LABELS = {
    'four_kans': '暗槓・明槓・加槓・四槓子サンプル',
    'rinshan': '嶺上ツモ',
    'chankan': '搶槓',
    'double_ron': 'ダブロン',
    'nagashi_mangan': '流し満貫',
    'triple_ron': '三家和了',
    'four_winds_draw': '四風連打',
    'four_riichi_draw': '四家立直',
    'nine_terminals_draw': '九種九牌',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract official Tenhou sample-page JSON payloads into data/raw/ with a raw manifest.',
    )
    parser.add_argument(
        '--html-file',
        required=True,
        help='Local path to a saved https://tenhou.net/mjlog.html page.',
    )
    parser.add_argument(
        '--snapshot-id',
        required=True,
        help='Raw snapshot id to create under data/raw/tenhou/.',
    )
    parser.add_argument(
        '--samples',
        nargs='+',
        choices=sorted(SAMPLE_LABELS) + ['all'],
        default=['all'],
        help='Which built-in sample labels to extract.',
    )
    parser.add_argument(
        '--output-root',
        default='data/raw/tenhou',
        help='Root directory for extracted raw sample files.',
    )
    parser.add_argument(
        '--manifest-root',
        default='data/manifests/raw/tenhou',
        help='Root directory for raw manifests.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing extracted files and manifest.',
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_href(html: str, label: str) -> str:
    marker = f'{label}</a>'
    if marker not in html:
        raise ValueError(f'could not find sample label in html: {label}')
    head = html.split(marker, 1)[0]
    return head.rsplit('<a href="', 1)[1].split('">', 1)[0]


def extract_sample_object(html: str, label: str) -> dict:
    href = extract_href(html, label)
    json_part = href.split('#json=', 1)[1].split('&ts=', 1)[0]
    return json.loads(unquote(json_part))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    html = Path(args.html_file).read_text(encoding='utf-8')

    sample_slugs = list(SAMPLE_LABELS) if args.samples == ['all'] else args.samples
    output_dir = ROOT / args.output_root / args.snapshot_id
    manifest_path = ROOT / args.manifest_root / f'{args.snapshot_id}.json'

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f'output directory is not empty, re-run with --overwrite: {output_dir}')
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f'manifest already exists, re-run with --overwrite: {manifest_path}')

    files = []
    total_bytes = 0
    for slug in sample_slugs:
        label = SAMPLE_LABELS[slug]
        obj = extract_sample_object(html, label)
        output_path = output_dir / f'tenhou_editor_sample_{slug}.json'
        write_json(output_path, obj)
        byte_size = output_path.stat().st_size
        total_bytes += byte_size
        files.append({
            'relative_path': str(output_path.relative_to(ROOT)),
            'sample_slug': slug,
            'sample_label': label,
            'content_type': 'tenhou_editor_sample_json',
            'byte_size': byte_size,
            'sha256': compute_sha256(output_path),
            'rule_display': (obj.get('rule') or {}).get('disp'),
            'title': obj.get('title'),
            'round_count': len(obj.get('log') or []),
        })

    manifest = {
        'source': 'tenhou',
        'snapshot_id': args.snapshot_id,
        'acquired_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'acquired_by': 'codex',
        'official_access_path': 'https://tenhou.net/mjlog.html',
        'usage_status': 'format-study-only',
        'notes': [
            'Official Tenhou sample-page JSON extracted for Step 3 converter validation.',
            'These are format-validation samples, not real ladder-corpus imports.',
        ],
        'file_count': len(files),
        'total_bytes': total_bytes,
        'files': files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(output_dir.relative_to(ROOT))
    print(manifest_path.relative_to(ROOT))


if __name__ == '__main__':
    main()
