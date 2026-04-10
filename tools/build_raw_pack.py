#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'
os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import load_full_config  # noqa: E402
from raw_store import build_raw_pack  # noqa: E402
from step6_experiments import configured_split_lists  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a raw-pack file containing exact gz bytes plus an index.',
    )
    parser.add_argument('--config', default='', help='Optional config used to resolve a dataset split.')
    parser.add_argument('--split', default='train', help='Dataset split to pack when --config is used.')
    parser.add_argument('--file-list', default='', help='Optional newline-delimited path list to pack.')
    parser.add_argument('--root-dir', default='', help='Root directory for relative paths in --file-list.')
    parser.add_argument('--output-pack', required=True, help='Output raw-pack payload file.')
    parser.add_argument('--output-index', required=True, help='Output raw-pack index JSON.')
    parser.add_argument('--limit', type=int, default=0, help='Optional file limit for small test packs.')
    parser.add_argument('--summary-json', default='', help='Optional summary output path.')
    return parser.parse_args()


def _load_paths_from_file_list(file_list_path: str, root_dir: str) -> list[str]:
    root = Path(root_dir).expanduser().resolve() if root_dir else None
    file_list = []
    for line in Path(file_list_path).read_text(encoding='utf-8').splitlines():
        path = line.strip()
        if not path:
            continue
        candidate = Path(path).expanduser()
        if root is not None and not candidate.is_absolute():
            candidate = root / candidate
        file_list.append(str(candidate.resolve()))
    return file_list


def resolve_file_list(*, config_path: str, split: str, file_list_path: str, root_dir: str) -> list[str]:
    if config_path:
        _resolved, full_config = load_full_config(config_path)
        split_lists = configured_split_lists(full_config, splits=[split])
        return list(split_lists[split])
    if file_list_path:
        return _load_paths_from_file_list(file_list_path, root_dir)
    raise ValueError('either --config or --file-list is required')


def main():
    args = parse_args()
    file_list = resolve_file_list(
        config_path=args.config,
        split=args.split,
        file_list_path=args.file_list,
        root_dir=args.root_dir,
    )
    if args.limit > 0:
        file_list = file_list[:args.limit]
    summary = build_raw_pack(
        file_list,
        pack_path=args.output_pack,
        index_path=args.output_index,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
