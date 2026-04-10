#!/usr/bin/env python

import json
import sys
from argparse import ArgumentParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_dataset import (  # noqa: E402
    build_actor_filter_map,
    normalize_file_list,
    save_actor_filter_index,
)


def parse_args():
    parser = ArgumentParser(
        description='Build a precomputed actor-filter index for Step 5 BC splits.',
    )
    parser.add_argument(
        '--manifest',
        required=True,
        help='Normalized manifest JSONL used to read player_dan metadata.',
    )
    parser.add_argument(
        '--list',
        dest='list_files',
        action='append',
        default=[],
        help='Split file list to include in the index. Repeat for train/val/test.',
    )
    parser.add_argument(
        '--root-dir',
        default='',
        help='Optional root directory used to resolve relative paths in --list files.',
    )
    parser.add_argument(
        '--min-actor-dan',
        type=int,
        required=True,
        help='Minimum acting-player dan to keep in the index.',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output .pth file for the saved actor-filter index.',
    )
    parser.add_argument(
        '--summary',
        default='',
        help='Optional JSON summary output path.',
    )
    return parser.parse_args()


def filtered_trimmed_lines(lines):
    return [line.strip() for line in lines if line.strip()]


def load_path_list(filename: str, root_dir: str = '') -> list[str]:
    with open(filename, encoding='utf-8') as f:
        paths = filtered_trimmed_lines(f)
    if root_dir:
        return [
            p if Path(p).is_absolute() else str(Path(root_dir) / p)
            for p in paths
        ]
    return paths


def main():
    args = parse_args()
    if not args.list_files:
        raise ValueError('provide at least one --list input')

    file_lists = [
        normalize_file_list(load_path_list(list_file, args.root_dir))
        for list_file in args.list_files
    ]
    actor_filter_map, summary = build_actor_filter_map(
        manifest_path=args.manifest,
        file_lists=file_lists,
        min_actor_dan=args.min_actor_dan,
        inputs_are_normalized=True,
    )
    save_actor_filter_index(
        args.output,
        actor_filter_map=actor_filter_map,
        summary=summary,
    )

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
