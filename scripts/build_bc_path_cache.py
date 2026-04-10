#!/usr/bin/env python

import json
import sys
from argparse import ArgumentParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_dataset import normalize_file_list, save_path_cache  # noqa: E402


def parse_args():
    parser = ArgumentParser(
        description='Build a normalized path cache for Step 5 BC split files.',
    )
    parser.add_argument('--train-list', required=True, help='Train split text file.')
    parser.add_argument('--val-list', required=True, help='Validation split text file.')
    parser.add_argument('--test-list', default='', help='Optional test split text file.')
    parser.add_argument(
        '--root-dir',
        default='',
        help='Optional root directory used to resolve relative paths inside split files.',
    )
    parser.add_argument('--output', required=True, help='Output .pth path cache file.')
    parser.add_argument('--summary', default='', help='Optional JSON summary output path.')
    return parser.parse_args()


def load_path_list(filename: str, root_dir: str = '') -> list[str]:
    with open(filename, encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]
    if root_dir:
        return [
            p if Path(p).is_absolute() else str(Path(root_dir) / p)
            for p in paths
        ]
    return paths


def main():
    args = parse_args()
    split_sources = {
        'train': args.train_list,
        'val': args.val_list,
    }
    if args.test_list:
        split_sources['test'] = args.test_list

    split_lists = {
        split_name: normalize_file_list(load_path_list(list_file, args.root_dir), desc=f'PATHS-{split_name.upper()}')
        for split_name, list_file in split_sources.items()
    }
    save_path_cache(
        args.output,
        split_lists=split_lists,
        source_files=split_sources,
    )

    summary = {
        'format': 'bc_path_cache_v1',
        'output': args.output,
        'root_dir': args.root_dir,
        'source_files': split_sources,
        'counts': {
            split_name: len(file_list)
            for split_name, file_list in split_lists.items()
        },
    }
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
