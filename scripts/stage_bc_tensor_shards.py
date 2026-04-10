#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'
os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import load_full_config  # noqa: E402
from bc_stage import ensure_stage_cache, stage_manifest_paths, stage_required_splits  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage BC tensor shards from the configured raw gz corpus.',
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the BC config TOML.',
    )
    parser.add_argument(
        '--split',
        action='append',
        choices=('train', 'val', 'test'),
        help='Split(s) to stage. Defaults to bc.stage.required_splits.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Rebuild the requested split cache even if a manifest already exists.',
    )
    parser.add_argument(
        '--output-json',
        help='Optional path to write the stage summary JSON.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    _, full_config = load_full_config(args.config)
    splits = stage_required_splits(full_config, requested_splits=args.split)
    logging.info(
        'stage request config=%s splits=%s force=%s',
        args.config,
        ','.join(splits),
        bool(args.force),
    )
    manifest_paths = stage_manifest_paths(full_config, splits=splits)
    manifests = ensure_stage_cache(
        full_config,
        splits=splits,
        force=args.force,
    )
    summary = {
        split: {
            'manifest_path': str(manifest_paths[split].resolve()),
            'shard_count': int(manifest.get('shard_count', 0)),
            'file_count': int(sum(int(shard.get('file_count', 0)) for shard in (manifest.get('shards') or []))),
            'sample_count': int(manifest.get('sample_count', 0)),
            'size_gib': float(manifest.get('size_bytes', 0)) / (1024 ** 3),
            'backend': manifest.get('backend', ''),
        }
        for split, manifest in manifests.items()
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        logging.info('wrote stage summary json to %s', output_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
