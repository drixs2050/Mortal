#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

from bc_conversion_bench import run_conversion_vs_training_benchmark  # noqa: E402
from bc_campaign import load_full_config  # noqa: E402


def parse_args():
    def parse_optional_bool(value: str) -> bool:
        lowered = str(value).strip().lower()
        if lowered in ('1', 'true', 'yes', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'off'):
            return False
        raise argparse.ArgumentTypeError(f'expected boolean value, got: {value}')

    parser = argparse.ArgumentParser(
        description='Benchmark the exact raw loader path up to CPU train batches, then replay those exact batches through the GPU training step.',
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the BC TOML config to benchmark.',
    )
    parser.add_argument(
        '--split',
        default='train',
        choices=('train', 'val', 'test'),
        help='Dataset split to benchmark.',
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=192,
        help='Number of raw files to benchmark after optional sharding.',
    )
    parser.add_argument(
        '--sample-strategy',
        default='round_robin',
        choices=('round_robin', 'head'),
        help='How to choose benchmark files from the selected split.',
    )
    parser.add_argument(
        '--shard-world-size',
        type=int,
        default=0,
        help='Optional file-list sharding world size to mimic one DDP rank. Defaults to bc.launch.nproc_per_node.',
    )
    parser.add_argument(
        '--shard-rank',
        type=int,
        default=0,
        help='Shard rank to benchmark when --shard-world-size > 1.',
    )
    parser.add_argument(
        '--device',
        default='',
        help='Torch device to benchmark on. Defaults to bc.control.device.',
    )
    parser.add_argument(
        '--output-json',
        default='',
        help='Optional path to write the benchmark summary JSON.',
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=0,
        help='Optional cap on the number of produced CPU batches to benchmark. Defaults to all batches from the selected raw files.',
    )
    parser.add_argument(
        '--override-shuffle',
        type=parse_optional_bool,
        default=None,
        help='Optional override for the producer-stage shuffle flag.',
    )
    parser.add_argument(
        '--override-pin-memory',
        type=parse_optional_bool,
        default=None,
        help='Optional override for the producer-stage DataLoader pin_memory flag.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _resolved_config_path, full_config = load_full_config(args.config)
    bc_cfg = full_config.get('bc') or {}
    launch_cfg = bc_cfg.get('launch') or {}
    control_cfg = bc_cfg.get('control') or {}

    shard_world_size = int(args.shard_world_size or launch_cfg.get('nproc_per_node', 1) or 1)
    device = str(args.device or control_cfg.get('device', 'cpu'))

    summary = run_conversion_vs_training_benchmark(
        config_path=args.config,
        split=args.split,
        sample_size=args.sample_size,
        sample_strategy=args.sample_strategy,
        shard_world_size=shard_world_size,
        shard_rank=int(args.shard_rank),
        device=device,
        max_batches=int(args.max_batches),
        shuffle_override=args.override_shuffle,
        pin_memory_override=args.override_pin_memory,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
