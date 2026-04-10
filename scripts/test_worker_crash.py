#!/usr/bin/env python
"""Targeted test to diagnose DataLoader worker crashes with num_workers >= 6.
Runs a single GPU training-like loop and captures worker exit signals."""

import os
import sys
import signal
import faulthandler
import traceback

# Enable faulthandler so segfaults produce a traceback
faulthandler.enable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MORTAL_DIR = os.path.join(ROOT, 'mortal')
if MORTAL_DIR not in sys.path:
    sys.path.insert(0, MORTAL_DIR)

os.environ.setdefault('MORTAL_CFG', os.path.join(MORTAL_DIR, 'config.example.toml'))

import torch
import torch.utils.data
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def worker_init_fn_debug(*args, **kwargs):
    """Enhanced worker_init_fn with crash diagnostics."""
    import faulthandler
    faulthandler.enable()

    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    worker_id = worker_info.id

    # Scale Rayon threads
    cpu_count = os.cpu_count() or 64
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    total_workers = num_workers * world_size
    rayon_threads = max(1, cpu_count // max(1, total_workers))
    os.environ['RAYON_NUM_THREADS'] = str(rayon_threads)

    logging.info(f'Worker {worker_id}/{num_workers}: PID={os.getpid()}, '
                 f'RAYON_NUM_THREADS={rayon_threads}')

    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / num_workers))
    start = worker_id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
    logging.info(f'Worker {worker_id}: assigned {len(dataset.file_list)} files')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batches', type=int, default=200)
    parser.add_argument('--rayon-threads', type=int, default=0,
                        help='Override RAYON_NUM_THREADS (0 = auto-scale)')
    args = parser.parse_args()

    if args.rayon_threads > 0:
        os.environ['RAYON_NUM_THREADS'] = str(args.rayon_threads)
        logging.info(f'Forced RAYON_NUM_THREADS={args.rayon_threads}')

    from bc_campaign import load_full_config
    config_path = os.path.join(ROOT, 'configs/step6_bc_large_preflight_full8dan_r5_control_b4.toml')
    _, full_config = load_full_config(config_path)
    dataset_cfg = full_config['bc']['dataset']

    from dataloader import FileBCDataset, worker_init_fn
    from bc_dataset import BCDatasetConfig

    bc_cfg = BCDatasetConfig.from_dict(dataset_cfg)

    # Build file list
    from train_bc import build_file_list
    all_files = build_file_list(full_config, split='train')
    logging.info(f'Total files: {len(all_files)}')

    dataset = FileBCDataset(
        file_list=all_files,
        bc_config=bc_cfg,
        oracle=bool(full_config['bc']['dataset'].get('oracle', False)),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn_debug,
        multiprocessing_context='spawn',
        pin_memory=True,
    )

    logging.info(f'Starting iteration: num_workers={args.num_workers}, batches={args.batches}')
    device = torch.device('cuda:0')

    t0 = time.monotonic()
    batch_count = 0
    sample_count = 0
    try:
        for batch in loader:
            # Move to GPU
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, non_blocking=True)
            batch_count += 1
            if isinstance(batch, dict) and 'obs' in batch:
                sample_count += batch['obs'].shape[0]
            else:
                sample_count += 8192  # estimate

            if batch_count % 10 == 0:
                elapsed = time.monotonic() - t0
                sps = sample_count / elapsed if elapsed > 0 else 0
                logging.info(f'Batch {batch_count}/{args.batches}: '
                             f'{sps:.1f} sps, mem={torch.cuda.memory_allocated(device)/1e9:.1f}G')

            if batch_count >= args.batches:
                break
    except Exception as e:
        logging.error(f'CRASHED at batch {batch_count}: {e}')
        traceback.print_exc()
        return 1

    elapsed = time.monotonic() - t0
    sps = sample_count / elapsed if elapsed > 0 else 0
    logging.info(f'COMPLETED: {batch_count} batches, {sample_count} samples, '
                 f'{sps:.1f} sps in {elapsed:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
