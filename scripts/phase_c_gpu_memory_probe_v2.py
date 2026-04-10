#!/usr/bin/env python
"""Phase C v2: Find maximum viable batch_size for each model config on A100-40GB."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mortal'))

import torch
from model import Brain, DQN
from libriichi.consts import obs_shape, ACTION_SPACE


def measure_one(resnet_kwargs: dict, batch_size: int, device: str) -> float | None:
    """Returns peak allocated GiB, or None if OOM."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    try:
        brain = Brain(version=4, **resnet_kwargs).to(device)
        dqn = DQN(version=4, hidden_dim=brain.hidden_dim).to(device)
        brain.train()
        dqn.train()

        in_ch = obs_shape(4)[0]
        x = torch.randn(batch_size, in_ch, 34, device=device, dtype=torch.bfloat16)
        mask = torch.ones(batch_size, ACTION_SPACE, device=device, dtype=torch.bool)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            phi = brain.actv(brain.encoder(x))
            q = dqn(phi, mask)
            loss = q.sum()
        loss.backward()

        peak = torch.cuda.max_memory_allocated(device) / 1024**3
    except torch.cuda.OutOfMemoryError:
        peak = None
    finally:
        # Aggressive cleanup
        for name in list(locals()):
            if name not in ('peak', 'device'):
                try:
                    del locals()[name]
                except:
                    pass
        torch.cuda.empty_cache()
        gc.collect()

    return peak


def find_max_batch(name: str, resnet_kwargs: dict, device: str, gpu_limit: float):
    """Binary search for max batch size that fits in GPU memory with DDP headroom."""
    ddp_overhead = 2.5  # GiB estimated for gradient buckets
    target = gpu_limit - ddp_overhead

    params = sum(p.numel() for p in Brain(version=4, **resnet_kwargs).parameters())
    params += sum(p.numel() for p in DQN(version=4, hidden_dim=resnet_kwargs.get('hidden_dim', 1024)).parameters())

    # Test batch sizes from large to small
    batch_sizes = [8192, 6144, 4096, 3072, 2048, 1536, 1024, 512]
    results = []

    for bs in batch_sizes:
        peak = measure_one(resnet_kwargs, bs, device)
        if peak is not None:
            fits = peak + ddp_overhead < gpu_limit
            results.append((bs, peak, fits))
            if fits:
                break  # Found the max that fits
        else:
            results.append((bs, None, False))

    return name, params, results


CONFIGS = [
    ('baseline_192ch_40b', dict(conv_channels=192, num_blocks=40)),
    ('2x_256ch_40b', dict(conv_channels=256, num_blocks=40)),
    ('3x_256ch_60b_bn48_hd1536', dict(conv_channels=256, num_blocks=60, bottleneck_channels=48, hidden_dim=1536)),
    ('5x_384ch_40b_bn64_hd2048', dict(conv_channels=384, num_blocks=40, bottleneck_channels=64, hidden_dim=2048)),
    ('10x_384ch_80b_bn64_hd2048', dict(conv_channels=384, num_blocks=80, bottleneck_channels=64, hidden_dim=2048)),
    ('10x_wide_512ch_48b', dict(conv_channels=512, num_blocks=48, bottleneck_channels=64, hidden_dim=2048)),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = args.device
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f'GPU: {gpu_name} ({gpu_mem:.1f} GiB)')
    print(f'DDP overhead estimate: 2.5 GiB')
    print()

    all_results = []
    for name, kwargs in CONFIGS:
        name, params, results = find_max_batch(name, kwargs, device, gpu_mem)
        all_results.append((name, params, results))

    # Summary table
    print()
    print('=' * 110)
    print('PHASE C: GPU MEMORY SCALING SUMMARY')
    print('=' * 110)
    print(f'{"Config":<35} {"Params":>8} {"Max_BS":>7} {"Peak_GiB":>9} {"DDP_est":>8} '
          f'{"Headroom":>9} {"grad_acc":>8} {"eff_batch":>9}')
    print('-' * 110)

    for name, params, results in all_results:
        # Find the largest batch that fits
        best = None
        for bs, peak, fits in results:
            if fits:
                best = (bs, peak)
                break

        if best:
            bs, peak = best
            ddp_est = peak + 2.5
            headroom = gpu_mem - ddp_est
            # Calculate grad_accum needed to match baseline effective batch of 16384 (8192 per GPU × 2 GPUs)
            grad_acc = max(1, 8192 // bs)
            eff = bs * grad_acc * 2  # 2 GPUs
            print(f'{name:<35} {params/1e6:>7.1f}M {bs:>7} {peak:>8.2f}G {ddp_est:>7.2f}G '
                  f'{headroom:>8.1f}G {grad_acc:>8} {eff:>9,}')
        else:
            print(f'{name:<35} {params/1e6:>7.1f}M {"N/A":>7} — Does not fit at any tested batch size')

    # Detailed breakdown
    print()
    print('DETAILED MEASUREMENTS:')
    print('-' * 80)
    for name, params, results in all_results:
        print(f'\n  {name} ({params/1e6:.1f}M params):')
        for bs, peak, fits in results:
            if peak is not None:
                status = 'FITS' if fits else 'TOO BIG'
                print(f'    bs={bs:>5}: {peak:.2f} GiB  ({status})')
            else:
                print(f'    bs={bs:>5}: OOM')


if __name__ == '__main__':
    main()
