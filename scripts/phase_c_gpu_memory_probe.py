#!/usr/bin/env python
"""Phase C: Measure actual GPU memory for each probe config using real model.py.

Run AFTER the 2000-step stability test finishes (needs free GPUs).
Usage: python scripts/phase_c_gpu_memory_probe.py --device cuda:0
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mortal'))

import torch
from model import Brain, DQN

CONFIGS = [
    # (name, resnet_kwargs, batch_size)
    ('baseline_192ch_40b', dict(conv_channels=192, num_blocks=40), 8192),
    ('2x_256ch_40b', dict(conv_channels=256, num_blocks=40), 8192),
    ('3x_256ch_60b_bn48_hd1536', dict(conv_channels=256, num_blocks=60, bottleneck_channels=48, hidden_dim=1536), 8192),
    ('5x_384ch_40b_bn64_hd2048', dict(conv_channels=384, num_blocks=40, bottleneck_channels=64, hidden_dim=2048), 8192),
    ('10x_384ch_80b_bn64_hd2048', dict(conv_channels=384, num_blocks=80, bottleneck_channels=64, hidden_dim=2048), 8192),
    ('10x_wide_512ch_48b', dict(conv_channels=512, num_blocks=48, bottleneck_channels=64, hidden_dim=2048), 8192),
    # Reduced batch sizes for large models
    ('10x_384ch_80b_bs4096', dict(conv_channels=384, num_blocks=80, bottleneck_channels=64, hidden_dim=2048), 4096),
    ('10x_wide_512ch_48b_bs4096', dict(conv_channels=512, num_blocks=48, bottleneck_channels=64, hidden_dim=2048), 4096),
    ('10x_384ch_80b_bs2048', dict(conv_channels=384, num_blocks=80, bottleneck_channels=64, hidden_dim=2048), 2048),
    ('10x_wide_512ch_48b_bs2048', dict(conv_channels=512, num_blocks=48, bottleneck_channels=64, hidden_dim=2048), 2048),
]


def measure(name: str, resnet_kwargs: dict, batch_size: int, device: str) -> dict:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    brain = Brain(version=4, **resnet_kwargs).to(device)
    dqn = DQN(version=4, hidden_dim=brain.hidden_dim).to(device)
    brain.train()
    dqn.train()

    brain_params = sum(p.numel() for p in brain.parameters())
    dqn_params = sum(p.numel() for p in dqn.parameters())

    from libriichi.consts import obs_shape
    in_ch = obs_shape(4)[0]  # 1012

    x = torch.randn(batch_size, in_ch, 34, device=device, dtype=torch.bfloat16)
    from libriichi.consts import ACTION_SPACE
    mask = torch.ones(batch_size, ACTION_SPACE, device=device, dtype=torch.bool)

    # Simulate training step with AMP
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        phi = brain.actv(brain.encoder(x))
        q = dqn(phi, mask)
        loss = q.sum()

    loss.backward()

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_rsv = torch.cuda.max_memory_reserved(device) / 1024**3

    del brain, dqn, x, mask, phi, q, loss
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'name': name,
        'params': brain_params + dqn_params,
        'batch_size': batch_size,
        'peak_alloc_gib': round(peak_alloc, 2),
        'peak_rsv_gib': round(peak_rsv, 2),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = args.device
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f'GPU: {gpu_name} ({gpu_mem:.1f} GiB)')
    print()

    print(f'{"Name":<35} {"Params":>8} {"Batch":>5} {"Peak_alloc":>10} {"Peak_rsv":>10} '
          f'{"Fits_DDP":>8} {"Headroom":>8}')
    print('-' * 95)

    for name, kwargs, bs in CONFIGS:
        try:
            r = measure(name, kwargs, bs, device)
            # DDP adds ~2-3 GiB for gradient buckets on A100
            ddp_est = r['peak_alloc_gib'] + 2.5
            fits = 'YES' if ddp_est < gpu_mem else 'MAYBE' if ddp_est < gpu_mem + 1 else 'NO'
            headroom = gpu_mem - ddp_est
            print(f'{name:<35} {r["params"]/1e6:>7.1f}M {bs:>5} '
                  f'{r["peak_alloc_gib"]:>9.2f}G {r["peak_rsv_gib"]:>9.2f}G '
                  f'{fits:>8} {headroom:>7.1f}G')
        except torch.cuda.OutOfMemoryError:
            print(f'{name:<35} {"":>8} {bs:>5} {"OOM":>10} {"":>10} {"NO":>8} {"":>8}')
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f'{name:<35} ERROR: {e}')
            torch.cuda.empty_cache()
            gc.collect()

    print()
    print('Notes:')
    print(f'  Fits_DDP = peak_alloc + 2.5 GiB DDP overhead < {gpu_mem:.1f} GiB')
    print('  If Fits_DDP=NO at bs=8192, try bs=4096 with grad_accum_steps=2')


if __name__ == '__main__':
    main()
