#!/usr/bin/env python
"""Phase C v3: Probe larger models (15x-30x) at various batch sizes."""

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
        for v in ['brain', 'dqn', 'x', 'mask', 'phi', 'q', 'loss']:
            if v in dir():
                exec(f'del {v}')
        torch.cuda.empty_cache()
        gc.collect()

    return peak


CONFIGS = [
    # Already measured — include for completeness
    ('10x_384ch_80b', dict(conv_channels=384, num_blocks=80, bottleneck_channels=64, hidden_dim=2048)),
    ('10x_wide_512ch_48b', dict(conv_channels=512, num_blocks=48, bottleneck_channels=64, hidden_dim=2048)),
    # New: 15x range
    ('15x_512ch_60b_bn96_hd2048', dict(conv_channels=512, num_blocks=60, bottleneck_channels=96, hidden_dim=2048)),
    ('15x_v2_384ch_100b_bn64_hd2048', dict(conv_channels=384, num_blocks=100, bottleneck_channels=64, hidden_dim=2048)),
    ('15x_v3_512ch_48b_bn96_hd4096', dict(conv_channels=512, num_blocks=48, bottleneck_channels=96, hidden_dim=4096)),
    # New: 20x range
    ('20x_512ch_80b_bn128_hd4096', dict(conv_channels=512, num_blocks=80, bottleneck_channels=128, hidden_dim=4096)),
    ('20x_v2_768ch_40b_bn96_hd2048', dict(conv_channels=768, num_blocks=40, bottleneck_channels=96, hidden_dim=2048)),
    # New: 30x (extreme)
    ('30x_768ch_60b_bn128_hd4096', dict(conv_channels=768, num_blocks=60, bottleneck_channels=128, hidden_dim=4096)),
]

BATCH_SIZES = [8192, 6144, 4096, 3072, 2048, 1536, 1024, 512]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = args.device
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    ddp_overhead = 2.5
    print(f'GPU: {gpu_name} ({gpu_mem:.1f} GiB)')
    print()

    all_results = []
    for name, kwargs in CONFIGS:
        params = sum(p.numel() for p in Brain(version=4, **kwargs).parameters())
        params += sum(p.numel() for p in DQN(version=4, hidden_dim=kwargs.get('hidden_dim', 1024)).parameters())

        best_bs = None
        best_peak = None
        for bs in BATCH_SIZES:
            peak = measure_one(kwargs, bs, device)
            if peak is not None and peak + ddp_overhead < gpu_mem:
                best_bs = bs
                best_peak = peak
                break

        all_results.append((name, kwargs, params, best_bs, best_peak))

    print(f'{"Config":<40} {"Params":>8} {"Max_BS":>7} {"Peak_GiB":>9} {"DDP_est":>8} '
          f'{"Headroom":>9} {"grad_acc":>8} {"eff_batch":>9}')
    print('-' * 115)

    for name, kwargs, params, best_bs, best_peak in all_results:
        if best_bs:
            ddp_est = best_peak + ddp_overhead
            headroom = gpu_mem - ddp_est
            grad_acc = max(1, 8192 // best_bs)
            eff = best_bs * grad_acc * 2
            print(f'{name:<40} {params/1e6:>7.1f}M {best_bs:>7} {best_peak:>8.2f}G {ddp_est:>7.2f}G '
                  f'{headroom:>8.1f}G {grad_acc:>8} {eff:>9,}')
        else:
            print(f'{name:<40} {params/1e6:>7.1f}M  — Does not fit at any batch size down to 512')


if __name__ == '__main__':
    main()
