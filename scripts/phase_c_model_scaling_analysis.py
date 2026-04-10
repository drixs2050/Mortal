#!/usr/bin/env python
"""Phase C: Comprehensive model scaling analysis across ALL tunable dimensions.

Explores: conv_channels, num_blocks, bottleneck_channels, hidden_dim,
          attention_ratio, DQN head depth, batch_size.

Reports: param counts, peak GPU memory (forward+backward), estimated throughput impact.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import nn, Tensor

# ---------------------------------------------------------------------------
# Flexible model builder (mirrors mortal/model.py but with all dims tunable)
# ---------------------------------------------------------------------------

class FlexChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=True),
            nn.Mish(inplace=True),
            nn.Linear(channels // ratio, channels, bias=True),
        )

    def forward(self, x: Tensor):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        return weight.unsqueeze(-1) * x


class FlexResBlock(nn.Module):
    def __init__(self, channels, *, kernel_size=3, attention_ratio=16):
        super().__init__()
        self.res_unit = nn.Sequential(
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
        )
        self.ca = FlexChannelAttention(channels, ratio=attention_ratio)

    def forward(self, x):
        out = self.res_unit(x)
        out = self.ca(out)
        return out + x


class FlexResNet(nn.Module):
    def __init__(self, in_channels, conv_channels, num_blocks, *,
                 bottleneck_channels=32, hidden_dim=1024,
                 kernel_size=3, attention_ratio=16, spatial_dim=34):
        super().__init__()
        blocks = [FlexResBlock(conv_channels, kernel_size=kernel_size,
                               attention_ratio=attention_ratio)
                  for _ in range(num_blocks)]
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            *blocks,
            nn.BatchNorm1d(conv_channels, momentum=0.01, eps=1e-3),
            nn.Mish(inplace=True),
            nn.Conv1d(conv_channels, bottleneck_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(bottleneck_channels * spatial_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class FlexModel(nn.Module):
    """Brain + DQN, all dims configurable."""
    def __init__(self, in_channels=1012, conv_channels=192, num_blocks=40,
                 bottleneck_channels=32, hidden_dim=1024,
                 kernel_size=3, attention_ratio=16, spatial_dim=34,
                 dqn_hidden=0, action_space=47):
        super().__init__()
        self.encoder = FlexResNet(
            in_channels, conv_channels, num_blocks,
            bottleneck_channels=bottleneck_channels,
            hidden_dim=hidden_dim, kernel_size=kernel_size,
            attention_ratio=attention_ratio, spatial_dim=spatial_dim,
        )
        self.actv = nn.Mish(inplace=True)
        if dqn_hidden > 0:
            self.dqn = nn.Sequential(
                nn.Linear(hidden_dim, dqn_hidden),
                nn.Mish(inplace=True),
                nn.Linear(dqn_hidden, 1 + action_space),
            )
        else:
            self.dqn = nn.Linear(hidden_dim, 1 + action_space)

    def forward(self, x):
        phi = self.actv(self.encoder(x))
        return self.dqn(phi)


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    conv_channels: int = 192
    num_blocks: int = 40
    bottleneck_channels: int = 32
    hidden_dim: int = 1024
    kernel_size: int = 3
    attention_ratio: int = 16
    dqn_hidden: int = 0  # 0 = single linear (v4 style)
    batch_size: int = 8192

    def build(self, in_channels=1012) -> FlexModel:
        return FlexModel(
            in_channels=in_channels,
            conv_channels=self.conv_channels,
            num_blocks=self.num_blocks,
            bottleneck_channels=self.bottleneck_channels,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            attention_ratio=self.attention_ratio,
            dqn_hidden=self.dqn_hidden,
        )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_size_mb(model: nn.Module) -> float:
    """Size of params in MB (float32)."""
    return count_params(model) * 4 / 1024 / 1024


def measure_gpu_memory(spec: ModelSpec, device='cuda:0', dtype=torch.bfloat16) -> dict:
    """Measure peak GPU memory for forward+backward pass."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    model = spec.build().to(device)
    model.train()

    # Simulate AMP bfloat16 forward+backward
    x = torch.randn(spec.batch_size, 1012, 34, device=device, dtype=dtype)

    with torch.amp.autocast('cuda', dtype=dtype):
        out = model(x)
        loss = out.sum()

    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 / 1024

    del model, x, out, loss
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'peak_allocated_mb': round(peak_mb, 1),
        'peak_reserved_mb': round(peak_reserved_mb, 1),
        'peak_allocated_gib': round(peak_mb / 1024, 2),
    }


# ---------------------------------------------------------------------------
# Spec definitions: comprehensive exploration
# ---------------------------------------------------------------------------

def get_all_specs() -> list[ModelSpec]:
    specs = []

    # ===== BASELINE =====
    specs.append(ModelSpec(name='baseline_192ch_40b'))

    # ===== AXIS 1: Width (conv_channels) =====
    for ch in [256, 384, 512, 768]:
        specs.append(ModelSpec(name=f'width_{ch}ch_40b', conv_channels=ch))

    # ===== AXIS 2: Depth (num_blocks) =====
    for nb in [48, 60, 80, 100]:
        specs.append(ModelSpec(name=f'depth_192ch_{nb}b', num_blocks=nb))

    # ===== AXIS 3: Bottleneck channels =====
    for bn in [48, 64, 96, 128]:
        specs.append(ModelSpec(name=f'bottleneck_{bn}', bottleneck_channels=bn))

    # ===== AXIS 4: Hidden dimension =====
    for hd in [1536, 2048, 4096]:
        specs.append(ModelSpec(name=f'hidden_{hd}', hidden_dim=hd))

    # ===== AXIS 5: Attention ratio =====
    for ar in [8, 4]:
        specs.append(ModelSpec(name=f'attn_ratio_{ar}', attention_ratio=ar))

    # ===== AXIS 6: DQN head depth =====
    for dh in [256, 512]:
        specs.append(ModelSpec(name=f'dqn_hidden_{dh}', dqn_hidden=dh))

    # ===== AXIS 7: Kernel size =====
    specs.append(ModelSpec(name='kernel_5', kernel_size=5))

    # ===== COMBINED: Scaled-up candidates =====
    # ~2x params (~21M)
    specs.append(ModelSpec(name='2x_256ch_40b_hd1024',
                           conv_channels=256, num_blocks=40))
    # ~3x params (~32M)
    specs.append(ModelSpec(name='3x_256ch_60b_hd1536',
                           conv_channels=256, num_blocks=60, hidden_dim=1536,
                           bottleneck_channels=48))
    # ~5x params (~54M)
    specs.append(ModelSpec(name='5x_384ch_40b_hd2048',
                           conv_channels=384, num_blocks=40, hidden_dim=2048,
                           bottleneck_channels=64))
    # ~10x params (~108M)
    specs.append(ModelSpec(name='10x_384ch_80b_hd2048',
                           conv_channels=384, num_blocks=80, hidden_dim=2048,
                           bottleneck_channels=64))
    # ~10x v2 (wider, shallower)
    specs.append(ModelSpec(name='10x_v2_512ch_48b_hd2048',
                           conv_channels=512, num_blocks=48, hidden_dim=2048,
                           bottleneck_channels=64))
    # ~15x params
    specs.append(ModelSpec(name='15x_512ch_60b_hd2048',
                           conv_channels=512, num_blocks=60, hidden_dim=2048,
                           bottleneck_channels=96))
    # ~20x params (aggressive)
    specs.append(ModelSpec(name='20x_512ch_80b_hd4096',
                           conv_channels=512, num_blocks=80, hidden_dim=4096,
                           bottleneck_channels=128))
    # ~30x (extreme — likely won't fit)
    specs.append(ModelSpec(name='30x_768ch_60b_hd4096',
                           conv_channels=768, num_blocks=60, hidden_dim=4096,
                           bottleneck_channels=128))

    # ===== BATCH SIZE VARIANTS for biggest viable model =====
    for bs in [4096, 2048]:
        specs.append(ModelSpec(name=f'10x_384ch_80b_bs{bs}',
                               conv_channels=384, num_blocks=80, hidden_dim=2048,
                               bottleneck_channels=64, batch_size=bs))
        specs.append(ModelSpec(name=f'15x_512ch_60b_bs{bs}',
                               conv_channels=512, num_blocks=60, hidden_dim=2048,
                               bottleneck_channels=96, batch_size=bs))

    return specs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--measure-gpu', action='store_true',
                        help='Actually measure GPU memory (slower, needs GPU)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--filter', default='', help='Only run specs containing this string')
    args = parser.parse_args()

    specs = get_all_specs()
    if args.filter:
        specs = [s for s in specs if args.filter in s.name]

    print('=' * 120)
    print('PHASE C: COMPREHENSIVE MODEL SCALING ANALYSIS')
    print('=' * 120)
    print(f'  in_channels=1012, spatial_dim=34, action_space=47')
    print(f'  Current baseline: 192ch / 40 blocks / bottleneck=32 / hidden=1024 / attn_ratio=16')
    print(f'  A100-40GB limit: ~38 GiB usable per GPU')
    print()

    # Table 1: Parameter counts (no GPU needed)
    print(f'{"Name":<35} {"ch":>4} {"blk":>4} {"bn":>3} {"hid":>5} {"ar":>3} '
          f'{"ks":>2} {"dqn_h":>5} {"bs":>5} '
          f'{"Params":>12} {"M_params":>8} {"vs_base":>8}')
    print('-' * 120)

    baseline_params = None
    results = []
    for spec in specs:
        model = spec.build()
        params = count_params(model)
        if baseline_params is None:
            baseline_params = params
        ratio = params / baseline_params

        r = {
            'spec': spec,
            'params': params,
            'ratio': ratio,
        }
        results.append(r)

        print(f'{spec.name:<35} {spec.conv_channels:>4} {spec.num_blocks:>4} '
              f'{spec.bottleneck_channels:>3} {spec.hidden_dim:>5} {spec.attention_ratio:>3} '
              f'{spec.kernel_size:>2} {spec.dqn_hidden:>5} {spec.batch_size:>5} '
              f'{params:>12,} {params/1e6:>7.1f}M {ratio:>7.1f}x')
        del model

    if not args.measure_gpu:
        print()
        print('Run with --measure-gpu to measure actual GPU memory usage.')
        print('This will instantiate each model on GPU with forward+backward pass.')
        return

    # Table 2: GPU memory measurement
    print()
    print('=' * 120)
    print('GPU MEMORY MEASUREMENT (forward + backward, bfloat16 AMP)')
    print('=' * 120)
    print(f'{"Name":<35} {"Params":>8} {"Batch":>5} {"Peak_alloc":>10} {"Peak_rsv":>10} '
          f'{"Fits_40GB":>9} {"Fits_2GPU":>9}')
    print('-' * 120)

    for r in results:
        spec = r['spec']
        try:
            mem = measure_gpu_memory(spec, device=args.device)
            r.update(mem)
            fits_40 = 'YES' if mem['peak_allocated_gib'] < 38 else 'NO'
            # For DDP: each GPU needs its own copy, plus grad buffers
            # Rule of thumb: if single-GPU peak < 35 GiB, DDP should fit
            fits_ddp = 'YES' if mem['peak_allocated_gib'] < 35 else 'MAYBE' if mem['peak_allocated_gib'] < 38 else 'NO'
            print(f'{spec.name:<35} {r["params"]/1e6:>7.1f}M {spec.batch_size:>5} '
                  f'{mem["peak_allocated_mb"]:>9.0f}MB {mem["peak_reserved_mb"]:>9.0f}MB '
                  f'{fits_40:>9} {fits_ddp:>9}')
        except torch.cuda.OutOfMemoryError:
            print(f'{spec.name:<35} {r["params"]/1e6:>7.1f}M {spec.batch_size:>5} '
                  f'{"OOM":>9} {"OOM":>9} {"NO":>9} {"NO":>9}')
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f'{spec.name:<35} ERROR: {e}')
            torch.cuda.empty_cache()
            gc.collect()

    print()
    print('Notes:')
    print('  - Peak_alloc = torch.cuda.max_memory_allocated (actual tensor memory)')
    print('  - Peak_rsv = torch.cuda.max_memory_reserved (allocator overhead)')
    print('  - Fits_40GB = peak_alloc < 38 GiB (leaves headroom)')
    print('  - Fits_2GPU = peak_alloc < 35 GiB (DDP overhead ~2-3 GiB for grad buckets)')
    print('  - All measurements are per-GPU with bfloat16 AMP')


if __name__ == '__main__':
    main()
