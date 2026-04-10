#!/usr/bin/env python3
"""Step 8 Tier 0 - Legal gameplay smoke test.

Loads a single Step 7 BC checkpoint as both challenger and champion, runs
seed_count=1 (4 hanchans) of OneVsThree self-play on CPU, and verifies the
arena loop completes cleanly.

Pass criteria:
  - no crashes, no exceptions
  - all 4 hanchans finish
  - mjai logs are written and parse back into Stat without error

Runs on the RTX 3070 (cuda:0 inside this process when invoked with
`CUDA_VISIBLE_DEVICES=2`) to avoid interfering with the live Step 7
training that is currently using both A100s. Falls back to CPU if no
CUDA device is visible.

This script is intentionally self-contained: no MORTAL_CFG, no shared
config, no production code paths touched. It builds Brain/DQN directly
from the saved checkpoint's config so it works for any model shape,
including the Step 7 10x-wide 512ch/48b/bn64/hd2048 architecture.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / "mortal"
sys.path.insert(0, str(MORTAL_DIR))


def parse_args():
    p = argparse.ArgumentParser(description="Step 8 Tier 0 gameplay smoke")
    p.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts/checkpoints/step7_bc_full_9dan_full_eval_best.pth"),
        help="path to a Step 7 BC checkpoint",
    )
    p.add_argument("--seed-start", type=int, default=10000)
    p.add_argument("--seed-key", type=lambda s: int(s, 0), default=0x2000)
    p.add_argument("--seed-count", type=int, default=1, help="seed_count for OneVsThree (1 -> 4 hanchans)")
    p.add_argument(
        "--device",
        default="auto",
        help="torch device. 'auto' picks cuda:0 if available, else cpu.",
    )
    p.add_argument(
        "--enable-amp",
        default="auto",
        choices=("auto", "true", "false"),
        help="bfloat16 AMP. 'auto' enables on cuda, disables on cpu.",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts/reports/step8/smoke"),
    )
    return p.parse_args()


def _import_step8_harness():
    """Imported lazily so torch errors surface at main() rather than import time."""
    from step8_harness import load_engine_from_checkpoint
    return load_engine_from_checkpoint


def main():
    args = parse_args()

    import torch
    from libriichi.arena import OneVsThree
    from libriichi.stat import Stat
    load_engine_from_checkpoint = _import_step8_harness()

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / f"tier0_n{args.seed_count}_{timestamp}"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.enable_amp == "auto":
        enable_amp = device.type == "cuda"
    else:
        enable_amp = args.enable_amp == "true"

    print(f"device={device} enable_amp={enable_amp}")
    if device.type == "cuda":
        print(f"cuda_device_name={torch.cuda.get_device_name(device)}")
        print(f"cuda_device_capability={torch.cuda.get_device_capability(device)}")
    print(f"log_dir={log_dir}")
    print()

    challenger = load_engine_from_checkpoint(
        args.checkpoint, device=device, name="step7_challenger", enable_amp=enable_amp
    )
    print()
    champion = load_engine_from_checkpoint(
        args.checkpoint, device=device, name="step7_champion", enable_amp=enable_amp
    )
    print()

    seed_pair = (args.seed_start, args.seed_key)
    print(f"seed_start={seed_pair} seed_count={args.seed_count}")
    print("running OneVsThree.py_vs_py...")
    print()

    started_at = time.perf_counter()
    env = OneVsThree(disable_progress_bar=False, log_dir=str(log_dir))
    rankings = env.py_vs_py(
        challenger=challenger,
        champion=champion,
        seed_start=seed_pair,
        seed_count=args.seed_count,
    )
    wall = time.perf_counter() - started_at

    print()
    print(f"rankings={rankings} wall={wall:.1f}s")

    log_files = sorted(log_dir.glob("*.json.gz"))
    print(f"hanchans_logged={len(log_files)}")
    if not log_files:
        print("FAIL: no mjai logs were written")
        sys.exit(1)

    expected_hanchans = args.seed_count * 4
    if len(log_files) != expected_hanchans:
        print(f"FAIL: expected {expected_hanchans} hanchans, got {len(log_files)}")
        sys.exit(1)

    print("running Stat.from_dir on challenger logs...")
    stat = Stat.from_dir(str(log_dir), "step7_challenger", True)

    # Standard tenhou pt distribution for 4-player hanchan with okka:
    # 1st=+90, 2nd=+45, 3rd=0, 4th=-135
    tenhou_pts = [90, 45, 0, -135]

    # Stat exposes metrics as PyO3 properties (no parens) except avg_pt
    # which takes a pt distribution argument.
    metrics = {
        "rankings": list(rankings),
        "total_hanchans": len(log_files),
        "challenger_avg_rank": stat.avg_rank,
        "challenger_avg_pt": stat.avg_pt(tenhou_pts),
        "challenger_rank_1_rate": stat.rank_1_rate,
        "challenger_rank_2_rate": stat.rank_2_rate,
        "challenger_rank_3_rate": stat.rank_3_rate,
        "challenger_rank_4_rate": stat.rank_4_rate,
        "challenger_agari_rate": stat.agari_rate,
        "challenger_houjuu_rate": stat.houjuu_rate,
        "challenger_riichi_rate": stat.riichi_rate,
        "challenger_tobi_rate": stat.tobi_rate,
    }

    summary = {
        "schema_version": "tier0_smoke.v1",
        "tier": 0,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "enable_amp": enable_amp,
        "seed_start": list(seed_pair),
        "seed_count": args.seed_count,
        "wall_seconds": wall,
        "metrics": metrics,
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        "passed": True,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print()
    print("=" * 60)
    print("TIER 0 SMOKE PASS")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print()
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
