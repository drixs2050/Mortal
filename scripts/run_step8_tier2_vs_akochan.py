#!/usr/bin/env python3
"""Step 8 Tier 2 - Step 7 Mortal vs akochan.

Runs the Step 7 BC checkpoint as challenger against three copies of
akochan as champions, via OneVsThree.py_vs_ako. Akochan is the only
publicly-available strong rule-based reference for riichi mahjong, so
this is the load-bearing external strength measurement for Step 8.

Akochan must be built externally (see plans/step_08 section 4) and
the following environment variables must be set BEFORE running this
script:

  AKOCHAN_DIR        absolute path to the akochan build dir
                     (containing system.exe and libai.so)
  AKOCHAN_TACTICS    relative-to-AKOCHAN_DIR path to a tactics JSON
                     file (default 'setup_mjai.json')
  LD_LIBRARY_PATH    must include AKOCHAN_DIR so the akochan
                     subprocess can find libai.so

The Mortal challenger runs on the RTX 3070 (cuda:0 inside this process
when invoked with CUDA_VISIBLE_DEVICES=2). The akochan subprocesses
run on CPU.

For seed_count=N, the arena spawns 4*N parallel games, each with up
to 3 akochan subprocesses (one per opponent seat). Wall-clock per
game depends on akochan's per-decision search depth and is the main
unknown for tier 2 -- start with seed_count=1 as a timing probe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / "mortal"
sys.path.insert(0, str(MORTAL_DIR))


def parse_args():
    p = argparse.ArgumentParser(description="Step 8 Tier 2: Step 7 vs 3x akochan")
    p.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts/checkpoints/step7_bc_full_9dan_full_eval_best.pth"),
        help="path to a Step 7 BC checkpoint",
    )
    p.add_argument("--seed-start", type=int, default=10000)
    p.add_argument("--seed-key", type=lambda s: int(s, 0), default=0x2000)
    p.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="seed_count for OneVsThree (default 1 -> 4 hanchans, "
        "use this as a timing probe; expect MUCH slower than tier 0/1/3 "
        "because akochan does Monte Carlo search per decision)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="torch device for the Mortal challenger. 'auto' picks cuda:0 if available, else cpu.",
    )
    p.add_argument(
        "--enable-amp",
        default="auto",
        choices=("auto", "true", "false"),
        help="bfloat16 AMP for Mortal. 'auto' enables on cuda, disables on cpu.",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts/reports/step8/smoke"),
    )
    return p.parse_args()


def stat_to_dict(stat, tenhou_pts):
    return {
        "avg_rank": stat.avg_rank,
        "avg_pt": stat.avg_pt(tenhou_pts),
        "rank_1_rate": stat.rank_1_rate,
        "rank_2_rate": stat.rank_2_rate,
        "rank_3_rate": stat.rank_3_rate,
        "rank_4_rate": stat.rank_4_rate,
        "agari_rate": stat.agari_rate,
        "houjuu_rate": stat.houjuu_rate,
        "riichi_rate": stat.riichi_rate,
        "tobi_rate": stat.tobi_rate,
    }


def main():
    args = parse_args()

    # Validate akochan environment up front so we fail fast.
    akochan_dir = os.environ.get("AKOCHAN_DIR")
    if not akochan_dir:
        sys.exit("FAIL: AKOCHAN_DIR is not set. export AKOCHAN_DIR=/path/to/akochan first.")
    akochan_dir_path = Path(akochan_dir)
    if not akochan_dir_path.is_dir():
        sys.exit(f"FAIL: AKOCHAN_DIR does not exist: {akochan_dir}")
    system_exe = akochan_dir_path / "system.exe"
    if not system_exe.is_file():
        sys.exit(f"FAIL: missing {system_exe}")
    libai_so = akochan_dir_path / "libai.so"
    if not libai_so.is_file():
        sys.exit(f"FAIL: missing {libai_so}")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if str(akochan_dir_path) not in ld_path.split(":"):
        sys.exit(
            f"FAIL: LD_LIBRARY_PATH does not include AKOCHAN_DIR.\n"
            f"  current: {ld_path!r}\n"
            f"  fix:     export LD_LIBRARY_PATH={akochan_dir_path}:$LD_LIBRARY_PATH"
        )
    tactics = os.environ.get("AKOCHAN_TACTICS", "setup_mjai.json")
    tactics_full = akochan_dir_path / tactics if not Path(tactics).is_absolute() else Path(tactics)
    if not tactics_full.is_file():
        sys.exit(f"FAIL: AKOCHAN_TACTICS file does not exist: {tactics_full}")

    print("akochan environment:")
    print(f"  AKOCHAN_DIR     = {akochan_dir}")
    print(f"  AKOCHAN_TACTICS = {tactics} ({tactics_full})")
    print(f"  LD_LIBRARY_PATH = {ld_path}")
    print()

    import torch
    from libriichi.arena import OneVsThree
    from libriichi.stat import Stat

    from step8_harness import load_engine_from_checkpoint

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / f"tier2_n{args.seed_count}_{timestamp}"
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
        args.checkpoint,
        device=device,
        name="step7_challenger",
        enable_amp=enable_amp,
    )
    print()

    seed_pair = (args.seed_start, args.seed_key)
    expected_hanchans = args.seed_count * 4
    print(f"seed_start={seed_pair} seed_count={args.seed_count} expected_hanchans={expected_hanchans}")
    print()
    print("running OneVsThree.py_vs_ako (Step 7 challenger vs 3x akochan)...")
    print("(akochan does real Monte Carlo search per decision -- this WILL be slow)")
    print()

    started_at = time.perf_counter()
    env = OneVsThree(disable_progress_bar=False, log_dir=str(log_dir))
    rankings = env.py_vs_ako(
        engine=challenger,
        seed_start=seed_pair,
        seed_count=args.seed_count,
    )
    wall = time.perf_counter() - started_at

    print()
    print(f"rankings={rankings} wall={wall:.1f}s")
    per_game = wall / max(expected_hanchans, 1)
    print(f"per_game={per_game:.1f}s ({per_game / 60:.1f} min)")

    log_files = sorted(log_dir.glob("*.json.gz"))
    print(f"hanchans_logged={len(log_files)}")
    if not log_files:
        print("FAIL: no mjai logs were written")
        sys.exit(1)

    if len(log_files) != expected_hanchans:
        print(f"WARN: expected {expected_hanchans} hanchans, got {len(log_files)}")

    print("running Stat.from_dir on challenger logs...")
    challenger_stat = Stat.from_dir(str(log_dir), "step7_challenger", True)
    print("running Stat.from_dir on akochan logs...")
    # akochan agent reports its name as "akochan" by default; if libriichi
    # wraps it differently this will fall back to empty stats
    try:
        akochan_stat = Stat.from_dir(str(log_dir), "akochan", True)
    except Exception:
        akochan_stat = None

    tenhou_pts = [90, 45, 0, -135]
    challenger_metrics = stat_to_dict(challenger_stat, tenhou_pts)
    akochan_metrics = stat_to_dict(akochan_stat, tenhou_pts) if akochan_stat else None

    summary = {
        "schema_version": "tier2_smoke.v1",
        "tier": 2,
        "challenger_checkpoint": str(Path(args.checkpoint).resolve()),
        "champion": "akochan (3x via py_vs_ako)",
        "akochan_dir": akochan_dir,
        "akochan_tactics": str(tactics_full),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "enable_amp": enable_amp,
        "seed_start": list(seed_pair),
        "seed_count": args.seed_count,
        "total_hanchans": len(log_files),
        "rankings": list(rankings),
        "wall_seconds": wall,
        "wall_seconds_per_game": per_game,
        "challenger_metrics": challenger_metrics,
        "akochan_metrics": akochan_metrics,
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print()
    print("=" * 60)
    print("TIER 2 RESULT")
    print("=" * 60)
    print()
    print(f"rankings (challenger): {rankings}")
    print(f"total hanchans: {len(log_files)}")
    print(f"wall: {wall:.1f}s ({per_game:.1f}s per game)")
    print()
    print("step7_challenger metrics:")
    print(json.dumps(challenger_metrics, indent=2))
    if akochan_metrics is not None:
        print()
        print("akochan metrics:")
        print(json.dumps(akochan_metrics, indent=2))
    print()
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
