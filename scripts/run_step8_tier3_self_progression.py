#!/usr/bin/env python3
"""Step 8 Tier 3 - Self-progression test.

Runs the latest Step 7 BC checkpoint as challenger against an earlier
stage snapshot of the same architecture as champion. This is the
internal progress check: has training produced a meaningfully stronger
model between the two points?

The comparison is 1 challenger vs 3 champions (OneVsThree topology),
so a "neutral" challenger would finish with rank_1_rate = 0.25 (one
rank in every four, on average). A meaningfully stronger challenger
should show:

  - rank_1_rate > 0.25 (ideally > 0.35 for ~2sigma at n=100)
  - avg_rank < 2.5
  - avg_pt > 0

Because challenger and champion are the same architecture at different
training points, they produce genuinely different policies, so all 4
rotations per wall are different games. seed_count=25 gives 100
independent games.

Runs on the RTX 3070 (cuda:0 inside this process when invoked with
`CUDA_VISIBLE_DEVICES=2`) to avoid interfering with live Step 7
training on both A100s.
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
    p = argparse.ArgumentParser(description="Step 8 Tier 3 self-progression test")
    p.add_argument(
        "--challenger",
        default=str(ROOT / "artifacts/checkpoints/step7_bc_full_9dan_full_eval_best.pth"),
        help="path to the newer Step 7 BC checkpoint",
    )
    p.add_argument(
        "--champion",
        default=str(ROOT / "artifacts/checkpoints/step7_bc_full_9dan_stage/stage_step_00008000.pth"),
        help="path to an earlier Step 7 stage snapshot",
    )
    p.add_argument("--seed-start", type=int, default=10000)
    p.add_argument("--seed-key", type=lambda s: int(s, 0), default=0x2000)
    p.add_argument(
        "--seed-count",
        type=int,
        default=25,
        help="seed_count for OneVsThree (default 25 -> 100 hanchans)",
    )
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

    import torch
    from libriichi.arena import OneVsThree
    from libriichi.stat import Stat

    from step8_harness import load_engine_from_checkpoint

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / f"tier3_n{args.seed_count}_{timestamp}"
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
        args.challenger,
        device=device,
        name="step7_newer",
        enable_amp=enable_amp,
    )
    print()
    champion = load_engine_from_checkpoint(
        args.champion,
        device=device,
        name="step7_earlier",
        enable_amp=enable_amp,
    )
    print()

    seed_pair = (args.seed_start, args.seed_key)
    expected_hanchans = args.seed_count * 4
    print(f"seed_start={seed_pair} seed_count={args.seed_count} expected_hanchans={expected_hanchans}")
    print("running OneVsThree.py_vs_py (newer vs 3x earlier)...")
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

    if len(log_files) != expected_hanchans:
        print(f"FAIL: expected {expected_hanchans} hanchans, got {len(log_files)}")
        sys.exit(1)

    tenhou_pts = [90, 45, 0, -135]

    print("running Stat.from_dir on challenger logs...")
    challenger_stat = Stat.from_dir(str(log_dir), "step7_newer", True)
    print("running Stat.from_dir on champion logs...")
    champion_stat = Stat.from_dir(str(log_dir), "step7_earlier", True)

    challenger_metrics = stat_to_dict(challenger_stat, tenhou_pts)
    champion_metrics = stat_to_dict(champion_stat, tenhou_pts)

    # Soft gates: report but do not block script exit. Tier 3 is
    # exploratory, not correctness-critical.
    gate_results = {
        "rank_1_rate_above_neutral": {
            "passed": challenger_metrics["rank_1_rate"] > 0.25,
            "value": challenger_metrics["rank_1_rate"],
            "threshold": 0.25,
        },
        "avg_rank_below_neutral": {
            "passed": challenger_metrics["avg_rank"] < 2.5,
            "value": challenger_metrics["avg_rank"],
            "threshold": 2.5,
        },
        "avg_pt_above_zero": {
            "passed": challenger_metrics["avg_pt"] > 0.0,
            "value": challenger_metrics["avg_pt"],
            "threshold": 0.0,
        },
        "rank_1_rate_ge_0.35_2sigma": {
            "passed": challenger_metrics["rank_1_rate"] >= 0.35,
            "value": challenger_metrics["rank_1_rate"],
            "threshold": 0.35,
            "note": "~2sigma above neutral at n=100",
        },
    }

    all_soft_gates_passed = all(g["passed"] for g in gate_results.values())

    summary = {
        "schema_version": "tier3_smoke.v1",
        "tier": 3,
        "challenger_checkpoint": str(Path(args.challenger).resolve()),
        "champion_checkpoint": str(Path(args.champion).resolve()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "enable_amp": enable_amp,
        "seed_start": list(seed_pair),
        "seed_count": args.seed_count,
        "total_hanchans": len(log_files),
        "rankings": list(rankings),
        "wall_seconds": wall,
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "gate_results": gate_results,
        "all_soft_gates_passed": all_soft_gates_passed,
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print()
    print("=" * 60)
    print(f"TIER 3 {'PASS' if all_soft_gates_passed else 'INCONCLUSIVE'}")
    print("=" * 60)
    print()
    print(f"rankings (challenger across 100 games): {rankings}")
    print(f"total hanchans: {len(log_files)}")
    print()
    print("challenger (newer) metrics:")
    print(json.dumps(challenger_metrics, indent=2))
    print()
    print("champion (earlier) metrics:")
    print(json.dumps(champion_metrics, indent=2))
    print()
    print("gate results:")
    for name, result in gate_results.items():
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"  [{mark}] {name}: value={result['value']:.4f} threshold={result['threshold']}")
        if result.get("note"):
            print(f"         note: {result['note']}")
    print()
    print(f"summary written to {summary_path}")

    # Tier 3 is exploratory: always exit 0, let the operator interpret.


if __name__ == "__main__":
    main()
