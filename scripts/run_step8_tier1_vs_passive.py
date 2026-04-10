#!/usr/bin/env python3
"""Step 8 Tier 1 - Crush-the-passive smoke test.

Runs the Step 7 BC checkpoint as challenger against three copies of
`ExampleMjaiLogEngine` (tsumogiri-only, never calls, never riichis,
never wins). This answers two questions with one test:

  1. Strength floor: "Has the model learned to actually win hands, or is
     it just emitting legal actions?" A BC model trained on 9dan Phoenix
     data should crush a passive opponent - rank_1_rate >= 0.95.

  2. Correctness check: "Does the mjai-log engine_type dispatch wire up
     end-to-end?" If the passive opponent is wired correctly, it never
     agaris, so there is nothing for the challenger to deal into. The
     challenger's houjuu_rate MUST be exactly 0.000. If it is not, we
     have an MjaiLogBatchAgent wiring bug, not a model bug.

Because challenger and champion are different engine types, each of the
4 seat-rotations at every wall produces a genuinely different game (the
challenger gets different starting tiles from each seat). So
seed_count=25 gives 100 truly independent games of data, unlike tier 0
self-play where rotations collapse to 1.

Runs on the RTX 3070 (cuda:0 inside this process when invoked with
`CUDA_VISIBLE_DEVICES=2`) to avoid interfering with the live Step 7
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
    p = argparse.ArgumentParser(description="Step 8 Tier 1 vs passive smoke")
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


def main():
    args = parse_args()

    import torch
    from libriichi.arena import OneVsThree
    from libriichi.stat import Stat

    from step8_harness import load_engine_from_checkpoint
    from engine import ExampleMjaiLogEngine

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / f"tier1_n{args.seed_count}_{timestamp}"
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

    # Challenger: the real Step 7 BC model
    challenger = load_engine_from_checkpoint(
        args.checkpoint,
        device=device,
        name="step7_challenger",
        enable_amp=enable_amp,
    )
    print()

    # Champion: passive tsumogiri-only bot (never wins)
    champion = ExampleMjaiLogEngine(name="passive_tsumogiri")
    print(f"[passive_tsumogiri] engine_type={champion.engine_type}")
    print()

    seed_pair = (args.seed_start, args.seed_key)
    expected_hanchans = args.seed_count * 4
    print(f"seed_start={seed_pair} seed_count={args.seed_count} expected_hanchans={expected_hanchans}")
    print("running OneVsThree.py_vs_py (Step 7 vs 3x passive)...")
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

    print("running Stat.from_dir on challenger logs...")
    challenger_stat = Stat.from_dir(str(log_dir), "step7_challenger", True)

    print("running Stat.from_dir on passive opponent logs...")
    passive_stat = Stat.from_dir(str(log_dir), "passive_tsumogiri", True)

    tenhou_pts = [90, 45, 0, -135]

    challenger_metrics = {
        "avg_rank": challenger_stat.avg_rank,
        "avg_pt": challenger_stat.avg_pt(tenhou_pts),
        "rank_1_rate": challenger_stat.rank_1_rate,
        "rank_2_rate": challenger_stat.rank_2_rate,
        "rank_3_rate": challenger_stat.rank_3_rate,
        "rank_4_rate": challenger_stat.rank_4_rate,
        "agari_rate": challenger_stat.agari_rate,
        "houjuu_rate": challenger_stat.houjuu_rate,
        "riichi_rate": challenger_stat.riichi_rate,
        "tobi_rate": challenger_stat.tobi_rate,
    }
    passive_metrics = {
        "avg_rank": passive_stat.avg_rank,
        "avg_pt": passive_stat.avg_pt(tenhou_pts),
        "rank_1_rate": passive_stat.rank_1_rate,
        "rank_2_rate": passive_stat.rank_2_rate,
        "rank_3_rate": passive_stat.rank_3_rate,
        "rank_4_rate": passive_stat.rank_4_rate,
        "agari_rate": passive_stat.agari_rate,
        "houjuu_rate": passive_stat.houjuu_rate,
        "riichi_rate": passive_stat.riichi_rate,
        "tobi_rate": passive_stat.tobi_rate,
    }

    # Pass criteria
    gate_results = {}
    failures = []

    # 1. Correctness: passive opponents cannot agari, so challenger cannot deal in
    if challenger_metrics["houjuu_rate"] == 0.0:
        gate_results["houjuu_rate_is_zero"] = {"passed": True, "value": 0.0}
    else:
        gate_results["houjuu_rate_is_zero"] = {
            "passed": False,
            "value": challenger_metrics["houjuu_rate"],
            "explanation": "challenger dealt in to a passive opponent - mjai-log wiring bug?",
        }
        failures.append("houjuu_rate_is_zero")

    # 2. Strength floor: model should crush a passive opponent
    if challenger_metrics["rank_1_rate"] >= 0.95:
        gate_results["rank_1_rate_ge_0.95"] = {
            "passed": True,
            "value": challenger_metrics["rank_1_rate"],
        }
    else:
        gate_results["rank_1_rate_ge_0.95"] = {
            "passed": False,
            "value": challenger_metrics["rank_1_rate"],
            "explanation": "model is not crushing a passive opponent - strength floor not met",
        }
        failures.append("rank_1_rate_ge_0.95")

    # 3. Passive opponents should never agari
    if passive_metrics["agari_rate"] == 0.0:
        gate_results["passive_never_agaris"] = {"passed": True, "value": 0.0}
    else:
        gate_results["passive_never_agaris"] = {
            "passed": False,
            "value": passive_metrics["agari_rate"],
            "explanation": "ExampleMjaiLogEngine somehow agari'd - mjai-log dispatch bug?",
        }
        failures.append("passive_never_agaris")

    passed = len(failures) == 0

    summary = {
        "schema_version": "tier1_smoke.v1",
        "tier": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "opponent": "ExampleMjaiLogEngine (passive tsumogiri)",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "enable_amp": enable_amp,
        "seed_start": list(seed_pair),
        "seed_count": args.seed_count,
        "total_hanchans": len(log_files),
        "rankings": list(rankings),
        "wall_seconds": wall,
        "challenger_metrics": challenger_metrics,
        "passive_metrics": passive_metrics,
        "gate_results": gate_results,
        "passed": passed,
        "failures": failures,
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print()
    print("=" * 60)
    print(f"TIER 1 SMOKE {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    print()
    print("rankings (challenger):", rankings)
    print(f"total hanchans: {len(log_files)}")
    print()
    print("challenger metrics:")
    print(json.dumps(challenger_metrics, indent=2))
    print()
    print("passive_tsumogiri metrics:")
    print(json.dumps(passive_metrics, indent=2))
    print()
    print("gate results:")
    for name, result in gate_results.items():
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"  [{mark}] {name}: value={result['value']}")
        if not result["passed"]:
            print(f"         reason: {result.get('explanation', '')}")
    print()
    print(f"summary written to {summary_path}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
