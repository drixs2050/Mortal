#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / "mortal"
DEFAULT_CONFIG = ROOT / "configs" / "step1_smoke.toml"


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def unique_paths(cfg: dict) -> list[Path]:
    paths = [
        Path(cfg["control"]["state_file"]),
        Path(cfg["control"]["best_state_file"]),
        Path(cfg["baseline"]["train"]["state_file"]),
        Path(cfg["baseline"]["test"]["state_file"]),
        Path(cfg["grp"]["state_file"]),
        Path(cfg["control"]["tensorboard_dir"]),
        Path(cfg["grp"]["control"]["tensorboard_dir"]),
        Path(cfg["test_play"]["log_dir"]),
        Path(cfg["train_play"]["default"]["log_dir"]),
        Path(cfg["online"]["server"]["buffer_dir"]),
        Path(cfg["online"]["server"]["drain_dir"]),
        Path(cfg["dataset"]["file_index"]),
        Path(cfg["grp"]["dataset"]["file_index"]),
        Path(cfg["step1"]["fixture_log_dir"]),
        Path(cfg["1v3"]["log_dir"]),
    ]
    out = []
    seen = set()
    for item in paths:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def ensure_dirs(cfg: dict) -> None:
    for target in unique_paths(cfg):
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)


def import_runtime(config_path: Path):
    os.environ.setdefault("MORTAL_CFG", str(config_path))
    if str(MORTAL_DIR) not in sys.path:
        sys.path.insert(0, str(MORTAL_DIR))
    try:
        import torch
        from torch import nn, optim
        from torch.amp import GradScaler

        from engine import MortalEngine
        from libriichi.arena import OneVsThree
        from lr_scheduler import LinearWarmUpCosineAnnealingLR
        from model import AuxNet, Brain, DQN, GRP
    except Exception as exc:
        raise SystemExit(
            "Failed to import the Mortal runtime.\n"
            "Install Rust, install torch, and build/copy libriichi before running this script.\n"
            f"Details: {exc}"
        ) from exc

    return {
        "torch": torch,
        "nn": nn,
        "optim": optim,
        "GradScaler": GradScaler,
        "MortalEngine": MortalEngine,
        "OneVsThree": OneVsThree,
        "LinearWarmUpCosineAnnealingLR": LinearWarmUpCosineAnnealingLR,
        "AuxNet": AuxNet,
        "Brain": Brain,
        "DQN": DQN,
        "GRP": GRP,
    }


def build_policy_checkpoint(cfg: dict, rt: dict) -> dict:
    torch = rt["torch"]
    nn = rt["nn"]
    optim = rt["optim"]
    GradScaler = rt["GradScaler"]
    Brain = rt["Brain"]
    DQN = rt["DQN"]
    AuxNet = rt["AuxNet"]
    Scheduler = rt["LinearWarmUpCosineAnnealingLR"]

    version = cfg["control"]["version"]
    mortal = Brain(
        version=version,
        conv_channels=cfg["resnet"]["conv_channels"],
        num_blocks=cfg["resnet"]["num_blocks"],
    )
    dqn = DQN(version=version)
    aux_net = AuxNet((4,))

    all_models = (mortal, dqn, aux_net)
    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith("weight"):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))

    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg["optim"]["weight_decay"]},
            {"params": no_decay_params},
        ],
        lr=1,
        weight_decay=0,
        betas=tuple(cfg["optim"]["betas"]),
        eps=cfg["optim"]["eps"],
    )
    scheduler = Scheduler(optimizer, **cfg["optim"]["scheduler"])
    device_type = str(cfg["control"]["device"]).split(":", 1)[0]
    scaler = GradScaler(device_type, enabled=cfg["control"]["enable_amp"])

    return {
        "mortal": mortal.state_dict(),
        "current_dqn": dqn.state_dict(),
        "aux_net": aux_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "steps": 0,
        "timestamp": datetime.now(tz=timezone.utc).timestamp(),
        "best_perf": {"avg_rank": 4.0, "avg_pt": -135.0},
        "config": cfg,
        "tag": "step1-bootstrap-main",
    }


def build_grp_checkpoint(cfg: dict, rt: dict) -> dict:
    optim = rt["optim"]
    GRP = rt["GRP"]

    grp = GRP(**cfg["grp"]["network"])
    optimizer = optim.AdamW(grp.parameters(), lr=cfg["grp"]["optim"]["lr"])
    return {
        "model": grp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": 0,
        "timestamp": datetime.now(tz=timezone.utc).timestamp(),
    }


def write_checkpoints(cfg: dict, rt: dict, force: bool) -> None:
    torch = rt["torch"]

    main_path = Path(cfg["control"]["state_file"])
    best_path = Path(cfg["control"]["best_state_file"])
    baseline_path = Path(cfg["baseline"]["train"]["state_file"])
    baseline_test_path = Path(cfg["baseline"]["test"]["state_file"])
    grp_path = Path(cfg["grp"]["state_file"])

    if force or not main_path.exists():
        policy_state = build_policy_checkpoint(cfg, rt)
        torch.save(policy_state, main_path)
        print(f"wrote main policy checkpoint to {main_path}")
    else:
        print(f"main policy checkpoint already exists at {main_path}, skipping")

    sibling_paths = []
    seen = set()
    for sibling in (best_path, baseline_path, baseline_test_path):
        if sibling not in seen:
            seen.add(sibling)
            sibling_paths.append(sibling)

    for sibling in sibling_paths:
        if force or not sibling.exists():
            shutil.copy2(main_path, sibling)
            print(f"synced policy checkpoint into {sibling}")
        else:
            print(f"checkpoint already exists at {sibling}, skipping")

    if force or not grp_path.exists():
        grp_state = build_grp_checkpoint(cfg, rt)
        torch.save(grp_state, grp_path)
        print(f"wrote GRP checkpoint to {grp_path}")
    else:
        print(f"GRP checkpoint already exists at {grp_path}, skipping")


def load_engine_from_checkpoint(checkpoint_path: Path, rt: dict):
    torch = rt["torch"]
    Brain = rt["Brain"]
    DQN = rt["DQN"]
    MortalEngine = rt["MortalEngine"]

    state = torch.load(checkpoint_path, weights_only=True, map_location=torch.device("cpu"))
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    brain = Brain(
        version=version,
        conv_channels=cfg["resnet"]["conv_channels"],
        num_blocks=cfg["resnet"]["num_blocks"],
    ).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return MortalEngine(
        brain,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device("cpu"),
        enable_amp=False,
        enable_rule_based_agari_guard=True,
        name=checkpoint_path.stem,
    )


def generate_fixture(cfg: dict, rt: dict, force: bool) -> None:
    OneVsThree = rt["OneVsThree"]
    fixture_dir = Path(cfg["step1"]["fixture_log_dir"])
    fixture_dir.mkdir(parents=True, exist_ok=True)

    existing_logs = sorted(fixture_dir.glob("*.json.gz"))
    if existing_logs and not force:
        print(f"fixture logs already exist under {fixture_dir}, skipping")
        return

    for old_file in existing_logs:
        old_file.unlink()

    challenger = load_engine_from_checkpoint(Path(cfg["control"]["state_file"]), rt)
    champion = load_engine_from_checkpoint(Path(cfg["baseline"]["train"]["state_file"]), rt)
    env = OneVsThree(
        disable_progress_bar=True,
        log_dir=str(fixture_dir),
    )
    rankings = env.py_vs_py(
        challenger=challenger,
        champion=champion,
        seed_start=(cfg["step1"]["fixture_seed_start"], cfg["step1"]["fixture_seed_key"]),
        seed_count=1,
    )
    created = sorted(fixture_dir.glob("*.json.gz"))
    if not created:
        raise SystemExit(f"fixture generation finished without any log files in {fixture_dir}")
    print(f"generated {len(created)} fixture logs under {fixture_dir}; challenger rankings: {rankings}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Step 1 local checkpoints and tiny fixtures.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config file to use (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--mode",
        choices=("checkpoints", "all"),
        default="all",
        help="Whether to generate only checkpoints or checkpoints plus the tiny fixture.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoints and fixture logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    rt = import_runtime(args.config)
    write_checkpoints(cfg, rt, args.force)
    if args.mode == "all":
        generate_fixture(cfg, rt, args.force)


if __name__ == "__main__":
    main()
