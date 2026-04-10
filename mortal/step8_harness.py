"""Step 8 evaluation harness — minimal shared library surface.

Scope for now is deliberately narrow: everything that the tier 0 / 1 / 2 / 3
smoke scripts need in common, and nothing else. The fuller library surface
sketched in plans/step_08_evaluation_harness_and_regression_suite_plan.md
section 11a (load_eval_pool, run_match, evaluate_checkpoint,
register_checkpoint) will be built out as later phases of Step 8 require
them. Keeping this module small now avoids committing to signatures that
might need to change once we know more.

Current public API:
  - load_engine_from_checkpoint(...) -> MortalEngine

Reads ALL resnet kwargs (conv_channels, num_blocks, bottleneck_channels,
hidden_dim) from the saved checkpoint config, which is needed for the
Step 7 10x-wide 512ch/48b/bn64/hd2048 architecture. The legacy
mortal/player.py load path only forwards conv_channels + num_blocks and
would silently build a wrong-shape Brain for this model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from model import Brain, DQN
from engine import MortalEngine


def load_engine_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    name: str,
    enable_amp: bool = True,
    enable_rule_based_agari_guard: bool = True,
    enable_quick_eval: bool = True,
    verbose: bool = True,
) -> MortalEngine:
    """Load a BC checkpoint and return a deterministic-mode MortalEngine.

    Deterministic mode means:
      - boltzmann_epsilon = 0 (pure argmax)
      - boltzmann_temp    = 1 (ignored at eps=0)
      - top_p             = 1 (ignored at eps=0)
      - stochastic_latent = False

    AMP is on by default on CUDA (bfloat16 recommended for Ampere+).
    Reading ALL resnet kwargs from the saved config makes this work for
    any width / depth, including the Step 7 10x-wide architecture that
    mortal/player.py's load_engine cannot handle.

    Parameters
    ----------
    checkpoint_path:
        Path to a Step 5+ BC checkpoint with a `config.resnet` block and
        `mortal` / `current_dqn` state dicts.
    device:
        Torch device to load onto.
    name:
        Engine name used in mjai logs (e.g. "step7_challenger").
    enable_amp:
        Passed through to MortalEngine. On cuda+bf16 this is essentially
        free and recommended.
    enable_rule_based_agari_guard:
        Passed through. Recommended True for evaluation.
    enable_quick_eval:
        Passed through. Recommended True for evaluation.
    verbose:
        Print checkpoint metadata to stdout on load.
    """
    ckpt_path = Path(checkpoint_path)
    state = torch.load(ckpt_path, weights_only=True, map_location=device)

    cfg = state["config"]
    resnet_cfg = cfg["resnet"]
    bc_cfg = cfg.get("bc", {})
    control_cfg = bc_cfg.get("control", {})
    dataset_cfg = bc_cfg.get("dataset", {})
    version = control_cfg.get("version", 4)
    oracle = dataset_cfg.get("oracle", False)

    if verbose:
        print(f"[{name}] checkpoint={ckpt_path}")
        print(f"[{name}] version={version} oracle={oracle}")
        print(f"[{name}] resnet={resnet_cfg}")
        print(f"[{name}] saved_step={state.get('steps')} best_perf={state.get('best_perf')}")

    brain = Brain(
        version=version,
        is_oracle=oracle,
        conv_channels=resnet_cfg["conv_channels"],
        num_blocks=resnet_cfg["num_blocks"],
        bottleneck_channels=resnet_cfg.get("bottleneck_channels", 32),
        hidden_dim=resnet_cfg.get("hidden_dim", 1024),
    ).eval()
    dqn = DQN(version=version, hidden_dim=brain.hidden_dim).eval()

    # If the state_dict shape disagrees with what we built, load_state_dict
    # will raise loudly here — exactly the failure mode we want surfaced.
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    brain = brain.to(device)
    dqn = dqn.to(device)

    return MortalEngine(
        brain,
        dqn,
        is_oracle=oracle,
        version=version,
        device=device,
        enable_amp=enable_amp,
        enable_quick_eval=enable_quick_eval,
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        stochastic_latent=False,
        boltzmann_epsilon=0,
        boltzmann_temp=1.0,
        top_p=1.0,
        name=name,
    )
