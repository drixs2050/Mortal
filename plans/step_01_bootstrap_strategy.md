# Step 1 Bootstrap Strategy

## Goal
Make Step 1 runnable with zero external checkpoints and zero public model dependencies.

## Strategy
Use locally generated smoke-test assets:
- A main policy checkpoint for inference and offline trainer bootstrapping
- A baseline checkpoint for `TestPlayer` and `TrainPlayer`
- A GRP checkpoint for `RewardCalculator` and dataset loading
- A tiny self-play fixture generated locally after the runtime can execute `OneVsThree`

## Helper Script
Script:
- `scripts/step1_bootstrap.py`

Config:
- `configs/step1_smoke.toml`

Default behavior:
1. Ensure required output directories exist.
2. Generate a valid policy checkpoint with matching optimizer, scheduler, and scaler state.
3. Generate a valid GRP checkpoint with matching optimizer state.
4. Generate one tiny self-play fixture in `artifacts/fixtures/logs/`.

## Why This Matters
Several current code paths eagerly load checkpoints:
- `mortal/mortal.py` loads `control.state_file`
- `mortal/player.py` loads `baseline.train.state_file` and `baseline.test.state_file`
- `mortal/dataloader.py` loads `grp.state_file`

Without local bootstrap assets, Step 1 smoke tests cannot even reach the first real runtime check.

## Expected Usage
After `torch` and `libriichi` are available:

```bash
export MORTAL_CFG=/home/drixs2050/Documents/Mortal/configs/step1_smoke.toml
python scripts/step1_bootstrap.py --force
```

If only checkpoints are needed:

```bash
python scripts/step1_bootstrap.py --mode checkpoints --force
```
