# Step 1 Gap Report

Date: 2026-03-28

## Current Status
- No Step 1 blocking failures remain on the single-A100 smoke or soak path.
- There is still a GPU index-order discrepancy between raw `nvidia-smi` output and torch-visible logical order, so configs should follow the torch-visible order used by the `mortal` env.
- Step 1 exit criteria are now satisfied and Step 2 can begin.

## Resolved During Step 1
- `mortal/train_grp.py` no longer fails on a single A100 after removing worker-side pinning and switching the Step 1 profile to conservative dataloading.
- `mortal/train.py` no longer appears stalled on a single A100; it enters training and test-play correctly when run directly through the `mortal` env's Python.
- Longer single-A100 validation now passes for `mortal/one_vs_three.py`, `mortal/train_grp.py`, and `mortal/train.py`.
- The offline trainer can now save an updated best checkpoint during Step 1 validation.
- A 10-minute single-A100 soak run completed on `cuda:0` without a crash or hang before timeout.
- The Step 2 handoff/backlog is now written under `plans/step_02_handoff_and_backlog.md`.

## Confirmed Repo-Level Dependencies
- The training stack depends on a built `libriichi` shared library in `mortal/`.
- The dataloader requires a GRP checkpoint before offline training can iterate samples.
- The test/train self-play helpers require baseline checkpoints before evaluation can run.
- Offline inference requires a model checkpoint at `control.state_file`.

## Known Codebase Risks Already Identified
- `mortal/train.py` documents a known online-training hang after test-play in online mode.
- `environment.yml` was missing `numpy` before Step 1 edits.
- Current docs cover basic build flow but not a fully reproducible workstation bring-up.

## What Is Implemented So Far
- Local Step 1 directory layout has been created under `artifacts/`, `data/`, `configs/`, and `scripts/`.
- A single-A100-first local smoke config now exists at `configs/step1_smoke.toml`.
- A local bootstrap helper now exists at `scripts/step1_bootstrap.py`.
- The `mortal` Conda env has been created and populated with the repo's Python dependencies plus `torch`.
- Rust has been installed via `rustup`.
- `libriichi` has been built in release mode and copied into `mortal/libriichi.so`.
- Local bootstrap checkpoints and a tiny self-play fixture have been generated successfully.
- Step 1 environment and bootstrap notes have been written under `plans/`.
- A dedicated Step 2 handoff/backlog now exists at `plans/step_02_handoff_and_backlog.md`.

## Ongoing Guardrails After Step 1
- Keep configs aligned with torch-visible devices while maintaining a single-GPU default path.
- Keep `cuda:1` out of the default workflow until a later stage that explicitly covers second-GPU usage.
- Leave the known online-mode hang investigation for a later step unless it blocks immediate Step 2 work.
