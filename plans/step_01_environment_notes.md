# Step 1 Environment Notes

Date: 2026-03-28

## Current Probe Status
- `conda` is installed at `/home/drixs2050/anaconda3/bin/conda`.
- The currently active Python is `/home/drixs2050/anaconda3/bin/python`.
- `python --version` reports `Python 3.12.4`.
- A dedicated Conda env now exists at `/home/drixs2050/anaconda3/envs/mortal`.
- `torch`, `numpy`, `toml`, and `tqdm` import correctly inside the `mortal` env.
- `cargo 1.94.1` and `rustc 1.94.1` are installed and working.
- `libriichi` builds successfully in release mode and imports from `mortal/libriichi.so`.
- Out-of-sandbox GPU checks confirm CUDA is available in the `mortal` env.
- Torch-visible logical device order in the `mortal` env is:
  - `cuda:0` = NVIDIA A100-SXM4-40GB
  - `cuda:1` = NVIDIA A100-SXM4-40GB
  - `cuda:2` = NVIDIA GeForce RTX 3070
- Default Step 1 policy is to use only `cuda:0`.
- `cuda:1` should stay out of the default path until the single-GPU bring-up is stable.
- `cuda:2` should be ignored for training/evaluation.
- For long-running training smoke checks, calling the env's Python binary directly produced clearer output than `conda run`.
- Longer host-shell validation has now succeeded for `mortal/one_vs_three.py`, `mortal/train_grp.py`, and `mortal/train.py` on `cuda:0`.

## Required Tooling For This Repo
- Python via Conda
- Python packages: `torch`, `numpy`, `toml`, `tqdm`, `tensorboard`
- Rust toolchain via `rustup`: `rustc` and `cargo`
- System linker/build tools needed for the Rust `PyO3` extension build

## Repo Notes
- `environment.yml` now includes `numpy`, which the repo imports directly.
- `torch` is still intentionally handled outside `environment.yml` because the final install path may depend on the CUDA/runtime situation.
- `configs/step1_smoke.toml` is aligned to the torch-visible order but defaults to a single-A100 bring-up on `cuda:0`.

## Commands That Already Worked
```bash
which conda
which python
python --version
python -c "import numpy, toml, tqdm"
conda env list
```

## Immediate Next Environment Actions
1. Keep using the `mortal` env for Step 1 smoke work.
2. Keep the default bring-up on `cuda:0`.
3. Run a longer soak on `cuda:0` before introducing `cuda:1`.
