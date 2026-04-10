# Step 1 Baseline Metrics

Date: 2026-03-28

## Environment Probe Snapshot

| Metric | Status | Value |
| --- | --- | --- |
| Conda available | Pass | `/home/drixs2050/anaconda3/bin/conda` |
| Python available | Pass | `/home/drixs2050/anaconda3/bin/python` |
| Python version | Pass | `3.12.4` |
| Dedicated `mortal` env | Pass | `/home/drixs2050/anaconda3/envs/mortal` |
| `numpy` import | Pass | imports in current Python and in the `mortal` env |
| `toml` import | Pass | imports in current Python and in the `mortal` env |
| `tqdm` import | Pass | imports in current Python and in the `mortal` env |
| `torch` import | Pass | `2.11.0+cu130` in the `mortal` env |
| Rust toolchain | Pass | `cargo 1.94.1`, `rustc 1.94.1` |
| GPU runtime probe | Pass | torch sees `cuda:0` and `cuda:1` as A100s, `cuda:2` as RTX 3070 |

## Runtime Metrics To Fill After Environment Bring-Up

| Metric | Status | Value |
| --- | --- | --- |
| `libriichi` import | Pass | `0.1.0`, release profile |
| Bootstrap checkpoint generation | Pass | `step1_main`, `step1_best`, `step1_baseline`, `step1_grp` created |
| Tiny fixture generation | Pass | 4 gz logs created under `artifacts/fixtures/logs/` |
| Offline dataloader smoke check | Pass | first sample loaded successfully |
| `mortal.py` startup smoke | Pass | loads config and checkpoint, exits cleanly with empty stdin |
| `OneVsThree` smoke check | Pass | 4 hanchans completed on the single-A100 profile, challenger rankings `[1 1 1 1]` |
| `OneVsThree` longer validation | Pass | 30s on `cuda:0`, challenger rankings `[1 0 3 0]`, avg rank `2.5`, avg pt `22.5` |
| Offline trainer smoke run | Pass | enters training and test-play on `cuda:0` during the timeout window |
| Offline trainer longer validation | Pass | 90s on `cuda:0`, progressed `88 -> 136`, repeated train/eval/test-play cycles, saved new `step1_best.pth` |
| Offline trainer soak run | Pass | 10m on `cuda:0`, progressed `136 -> 456`, completed 39 logged eval cycles, saved new best twice, timeout only |
| GRP trainer smoke run | Pass | trains on `cuda:0` after removing worker-side pinning and using conservative dataloading |
| GRP trainer longer validation | Pass | 60s on `cuda:0`, progressed `10,480 -> 17,712`, repeated `TRAIN` and `VAL` cycles until timeout |
| Checkpoint disk footprint | Pass | `18M` |
| Tiny fixture disk footprint | Pass | `212K` |

## Bring-Up Policy

| Policy | Status | Value |
| --- | --- | --- |
| Default training/eval card | Pass | `cuda:0` |
| Second A100 usage | Pass | hold back until single-GPU path is stable |
| RTX 3070 usage | Pass | ignore for training/eval |
