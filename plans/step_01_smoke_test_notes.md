# Step 1 Smoke Test Notes

Date: 2026-03-28

Config used:
- `/home/drixs2050/Documents/Mortal/configs/step1_smoke.toml`

Environment used:
- Conda env: `mortal`
- Runtime mode at first pass: CPU-only
- Updated device plan: use only torch-visible `cuda:0` first, keep `cuda:1` for later scale-up, ignore `cuda:2` because it maps to the RTX 3070

## Passed Checks

### 1. `libriichi` build and import
Result:
- `cargo build -p libriichi --lib --release` succeeded
- `mortal/libriichi.so` was copied successfully
- `import libriichi` worked in the `mortal` env

### 2. Bootstrap assets
Result:
- `scripts/step1_bootstrap.py --force` succeeded
- Created:
  - `artifacts/checkpoints/step1_main.pth`
  - `artifacts/checkpoints/step1_best.pth`
  - `artifacts/checkpoints/step1_baseline.pth`
  - `artifacts/checkpoints/step1_grp.pth`
- Created 4 tiny fixture logs under `artifacts/fixtures/logs/`

### 3. Inference entrypoint
Command shape:
- `MORTAL_CFG=... conda run -n mortal python mortal/mortal.py 0`

Result:
- Entry point started and exited cleanly with empty stdin

### 4. Offline dataloader
Result:
- Loaded a real sample from the tiny fixture corpus
- Observed:
  - `file_count = 4`
  - `obs_shape = (1012, 34)`
  - `mask_shape = (46,)`

### 5. `1v3` evaluation entrypoint
Command shape:
- `MORTAL_CFG=... conda run -n mortal python mortal/one_vs_three.py`

Result:
- Completed 4 hanchans
- Challenger rankings: `[1 1 1 1]`
- Logs written under `artifacts/eval_logs/one_vs_three/`
- Re-ran successfully on the single-A100 default profile using `cuda:0`

### 6. Longer `1v3` validation on the real Step 1 GPU path
Command shape:
- `timeout 30s /home/drixs2050/anaconda3/envs/mortal/bin/python -u mortal/one_vs_three.py`

Result:
- Completed cleanly on `cuda:0`
- Challenger rankings: `[1 0 3 0]`
- Aggregate result: `2.5` avg rank, `22.5pt`

## Resolved During This Round

### 1. GRP trainer smoke run
Command shape:
- `timeout 25s ... /home/drixs2050/anaconda3/envs/mortal/bin/python -u mortal/train_grp.py`

Result:
- Now trains successfully on `cuda:0`

Error summary:
- Previous CPU attempt: `RuntimeError: No CUDA GPUs are available`
- Previous single-A100 attempt before the fix: `torch.AcceleratorError: CUDA error: initialization error`

Why it matters:
- The fix was to stop pinning memory inside the collate worker path and to use conservative Step 1 dataloader settings.

### 2. Offline trainer smoke run
Command shape:
- `timeout 25s ... /home/drixs2050/anaconda3/envs/mortal/bin/python -u mortal/train.py`

Result:
- Now logs correctly, enters training, and reaches test-play on `cuda:0`

Why it matters:
- The earlier "silent stall" result was not representative of the actual single-GPU training path once the dataloader settings were made conservative and the env Python was used directly.

## Longer Single-A100 Validation

### 1. GRP trainer longer run
Command shape:
- `timeout 60s /home/drixs2050/anaconda3/envs/mortal/bin/python -u mortal/train_grp.py`

Result:
- Ran continuously on `cuda:0` until the intentional timeout
- Loaded the local Step 1 checkpoint and fixture corpus cleanly
- Progressed from `total steps: 10,480` to `total steps: 17,712`
- Repeated `TRAIN` and `VAL` cycles without crashing or hanging

### 2. Offline trainer longer run
Command shape:
- `timeout 90s /home/drixs2050/anaconda3/envs/mortal/bin/python -u mortal/train.py`

Result:
- Ran continuously on `cuda:0` until the intentional timeout
- Progressed from `total steps: 88` to `total steps: 136`
- Repeatedly entered `one_vs_three` test-play and returned to training
- Observed test-play summaries:
  - `avg rank: 2.25`, `avg pt: 11.25`
  - `avg rank: 2.125`, `avg pt: 28.125`
  - `avg rank: 2.625`, `avg pt: -5.625`
  - `avg rank: 2.25`, `avg pt: 11.25`
  - `avg rank: 1.875`, `avg pt: 39.375`
- Saved a new best checkpoint to `artifacts/checkpoints/step1_best.pth`

### 3. Offline trainer soak run
Command shape:
- `timeout 600s /home/drixs2050/anaconda3/envs/mortal/bin/python -u mortal/train.py > artifacts/tmp/step1_train_soak.log`

Result:
- Ran on `cuda:0` until the intentional timeout
- Progressed from `total steps: 136` to `total steps: 456`
- Completed 39 logged `one_vs_three` evaluation cycles during the soak
- Saved a new best checkpoint twice during the run
- Did not crash or hang before the timeout boundary

Notes:
- The soak log is stored at `artifacts/tmp/step1_train_soak.log`
- The final in-progress eval had already started when the timeout landed, so the 39-cycle count reflects completed logged evals

## Follow-Up
- User confirmed GPU availability from the `mortal` env.
- Out-of-sandbox checks confirm torch-visible `cuda:0` and `cuda:1` are the A100s.
- The default next smoke pass should run on `cuda:0` only before any multi-GPU split is attempted.
- The longer soak run on `cuda:0` completed cleanly, so Step 1 can now hand off to Step 2 without introducing `cuda:1`.

## Artifacts Snapshot
- Checkpoints: `18M`
- Tiny fixture logs: `212K`
- `1v3` eval logs: `216K`
