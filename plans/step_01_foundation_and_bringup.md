# Step 1 Plan: Foundation And Baseline Bring-Up

## Why This Is First
Before collecting a huge corpus or launching expensive training, we need to prove that this repo can be run, measured, and debugged cleanly on your workstation. Right now the codebase has important building blocks, but the operational layer is still missing.

Assumption for this step:
- Do not assume any public pretrained checkpoints are available. Step 1 must be able to bootstrap from local builds and locally created smoke-test assets.

Core language/tooling stack for this repo:
- Python is required for training, inference, configs, and orchestration.
- Rust is required to build `libriichi`, which is the fast mahjong core exposed to Python.
- No additional programming language toolchain is required for the core training path.
- Docker and mdBook are optional and not required for Step 1.

## Step 1 Objectives
1. Make the repo reproducibly runnable on this machine.
2. Replace placeholder config paths with local workstation profiles.
3. Verify the current end-to-end capabilities that already exist.
4. Produce a short gap report that tells us what to build next.
5. Keep the initial bring-up on one GPU before introducing parallel or multi-GPU complexity.

## Scope
Included in Step 1:
- Environment setup
- Build and smoke tests
- Local config creation
- Baseline command validation
- Artifact directory layout
- Current-gap audit

Not included in Step 1:
- Large-scale data acquisition
- Full supervised training campaign
- RL algorithm changes
- Tenhou automation

## Concrete Tasks

### 1. Create A Stable Project Layout
Create or standardize repo-local paths for:
- `artifacts/checkpoints/`
- `artifacts/tensorboard/`
- `artifacts/eval_logs/`
- `artifacts/train_play_logs/`
- `artifacts/test_play_logs/`
- `artifacts/online_buffer/`
- `artifacts/online_drain/`
- `artifacts/tmp/`
- `artifacts/fixtures/`
- `data/raw/`
- `data/normalized/`
- `data/manifests/`
- `configs/`
- `plans/`

Deliverable:
- A documented directory convention so every later script knows where to read and write.

### 2. Build The Workstation Environment
Install and verify:
- Conda Python environment from `environment.yml`
- Missing runtime Python packages actually used by the repo, especially `numpy`
- PyTorch with CUDA support for the installed GPUs
- Rust toolchain via `rustup`, including `rustc` and `cargo`
- System build tools and linker needed for the Rust/PyO3 extension build
- `libriichi` build output copied into `mortal/`

Validation:
- `import libriichi` works from the project environment.
- `python mortal.py <ID>` can start with a real config and checkpoint path.
- `python -c "import torch, numpy, toml, tqdm"` works in the same environment.

Deliverable:
- A short environment setup note with exact commands that worked on this workstation and a list of required toolchains/packages.

### 3. Create Real Local Config Profiles
Create actual config files for this machine, derived from `mortal/config.example.toml`, for at least:
- GRP training sanity checks
- Offline training sanity checks
- Offline evaluation / self-play checks
- Future online training placeholders

Important decisions to lock now:
- Where checkpoints live
- Which single GPU is the default bring-up card
- Which jobs remain single-GPU until the pipeline is stable
- Where test-play and train-play logs go
- Where dataset indexes and manifests live

Deliverable:
- One or more working config files under `configs/`.

### 4. Prepare Bootstrap Smoke-Test Assets
Decide how the required inputs for Step 1 validation will be satisfied:
- A main model checkpoint for `control.state_file`
- Baseline checkpoints for `baseline.train.state_file` and `baseline.test.state_file`
- A GRP checkpoint for `grp.state_file`
- A tiny gz-log fixture for dataloader tests

Allowed approaches:
- Generate minimal bootstrap checkpoints from randomly initialized networks only to validate load paths
- Produce the tiny log fixture from a very small self-play run once the baseline checkpoint path is working
- If private/internal checkpoints later become available, treat them as optional conveniences rather than a dependency of Step 1

Deliverable:
- A documented bootstrap strategy that is runnable even with zero external checkpoints.

### 5. Run Minimal Validation Commands
Run small, cheap checks instead of full training:
- Rust test/build check
- Python import check
- GRP model load or tiny-train sanity check
- Offline data loader sanity check on the tiny fixture from Task 4
- `OneVsThree` evaluation sanity check
- Single checkpoint save/load sanity check

Deliverable:
- A short pass/fail list of what currently works.

### 6. Audit The Existing Training Path
Inspect and document:
- Whether offline training runs with a tiny dataset
- Whether GRP is a hard prerequisite for all offline training
- What single-GPU issues must be solved before multi-GPU support is worth attempting
- What the current online trainer-server-worker path can already do
- The known online hang in `mortal/train.py`

Deliverable:
- A repo-local gap report listing blockers, risks, and likely fixes.

### 7. Define Baseline Metrics For Future Work
Record the first minimal metrics we will use in Step 2+:
- Build success
- Import success
- Self-play eval command success
- Samples/sec or batches/sec for a smoke run
- Replay generation throughput for tiny self-play
- Disk footprint of logs/checkpoints

Deliverable:
- A small baseline metrics table that future changes can compare against.

### 8. Prepare The Step 2 Handoff
At the end of Step 1, write down:
- Which data formats the repo already accepts
- Which raw sources still need converters
- Whether current configs and scripts are good enough to scale
- What must be fixed before data collection and SL training start

Deliverable:
- A clear Step 2 backlog focused on data strategy and ingestion.

## Suggested Execution Order
1. Create the directory layout.
2. Build the environment and `libriichi`.
3. Create the configs.
4. Prepare bootstrap checkpoints and the tiny fixture.
5. Run smoke tests.
6. Audit the training and self-play paths.
7. Record findings and Step 2 blockers.

## Exit Criteria
Step 1 is complete when all of the following are true:
- We have working local config files instead of placeholder paths.
- We have a concrete way to satisfy the checkpoint and fixture dependencies used by the smoke tests without relying on public weights.
- We can build and import the Rust/Python stack on this workstation.
- We can run at least one cheap end-to-end sanity check through the current training/eval stack.
- We have a written list of current blockers, especially around data and online RL.
- We know exactly what Step 2 needs to produce.

## Expected Outputs From Step 1
- Working local config files
- Bootstrap checkpoint and tiny-fixture notes
- Environment setup notes
- Smoke-test notes
- Gap report
- Baseline metrics snapshot

## What Step 2 Will Likely Be
Step 2 should be "Data strategy and ingestion design": choose sources, define usage rules, define canonical schema, and build the raw-to-normalized pipeline that the existing `GameplayLoader` and training stack can consume.
