# Step 6 Plan: Training Pipeline Hardening

Date: 2026-03-29

## Purpose
Turn the now-working Step 5 BC baseline path into a training pipeline that can support longer and more expensive campaigns without constant manual babysitting.

Step 6 is not about changing the supervised objective.
It is about making the existing BC path operationally reliable, measurable, and easier to scale.
That includes checking whether the current model family can be scaled up cleanly before we commit to the first full-scale supervised campaign.

## Starting Point
Current BC status at Step 6 handoff:
- Step 5 is complete for the current baseline scope
- Step 4B is now complete for the current four-year Phoenix scope
- the full merged release `tenhou_phoenix_4y_full_e01_e16_v0` now exists with QA, split, overlap, path-cache, and actor-filter artifacts
- `mortal/train_bc.py` and `mortal/eval_bc.py` both run successfully on real Phoenix corpora, with current reproducible bounded baseline metrics first established on the earlier partial `E01` through `E09` checkpoint
- debug, medium, and large bounded BC configs already exist
- the BC runtime already has:
  - sampled `VAL`
  - deterministic `BEST-*` checkpoint promotion
  - resume-safe checkpoints
  - TensorBoard logging
  - WandB logging
  - throughput and CUDA memory reporting
  - explicit `bfloat16` AMP
  - TF32
  - fused AdamW
  - optional `torch.compile`
  - cosine scheduling with dynamic warmup ratio support

Implementation status as of `2026-03-29`:
- the Step 6 config ladder now exists in `configs/step6_bc_debug.toml` through `configs/step6_bc_full_9dan.toml`
- the full-release `phoenix_hanchan_9dan_actor_any` split is now materialized with split files, overlap report, path cache, and `actor_filter_min18`
- `mortal/train_bc.py` now supports:
  - auto-enabled distributed startup from `WORLD_SIZE`
  - `LOCAL_RANK` CUDA binding
  - DDP wrapping of both `Brain` and `DQN`
  - rank-aware round-robin train-list sharding
  - rank-0-only validation, checkpointing, TensorBoard, and WandB writes
  - explicit `seed` and `grad_accum_steps`
  - config-fingerprint resume validation
  - wall-clock runtime caps via `bc.control.max_runtime_seconds`
- the Step 6 one-command launcher now exists at `scripts/run_bc_campaign.py`
- the launcher validates required data artifacts, runs training, runs final full `val` and `test` eval, and writes a campaign-summary JSON
- the RAM-first loader tuning path now has a dedicated gated runner at `scripts/run_step6_experiment_ladder.py`; by default it stops after Phase A so loader optimization can proceed safely before any longer dual-A100 rehearsal
- unit coverage now includes Step 6 runtime helpers and launcher command assembly
- the checked-in Step 6 `max_steps` values are now usable launch budgets, but they are still corpus-scaled estimates until the exact full `step_counts_v1.json` builds finish for the full `9dan` split

Execution status as of `2026-03-30`:
- the real Step 6 `debug` campaign completed end to end on an A100 and produced final `val`, final `test`, and campaign-summary artifacts
- the real Step 6 `medium` campaign has been launched on the full-release `8dan` cohort on an A100
- the `medium` resume path has now been proven with a deliberate interruption after the first saved checkpoint at step `200`, followed by a successful relaunch from `current_step = 200`
- the `medium` run was later stopped manually after proving stability and resume behavior, and the repo now includes a synthesized interrupted campaign summary plus a checkpoint-derived snapshot report for that run
- `scripts/run_bc_campaign.py` now records an explicit `interrupted` campaign summary for `KeyboardInterrupt` so resume-proof runs leave truthful launcher artifacts behind
- `mortal/train_bc.py` now writes BC checkpoints atomically so save windows do not expose half-written `state_file` payloads to future resume attempts or operator inspection
- the dual-A100 large-run plan now uses no gradient accumulation by default and starts with a manual post-`medium` 8192-per-GPU preflight gate before falling back to 4096 per GPU only if the preflight OOMs
- the repo now includes `configs/step6_bc_large_preflight_full8dan.toml`, `configs/step6_bc_large_bounded_full8dan_8192.toml`, and `scripts/run_step6_large_after_medium.py`, although the live Step 6 closeout flow is being driven manually rather than by the watcher helper
- the manual dual-A100 10-minute preflight at 8192 per GPU completed cleanly, including sampled `VAL` and deterministic `BEST-VAL`, without any OOM
- the real Step 6 `large bounded` DDP campaign is now running on both A100s with the 8192-per-GPU profile and a 2-hour runtime cap

Known remaining weaknesses:
- the gameplay loader is still multiprocessing-fragile, so the checked-in BC profiles run with conservative worker settings
- validation can still consume too much wall-clock on larger profiles if cadence and sample budgets are not tuned carefully
- large held-out sweeps are still bounded by default rather than standardized as a final full-eval policy
- the runtime can be launched and resumed safely, but the project still lacks a fully hardened long-run operating recipe

## Goals
1. Make long BC runs predictable on this workstation.
2. Reduce avoidable wall-clock overhead from evaluation and logging.
3. Standardize what counts as a small, medium, large, and later full-scale run.
4. Make resume/restart behavior fully trustworthy.
5. Capture enough runtime and failure information that future long runs can be debugged quickly.
6. Determine whether a larger version of the current `Brain` + `DQN` family is operationally viable on this hardware.
7. Finish Step 6 with a launch-ready plan for the first full-scale supervised run on both A100s.

## Scope
In scope:
- run-profile cleanup and naming
- validation and checkpoint cadence tuning
- evaluation budgeting and full-eval policy
- resume safety and restart expectations
- runtime observability and reporting
- dataloader stability and known-safe settings
- explicit runbook guidance for longer campaigns
- scaling experiments within the current model family, such as larger `resnet` width/depth while keeping the BC objective unchanged
- preparation for the first dual-A100 supervised launch once the single-GPU path and config policy are stable

Out of scope:
- new objectives
- new encoder families
- multi-head policy redesign
- teacher distillation
- RL algorithm work

## Main Questions To Resolve
1. What are the official debug, medium, large, and later full-scale BC profiles?
2. What validation cadence keeps checkpoint quality acceptable without wasting too much wall-clock?
3. When should we run bounded eval versus full held-out eval?
4. What worker settings are officially considered stable for long runs?
5. What metrics and artifacts must every serious run leave behind?
6. How much larger can the current model family become before runtime cost or instability outweighs the benefit?
7. What is the exact readiness checklist for moving from hardened single-GPU BC runs to the first dual-A100 full-scale supervised launch?

## Recommended Work

### 1. Lock Run Tiers
Define and document the official purposes of:
- debug
- medium
- large bounded
- later full-scale

Each tier should have:
- target corpus
- step budget
- train batch size
- model size
- sampled `VAL` cadence and batch count
- `BEST-*` cadence and batch count
- standalone eval defaults

The run-tier table should make it explicit which profiles are:
- single-GPU stabilization profiles
- larger-model exploration profiles
- dual-A100 launch-candidate profiles

Initial Step 6 run-tier table draft:

| Tier | Proposed config | Primary purpose | Corpus | Imitation target | Model | Train batch | Step budget | Eval contract | Loader policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `debug` | `configs/step6_bc_debug.toml` | fastest real-data trainer sanity check after code/config changes | Batch D `phoenix_hanchan_8dan_actor_any` | `八段+` acting seats (`min_actor_dan = 17`) | `64` channels / `4` blocks | `128` | `40` | sampled `VAL` = `10` batches; deterministic `BEST-VAL` every `20` steps | `num_workers = 0`, `eval_num_workers = 0` |
| `medium` | `configs/step6_bc_medium_full8dan.toml` | single-GPU stabilization on the full merged corpus | full-release `phoenix_hanchan_8dan_actor_any` | `八段+` acting seats (`min_actor_dan = 17`) | `192` channels / `40` blocks | `2048` | `500` to `1000` | sampled `VAL` = `16` to `32` batches; deterministic `BEST-VAL` every `400` steps on `32` to `64` batches | single-process default |
| `large bounded` | `configs/step6_bc_large_bounded_full8dan.toml` | serious throughput, checkpoint, and cadence tuning on the full merged corpus | full-release `phoenix_hanchan_8dan_actor_any` | `八段+` acting seats (`min_actor_dan = 17`) | `192` channels / `40` blocks baseline | `8192` | `1000` to `1500` | sampled `VAL` = `8` to `16` batches; deterministic `BEST-VAL` every `200` to `400` steps on `16` to `32` batches | single-process default |
| `full` | `configs/step6_bc_full_9dan.toml` | Step 7 launch-candidate supervised run | full-release `phoenix_hanchan_9dan_actor_any` | `九段+` acting seats (`min_actor_dan = 18`) | Step 6 baseline winner or scaling-check winner | `8192+` after probe | multi-day launch candidate | sampled `VAL` = `4` to `8` batches at low frequency; deterministic `BEST-VAL` every `500` to `1000` steps on `16` to `32` batches; require full held-out sweeps before final promotion | start single-process; relax only after proof |

Run-tier policy:
- Keep the checked-in Step 5 configs unchanged on the partial `E01` through `E09` release so the recorded baseline metrics remain reproducible.
- Use the full merged `E01` through `E16` release as the default Step 6 data foundation for all new configs.
- Keep Step 6 `debug`, `medium`, and `large bounded` on the current `八段+ any-player` game cohort so scale-up comparisons stay anchored to the established Step 5 baseline path.
- Treat the first true full-scale launch-candidate run as a stricter `九段+ any-player` game cohort with `min_actor_dan = 18`.
- Keep the operational runtime cap for `medium` at `12` hours, but cap `large bounded` and the large-model probes at `2` hours for Step 6 closeout instead of extending the large tier to `24` hours.
- Keep `grad_accum_steps = 1` for the checked-in DDP configs.
- After `medium` completes, run a short 8192-per-GPU DDP preflight on both A100s.
- If that preflight finishes without OOM, use the 8192-per-GPU 2-hour large-bounded config.
- If that preflight OOMs, fall back to the 4096-per-GPU 2-hour large-bounded config.
- A Step 6 size check on the full merged release shows that this stricter `九段+ any-player` cohort still contains `256815` games, split as `205452` train / `25681` val / `25682` test, so the stricter launch-candidate policy is now practical.
- Materialize a permanent `phoenix_hanchan_9dan_actor_any` split family plus `path_cache_v1` and `actor_filter_min18` before the first full-scale launch.
- Keep larger-model scaling probes as sidecar bounded configs rather than separate promotion tiers so the main run ladder stays simple and comparable.

### 2. Harden Validation Policy
Treat validation as an explicit runtime budget, not as an afterthought.

Need to decide:
- sampled `VAL` frequency for large and later full runs
- `BEST-*` frequency for large and later full runs
- when larger eval batch sizes help and when they become unstable
- when to require full held-out sweeps instead of bounded evals

### 3. Harden Resume And Checkpoint Policy
Make sure every serious run has a clearly defined:
- primary `state_file`
- promoted `best_state_file`
- expected resume behavior
- expected behavior when config changes require a fresh run instead of resume

### 4. Standardize Runtime Observability
Every serious run should capture:
- train throughput
- eval throughput
- CUDA memory behavior
- current LR
- best checkpoint metrics
- clear run notes and config identity

### 5. Dataloader Stability
The current BC runtime is stable in single-process mode on the merged Phoenix corpus.
This should be treated as the default known-safe operating mode until a separate focused effort proves that worker-process loading is reliable enough to re-enable for long runs.

### 6. Step 4B / Step 6 Coordination
Step 4B is no longer a blocker.
The full release now exists, so Step 6 should treat `tenhou_phoenix_4y_full_e01_e16_v0` as the default data foundation and define how the current bounded BC recipe should be retuned for the larger corpus and longer run lengths.
For the current phase:
- use `phoenix_hanchan_8dan_actor_any` for debug, medium, and large bounded stability work
- use a stricter `phoenix_hanchan_9dan_actor_any` plus `min_actor_dan = 18` policy for the first full-scale launch candidate

### 7. Current-Model Scaling Checks
Step 6 should include bounded experiments that answer:
- can the current `Brain` + `DQN` family be scaled up without destabilizing training?
- what are the throughput, memory, and validation-cost tradeoffs of a larger model on one A100?
- is the larger current-model variant worth carrying forward into the first full-scale supervised launch?

These checks should stay within the current model family.
This is still pipeline hardening, not architecture exploration in the broader Step 5 sense.

### 8. Dual-A100 Launch Readiness
The end of Step 6 should produce a concrete launch recipe for the first full-scale supervised campaign on both A100s.

That recipe should include:
- the chosen model size
- the chosen single-GPU or dual-GPU training strategy
- the official full-scale train/eval cadence
- expected checkpoint cadence and restart behavior
- expected runtime and storage footprint
- the exact command/config pair to launch the first serious full-scale supervised campaign

The actual long campaign is best treated as the Step 6 to Step 7 handoff:
- Step 6 proves the path is ready
- Step 7 uses that hardened path to run the serious supervised campaign and compare baselines

## Exit Criteria
- We have clearly documented BC run tiers with known-safe settings.
- We have a written validation and checkpoint policy for larger runs.
- We have a known-safe resume/restart workflow for serious runs.
- We have enough runtime instrumentation and runbook guidance to launch longer BC campaigns confidently.
- We have a clear answer on whether a larger version of the current model family should be used for the first full-scale supervised campaign.
- We have a launch-ready dual-A100 full-scale supervised recipe, even if the multi-day campaign itself starts at the Step 6 to Step 7 handoff.
- We are ready to enter Step 7 with a hardened training path instead of continuing to debug the pipeline itself.
