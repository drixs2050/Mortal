# Step 5 Plan: Supervised Objective Design

Date: 2026-03-29

## Purpose
Lock, stabilize, and evaluate the first true supervised human-imitation baseline on top of the expanded Phoenix corpus rather than treating the existing offline RL trainer as the supervised path.

This stage starts once the data program has demonstrated:
- a stable pilot release (`RC0`)
- a larger Batch D scaling checkpoint
- a training-scale intermediate Batch E checkpoint with successful multi-batch merge validation

Step 5 does not require the final full four-year merge to be complete before design and first implementation work begins.
It does require enough corpus scale to avoid designing around a tiny pilot-only slice.
For the current phase, the priority is not broader exploration.
The priority is to make the existing BC path run correctly, measure it correctly, and establish a trustworthy supervised baseline.

## Starting Point
Current repo reality:
- `mortal/train.py` is an offline RL / CQL-style trainer on replay-derived targets.
- `mortal/train_grp.py` is a supervised support model, not the main action-imitation trainer.
- `plans/step_02_sl_objective_decision.md` already established that the first real supervised baseline should be explicit behavior cloning.

Current corpus status relevant to Step 5:
- Step 4 `RC0` is closed and remains the stable pilot reference.
- Step 4B has already produced a training-scale intermediate checkpoint:
  - `9` completed Batch E normalized batches
  - `412303` accepted games across `E01` through `E09`
  - a duplicate-free partial merge over `E01` through `E09`
  - merged train/val/test split files and player-overlap reporting for that partial release

## Current Implementation Checkpoint
Initial Step 5 implementation is now present in the repo:
- `mortal/train_bc.py` provides a separate behavior-cloning trainer entrypoint
- `mortal/eval_bc.py` provides a separate offline held-out evaluator for saved BC checkpoints
- `mortal/dataloader.py` now includes `ActionFileDatasetsIter`, a lighter gameplay iterator that skips GRP/reward loading and yields `obs`, `actions`, and legal `masks`
- `mortal/model.py` now exposes `DQN.action_logits`, so BC can optimize masked human-action imitation losses while still saving checkpoints in the existing `Brain` + `DQN` format
- `configs/step5_bc_debug.toml` and `configs/step5_bc_medium.toml` provide runnable workstation profiles for debug and first-baseline BC work
- `configs/step5_bc_large.toml` now provides the next larger bounded BC run on the same partial Step 4B corpus without overwriting the medium baseline settings
- `configs/step5_bc_medium.toml` now targets the merged partial Step 4B split built from `E01` through `E09`
- Step 5 split generation now filters the game cohort first to tables with at least one `八段+` player and only then performs chronological train/val/test splitting
- the BC dataset path now filters imitation targets to acting players with `八段` and above using the normalized manifest `player_dan` metadata
- Step 5 now supports a precomputed actor-filter index artifact so seat-level eligibility preprocessing can be done offline instead of during every training run
- Step 5 now also supports a precomputed normalized path-cache artifact so repeated runs can skip rebuilding large absolute file lists during startup
- Step 5 also supports an optional precomputed learnable-step summary artifact if we later want exact offline-eval totals, but it is no longer part of the default baseline workflow
- the BC trainer currently reports:
  - held-out negative log-likelihood
  - held-out masked action accuracy
  - held-out top-k action accuracy
  - raw-logit legality rate before masking
  - per-decision-family validation accuracy slices
- the BC trainer now distinguishes:
  - sampled in-training validation for fast feedback
  - deterministic offline `BEST-*` evaluation for checkpoint selection
- the BC trainer now logs throughput, and CUDA runs also log memory statistics
- the first smoke test completed successfully on real Batch D split files with a bounded CPU config
- the first smoke evaluation pass completed successfully on a saved BC checkpoint with the new offline evaluator

## Core Decision To Lock
The project default for the first supervised baseline is:
- explicit behavior cloning on strong human Phoenix actions
- not a relabeling of the existing offline RL trainer as "supervised"
- not a simultaneous exploration of multiple new objectives or architectures

## Main Questions To Resolve
1. Which configs and run sizes should define the debug, medium, and first serious BC runs?
2. How should legal-action masking be validated during training and evaluation?
3. Which held-out metrics are required before we call the baseline usable?
4. What checkpoint-selection rule should define the first official BC baseline?
5. Which evaluation slices are mandatory before we compare later ideas against this baseline?

## Recommended First Baseline
Primary target:
- masked legal-action prediction from human logs

First decision categories to support:
- discard choice
- call / no-call decisions
- riichi / no-riichi decisions
- agari / pass decisions

Optional auxiliary targets for the first pass:
- none by default for the initial stabilization pass

Deferred exploration candidates:
- next-rank prediction
- score or placement-related auxiliary heads
- GRP-informed auxiliary targets
- multi-head policy outputs
- policy-specific heads instead of the shared `DQN` action head
- teacher/search/value distillation targets

## Metrics
Minimum held-out metrics:
- masked action accuracy
- top-k masked action accuracy
- negative log-likelihood
- legality correctness

Useful slice metrics:
- discard-only accuracy
- call decision accuracy
- riichi decision accuracy
- agari decision accuracy

Do not treat training loss alone as progress.

Required run-health checks:
- successful checkpoint save and reload
- successful validation during training
- stable resume behavior from `state_file`
- no legality-mask failures on held-out data
- reproducible run notes for every reported result

## Implementation Direction
Preferred implementation:
- a separate BC entrypoint first

Why:
- cleaner metrics and checkpoint interpretation
- less risk of muddying the existing offline RL trainer
- easier A/B comparison between BC and offline RL on the same corpus
- keeps the saved checkpoint format close enough to the existing `Brain` + `DQN` inference stack that later evaluation integration stays straightforward

## Immediate Step 5 Tasks
1. Add and validate debug and medium BC configs on real Phoenix split files.
2. Standardize the evaluation contract for held-out BC runs:
   - metric names
   - validation cadence
   - checkpoint-selection rule
   - decision-family slices
3. Run the first bounded BC experiments on real merged Phoenix data.
4. Record the first reproducible baseline result with config, corpus slice, and metrics.
5. Confirm the current BC path is trustworthy before changing objectives or model structure.
6. Treat the current medium/large eval cadence as an experimental checkpoint cadence, not the final large-scale recipe; later long runs should reduce validation frequency or validation window size to keep wall-clock cost under control.

## Out Of Scope For This Pass
Do not expand scope yet into:
- multi-head policy redesign
- auxiliary supervision experiments
- new encoder families
- search-distillation or solver-style targets
- broad architecture exploration inspired by Suphx, NAGA, poker systems, or other external systems

Those ideas remain useful, but they should be evaluated only after the current BC baseline is stable and measured cleanly.

## Exit Criteria
- We have a locked BC objective and evaluation contract.
- We have working debug and medium BC runs on real data.
- We have at least one reproducible baseline result with saved config, checkpoint, and held-out metrics.
- We have enough confidence in the current pipeline to treat later objective or architecture exploration as true ablations rather than pipeline debugging.

## Completion Checkpoint
Step 5 exit criteria are now satisfied for the current baseline scope.

Completed outcomes:
- the BC objective is locked as masked human-action imitation on the existing `Brain` + `DQN` stack
- the BC evaluation contract is locked around sampled in-training `VAL`, deterministic `BEST-*` checkpoint promotion, and standalone held-out `eval_bc.py` reports
- the Step 5 data path is now stable on the merged partial Step 4B Phoenix release built from `E01` through `E09`
- the Step 5 data cohort is locked to games with at least one `八段+` player before chronological splitting, with runtime imitation restricted to `八段+` acting seats
- the expensive split-path normalization and actor-seat preprocessing are now moved out of default training startup via `path_cache` and `actor_filter_index` artifacts
- the BC runtime now supports resume-safe checkpoints, TensorBoard logging, WandB logging, throughput/memory reporting, bounded standalone eval defaults, explicit `bfloat16` AMP, TF32, fused AdamW, optional `torch.compile`, and a cosine schedule with dynamic warmup ratio support

Reproducible baseline results now on disk:
- medium bounded baseline:
  - val: accuracy `0.75237`, NLL `0.66867`, top-3 `0.96059`
  - test: accuracy `0.74855`, NLL `0.67767`, top-3 `0.95966`
- large bounded `500`-step baseline:
  - val: accuracy `0.75836`, NLL `0.66959`, top-3 `0.96272`
  - test: accuracy `0.75807`, NLL `0.67292`, top-3 `0.96212`

Current interpretation:
- the BC pipeline is now trustworthy enough to stop treating Step 5 as pipeline bring-up
- the large bounded run slightly improved held-out top-1 and top-3 accuracy over the medium baseline while keeping val and test closely aligned
- the remaining limiting factor is not basic correctness; it is long-run runtime reliability, validation cost, and operational hardening

Handoff to Step 6:
- Step 6 should now focus on hardening the long-run BC training path rather than redesigning the objective
- broader objective or architecture exploration remains intentionally deferred until after Step 6
