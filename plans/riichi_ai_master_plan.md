# Riichi AI Master Plan

## Goal
Build a strong riichi mahjong AI on top of this `Mortal` codebase, first with supervised learning from strong human games and then with self-play reinforcement learning, with a later optional integration layer for platform play/testing.

## Program Status
As of `2026-03-30`:
- Step 1 is complete.
- Step 2 is complete.
- Step 3 exit criteria are satisfied for the current four-player Phoenix hanchan pilot slice.
- Step 4 is now formally closed for the current Batch C `RC0` pilot release candidate.
- Step 4B is now formally closed for the current four-year Phoenix corpus target.
- Batch D remains the first January-scale Step 4B checkpoint and the recent-first multi-batch collection automation is now proven.
- The full Batch E recent-first Phoenix sweep is now complete for the current scope:
  - `16` normalized recent-first batches completed across `E01` through `E16`
  - `694346` merged rows across the full release `tenhou_phoenix_4y_full_e01_e16_v0`
  - `0` duplicate rows at merge time
  - merged QA, split, and player-overlap artifacts now exist for the full release
  - the current Step 6 BC path also has full-corpus `path_cache_v1` and `actor_filter_min17` helper artifacts
- Step 5 is now complete for the current baseline scope:
  - a separate `mortal/train_bc.py` behavior-cloning trainer exists and is running successfully on real Phoenix data
  - a separate `mortal/eval_bc.py` offline evaluator exists and is producing reproducible held-out BC reports
  - workstation-ready `configs/step5_bc_debug.toml`, `configs/step5_bc_medium.toml`, and `configs/step5_bc_large.toml` profiles now exist
  - the BC path reuses the existing `Brain` + `DQN` stack but trains explicit masked human-action imitation losses instead of offline RL targets
  - the current BC path imitates only acting players with `八段` and above (`min_actor_dan = 17`) using manifest-backed actor filters
  - the Step 5 data path now uses a prefiltered `八段+ any-player` split, a normalized path cache, and a precomputed actor-filter index so expensive preprocessing is moved out of the training loop
  - the BC pipeline now has resume-safe checkpoints, sampled `VAL`, deterministic `BEST-*` checkpoint promotion, standalone held-out eval, TensorBoard logging, and WandB logging
  - the checked-in CUDA Step 5 profiles now use explicit `bfloat16` AMP, TF32, fused AdamW, and optional `torch.compile`
  - the current large bounded baseline also uses a cosine schedule with dynamic warmup ratio support
  - a real debug smoke run completed successfully on Batch D
  - a real medium baseline completed successfully on the merged partial Step 4B release
  - a larger bounded `500`-step BC run completed successfully on the same merged partial release
  - the current large bounded held-out reports are:
    - val: accuracy `0.75836`, NLL `0.66959`, top-3 `0.96272`
    - test: accuracy `0.75807`, NLL `0.67292`, top-3 `0.96212`
- The next active focuses are:
  - Step 6 training-pipeline hardening on top of the now-working BC baseline and the completed full Step 4B corpus release
  - retuning the bounded BC recipe for the full `E01` through `E16` release, including validation cadence, dataloader policy, and run-tier definitions
  - using the larger corpus to tighten the eventual full-scale launch-candidate imitation target to `九段+` acting seats
  - current-model scaling checks and dual-A100 launch readiness for the first full-scale supervised campaign
  - defer broader objective and architecture exploration until the hardened BC pipeline is ready for longer campaigns
- The current Step 6 implementation work now includes:
  - a checked-in Step 6 config ladder for debug, medium, large bounded, model probes, and the full `9dan` launch candidate
  - a one-command campaign runner at `scripts/run_bc_campaign.py`
  - a gated Step 6 experiment runner at `scripts/run_step6_experiment_ladder.py` for subset-based RAM-first loader tuning before the longer large-run and model-probe phases
  - DDP-aware BC training support in `mortal/train_bc.py`
  - a permanent full-release `phoenix_hanchan_9dan_actor_any` split family with `path_cache_v1`, `actor_filter_min18`, and overlap artifacts
  - a real Step 6 `debug` campaign completed end to end on an A100 with final `val`, final `test`, and campaign-summary artifacts
  - a real Step 6 `medium` campaign is now underway on the full `8dan` corpus, and its resume path has already been proven via a deliberate interrupt/relaunch from the first saved checkpoint at step `200`
  - Step 6 launcher interruption summaries and BC checkpoint writes have now been hardened further based on real run behavior
  - the manual-stop `medium` handoff now has a synthesized interrupted campaign summary and checkpoint-derived snapshot report
  - the post-`medium` large-run policy now uses a short manual 8192-per-GPU DDP preflight and falls back to 4096 per GPU only if that gate OOMs
  - that 8192-per-GPU dual-A100 preflight has now completed cleanly, and the live large-bounded DDP proof run is capped at 2 hours rather than 12
- Current ingestion and dataset support covers a validated four-player Tenhou conversion slice with manifest output, behavior-based regression tests, an XML-only converter path, live progress reporting, a one-command pipeline runner, a recent-first multi-batch series driver, merged-release tooling, the `36`-game Batch C `RC0` cohort, the `372`-game Batch D scale-up checkpoint, and the full `E01` through `E16` merged four-year Phoenix release with QA, split outputs, overlap reporting, and BC helper artifacts.
- That `RC0` cohort is suitable for pipeline validation and early objective design, not for serious supervised-training-scale conclusions.

## Current Repo Snapshot
What already exists in this repo:
- `libriichi` provides the game rules, state tracking, dataset extraction, mjai interface, statistics, and self-play arena.
- `mortal/train.py` already trains the main model from log-derived gameplay data.
- `mortal/train_grp.py` already trains the GRP model used by the reward calculator.
- `mortal/client.py` and `mortal/server.py` sketch an online training loop with replay submission.
- `mortal/player.py` already supports `OneVsThree` evaluation against a baseline agent.

What is still missing or not production-ready:
- A fully standardized end-to-end config story outside the current Step 5 and Step 6 BC profiles. The original `mortal/config.example.toml` remains placeholder-oriented.
- A reproducible workstation setup for training, testing, logging, and artifact storage.
- A checkpoint bootstrap path that does not rely on externally available public weights.
- A documented corpus-refresh policy beyond the current `2022-03-29` through `2026-03-18` Phoenix release, plus any stricter future evaluation split families.
- A clear supervised learning curriculum and evaluation suite.
- A robust large-scale RL pipeline. The current online flow is useful, but still looks experimental.
- A fully validated dual-A100 operating record for the new Step 6 DDP BC path. The support now exists in the repo, but the real `large bounded`, probe, and `full 9dan` campaigns are still pending.
- A mature experiment registry, model registry, and runbook.
- Platform integration and safety controls for later Tenhou-side testing.

## Guiding Principles
1. Make the system reproducible before making it bigger.
2. Turn every major stage into a measurable pipeline with artifacts, metrics, and exit criteria.
3. Keep platform automation separate from core AI training so the model can be developed and evaluated offline first.
4. Prefer small validation runs before long expensive runs.
5. Do not trust data or training improvements until they survive holdout and self-play evaluation.
6. Stabilize the full pipeline on one GPU before introducing multi-GPU or distributed complexity.

## Step-By-Step Program Plan

### Step 1. Foundation And Baseline Bring-Up
Purpose:
Make the current repo runnable end-to-end on this workstation in a reproducible way.

Work:
- Build the Python and Rust environment.
- Build and import `libriichi`.
- Create real local config profiles and artifact directories.
- Prepare the minimal checkpoints and tiny fixtures needed for smoke tests without assuming public weights exist.
- Run smoke tests for inference, dataset loading, GRP, offline training, and self-play evaluation.
- Document current breakages and bottlenecks.

Exit criteria:
- We can run the core repo commands locally with known configs.
- We have a written baseline status report and a stable directory layout.

### Step 2. Data Strategy And Governance
Purpose:
Decide exactly what human data we will use and how we are allowed to use it.

Work:
- Pick target sources of high-level human games.
- Define legal, terms-of-service, privacy, and retention rules for each source.
- Define what metadata we need: source, date, room/rank, player tags, game quality flags.
- Confirm that target-source elite-game metadata, such as Tenhou lobby/dan/rate fields, can be preserved through manifests before large-scale conversion starts.
- Define storage budget and retention tiers for raw logs, normalized logs, and derived tensors.

Exit criteria:
- We have a written source policy, a canonical raw data schema, and a clear manifest path for elite-game filtering metadata.

### Step 3. Data Ingestion And Canonicalization
Purpose:
Turn raw game records into a clean canonical dataset that the repo can consume.

Work:
- Implement or adapt parsers/converters into the repo's expected log format.
- Keep the active ingestion scope on four-player hanchan data until the main pipeline is stable; defer sanma-only branches unless they become necessary for the target corpus.
- Add validation for corrupt, incomplete, or duplicated games.
- Add deterministic normalization and compression rules.
- Generate manifests for every shard with counts, checksums, date ranges, and source tags.

Exit criteria:
- We can ingest raw data into a versioned normalized corpus without manual cleanup.

### Step 4. Dataset QA, Splits, And Sampling Policy
Purpose:
Make the dataset trustworthy for training and evaluation.

Work:
- Create train/validation/test splits by time and by player leakage policy.
- Build QA reports: game counts, player counts, action frequencies, rank distributions, round lengths, malformed rate.
- Decide filtering rules for weak rooms, low-quality logs, short games, and edge-case rule sets.
- Define sampling weights across sources, years, and player strength tiers.
- Generate manifest-driven curated file lists for target cohorts such as Tenhou high-dan / Phoenix-room subsets.

Exit criteria:
- We have a versioned dataset release with QA reports and stable split files.

### Step 4B. Corpus Expansion And Dataset Scaling
Purpose:
Scale from the pilot release candidate into a real supervised-training corpus.

Work:
- Expand official Tenhou Phoenix/high-dan four-player hanchan collection over a much broader date range.
- Reuse the existing ingestion, manifest, QA, and split pipeline instead of inventing a second path.
- Produce larger versioned dataset releases with the same cohort boundaries and leakage reporting.
- Track collection coverage by date, ruleset, room, dan tier, and failure categories.
- Decide when the corpus is large enough to support serious supervised runs instead of debug-only experiments.

Exit criteria:
- We have a training-scale elite corpus, not just a pilot slice, with the same QA and split guarantees as `RC0`.

### Step 5. Supervised Learning Objective Design
Purpose:
Lock and validate the first strong human-imitation baseline before exploring richer objectives or new model families.

Work:
- Keep the current explicit BC path as the default supervised baseline until it has been trained and evaluated cleanly.
- Add workstation-ready debug and medium configs for the BC trainer.
- Verify that training, validation, checkpointing, and resume behavior all work correctly on real Phoenix splits.
- Define and standardize the held-out metric suite beyond loss, such as action-match accuracy on key decision categories.
- Run the first bounded supervised baselines and record reproducible results.
- Defer alternative objectives, auxiliary heads, and architecture exploration until the existing baseline is trustworthy.

Exit criteria:
- We have a working and reproducibly evaluated BC baseline, with documented configs, metrics, and run notes.

### Step 6. Training Pipeline Hardening
Purpose:
Make long training runs reliable on this machine.

Work:
- Add workstation-ready configs for small, medium, and full runs.
- Explore whether a larger version of the current model family is operationally worthwhile before the first full-scale supervised campaign.
- Add resume safety, periodic validation, and structured logging.
- Add throughput measurement for data loading, training, and self-play.
- Fix or isolate known issues, including the online-training hang called out in `mortal/train.py`.
- Keep multi-GPU support out of the default path until the single-GPU baseline is stable, then define the launch recipe for the first dual-A100 full-scale supervised run.

Exit criteria:
- We can launch repeatable training runs with known runtime behavior and reliable checkpoints, and we have a launch-ready recipe for the first full-scale supervised campaign on both A100s.

### Step 7. Baseline Supervised Runs
Purpose:
Produce the first serious model family trained from high-level human data.

Work:
- Run small debug runs, then medium ablations, then full training.
- First compare optimizer settings, sampling rules, and data/eval choices on the locked BC baseline.
- Only after the baseline is stable, explore architecture or objective changes such as multi-head policies or auxiliary targets.
- Track checkpoints, tensorboard logs, evaluation summaries, and selected best models.
- Freeze a supervised baseline that clearly beats the current starting point.

Exit criteria:
- We have a chosen SL baseline model with supporting metrics and reproducible run notes.

### Step 8. Evaluation Harness And Regression Suite
Purpose:
Prevent fake progress.

Work:
- Build a repeatable offline eval command for held-out logs.
- Build a self-play ladder using existing arena support.
- Standardize evaluation seeds and sample sizes for rank/pt confidence intervals.
- Add regression tests for inference correctness, action legality, and log parsing.

Exit criteria:
- Every candidate model can be evaluated by the same harness and compared historically.

### Step 9. Self-Play RL Environment Hardening
Purpose:
Turn the current online/self-play pieces into a stable RL system.

Work:
- Decide whether to keep the current trainer-server-worker design or replace parts of it.
- Separate replay generation, replay storage, training, and evaluation concerns.
- Add replay manifests, retention policy, and sample accounting.
- Support robust process restarts, partial failures, and monitoring.
- Move from pure challenger-vs-baseline play toward a better opponent pool or broader self-play regime when ready.

Exit criteria:
- Self-play data generation and RL training can run for long periods without manual babysitting.

### Step 10. RL Algorithm Development
Purpose:
Improve beyond supervised imitation.

Work:
- Start from the best SL checkpoint.
- Decide the first RL objective: refine current value-based approach, add policy improvement, or introduce a different online RL recipe.
- Tune reward design, opponent mixing, exploration schedule, and checkpoint selection rules.
- Run ablations to confirm RL is actually helping held-out and self-play metrics.

Exit criteria:
- We have at least one RL-finetuned model that is measurably better than the SL baseline.

### Step 11. Large-Scale Training Operations
Purpose:
Use the workstation efficiently and prepare for longer campaigns.

Work:
- First validate long-run reliability on one A100, then scale specific roles onto the second A100.
- Decide CPU worker counts and replay generation concurrency for the 3990X.
- Add process orchestration, health checks, log rotation, and storage cleanup.
- Add model registry conventions and checkpoint promotion rules.

Exit criteria:
- Long multi-day training/eval jobs can be run and recovered cleanly.

### Step 12. Platform Integration Layer
Purpose:
Create a separate bridge between the trained AI and external play/testing environments.

Work:
- Keep the core model behind a stable inference API.
- Build a local harness that feeds real platform events into the mjai-compatible agent loop.
- Add latency measurement, action timeout handling, reconnect handling, and full raw-event logging.
- Add manual-review and simulation modes before any live automation.
- Review platform rules and operational risk before enabling real unattended use.

Exit criteria:
- The platform adapter can be tested safely without entangling training code with site-specific automation.

### Step 13. Documentation, Runbooks, And Project Hygiene
Purpose:
Make future work faster instead of re-discovering everything each month.

Work:
- Write setup docs, runbooks, and experiment templates.
- Maintain a known-issues list and backlog.
- Record dataset versions, model versions, and evaluation methodology.
- Keep plans updated after each completed stage.

Exit criteria:
- Another session can pick up the project without reverse-engineering the whole repo.

## Recommended Execution Order Right Now
1. Treat the full Step 4B release as the stable dataset foundation for the next supervised campaign.
2. Use the current Step 5 BC baseline as the stable starting point.
3. Execute Step 6 to harden long-run training, evaluation cadence, dataloader policy, and runtime reliability before larger campaigns.
4. Enter Step 7 only after the hardened Step 6 run profiles and launch recipe are in place.

Longer-range order:
1. Do Step 2 and Step 3 before any serious source broadening or new corpus family.
2. Do Step 4 through Step 8 to establish a reliable supervised baseline.
3. Only then scale into Step 9 through Step 11 for RL.
4. Leave Step 12 until the model and evaluation stack are already strong and stable.

## Immediate Known Risks
- The repo currently depends on placeholder config paths, so nothing large-scale is ready out of the box.
- `environment.yml` is minimal and does not fully define the training environment.
- The current online training path documents a known hang in `mortal/train.py`.
- Current evaluation is baseline-centric and does not yet look like a full research benchmark suite.
- The current online play docs are sparse, and there is no separate Tenhou automation/testing layer in the repo yet.
