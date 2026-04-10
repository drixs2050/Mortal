# Step 4B Plan: Corpus Expansion And Dataset Scaling

Date: 2026-03-28

## Status Update
As of `2026-03-29`, Step 4B exit criteria are satisfied for the current four-year Phoenix yonma corpus target.

The formal closeout artifact is:
- `plans/step_04b_full_four_year_release.md`

Interpretation:
- Batch D remains the first successful Step 4B scale-up checkpoint
- the full merged release now exists as `tenhou_phoenix_4y_full_e01_e16_v0`
- Step 4B is closed for the current four-year Phoenix scope
- Step 6 is now the active stage

## Purpose
Scale beyond the current `RC0` pilot so the project has a real supervised-learning corpus rather than only a pipeline-validation checkpoint.

`RC0` proved that:
- the XML-only Tenhou ingestion path works
- the cohort definition is sane
- the split and QA machinery works

What `RC0` did not prove:
- that we have enough data for serious supervised learning
- that the current tiny three-day slice is representative
- that held-out metrics on this small split mean much

## Why This Needs To Be Its Own Stage
The current roadmap already has:
- Step 4 for pilot release and QA policy
- Step 5 for supervised-objective design

But there is still a missing practical stage between "pilot release candidate exists" and "serious supervised training can start":
- large-scale corpus expansion with the same ingestion and QA guarantees

This stage makes that work explicit so we do not accidentally treat `RC0` as a real training dataset.

## Starting Point
Current best pilot release:
- `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.jsonl`

Current cohort definition:
- source: Tenhou
- room: `鳳`
- ruleset: `鳳南喰赤`
- go type: `169`
- four-player only
- all players `>= 七段`

Current scale:
- `36` games
- `391` kyoku
- `38421` normalized events

## Objectives
1. Build the first training-scale Tenhou Phoenix/high-dan yonma hanchan corpus.
2. Keep the exact same cohort definition and QA standards unless we explicitly revise them.
3. Make larger releases routine and reproducible, not one-off manual snapshots.

## Workstreams

### Workstream A. Collection Sweep
Goal:
Stage a much broader set of official Phoenix archive references.

Work:
- select replay ids across many archive days, not just three days
- keep the cohort exact: `四鳳南喰赤－` / `鳳南喰赤` / `go_type = 169`
- spread collection across time so the dataset is not clustered into a tiny window

Suggested first target:
- at least thousands of games
- enough date spread to make train/val/test chronological splits meaningful

### Workstream B. Batch Ingestion At Scale
Goal:
Run the existing XML-only ingestion path on much larger snapshots.

Work:
- reuse:
  - `scripts/fetch_tenhou_scc_archives.py`
  - `scripts/select_tenhou_scc_refs.py`
  - `scripts/stage_tenhou_reference_batch.py`
  - `scripts/ingest_tenhou_snapshot.py`
  - `scripts/summarize_normalized_manifest.py`
  - `scripts/build_dataset_splits.py`
  - `scripts/summarize_split_overlap.py`
- record failure categories and watch for new unsupported branches
- keep Batch B available for converter spot checks if regressions appear

### Workstream E. Multi-Batch Automation
Goal:
Run large recent-first Phoenix collection sweeps without hand-editing date windows for every batch.

Work:
- use `scripts/run_tenhou_pipeline_series.sh` to launch repeated recent-to-old windows
- keep each batch bounded, for example `182` days and `100000` games, rather than one monolithic multi-year run
- keep archive publication lag in mind when choosing the newest batch end date; do not assume the current calendar date is already available in Tenhou's daily archive dump
- use concurrent archive fetch and replay staging settings that are conservative enough for long runs but much faster than single-threaded staging
- prefer `--stop-after ingest` on the first pass, then inspect summaries before QA/splits
- use `--min-date` when the corpus target is a fixed historical cutoff rather than an open-ended number of batches

### Workstream F. Post-Series Consolidation
Goal:
Turn successful per-batch ingests into one training-scale dataset release instead of leaving the corpus fragmented across batch ids.

Work:
- use `scripts/merge_normalized_manifests.py` to combine successful normalized manifests into a merged release manifest while preserving per-batch provenance
- validate that adjacent recent-first windows did not introduce duplicate `source_game_id` rows or date-window overlap mistakes
- generate QA, split, and player-overlap artifacts on the merged corpus, not just on individual batches
- define the naming/versioning policy for the first merged multi-batch Phoenix release
- keep per-batch manifests as regression and recovery checkpoints even after the merged release exists

### Workstream G. Intermediate Training-Scale Checkpoints
Goal:
Use partial multi-batch merged checkpoints to unblock supervised-objective work before the entire four-year sweep is finished.

Work:
- treat a duplicate-free partial merged release as sufficient evidence that the merge path works at training scale
- use those intermediate checkpoints for Step 5 design, trainer scaffolding, and bounded debug runs
- keep Step 4B formally open until the intended full multi-batch release and its QA artifacts are finished

### Workstream C. Release Scaling
Goal:
Turn larger ingested batches into dataset versions, not ad hoc folders.

Work:
- keep release notes for each scaled cohort
- compare each release against `RC0`
- monitor:
  - game counts
  - date coverage
  - ruleset purity
  - player overlap
  - failure rates

### Workstream D. Training Readiness Threshold
Goal:
Define when the corpus is big enough to stop being "pilot only."

Questions to answer:
- what minimum game count is required before serious SL runs start?
- what minimum date span is required for meaningful chronological holdout?
- do we need stricter player-disjoint eval splits before first medium/full supervised runs?

Practical guidance:
- do not use `RC0` alone for serious training claims
- use `RC0` for debug and objective design only
- do not begin Step 7-style serious supervised runs until this stage produces a much larger release

## Immediate Next Tasks
1. Treat `plans/step_04b_batch_d_scaling_checkpoint.md` as the first completed 4B scale-up checkpoint and preserve Batch D as the first January-scale regression and QA reference set.
2. Use the one-command runner and the recent-first multi-batch series driver for Batch E style collection rather than hand-building every wide sweep.
3. Keep the first active large-scale collection target on the lag-adjusted four-year Phoenix window from `2022-03-29` through the latest currently published archive date, not blindly through the current calendar day.
4. Inspect each newly completed batch summary before generating downstream QA/splits for that batch.
5. Add merged-release tooling before calling the full four-year sweep a finished dataset release.
6. Reassess training-readiness thresholds after the first training-scale merged release lands, not before.
7. Add stricter evaluation variants, such as stronger dan cuts or player-disjoint holdouts, after the larger Phoenix corpus is staged.
8. Once a duplicate-free partial multi-batch merge exists at training scale, allow Step 5 design and first baseline implementation work to proceed in parallel with the remaining Batch E collection.

## Guardrails
- Keep the default path on four-player Phoenix hanchan only.
- Do not broaden into sanma or mixed rulesets during this stage.
- Prefer reproducible batch selection over one-off cherry-picked samples.
- Preserve the current QA and split machinery; scale the pipeline, not the chaos.
