# Step 2 Manifest And Split Spec

Date: 2026-03-28

## Purpose
This document defines how Step 2 datasets should be described, audited, and split.
The point is to make future dataset releases inspectable and reproducible.

## Manifest Families

### 1. Raw snapshot manifests
Location:
- `data/manifests/raw/<source>/<snapshot_id>.json`

Purpose:
- describe what was imported into `data/raw/`
- preserve provenance and acquisition notes

Recommended fields:
- `source`
- `snapshot_id`
- `acquired_at`
- `acquired_by`
- `official_access_path`
- `usage_status`
- `file_count`
- `total_bytes`
- `notes`
- `files`: per-file entries with relative path, size, and hash when practical

### 2. Normalized dataset manifests
Location:
- `data/manifests/normalized/v1/<dataset_id>.jsonl`

Purpose:
- one row per normalized game file
- main index for QA, deduplication, and split generation

Recommended fields per row:
- `dataset_id`
- `relative_path`
- `source`
- `source_game_id`
- `raw_snapshot_id`
- `game_date`
- `year`
- `month`
- `ruleset`
- `room`
- `table_size`
- `player_names_present`
- `player_ids_hashed`
- `converter_version`
- `validation_status`
- `file_sha256`
- `byte_size`
- `event_count`
- `kyoku_count`
- `duplicate_group`
- `split`

### 3. Failure manifests
Location:
- `data/manifests/failures/<source>/<run_id>.jsonl`

Purpose:
- record why candidate source games were skipped or rejected

Required fields:
- `source`
- `source_game_id` when available
- `raw_snapshot_id`
- `error_category`
- `error_message`
- `run_id`

## Split Files
Location:
- `data/manifests/splits/<dataset_id>/train.txt`
- `data/manifests/splits/<dataset_id>/val.txt`
- `data/manifests/splits/<dataset_id>/test.txt`

Contents:
- one normalized relative path per line

These files should be generated from the normalized dataset manifest, not maintained by hand.

## Split Policy

### Primary rule
Use a deterministic temporal split once date information is available.

Recommended first policy:
- `train`: oldest 80 percent by game date
- `val`: next 10 percent
- `test`: newest 10 percent

Reason:
- it better matches the real future-facing usage case than a purely random split
- it reduces leakage from duplicate or near-duplicate logs collected in the same period

### Secondary anti-leakage rules
- Never split the same `source_game_id` across train, val, and test.
- Keep exact duplicates in the same duplicate group and assign the whole group to one split.
- If multiple normalized rows come from the same raw source game, assign them together.

### Optional future holdouts
After the first dataset is stable, consider adding:
- player-holdout evaluation
- source-holdout evaluation
- elite-room-only evaluation

These are useful, but they should not block the first Step 2 dataset release.

## QA Checklist
Every normalized dataset build should report:
- raw snapshot count
- candidate game count
- accepted game count
- rejected game count by error category
- duplicate count
- source mix
- year/month mix
- ruleset mix
- kyoku count distribution
- event count distribution

Loader-based QA:
- percent of files that load through `GameplayLoader`
- percent of files that load through `Grp`
- count of files rejected by each loader

Game-level QA:
- score sum sanity checks
- rank-label sanity checks
- nonzero action count
- complete `start_game` to `end_game` structure
- required `deltas` present for `hora` and `ryukyoku`

## Duplicate Detection Rules
The first duplicate key should use:
- `source`
- `source_game_id`

If `source_game_id` is unavailable or unreliable, use a stronger fallback key derived from:
- player identities or aliases
- game date when known
- early game fingerprint
- full normalized file hash after conversion

Important rule:
- deduplication should happen before split assignment

## Privacy And Identity Handling
- If platform policy or participant consent is sensitive, store hashed player IDs in manifests.
- Preserve a clear distinction between player labels inside normalized files and richer identity metadata in manifests.
- Do not require real names anywhere in the canonical pipeline.

## Release Convention
Each future dataset build should have:
- a stable `dataset_id`
- a manifest file
- split files
- a short QA summary
- a converter version or git commit reference

## Immediate Implementation Guidance
For the first real dataset attempt:
- keep manifests simple and machine-readable
- generate split files from the manifest
- make the QA report small but mandatory
- prefer deterministic behavior over clever heuristics
