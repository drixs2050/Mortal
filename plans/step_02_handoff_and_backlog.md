# Step 2 Handoff And Backlog

Date: 2026-03-28

## Purpose
This document closes out Step 1 and defines what Step 2 must produce before any serious supervised training campaign starts.

Step 1 proved that the repo can be built, configured, and smoke-tested on this workstation.
Step 2 is not about optimization yet.
Step 2 is about real data strategy, ingestion, and dataset quality.

## What The Repo Already Accepts

### 1. Main offline training input
The main offline trainer expects gzipped line-delimited JSON game logs:
- file pattern: `**/*.json.gz`
- each file is parsed by `libriichi.dataset.GameplayLoader`
- each game log is expanded into per-player gameplay samples

What the loader extracts today:
- observations
- legal-action masks
- chosen actions from the log
- per-step terminal distance metadata
- per-kyoku GRP-derived reward targets
- next-rank auxiliary targets

Implication:
- We already have a usable canonical training target format for offline replay-style training.
- Step 2 does not need to invent a new training file format from scratch unless we decide to.

### 2. GRP training input
The GRP trainer also expects gzipped line-delimited JSON game logs:
- file pattern: `**/*.json.gz`
- each log is parsed into per-kyoku game progression features
- labels are derived final placement / rank targets

Implication:
- A single normalized log corpus can feed both GRP training and the main offline trainer.

### 3. Optional loader-side filtering and augmentation
The current data path already supports:
- include filters by player name
- exclude filters by player name
- augmentation toggle at load time

Implication:
- We can keep the normalized corpus relatively generic and layer some training-time selection on top.

## What Step 1 Actually Used
Step 1 did not use human records.
It used tiny local fixture logs created from bootstrap self-play so we could validate the pipeline end to end.

Implication:
- The current setup proves mechanics, not model quality.
- No conclusions about strength should be drawn from Step 1 artifacts.

## What Raw Sources Still Need Converters
The repo expects normalized `.json.gz` logs in the `libriichi` event format.
That means any external source still needs a converter unless it already matches this schema.

Likely raw sources that need conversion work:
- Tenhou-style records
- Any downloaded archive with non-`libriichi` event structure
- Any source with missing metadata we care about for filtering and splits

Step 2 must decide:
- which sources we will actually use
- whether we are allowed to use them
- what metadata we must preserve during conversion

## What Current Configs And Scripts Are Good Enough For

### Good enough now
- local workstation bring-up
- tiny smoke datasets
- checkpoint bootstrap
- single-GPU validation on `cuda:0`
- basic offline trainer and GRP trainer execution

### Not good enough yet
- large dataset ingestion
- dataset versioning
- data QA and deduplication
- train/val/test split management
- manifest generation and checksums
- serious supervised experiment tracking
- multi-GPU training

## Must-Fix Items Before Data Collection And SL Training Start

### 1. Source policy
Write down:
- approved sources
- allowed usage rules
- retention policy
- attribution / compliance requirements

### 2. Canonical normalized schema
Define exactly:
- what the normalized `.json.gz` files contain
- whether one file equals one game or a grouped bundle
- what metadata must be preserved at the file or record level
- how source, date, room, and player-strength information is stored

### 3. Conversion pipeline
Implement or adapt:
- raw-record parser
- normalization into `libriichi`-compatible events
- validation for corrupt or partial games
- deterministic output naming and sharding

### 4. Dataset manifests and QA
Add:
- shard manifests
- counts and checksums
- date ranges
- duplicate detection
- malformed-log reports
- split files for train, validation, and test

### 5. Real supervised training plan
Based on the current repo, the main trainer is offline RL / offline value learning, not pure behavior cloning.
Before the planned human-record pretraining phase starts, we must decide whether to:
- keep the current trainer as the initial offline baseline
- add explicit behavior-cloning losses to it
- or create a separate supervised imitation trainer

### 6. Evaluation policy
Define the first real metrics for model selection:
- held-out log metrics
- self-play ladder metrics
- confidence intervals for avg rank / avg pt
- regression checks for parser and legality correctness

## Step 2 Deliverables
Step 2 should end with:
- a written data-source policy
- a canonical raw-data schema
- a canonical normalized-data schema
- a conversion/backfill plan
- a QA and split plan
- a decision on the first true supervised objective

## Recommended First Tasks For Step 2
1. List candidate human-data sources and usage constraints.
2. Inspect a small sample from each source and compare fields against the `libriichi` loader expectations.
3. Define the canonical normalized log schema and shard layout.
4. Design manifest, checksum, and split files.
5. Decide whether supervised pretraining means behavior cloning, current offline training, or a hybrid objective.

## Exit Signal For Moving Past Step 2
We should not start a serious training campaign until:
- we know exactly what real data we are allowed to use
- we can convert it into the repo's accepted format
- we have train/val/test splits and QA
- we know what "supervised pretraining" concretely means in this codebase
