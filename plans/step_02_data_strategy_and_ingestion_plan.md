# Step 2 Plan: Data Strategy And Ingestion

Date: 2026-03-28

## Why Plan Step 2 First
Step 2 is the first stage where a wrong decision can waste a lot of time later.
If we start collecting or converting data before we lock source policy, schema, and QA rules, we risk building a corpus that is hard to trust, hard to reproduce, or hard to train on.

The purpose of this plan is to make Step 2 deliberate:
- decide what data we can use
- decide what format we want long term
- decide how we validate it
- only then start ingestion work

## Step 2 Objectives
1. Define which human game sources are acceptable and useful.
2. Define the canonical raw and normalized schemas for this project.
3. Define the conversion pipeline from source records into `libriichi`-compatible `.json.gz` logs.
4. Define the dataset QA, manifest, and split policy.
5. Decide what "supervised pretraining" will concretely mean in this codebase before large-scale ingestion begins.

## Inputs We Already Have
- Step 1 proved the repo runs on this workstation.
- The current repo accepts normalized `.json.gz` logs through `GameplayLoader` and `Grp`.
- The current main trainer is an offline RL / offline value-learning path, not a pure behavior-cloning pipeline.
- A Step 2 handoff/backlog already exists in `plans/step_02_handoff_and_backlog.md`.

## Current Status
Completed so far:
- Workstream A now has a source catalog and a conservative source-policy draft.
- Workstream B now has a canonical raw/normalized schema spec tied to the current `GameplayLoader` and `Grp` requirements.
- Workstream C now has an initial conversion design with Tenhou chosen as the first technical format-study target.
- Workstream D now has a manifest/split spec, plus initial training-side support for manifest-driven file lists.
- Workstream E now has a written decision that true supervised pretraining should mean explicit behavior cloning rather than re-labeling the current offline RL trainer.

Important Tenhou-specific progress:
- one official raw Tenhou XML sample has been captured under `data/raw/`
- raw Tenhou `lobby`, `type`, `dan`, and `rate` fields have been confirmed present
- the current remaining gap is the first real parser/converter, not data-layout uncertainty

## Step 2 Scope

Included:
- source inventory
- usage-policy review
- schema design
- manifest design
- split policy
- conversion design
- QA design
- supervised-objective decision

Not included:
- full historical backfill
- long training runs
- RL finetuning
- multi-GPU training work
- Tenhou automation

## Workstreams

### Workstream A: Source Inventory And Usage Policy
Goal:
Decide what candidate human-record sources exist and what constraints apply to each.

Tasks:
- list candidate sources
- note source type, accessibility, expected strength level, time coverage, and metadata quality
- record usage constraints, retention limits, and any attribution requirements
- classify sources as:
  - approved
  - needs review
  - rejected

Deliverable:
- a source catalog with a clear status for each candidate source

Exit condition:
- we know which sources we are even allowed to prototype against

### Workstream B: Canonical Dataset Design
Goal:
Define the project's own raw and normalized data model.

Tasks:
- define what counts as raw data in this project
- define the normalized `.json.gz` event format we will store
- define file naming, sharding, compression, and directory layout
- define required metadata fields:
  - source
  - date
  - room / lobby / rank tier when available
  - ruleset flags
  - player tags or anonymized IDs when appropriate
  - converter version

Deliverable:
- a schema spec for raw data and normalized data

Exit condition:
- every later converter and manifest can target one stable schema

### Workstream C: Conversion Pipeline Design
Goal:
Define how external records become normalized training logs.

Tasks:
- map each approved source into the normalized schema
- identify source-specific parser requirements
- define deterministic conversion behavior
- define error categories:
  - malformed
  - partial
  - duplicate
  - unsupported rules
- define converter outputs and failure reports

Deliverable:
- a converter design doc and per-source mapping notes

Exit condition:
- we know how to implement ingestion without guessing at edge cases

### Workstream D: QA, Manifests, And Splits
Goal:
Make the dataset auditable and safe to train on.

Tasks:
- define shard manifests
- define checksums and counts
- define duplicate detection rules
- define train/val/test split policy
- define leakage policy by player and by time
- define QA reports:
  - game count
  - malformed rate
  - duplicate rate
  - source mix
  - year/month mix
  - player coverage
  - action frequency and round-length sanity checks

Deliverable:
- manifest spec, split spec, and QA checklist

Exit condition:
- a future dataset release can be inspected and reproduced

### Workstream E: Training-Objective Decision
Goal:
Resolve the mismatch between the roadmap's intended supervised phase and the repo's current trainer design.

Tasks:
- review whether the current `train.py` can serve as the first human-log baseline
- decide whether to:
  - keep the existing offline objective first
  - add explicit behavior-cloning losses
  - build a separate supervised imitation trainer
- define the first metrics for that choice

Deliverable:
- a written decision on the first real SL baseline objective

Exit condition:
- Step 3 and later training work know what targets they are preparing for

## Recommended Execution Order
1. Finish Workstream A first.
2. Use source samples from approved or reviewable candidates to inform Workstream B.
3. Lock schema before writing any serious converter logic.
4. Define QA and splits before large-scale conversion.
5. Resolve the supervised-objective decision before committing to a huge ingestion campaign.

## Step 2 Deliverables
- `source_catalog.md`
- `data_schema_spec.md`
- `conversion_design.md`
- `manifest_and_split_spec.md`
- `sl_objective_decision.md`

These can be separate files or a smaller set of combined docs, but all five topics must be covered.

## Exit Criteria
Step 2 is complete when:
- we have a reviewed source list with usage status
- we have a canonical normalized schema
- we know how approved sources map into that schema
- we have manifest, QA, and split rules
- we have decided what the first real supervised baseline means in this repo

## Immediate Next Actions
1. Verify the mapping from Tenhou raw codes into stable manifest fields, especially `GO type`, `lobby`, and `dan`.
2. Implement a read-only Tenhou XML parser that extracts one game's metadata and event skeleton without attempting full conversion yet.
3. Define the first normalized-manifest row shape for Tenhou outputs so elite-game subsets can be selected deterministically.
4. Implement the first minimal Tenhou-to-Mortal converter and validate its output through `GameplayLoader` and `Grp`.

## Guidance For This Session
The best use of the current session is now the Step 2 to Step 3 boundary work:
- keep Tenhou as the first converter target
- preserve raw elite-game metadata exactly before adding human-readable labels
- validate parser output against the current Mortal loaders as early as possible

We should still avoid any bulk collection campaign until the parser, manifests, and validation path are stable.
