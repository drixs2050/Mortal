# Step 4 Plan: Dataset QA And Release Candidate

Date: 2026-03-28

## Purpose
Turn the current normalized Tenhou pilot corpus into a trustworthy pilot dataset release candidate for later supervised-training design and pipeline validation.

Step 3 proved that we can ingest and normalize the current four-player Phoenix hanchan slice.
Step 4 is where we decide what subset is valid for training, how we split it, and which QA signals must exist before we call a dataset version usable.

## Current Starting Point
Available pilot corpora now include:
- Batch B:
  - `8` real elite replays with oracle-backed XML parity validation
- Batch C:
  - `36` real elite replays staged from official Phoenix raw-archive listings
  - XML-only ingestion
  - zero failures
  - QA summary and split outputs

The strongest current pilot artifact is:
- `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.jsonl`

## Current Status
As of `2026-03-28`:
- Batch B remains the converter-validation corpus for XML parity and edge-case spot checks.
- Batch C is the first Step 4 release-candidate cohort for the current supervised-data pilot slice.
- This Step 4 checkpoint is a pilot release candidate, not a production-scale supervised-learning corpus.
- The Batch C hanchan split now has a dedicated player-overlap QA artifact at:
  - `data/splits/v1/tenhou_phoenix_batch_c_v0/phoenix_hanchan_7dan_all/player_overlap_summary.json`
- The Step 4 release-candidate checkpoint is recorded in:
  - `plans/step_04_batch_c_release_candidate.md`

## Closure Decision
As of `2026-03-28`, Step 4 is formally closed for the current pilot scope.

That means:
- the Step 4 deliverables below are complete for the pilot `RC0` cohort
- further corpus growth belongs to Step 4B, not Step 4
- Step 5 may be prepared in parallel, but it should not be treated as the main active milestone until Step 4B produces a training-scale merged release

## Step 4 Deliverables
1. A written split policy for:
   - time ordering
   - player leakage handling
   - source cohort boundaries
2. A reusable QA checklist and summary format for normalized dataset versions.
3. A first release-candidate cohort definition for the initial supervised baseline.
4. Stable split outputs and file lists for that cohort.

## Workstreams

### Workstream A. Split Policy
Goal:
Lock the first non-accidental train/val/test policy.

Questions to answer:
- Is simple chronological splitting enough for the first release candidate?
- Do we need explicit player-disjoint validation/test sets, or is that a later stricter evaluation layer?
- Should Batch B and Batch C remain separate checkpoints, or should they be merged into one pilot manifest?

Suggested first answer:
- use chronological splitting as the default release candidate
- record player overlap statistics in QA
- add stricter player-disjoint evaluation as a later optional split family

### Workstream B. Cohort Definition
Goal:
Define the first exact supervised-learning cohort.

Suggested initial cohort:
- `source = tenhou`
- `room = 鳳`
- `ruleset = 鳳南喰赤`
- `go_type = 169`
- all players `>= 七段`
- four-player only

Questions to answer:
- Do we allow `天鳳位` inside the same first cohort? likely yes
- Do we require a minimum player rate floor in addition to dan? likely not yet

### Workstream C. QA Standards
Goal:
Define the minimum QA fields that must be present for every dataset version.

Required checks:
- row count
- date range
- room/ruleset/go_type counts
- dan and rate distributions
- kyoku and event-count summaries
- validation status counts
- failure manifest counts

Nice-to-have additions:
- player-name hash overlap report across splits
- duplicate-group collision report
- source-game-id date histogram

### Workstream D. Release Candidate Packaging
Goal:
Package the first dataset slice so later SL work can consume it directly.

Outputs:
- normalized manifest
- QA summary
- curated split directories
- short release note with cohort definition and caveats

## Step 4 Handoff
1. Treat Batch C `RC0` as the closed Step 4 pilot release artifact.
2. Keep Batch B available as the converter-validation checkpoint, not the training-release cohort.
3. Keep all larger-scale dataset growth inside `plans/step_04b_corpus_expansion_plan.md` so it is not confused with the pilot `RC0` checkpoint.
4. Do not treat Step 5 as fully active until Step 4B produces a training-scale merged release.

## Guardrails
- Keep the default cohort on four-player Phoenix hanchan only.
- Do not broaden into sanma or mixed rulesets during Step 4.
- Prefer documented cohort boundaries over maximizing row count too early.
- Keep replay-JSON oracles as validation references, not as a data dependency.
