# Step 4 Batch C Release Candidate

Date: 2026-03-28

## Purpose
Record the first Step 4 dataset release-candidate checkpoint for the current supervised-data pilot slice.

This note turns the current Batch C hanchan cohort into `RC0` for the current scope:
- source: Tenhou
- room: Phoenix
- ruleset: `鳳南喰赤`
- go type: `169`
- four-player only
- all players `>= 七段`

## Release Decision
- Batch B remains the converter-validation corpus.
- Batch C becomes the first dataset release candidate for the current pilot slice.
- `RC0` is a pipeline-validation and objective-design checkpoint, not a production-scale supervised-learning dataset.

Why:
- Batch B is the stronger XML-parity checkpoint because it still has replay-JSON oracle coverage.
- Batch C is the stronger training-data checkpoint because it is larger, XML-only, and already staged from official Phoenix raw-archive listings.

## Release Artifacts
- normalized manifest:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.jsonl`
- normalized summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.summary.json`
- normalized QA summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.qa_summary.json`
- curated split dir:
  - `data/splits/v1/tenhou_phoenix_batch_c_v0/phoenix_hanchan_7dan_all/`
- split overlap QA:
  - `data/splits/v1/tenhou_phoenix_batch_c_v0/phoenix_hanchan_7dan_all/player_overlap_summary.json`

## Split Policy
For `RC0`, use the existing chronological split as the default release policy:
- train: earliest accepted rows
- val: next accepted rows
- test: latest accepted rows

For the current `36`-game cohort, that produces:
- train: `28`
- val: `3`
- test: `5`

Important policy note:
- `RC0` is not player-disjoint.
- player leakage is measured and reported, not yet used as a hard exclusion rule.
- stricter player-disjoint evaluation can be added later as a separate split family once the corpus is larger.

## QA Snapshot
From the current release artifacts:
- accepted games: `36`
- rejected games: `0`
- failure categories: none
- total kyoku: `391`
- total normalized events: `38421`
- room counts: `鳳 = 36`
- ruleset counts: `鳳南喰赤 = 36`
- go type counts: `169 = 36`
- date range: `2026-01-01` through `2026-01-03`

## Player Overlap Snapshot
From `player_overlap_summary.json`:
- no source games are shared across train, val, and test
- train unique player hashes: `92`
- val unique player hashes: `12`
- test unique player hashes: `20`
- train/val shared unique player hashes: `3`
- train/test shared unique player hashes: `3`
- val/test shared unique player hashes: `0`
- three-way shared unique player hashes: `0`

Interpretation:
- the chronological split is mechanically clean at the game level
- player leakage exists but is limited in this small pilot cohort
- the cohort is good enough for a first release candidate, but not yet a strict generalization benchmark
- the cohort is far too small to count as a proper serious supervised-learning corpus by itself

## Caveats
- `RC0` is still only a pilot corpus and spans three archive days.
- the split sizes are too small for strong held-out claims by themselves.
- replay-JSON oracles are no longer required for ingestion, but Batch B should remain available for converter spot checks.
- sanma and mixed-ruleset support remain out of scope.

## Exit Signal
For the current pilot scope, Step 4 exit criteria are satisfied:
- we have a versioned dataset release candidate
- QA reports exist
- split files exist
- player-overlap reporting exists

## Closeout Decision
As of `2026-03-28`, this `RC0` note is the formal closeout artifact for Step 4.

Interpretation:
- Step 4 is closed for the pilot scope
- this document should remain the stable reference for the pilot release definition and caveats
- larger-scale Phoenix collection and merged-release work now belong to Step 4B
- Step 5 should not be treated as fully active until Step 4B yields a training-scale merged release

## Handoff
1. Keep the Batch B converter-validation path intact for future ingestion regressions.
2. Treat Step 4B as the active dataset stage for larger Phoenix collection and merged-release work.
3. Use this `RC0` cohort for pipeline validation and early objective-design reference only, not for serious training conclusions.
