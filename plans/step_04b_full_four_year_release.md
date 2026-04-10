# Step 4B Full Four-Year Release

Date: 2026-03-29

## Purpose
Record the first full Step 4B merged Phoenix corpus release for the current supervised-data program.

This note turns the completed Batch E recent-first sweep into the stable Step 4B release for the current scope:
- source: Tenhou
- room: Phoenix
- intended ruleset: `鳳南喰赤`
- intended go type: `169`
- four-player only
- merged from `E01` through `E16`

## Release Decision
- Batch D remains the first Step 4B scaling checkpoint.
- the partial `E01` through `E09` merged release remains the intermediate checkpoint that unblocked Step 5 work.
- `tenhou_phoenix_4y_full_e01_e16_v0` is now the formal full Step 4B dataset release.

Why:
- the recent-first sweep now covers the intended four-year window from `2022-03-29` through `2026-03-18`
- all `16` normalized Batch E manifests merge cleanly with `0` duplicate rows
- merged QA, split, player-overlap, and BC-helper artifacts now exist for the full release
- the corpus is now large enough that Step 6 can harden the actual training path against the intended full dataset rather than an intermediate checkpoint

## Release Artifacts
- merged normalized manifest:
  - `data/manifests/normalized/v1/tenhou_phoenix_4y_full_e01_e16_v0.jsonl`
- merge summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_4y_full_e01_e16_v0.merge_summary.json`
- merged QA summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_4y_full_e01_e16_v0.qa_summary.json`
- `七段+ all-player` split family:
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_7dan_all/`
- `七段+ all-player` overlap QA:
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_7dan_all/player_overlap_summary.json`
- `八段+ any-player` split family used by the current BC path:
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/`
- `八段+ any-player` overlap QA:
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/player_overlap_summary.json`
- BC path-cache artifact:
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/path_cache_v1.pth`
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/path_cache_v1.summary.json`
- BC actor-filter artifact:
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/actor_filter_min17.pth`
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/actor_filter_min17.summary.json`

## Merge Snapshot
From `tenhou_phoenix_4y_full_e01_e16_v0.merge_summary.json`:
- input manifests: `16`
- merged rows: `694346`
- duplicate rows: `0`
- merged date range: `2022-03-29T00:00:00` through `2026-03-18T23:00:00`

## QA Snapshot
From `tenhou_phoenix_4y_full_e01_e16_v0.qa_summary.json`:
- total merged rows: `694346`
- room counts:
  - `鳳`: `694346`
- ruleset counts:
  - `鳳南喰赤`: `694342`
  - `鳳南喰赤速`: `4`
- go type counts:
  - `169`: `694342`
  - `233`: `4`
- dan label counts:
  - `七段`: `1583359`
  - `八段`: `891372`
  - `九段`: `264305`
  - `十段`: `35692`
  - `天鳳位`: `2656`
- total kyoku: `7361558`
- total normalized events: `745672962`

Important note:
- the merged manifest preserves `4` rows that fall outside the intended `鳳南喰赤` / `go_type = 169` cohort
- the curated split families continue to enforce the exact cohort filters, so those `4` rows do not enter the release splits

## Split Snapshot
`七段+ all-player` release split:
- accepted rows: `694342`
- train: `555473`
- val: `69434`
- test: `69435`

`八段+ any-player` BC release split:
- accepted rows: `621103`
- train: `496882`
- val: `62110`
- test: `62111`

Important policy note:
- both split families are chronologically clean at the game level
- neither split family is player-disjoint
- player leakage is explicitly measured and reported rather than silently ignored

## Player Overlap Snapshot
From `phoenix_hanchan_8dan_actor_any/player_overlap_summary.json`:
- shared source games across any split pair: `0`
- train unique player hashes: `4257`
- val unique player hashes: `1663`
- test unique player hashes: `1663`
- train/val shared unique player hashes: `1390`
- train/test shared unique player hashes: `1190`
- val/test shared unique player hashes: `1178`
- three-way shared unique player hashes: `994`

Interpretation:
- the release is mechanically clean at the game level
- player leakage remains material in chronological splits at this corpus scale
- the overlap report is doing its job and should remain part of every future corpus refresh

## BC Helper Artifact Snapshot
From the `八段+ any-player` helper summaries:
- `path_cache_v1` covers:
  - train `496882`
  - val `62110`
  - test `62111`
- `actor_filter_min17` indexed all `621103` requested split files
- `actor_filter_min17` matched all `621103` split rows with `0` file-level exclusions

## Exit Signal
For the current four-year Phoenix scope, Step 4B exit criteria are satisfied:
- we have a training-scale elite corpus rather than a pilot-only checkpoint
- the merged release is duplicate-free
- merged QA reports exist
- stable split files exist
- player-overlap reporting exists
- the current BC path has the full-corpus helper artifacts it needs for Step 6

## Closeout Decision
As of `2026-03-29`, this note is the formal closeout artifact for Step 4B.

Interpretation:
- Step 4B is closed for the current four-year Phoenix yonma scope
- the partial `E01` through `E09` merge remains a useful historical checkpoint, but it is no longer the main dataset reference
- future corpus refresh work beyond `2026-03-18` should be treated as a new dataset-maintenance task, not unfinished Step 4B work
- Step 6 is now the active milestone

## Handoff
1. Use `tenhou_phoenix_4y_full_e01_e16_v0` as the default full-corpus data foundation for Step 6.
2. Keep `phoenix_hanchan_8dan_actor_any` plus `path_cache_v1` and `actor_filter_min17` as the default BC data path until a different evaluation policy is intentionally chosen.
3. Treat single-process loading as the known-safe default while Step 6 hardens the runtime.
4. Retune validation cadence, checkpoint policy, and model-size experiments against the full release rather than the earlier partial checkpoint.
5. Enter Step 7 only after the Step 6 hardening work yields a launch-ready full-scale training recipe.
