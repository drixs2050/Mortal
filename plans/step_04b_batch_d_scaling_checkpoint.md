# Step 4B Batch D Scaling Checkpoint

Date: 2026-03-28

## Purpose
Record the first real Step 4B corpus-expansion run after the Batch C `RC0` pilot.

Batch D was designed as the first larger-scale check that the XML-only Tenhou pipeline still works when the archive span grows from `3` days to `31`.

## Cohort
- source: `tenhou`
- room: `鳳`
- ruleset: `鳳南喰赤`
- go type: `169`
- four-player only
- all players `>= 七段`

Snapshot id:
- `2026-03-28_phoenix_batch_d`

Dataset id:
- `tenhou_phoenix_batch_d_v0`

## Archive Sweep
Archive dir:
- `/tmp/tenhou_scc_batch_d/`

Selection inputs:
- `31` daily `scc*.html.gz` files covering `2026-01-01` through `2026-01-31`

Selection outputs:
- archive list:
  - `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.archives.txt`
- archive summary:
  - `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.archives.json`
- refs file:
  - `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.refs.txt`
- selection summary:
  - `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.selection.json`

Selection result:
- `372` selected replay ids
- `12` qualifying replays from each of the `31` archive days

## Raw Staging Result
Raw snapshot outputs:
- `data/raw/tenhou/2026-03-28_phoenix_batch_d/`
- `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.json`

Result:
- `372` XML files staged
- raw manifest written successfully

## Ingestion Result
Normalized outputs:
- `data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.jsonl`
- `data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.summary.json`
- `data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.qa_summary.json`
- `data/manifests/failures/tenhou/tenhou_phoenix_batch_d_v0__2026-03-28_phoenix_batch_d.jsonl`

Final result after converter fix:
- candidate count: `372`
- accepted count: `372`
- rejected count: `0`
- failure categories: none

Important Batch D finding:
- the first ingestion pass surfaced `4` loader-validation failures
- cause: zero-delta `ryukyoku` outcomes were being emitted without a `deltas` field
- fix: `scripts/tenhou_xml.py` now always writes `deltas` for those `ryukyoku` events
- regression coverage: `tests/test_tenhou_converter.py`

## QA Snapshot
From `data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.qa_summary.json`:

- accepted games: `372`
- room counts:
  - `鳳`: `372`
- ruleset counts:
  - `鳳南喰赤`: `372`
- lobby counts:
  - `0`: `372`
- go type counts:
  - `169`: `372`
- dan label counts:
  - `七段`: `890`
  - `八段`: `487`
  - `九段`: `92`
  - `十段`: `18`
  - `天鳳位`: `1`
- player rate range:
  - min `2022.52`
  - max `2336.48`
  - mean `2159.094059139785`
- total kyoku count:
  - `4020`
- total normalized event count:
  - `410443`
- game date range:
  - `2026-01-01T00:00:00` to `2026-01-31T00:00:00`

## Split Snapshot
Hanchan-only elite split:
- output dir:
  - `data/splits/v1/tenhou_phoenix_batch_d_v0/phoenix_hanchan_7dan_all/`
- counts:
  - train `297`
  - val `37`
  - test `38`

Because Batch D was selected directly from `四鳳南喰赤－`, the all-elite and hanchan-only split families remain identical here.

## Player Overlap Snapshot
From `player_overlap_summary.json`:
- train unique player hashes: `467`
- val unique player hashes: `122`
- test unique player hashes: `116`
- train/val shared unique player hashes: `96`
- train/test shared unique player hashes: `84`
- val/test shared unique player hashes: `47`
- three-way shared unique player hashes: `38`

Interpretation:
- the chronological split is still clean at the game level
- player leakage is much more visible than in `RC0`
- this is expected for a one-month Phoenix slice and is a reminder that Batch D is a scaling checkpoint, not yet a strict generalization benchmark

## Practical Meaning
Batch D is a successful Step 4B checkpoint because it proves:
- the archive sweep can expand from `3` to `31` days cleanly
- the XML-only staging path handles hundreds of real replays, not just dozens
- the ingestion/QA/split pipeline remains reusable at this larger scale
- the first scale-up run can surface and quickly validate real converter bugs

## Remaining Limits
- Batch D is still only one month of data.
- `372` games is far larger than `RC0`, but still below a truly training-scale multi-month corpus.
- player leakage across chronological splits is now clearly non-trivial.

## Recommended Next Work
1. Treat Batch D as the first successful Step 4B scaling checkpoint.
2. Start Batch E as a recent-first multi-batch series of bounded `100k` windows rather than one monolithic multi-year run.
3. Keep the zero-delta `ryukyoku` regression covered so future corpus expansions do not reintroduce the bug.
4. Inspect each larger batch summary before generating downstream QA/splits for that batch.
