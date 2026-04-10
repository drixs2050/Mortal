# Step 3 Phoenix Batch C Validation

Date: 2026-03-28

## Purpose
Record the first broader XML-only Phoenix hanchan validation cohort after the eight-game pilot.

Batch C is the first Batch 3 checkpoint that:
- meets the target `20` to `50` game validation range
- is staged from official Phoenix raw-archive listings rather than only the tiny local replay-JSON pool
- proves the XML-only path can scale beyond the earlier eight-game parity set

## Raw Snapshot
- Raw snapshot id: `2026-03-28_phoenix_batch_c`
- Raw manifest: `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_c.json`
- Ref list: `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_c.refs.txt`
- Selection summary: `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_c.selection.json`

Selection profile:
- official `scc*.html.gz` Phoenix raw archives
- four-player only
- exact raw-archive ruleset label `四鳳南喰赤－`
- `12` selected replays per archive
- `36` total selected replays across:
  - `scc20260101.html.gz`
  - `scc20260102.html.gz`
  - `scc20260103.html.gz`

## Staging Method
- Official Phoenix raw-archive files were fetched into `/tmp/tenhou_scc_batch_c/`
- replay ids were selected with:
  - `scripts/select_tenhou_scc_refs.py`
- XML replays were staged with:
  - `scripts/stage_tenhou_reference_batch.py`

Important Batch C change:
- the stager now supports XML-only snapshots without replay-JSON oracles
- the selector now supports `--max-per-archive` so a cohort can span multiple raw-archive files instead of overfitting one day

## Ingestion Run
Command:

`PYTHONPATH=/home/drixs2050/Documents/Mortal/mortal /home/drixs2050/anaconda3/envs/mortal/bin/python scripts/ingest_tenhou_snapshot.py --raw-snapshot-id 2026-03-28_phoenix_batch_c --dataset-id tenhou_phoenix_batch_c_v0 --converter-version tenhou-xml-v0 --converter-source xml --overwrite`

Produced outputs:
- normalized manifest:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.jsonl`
- normalized summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.summary.json`
- normalized QA summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.qa_summary.json`
- failure manifest:
  - `data/manifests/failures/tenhou/tenhou_phoenix_batch_c_v0__2026-03-28_phoenix_batch_c.jsonl`

## Result
- candidate count: `36`
- accepted count: `36`
- rejected count: `0`
- failure categories: none

This is the first broader cohort showing that the XML-only four-player Phoenix hanchan path works on more than a tiny pilot.

## QA Summary
From `data/manifests/normalized/v1/tenhou_phoenix_batch_c_v0.qa_summary.json`:

- room counts:
  - `鳳`: `36`
- ruleset counts:
  - `鳳南喰赤`: `36`
- lobby counts:
  - `0`: `36`
- go type counts:
  - `169`: `36`
- dan label counts:
  - `七段`: `87`
  - `八段`: `49`
  - `九段`: `7`
  - `天鳳位`: `1`
- player rate range:
  - min `2027.24`
  - max `2286.17`
  - mean `2152.4040972222224`
- total kyoku count:
  - `391`
- total normalized event count:
  - `38421`
- game date range:
  - `2026-01-01T00:00:00` to `2026-01-03T00:00:00`

## Split Outputs
All-elite Phoenix split:
- output dir:
  - `data/splits/v1/tenhou_phoenix_batch_c_v0/phoenix_all_7dan_all/`
- filters:
  - `source = tenhou`
  - `room = 鳳`
  - `min_player_dan = 16`
  - `player_threshold_mode = all`
- counts:
  - train `28`
  - val `3`
  - test `5`

Hanchan-only elite Phoenix split:
- output dir:
  - `data/splits/v1/tenhou_phoenix_batch_c_v0/phoenix_hanchan_7dan_all/`
- filters:
  - `source = tenhou`
  - `room = 鳳`
  - `ruleset = 鳳南喰赤`
  - `go_type = 169`
  - `min_player_dan = 16`
  - `player_threshold_mode = all`
- counts:
  - train `28`
  - val `3`
  - test `5`

Because Batch C was selected directly from `四鳳南喰赤－`, the all-elite and hanchan-only splits are identical here.

## Useful Findings
- The XML-only path is no longer just a parity experiment; it now supports a `36`-game elite cohort end to end.
- Batch C surfaced a real metadata gap: XML-only manifest rows were missing human-readable `room` and `ruleset`.
- That gap is now fixed by inferring those fields from `go_type`, and the new behavior is covered by `tests/test_tenhou_converter.py`.
- This means manifest-driven curation works on XML-only corpora too, not only oracle-backed samples.

## Practical Meaning
For the current target slice:
- four-player
- Phoenix room
- hanchan
- high-dan ranked tables

the Step 3 ingestion path is now mechanically in good shape:
- official raw listings can produce replay ids
- official XML can be staged without saved replay JSON
- normalized logs, manifests, QA summaries, and split files can be generated reproducibly

## Remaining Limits
- Batch C is still a pilot cohort, not a real training corpus.
- The current selection only spans three January 2026 archive days.
- No real failure categories appeared in this cohort, so malformed-edge coverage still comes mostly from dedicated regression fixtures.
- Sanma and non-hanchan branches remain intentionally out of scope.

## Recommended Next Work
1. Treat the core Step 3 exit criteria as satisfied for the current four-player Phoenix hanchan slice.
2. Start Step 4 as the next active stage, using Batch C as the first stronger QA-and-split checkpoint.
3. Decide the first real release policy for:
   - time-based splits
   - player leakage policy
   - ruleset cohort boundaries
   - sampling weights for later larger corpora
4. Keep replay-JSON oracles only as spot-check references for edge-case validation, not as a dependency for routine ingestion.
