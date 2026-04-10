# Step 3 Phoenix Batch B Validation

Date: 2026-03-28

## Purpose
Record the first expanded pilot QA checkpoint for real four-player elite Tenhou ingestion.

Batch B is the next step after Batch A:
- it uses the full local elite replay reference set that is currently ingestible
- it is large enough to produce the first real split/file-list outputs
- it exposes one useful curation lesson: not all Phoenix-room games belong in the same training subset

## Raw Snapshot
- Raw snapshot id: `2026-03-28_phoenix_batch_b`
- Raw manifest: `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_b.json`
- Ref list: `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_b.refs.txt`

Selection profile:
- four-player only
- ranking lobby `0000`
- room display `鳳`
- tables made entirely of `七段` to `九段`

Selected replay ids:
- `2022013100gm-00a9-0000-af91b2de`
- `2022080600gm-00a9-0000-06406b7f`
- `2022080600gm-00a9-0000-b8ad3aee`
- `2022080601gm-00a9-0000-e3595545`
- `2022080818gm-00a9-0000-6c4ec7d1`
- `2022081017gm-00e1-0000-2df24853`
- `2022081121gm-00a9-0000-372fcc17`
- `2022081318gm-00a9-0000-6c91213c`

## Staging Method
- XML files were fetched from the official replay path:
  - `https://tenhou.net/0/log/?<source_game_id>`
- Matching replay JSON oracles were copied from the local `riichi` reference set:
  - `/tmp/riichi-src/riichi-0.1.0/data/t6-samples/random-games/`

Helper used:
- `scripts/stage_tenhou_reference_batch.py`

## Ingestion Run
Command:

`/home/drixs2050/anaconda3/envs/mortal/bin/python scripts/ingest_tenhou_snapshot.py --raw-snapshot-id 2026-03-28_phoenix_batch_b --dataset-id tenhou_phoenix_batch_b_v0 --overwrite`

Produced outputs:
- normalized manifest:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_b_v0.jsonl`
- normalized summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_b_v0.summary.json`
- normalized QA summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_b_v0.qa_summary.json`
- failure manifest:
  - `data/manifests/failures/tenhou/tenhou_phoenix_batch_b_v0__2026-03-28_phoenix_batch_b.jsonl`

## Result
- candidate count: `8`
- accepted count: `8`
- rejected count: `0`
- failure categories: none

This confirms that the current converter slice is stable on the full local elite reference set we currently have available.

## XML-Only Follow-Up
After the first oracle-backed run, the same eight-game batch was re-run through the XML-only path:

Command:

`PYTHONPATH=/home/drixs2050/Documents/Mortal/mortal /home/drixs2050/anaconda3/envs/mortal/bin/python scripts/ingest_tenhou_snapshot.py --raw-snapshot-id 2026-03-28_phoenix_batch_b --dataset-id tenhou_phoenix_batch_b_xml_v0_tmp --converter-version tenhou-xml-v0 --converter-source xml --normalized-root artifacts/tmp/xml_norm --manifest-root artifacts/tmp/xml_manifest --overwrite`

Result:
- candidate count: `8`
- accepted count: `8`
- rejected count: `0`
- failure categories: none

Supporting validation:
- `tests/test_tenhou_converter.py` now includes an XML-vs-oracle equivalence check across all eight replay ids in Batch B
- the full converter regression suite passes after the XML-only parity fixes

Practical implication:
- for the currently supported four-player slice, replay JSON is now a validation oracle rather than a hard runtime dependency
- the main remaining ceiling is batch size and coverage breadth, not the converter source path

## QA Summary
From `data/manifests/normalized/v1/tenhou_phoenix_batch_b_v0.qa_summary.json`:

- room counts:
  - `鳳`: `8`
- lobby counts:
  - `0`: `8`
- ruleset counts:
  - `鳳南喰赤`: `7`
  - `鳳東喰赤速`: `1`
- go type counts:
  - `169`: `7`
  - `225`: `1`
- dan label counts:
  - `七段`: `17`
  - `八段`: `12`
  - `九段`: `3`
- player rate range:
  - min `2008.89`
  - max `2288.83`
  - mean `2153.141875`
- total kyoku count:
  - `79`
- total normalized event count:
  - `7718`
- game date range:
  - `2022-01-31T00:00:00` to `2022-08-13T18:00:00`

## Split Outputs
All-elite Phoenix split:
- output dir:
  - `data/splits/v1/tenhou_phoenix_batch_b_v0/phoenix_all_7dan_all/`
- filters:
  - `source = tenhou`
  - `room = 鳳`
  - `min_player_dan = 16`
  - `player_threshold_mode = all`
- counts:
  - train `6`
  - val `1`
  - test `1`

Hanchan-only elite Phoenix split:
- output dir:
  - `data/splits/v1/tenhou_phoenix_batch_b_v0/phoenix_hanchan_7dan_all/`
- filters:
  - `source = tenhou`
  - `room = 鳳`
  - `ruleset = 鳳南喰赤`
  - `go_type = 169`
  - `min_player_dan = 16`
  - `player_threshold_mode = all`
- counts:
  - train `5`
  - val `1`
  - test `1`

## Practical Meaning
This batch is the first point where the ingestion path is doing more than raw conversion:
- we now have a small but real elite pilot corpus
- we now have a machine-readable QA checkpoint
- we now have curated file lists instead of only a manifest

This also exposed one important dataset-policy lesson:
- Phoenix room alone is not enough as a final training cohort definition
- ruleset and `go_type` matter too
- we should not blindly mix `鳳東喰赤速` into a `鳳南喰赤` hanchan baseline

## Remaining Limits
- The pilot still remains smaller than the target `20` to `50` game validation cohort.
- XML-only conversion is validated on the current supported four-player slice, but broader raw coverage still needs more real elite games before Step 3 can close.
- A clean `8/8` pilot is encouraging, but still not enough to declare Step 3 complete.

## Recommended Next Work
1. Grow beyond the local eight-game ceiling to a `20` to `50` game elite cohort using the XML-only path.
2. Keep future pilot subsets ruleset-aware so `鳳東喰赤速` and `鳳南喰赤` are not mixed accidentally.
3. Re-run the same QA and split workflow on that broader elite cohort.
4. Keep replay JSON oracles when available for diff-based validation, especially on kan-heavy edge cases.
5. Only then decide whether Step 3 is ready to hand off to Step 4.
