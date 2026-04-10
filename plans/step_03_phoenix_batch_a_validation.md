# Step 3 Phoenix Batch A Validation

Date: 2026-03-28

## Purpose
Record the first real four-player elite Tenhou mixed-batch ingestion run.

This note is separate from the single-sample validation note because this batch is:
- real ranking-lobby replay data
- Phoenix-room metadata confirmed through the saved replay JSON oracles
- large enough to test the batch ingestion path instead of only one-off conversion

## Raw Snapshot
- Raw snapshot id: `2026-03-28_phoenix_batch_a`
- Raw manifest: `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_a.json`
- Ref list: `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_a.refs.txt`

Selection profile:
- four-player only
- ranking lobby `0000`
- room display `鳳`
- player tables made of `七段` to `九段`

Selected replay ids:
- `2022013100gm-00a9-0000-af91b2de`
- `2022080600gm-00a9-0000-06406b7f`
- `2022080600gm-00a9-0000-b8ad3aee`
- `2022080818gm-00a9-0000-6c4ec7d1`
- `2022081017gm-00e1-0000-2df24853`

## Staging Method
- XML files were fetched from the official replay path:
  - `https://tenhou.net/0/log/?<source_game_id>`
- Matching replay JSON oracles were copied from the local `riichi` reference set:
  - `/tmp/riichi-src/riichi-0.1.0/data/t6-samples/random-games/`

The helper used for this staging run is:
- `scripts/stage_tenhou_reference_batch.py`

## Ingestion Run
Command:

`/home/drixs2050/anaconda3/envs/mortal/bin/python scripts/ingest_tenhou_snapshot.py --raw-snapshot-id 2026-03-28_phoenix_batch_a --dataset-id tenhou_phoenix_batch_a_v0 --overwrite`

Produced outputs:
- normalized manifest:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_a_v0.jsonl`
- normalized summary:
  - `data/manifests/normalized/v1/tenhou_phoenix_batch_a_v0.summary.json`
- failure manifest:
  - `data/manifests/failures/tenhou/tenhou_phoenix_batch_a_v0__2026-03-28_phoenix_batch_a.jsonl`

## Result
- candidate count: `5`
- accepted count: `5`
- rejected count: `0`
- failure categories: none

This means the first real Phoenix-style batch did not require manual cleanup or converter patching to ingest.

## Accepted Games
- `2022013100gm-00a9-0000-af91b2de`
  - ruleset `鳳南喰赤`
  - room `鳳`
  - dan labels `七段 / 七段 / 八段 / 八段`
  - normalized events `1212`
  - kyoku count `12`
- `2022080600gm-00a9-0000-06406b7f`
  - ruleset `鳳南喰赤`
  - room `鳳`
  - dan labels `八段 / 九段 / 七段 / 七段`
  - normalized events `1097`
  - kyoku count `12`
- `2022080600gm-00a9-0000-b8ad3aee`
  - ruleset `鳳南喰赤`
  - room `鳳`
  - dan labels `九段 / 七段 / 八段 / 七段`
  - normalized events `1162`
  - kyoku count `12`
- `2022080818gm-00a9-0000-6c4ec7d1`
  - ruleset `鳳南喰赤`
  - room `鳳`
  - dan labels `七段 / 八段 / 九段 / 八段`
  - normalized events `738`
  - kyoku count `8`
- `2022081017gm-00e1-0000-2df24853`
  - ruleset `鳳東喰赤速`
  - room `鳳`
  - dan labels `七段 / 七段 / 七段 / 八段`
  - normalized events `405`
  - kyoku count `4`

## Loader Validation
Every accepted game passed:
- `GameplayLoader`
- `Grp`

The normalized manifest rows record:
- `room = 鳳`
- `lobby = 0`
- `ranking_lobby = true`
- per-table dan labels
- per-file loader validation details

## Practical Meaning
This batch is stronger evidence than the earlier sample-page and single-sample work because it confirms:
- the ingestion path works on real four-player ranking-lobby Phoenix-style data
- elite-game metadata survives into the normalized manifest rows
- the current converter slice is already enough for at least one small real cohort without special-case repair

## Remaining Limits
- The batch is still tiny and should be treated as a pilot cohort, not a release dataset.
- The pipeline still depends on saved replay JSON oracles rather than XML-only reconstruction.
- Zero failures on five games is encouraging, but not enough to declare failure categories stable.

## Recommended Next Work
1. Expand from `5` replays to a broader real batch, such as `20` to `50`, while keeping the same four-player elite scope.
2. Generate the first pilot QA summary and split/file-list outputs from the resulting normalized manifest.
3. Only then decide whether Step 3 can be closed and Step 4 can become the active stage.
