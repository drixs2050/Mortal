# Step 4B Batch D Execution Plan

Date: 2026-03-28

## Purpose
Turn Step 4B from a strategic stage into the first concrete scaled corpus run.

Batch D is the first larger post-`RC0` sweep that should:
- reuse the exact Batch C cohort definition
- exercise the same XML-only ingestion path on a meaningfully larger archive span
- stay small enough to debug if new failure categories appear

This is still a scale-up checkpoint, not the final training-scale release.

## Batch D Definition

Snapshot id:
- `2026-03-28_phoenix_batch_d`

Dataset id:
- `tenhou_phoenix_batch_d_v0`

Scope:
- source: `tenhou`
- room: `鳳`
- ruleset: `鳳南喰赤`
- go type: `169`
- four-player only
- all players `>= 七段`

Archive sweep:
- start date: `2026-01-01`
- end date: `2026-01-31`
- target archive count: `31`
- selection cap: `12` replays per archive

Expected selection target:
- `372` candidate replay ids if every archive is present and yields at least `12` qualifying rows

## Why This Is The Right First 4B Step
- It expands from `3` archive days to `31`, which is enough to surface scale-related issues without jumping into an unbounded crawl.
- It keeps the exact same Phoenix hanchan slice as Batch C, so comparison is clean.
- It gives a full calendar month for chronological splits, which is already much more meaningful than `RC0`.
- It is still small enough that manual QA spot checks remain realistic if the converter hits a new unsupported branch.

## Output Paths

Archive fetch dir:
- `/tmp/tenhou_scc_batch_d/`

Archive manifest files:
- `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.archives.txt`
- `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.archives.json`

Selection files:
- `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.refs.txt`
- `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.selection.json`

Raw snapshot outputs:
- `data/raw/tenhou/2026-03-28_phoenix_batch_d/`
- `data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.json`

Normalized outputs:
- `data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.jsonl`
- `data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.summary.json`
- `data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.qa_summary.json`
- `data/manifests/failures/tenhou/tenhou_phoenix_batch_d_v0__2026-03-28_phoenix_batch_d.jsonl`

Split outputs:
- `data/splits/v1/tenhou_phoenix_batch_d_v0/phoenix_all_7dan_all/`
- `data/splits/v1/tenhou_phoenix_batch_d_v0/phoenix_hanchan_7dan_all/`

## Batch D Commands

Run from the repo root.

### Optional One-Command Wrapper

If you want to run fetch, select, stage, and ingest as one command:

```bash
python scripts/run_tenhou_pipeline.py \
  --snapshot-id 2026-03-28_phoenix_batch_d \
  --dataset-id tenhou_phoenix_batch_d_v0 \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --archive-dir /tmp/tenhou_scc_batch_d \
  --ruleset 四鳳南喰赤－ \
  --max-per-archive 12 \
  --limit 372 \
  --usage-status corpus-expansion-batch-d \
  --converter-version tenhou-xml-v0 \
  --converter-source xml \
  --overwrite
```

Add `--with-release-artifacts` if you also want QA, split generation, and overlap reporting in the same wrapper run.

### 1. Fetch daily archive files

```bash
python scripts/fetch_tenhou_scc_archives.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --output-dir /tmp/tenhou_scc_batch_d \
  --output-list data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.archives.txt \
  --summary data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.archives.json
```

### 2. Select Phoenix hanchan replay ids

```bash
python scripts/select_tenhou_scc_refs.py \
  --archive-list data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.archives.txt \
  --ruleset 四鳳南喰赤－ \
  --max-per-archive 12 \
  --limit 372 \
  --output-refs data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.refs.txt \
  --output-summary data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.selection.json
```

### 3. Stage the XML raw snapshot

```bash
PYTHONPATH=/home/drixs2050/Documents/Mortal/mortal \
python scripts/stage_tenhou_reference_batch.py \
  --snapshot-id 2026-03-28_phoenix_batch_d \
  --refs-file data/manifests/raw/tenhou/2026-03-28_phoenix_batch_d.refs.txt \
  --usage-status corpus-expansion-batch-d \
  --overwrite
```

### 4. Ingest the staged snapshot

```bash
PYTHONPATH=/home/drixs2050/Documents/Mortal/mortal \
python scripts/ingest_tenhou_snapshot.py \
  --raw-snapshot-id 2026-03-28_phoenix_batch_d \
  --dataset-id tenhou_phoenix_batch_d_v0 \
  --converter-version tenhou-xml-v0 \
  --converter-source xml \
  --overwrite
```

### 5. Build the normalized QA summary

```bash
python scripts/summarize_normalized_manifest.py \
  --manifest data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.jsonl \
  --output data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.qa_summary.json
```

### 6. Build the all-Phoenix split family

```bash
python scripts/build_dataset_splits.py \
  --manifest data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.jsonl \
  --output-dir data/splits/v1/tenhou_phoenix_batch_d_v0/phoenix_all_7dan_all \
  --source tenhou \
  --room 鳳 \
  --min-player-dan 16 \
  --player-threshold-mode all
```

### 7. Build the Phoenix hanchan split family

```bash
python scripts/build_dataset_splits.py \
  --manifest data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.jsonl \
  --output-dir data/splits/v1/tenhou_phoenix_batch_d_v0/phoenix_hanchan_7dan_all \
  --source tenhou \
  --room 鳳 \
  --ruleset 鳳南喰赤 \
  --go-type 169 \
  --min-player-dan 16 \
  --player-threshold-mode all
```

### 8. Build the split overlap report

```bash
python scripts/summarize_split_overlap.py \
  --manifest data/manifests/normalized/v1/tenhou_phoenix_batch_d_v0.jsonl \
  --split-dir data/splits/v1/tenhou_phoenix_batch_d_v0/phoenix_hanchan_7dan_all \
  --output data/splits/v1/tenhou_phoenix_batch_d_v0/phoenix_hanchan_7dan_all/player_overlap_summary.json
```

## Batch D Review Checklist
- Did the archive fetch succeed for most or all `31` dates?
- How many candidate refs were selected versus the `372` target?
- Are accepted rows still pure on:
  - `room = 鳳`
  - `ruleset = 鳳南喰赤`
  - `go_type = 169`
- Did new failure categories appear during ingestion?
- Is the failure rate still low enough that the current converter slice is trustworthy for larger sweeps?
- Does player leakage remain measured and bounded enough for chronological pilot evaluation?

## Exit Signal
Batch D is successful if:
- archive selection and XML staging work cleanly over the full January sweep
- ingestion remains mostly or fully successful
- the QA report preserves cohort purity
- the split outputs and overlap report are generated without special-case fixes

If Batch D succeeds, the next 4B move should be a wider Batch E style sweep over multiple months using the exact same commands and artifact structure.
