# Step 3 Sample Conversion Validation

Date: 2026-03-28

## Purpose
Record the first end-to-end Tenhou ingestion milestone:
- saved raw sample
- read-only parse
- minimal conversion
- loader validation
- normalized manifest output

## Inputs
- Raw snapshot id: `2026-03-28_sample_a`
- XML source: `data/raw/tenhou/2026-03-28_sample_a/2012060420gm-0009-10011-acfd4b57.xml`
- Official replay JSON: `data/raw/tenhou/2026-03-28_sample_a/2012060420gm-0009-10011-acfd4b57.mjlog2json.json`

## Implemented Tools
- `scripts/tenhou_xml.py`
  - read-only XML parsing
  - normalized manifest-row builder
  - XML-to-mjai conversion helpers
  - official-JSON-to-mjai conversion helpers
- `scripts/inspect_tenhou_xml.py`
  - summary and manifest-row inspection CLI
- `scripts/convert_tenhou_to_mjai.py`
  - direct single-file conversion CLI
- `scripts/ingest_tenhou_snapshot.py`
  - snapshot ingestion skeleton with normalized outputs and manifests
- `scripts/extract_tenhou_mjlog_samples.py`
  - extract official sample-page JSON payloads into `data/raw/` plus a raw manifest
- `tests/test_tenhou_converter.py`
  - behavior-based conversion and loader regression tests using saved official Tenhou samples

## Produced Outputs
- Normalized sample log:
  - `data/normalized/v1/source=tenhou/year=2012/month=06/2012060420gm-0009-10011-acfd4b57.json.gz`
- Normalized dataset manifest:
  - `data/manifests/normalized/v1/tenhou_official_json_sample_v0.jsonl`
- Failure manifest:
  - `data/manifests/failures/tenhou/tenhou_official_json_sample_v0__2026-03-28_sample_a.jsonl`
- Summary:
  - `data/manifests/normalized/v1/tenhou_official_json_sample_v0.summary.json`
- Official sample-page raw snapshot:
  - `data/raw/tenhou/2026-03-28_editor_samples_b/`
- Official sample-page raw manifest:
  - `data/manifests/raw/tenhou/2026-03-28_editor_samples_b.json`

## Validation Result
- The minimal converter produced `1352` normalized events for the sample game.
- `GameplayLoader` accepted the normalized file and returned `4` player views.
- Per-player action counts were:
  - `203`
  - `208`
  - `211`
  - `199`
- `Grp` accepted the normalized file and produced:
  - feature shape `14 x 7`
  - one game summary
- Snapshot ingestion summary:
  - candidate count `1`
  - accepted count `1`
  - rejected count `0`

## Expanded Sample Coverage
The widened converter was also validated against official Tenhou sample-page JSON payloads for:

- `four_kans`
  - converted successfully
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output
  - event coverage included `daiminkan`, `kakan`, `ankan`, and repeated `dora`
- `rinshan`
  - converted successfully
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output
- `chankan`
  - converted successfully
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output
- `double_ron`
  - converted successfully
  - emits two `hora` events before `end_kyoku`
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output
- `nagashi_mangan`
  - converted successfully
  - normalized as `ryukyoku` with nonzero deltas
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output
- `triple_ron`
  - converted successfully
  - normalized as zero-delta `ryukyoku`
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output
- `four_winds_draw`
  - converted successfully
  - normalized as zero-delta `ryukyoku`
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output
- `four_riichi_draw`
  - converted successfully
  - preserves four `reach` plus four `reach_accepted` events
  - normalized as zero-delta `ryukyoku`
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output
- `nine_terminals_draw`
  - converted successfully
  - normalized as zero-delta `ryukyoku`
  - `GameplayLoader` accepted the output
  - `Grp` accepted the output

This means the current Step 3 converter slice is no longer limited to:
- plain draw-discard games
- one single sample replay
- one single kan-free path

## Regression Test Command
Run:

`PYTHONPATH=/home/drixs2050/Documents/Mortal/mortal /home/drixs2050/anaconda3/envs/mortal/bin/python -m unittest tests.test_tenhou_converter -v`

Current regression coverage:
- `four_kans`
  - verifies `daiminkan`, `kakan`, `ankan`, and the expected dora-reveal sequence
- `rinshan`
  - verifies `ankan -> tsumo -> dora -> hora`
- `chankan`
  - verifies `kakan -> hora`
- `double_ron`
  - verifies two `hora` events and riichi-tied ura marker handling
- `nagashi_mangan`
  - verifies `ryukyoku` with nonzero deltas
- `triple_ron`, `four_winds_draw`, `four_riichi_draw`, `nine_terminals_draw`
  - verify normalization into zero-delta `ryukyoku`
- `four_riichi_draw`
  - verifies four `reach` and four `reach_accepted` events
- all committed supported samples
  - load through both `GameplayLoader` and `Grp`

## Supported Slice
Current minimal converter support is intentionally narrow:
- four-player complete games
- XML-only conversion for the supported slice
- replay JSON oracle available when we want diff-based validation
- standard draws and discards
- `chi`
- `pon`
- `daiminkan`
- `kakan`
- `ankan`
- `riichi`
- `hora`
- `ryukyoku`
- double ron
- nagashi mangan
- common abortive-draw outcomes from the official sample page

## Known Limits
- The validated sample is a custom-lobby game, not a Phoenix-room proof sample.
- Bulk ingestion QA still needs more varied samples before we treat failure categories as stable.
- `f`-encoded special actions are still not implemented or validated.
- Current research indicates Tenhou `f` is likely the sanma North-extraction branch, not a missing yonma core action.
- Because the active ingestion target is four-player hanchan data, `f` should be treated as a tracked out-of-scope path unless a real four-player counterexample is found.
- A deeper review found that some third-party replay JSON corpora can be lossy on post-kan dora indicators, so they are not a drop-in substitute for official `mjlog2json` in edge-case validation.
- Failure-path regression tests for unsupported cases are still missing.

## Recommended Next Work
1. Capture a real official `f` sample only if we decide to support sanma later, or if a four-player replay proves `f` matters for the current pipeline.
2. Capture a few more real replay samples with stronger-room metadata, not only editor-page format samples.
3. Add explicit failure-path regression fixtures for unsupported replay encodings, including `f` if we obtain a stable official example.
4. Run the ingestion skeleton on that mixed mini-batch and inspect the resulting failure rows before scaling up.
