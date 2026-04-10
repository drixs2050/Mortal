# Step 3 Plan: Tenhou Ingestion And Canonicalization

Date: 2026-03-28

## Purpose
This document starts Step 3 as a separate implementation stage.
Step 2 established policy, schema, and manifest direction.
Step 3 is where we turn Tenhou source records into a clean canonical dataset that Mortal can consume.

## Why Step 3 Starts With Tenhou
- Tenhou remains the strongest official format-study target for high-level riichi data.
- We now have both a raw XML sample and a saved official `mjlog2json` oracle for the same game.
- We have already confirmed that elite-game metadata does exist in the source path, but not all of it lives in the same raw field.

## What Is Already True Before Step 3
- `data/raw/tenhou/2026-03-28_sample_a/` contains:
  - one official replay XML sample
  - one saved official `mjlog2json` response for the same game
- The repo now supports manifest-driven training file lists.
- Step 2 docs define:
  - source policy
  - canonical normalized schema
  - conversion design
  - manifest and split rules

## Key Tenhou Findings To Carry Forward
- Ranking matches live in ranking lobby `0000` according to the official manual.
- Tenhou room class such as `一般`, `上級`, `特上`, and `鳳凰` is separate from the lobby number.
- Official replay JSON exposes:
  - raw `lobby`
  - human-readable `dan`
  - `rule.disp`
  - player `rate`
- Official replay client logic displays lobby values as:
  - `Lxxxx` for normal numbered lobbies
  - `Cxxxx` for custom lobbies when raw lobby id is greater than `10000`

Practical implication:
- We must preserve both raw `lobby` and rule/room metadata.
- Phoenix filtering should not be modeled as "lobby == X" without checking the room field.

## Step 3 Deliverables
1. A read-only Tenhou XML parser that extracts:
   - game identity
   - player metadata
   - raw room/lobby/rule metadata
   - per-round initialization
   - per-round event skeleton
2. A normalized-manifest row builder for Tenhou outputs.
3. A first minimal Tenhou-to-Mortal converter that emits `.json.gz` event logs for a supported subset of games.
4. Loader validation proving that converter output can be consumed by both:
   - `GameplayLoader`
   - `Grp`

## Recommended Implementation Sequence

### Phase 1. Read-only parse
Goal:
Parse Tenhou XML into manifest-ready metadata and compact round summaries.

Status:
- complete for the first saved sample through `scripts/tenhou_xml.py`
- exposed through `scripts/inspect_tenhou_xml.py`

Exit signal:
- we can inspect a raw sample without guessing at field meaning

### Phase 2. Manifest row shape
Goal:
Lock the first Tenhou normalized-manifest row fields before emitting converted logs.

Fields that must be present:
- `source`
- `source_game_id`
- `raw_snapshot_id`
- `year`
- `month`
- `lobby`
- `lobby_display`
- `go_type`
- `room`
- `ruleset`
- `player_dan`
- `player_rate`

Status:
- complete for the first normalized sample
- emitted through `scripts/ingest_tenhou_snapshot.py` into `data/manifests/normalized/v1/`

Exit signal:
- elite-game subsets can be selected from manifests without reading raw XML again

### Phase 3. Minimal converter
Goal:
Convert one supported Tenhou game into Mortal-ready `.json.gz` mjai events.

Suggested initial scope:
- four-player games only
- one validated rule family only
- complete games only
- no bulk processing yet

Status:
- complete for the first supported slice using saved official `mjlog2json` data
- current supported action set: normal draws/discards, `chi`, `pon`, `daiminkan`, `kakan`, `ankan`, `riichi`, `hora`, and `ryukyoku`
- current supported result slice also includes double ron, nagashi mangan, and common abortive-draw outcomes from the official sample page
- validated not only on the first replay sample, but also on official Tenhou sample-page JSONs for four kans, rinshan, chankan, double ron, nagashi mangan, triple ron, four winds, four riichi, and nine terminals
- backed by behavior-based regression tests in `tests/test_tenhou_converter.py`
- XML-only conversion is now implemented for the same supported four-player slice and matches the saved replay-JSON oracle on the full `2026-03-28_phoenix_batch_b` elite pilot
- current unsupported slice still includes `f`-encoded special actions and broader format coverage
- current research strongly suggests `f` is the sanma North-extraction branch, which is non-blocking for the current four-player pipeline unless a yonma counterexample appears

Exit signal:
- one sample log loads through both Mortal loaders

### Phase 4. Batch ingestion skeleton
Goal:
Scale from one converted game to a deterministic batch process.

Suggested outputs:
- normalized logs
- success manifest rows
- failure manifest rows
- summary counts

Status:
- started with `scripts/ingest_tenhou_snapshot.py`
- validated on `2026-03-28_sample_a` with one accepted sample and zero failures
- supported by `scripts/extract_tenhou_mjlog_samples.py` for repeatable official sample-page extraction into `data/raw/`
- validated on `2026-03-28_phoenix_batch_a` with five accepted real Phoenix-style replays and zero failures
- validated on `2026-03-28_phoenix_batch_b` with eight accepted real Phoenix-style replays and zero failures
- validated again on `2026-03-28_phoenix_batch_b` with `--converter-source xml`, eight accepted games, and zero failures in a repo-local temp run
- validated on `2026-03-28_phoenix_batch_c` with `36` accepted real Phoenix hanchan replays and zero failures
- raw staging for mixed replay batches is now supported by `scripts/stage_tenhou_reference_batch.py`
- official Phoenix raw-archive replay-id selection is now supported by `scripts/select_tenhou_scc_refs.py`
- deeper reference review is recorded in `plans/step_03_converter_reference_review.md`
- that review found and fixed `四槓散了` plus legacy wall-exhaust result handling
- that review also showed local third-party replay JSON can be lossy on post-kan dora, so it must not be treated as fully equivalent to official `mjlog2json`
- the first pilot QA artifact now exists at `data/manifests/normalized/v1/tenhou_phoenix_batch_b_v0.qa_summary.json`
- the split builder now supports `ruleset` and `go_type` filters so hanchan-only elite subsets can be derived from one manifest
- XML-only manifest rows now infer `room` and `ruleset` from `go_type`, so cohort filtering still works without replay-JSON oracles

Exit signal:
- small batch ingestion can run reproducibly on local samples

## Immediate Next Tasks
1. Treat the current four-player Phoenix hanchan ingestion slice as Step 3-complete enough for handoff.
2. Keep official `mjlog2json` oracles as validation references when available, but stop treating them as a runtime dependency for the supported yonma slice.
3. Keep derived training subsets ruleset-aware so `鳳南喰赤` and `鳳東喰赤速` are not mixed by accident.
4. Treat `f` as a future sanma-support task unless a real four-player replay proves it belongs in the current ingestion scope.
5. Move the active planning focus to `plans/step_04_dataset_qa_and_release_candidate_plan.md`.

## Guardrails
- Keep the default path single-source and single-GPU simple.
- Do not begin bulk collection yet.
- Preserve raw Tenhou values exactly whenever a human-readable mapping is still uncertain.
