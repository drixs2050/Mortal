# Step 2 Data Schema Spec

Date: 2026-03-28

## Purpose
This document defines the project's canonical data model for Step 2.
It covers both:
- raw source storage
- normalized training logs that the current Mortal codebase can actually load

The goal is to give every future converter one stable target.

## Key Repo Constraints
The current codebase already fixes several important choices:
- `mortal/train.py` and `mortal/dataloader.py` load `**/*.json.gz` files through `GameplayLoader`.
- `mortal/train_grp.py` loads the same file pattern through `Grp`.
- `GameplayLoader` parses line-delimited JSON events.
- `Grp` also parses line-delimited JSON events and requires valid end-of-kyoku score deltas.

Practical consequence:
- The canonical normalized format should stay "one gzipped line-delimited JSON event stream per complete game".
- File-level metadata that does not fit the current event schema should live in manifests, not inside ad hoc event fields.

## Canonical Storage Layout

### Raw source area
Use `data/raw/` for immutable source snapshots:

```text
data/raw/<source>/<snapshot_id>/...
```

Examples:
- `data/raw/tenhou/2026-03-28_sample_a/...`
- `data/raw/private_scrim/2026-04-02_event01/...`

Rules:
- Never edit raw files in place after import.
- Record provenance in a manifest alongside the snapshot.
- Keep source-specific formats exactly as obtained.

### Normalized area
Use `data/normalized/` for canonical Mortal-ready logs:

```text
data/normalized/v1/source=<source>/year=<YYYY>/month=<MM>/<source_game_id>.json.gz
```

Examples:
- `data/normalized/v1/source=tenhou/year=2026/month=03/2026032801gm-0009-0000-abcdef01.json.gz`
- `data/normalized/v1/source=private_scrim/year=2026/month=04/event01_table3_hanchan02.json.gz`

Rules:
- One file equals one complete four-player game.
- Do not bundle multiple games into one `.json.gz` file.
- Use deterministic filenames derived from source game identity.

## Raw Data Model
Raw data is source-specific and intentionally not standardized at the file-content level.
Instead, Step 2 standardizes the minimum metadata we must track for each raw snapshot:

- `source`
- `snapshot_id`
- `acquired_at`
- `acquired_by`
- `official_access_path`
- `usage_status`
- `format_notes`
- `file_count`
- `sha256` or per-file hashes when practical

This metadata belongs in manifests under `data/manifests/`, not inside raw log files.

## Canonical Normalized File Contract

### File type
- Extension: `.json.gz`
- Compression: gzip
- Content: UTF-8 line-delimited JSON
- One logical JSON record per line

### Record type
The normalized event stream should be compatible with `libriichi`'s current mjai-style `Event` enum, with optional metadata tolerated where the parser already allows it.

The supported event families today are:
- `start_game`
- `start_kyoku`
- `tsumo`
- `dahai`
- `chi`
- `pon`
- `daiminkan`
- `kakan`
- `ankan`
- `dora`
- `reach`
- `reach_accepted`
- `hora`
- `ryukyoku`
- `end_kyoku`
- `end_game`

### Required ordering
At minimum, each normalized file must follow this structure:
1. `start_game`
2. zero or more per-kyoku blocks, where each block starts with `start_kyoku`
3. an `end_kyoku` after each completed kyoku
4. `end_game` at the end

If the file does not represent a complete finished game, it should not be emitted into the canonical normalized corpus.

## Required Fields By Event

### `start_game`
Required now:
- `type = "start_game"`
- `names`: array of 4 player names or stable player labels

Optional now:
- `seed`

Notes:
- `GameplayLoader` uses `names` for filtering by included or excluded players.
- If source policy later requires anonymization, stable aliases may be used here, with richer identity metadata moved to manifests.

### `start_kyoku`
Required:
- `bakaze`
- `dora_marker`
- `kyoku`
- `honba`
- `kyotaku`
- `oya`
- `scores`
- `tehais`

Why all of these matter:
- `GameplayLoader` reconstructs legal states from the event stream.
- `Grp` derives per-kyoku features from round context and scores.
- Missing `tehais` or score state makes the log unusable for current loaders.

### Action events
Required according to the event type:
- actor identity
- target when the action is a call/win against another player
- tile fields such as `pai` and `consumed`
- `tsumogiri` for `dahai`

### `reach_accepted`
Required:
- `actor`

Why it matters:
- `Grp` adjusts final deltas for accepted riichi sticks while reconstructing rank labels.

### `hora`
Required:
- `actor`
- `target`
- `deltas`

Optional:
- `ura_markers`

Why `deltas` must be present:
- `Grp` requires them to recover final scores and final ranks.

### `ryukyoku`
Required:
- `deltas`

Why it matters:
- Same reason as `hora`: GRP reconstruction depends on valid end-of-kyoku score movement.

### `end_kyoku`
Required:
- event marker only

### `end_game`
Required:
- event marker only

## Optional Metadata Inside Events
The fixture logs produced in Step 1 include action-level `meta` fields such as:
- `q_values`
- `mask_bits`
- `is_greedy`
- `batch_size`
- `eval_time_ns`
- `shanten`
- `at_furiten`

Current rule:
- Optional action metadata is allowed for debugging or analysis.
- Training correctness must not depend on it.
- Source, room, ruleset, and converter provenance should still live in manifests unless and until the event schema is deliberately extended.

## Manifest-Backed File Metadata
Because the current event schema is intentionally narrow, the following metadata should be stored in manifests keyed by relative normalized path:
- `source`
- `source_game_id`
- `game_date` when known
- `acquired_at`
- `ruleset`
- `room` or lobby tier
- `table_size`
- `red_fives`
- `player_identities` or hashed IDs when allowed
- `converter_version`
- `raw_snapshot_id`
- `validation_status`

## Validation Rules For Canonical Normalized Logs
Before a normalized file is accepted:
- it must load through `GameplayLoader`
- it must load through `Grp`
- it must contain at least one `start_kyoku`
- every finished kyoku must end with `end_kyoku`
- the full game must end with `end_game`
- `hora` and `ryukyoku` must contain valid `deltas`
- the file must represent a supported four-player ruleset

## Out Of Scope For V1
Do not block Step 2 on these:
- embedding all file metadata directly into the event stream
- multi-game archive containers
- three-player support
- exotic rule variants outside the current Tenhou-style four-player target

## Immediate Implementation Guidance
For the first real converter target:
- keep normalized output strictly compatible with the existing `Event` schema
- keep one-file-per-game
- store richer provenance in manifests
- reject partial or ambiguous games instead of inventing fields on the fly
