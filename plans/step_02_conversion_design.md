# Step 2 Conversion Design

Date: 2026-03-28

## Purpose
This document defines the first converter design for Step 2.
It does not implement a parser yet.
It defines the stages, failure modes, and first-source mapping rules so that later code work can be deliberate.

## First Converter Target
Use Tenhou as the first technical format-study target.

Why Tenhou first:
- official replay viewing and replay download paths exist
- replay data is already a common reference point in riichi tooling
- Mortal itself targets Tenhou-style four-player rules
- it is the strongest current candidate for a technically workable action-level corpus

Important limitation:
- Step 2 still treats Tenhou as policy-sensitive.
- The first implementation goal is a tiny sample converter, not bulk backfill.

## Conversion Pipeline Stages

### Stage 1. Raw acquisition
Input:
- official replay download or officially reachable replay source

Output:
- immutable raw snapshot in `data/raw/<source>/<snapshot_id>/`

Also write manifest metadata:
- acquisition date
- source path
- source status
- any notes about policy or account context

### Stage 2. Source parse
Input:
- source-specific raw file

Output:
- source-native in-memory game model

Responsibilities:
- parse source identifiers
- parse player info
- parse per-kyoku boundaries
- parse actions, calls, riichi, dora, win, and draw outcomes
- extract enough score movement to recover `deltas`

### Stage 3. Normalize to Mortal event stream
Input:
- parsed source-native game model

Output:
- canonical line-delimited mjai-style event stream compatible with `libriichi`

Responsibilities:
- emit `start_game`
- emit one `start_kyoku` per kyoku with full initial state
- emit action events in gameplay order
- emit `hora` or `ryukyoku` with score deltas
- emit `end_kyoku`
- emit `end_game`

### Stage 4. Validation
Every normalized output candidate must:
- load through `GameplayLoader`
- load through `Grp`
- pass structural checks
- pass source-specific sanity checks

### Stage 5. Write output and manifests
Accepted outputs:
- normalized `.json.gz` file in `data/normalized/v1/...`
- manifest row in `data/manifests/...`

Rejected outputs:
- original raw file stays in `data/raw/...`
- failure is logged in a converter report

## Error Categories
Every rejected or skipped game should fall into one of these buckets:
- `policy_blocked`
- `malformed_raw`
- `partial_game`
- `unsupported_ruleset`
- `unsupported_table_size`
- `missing_required_scores`
- `duplicate_game`
- `normalization_error`
- `loader_validation_failed`

This is important because Step 2 needs auditable QA, not just a success count.

## Tenhou Mapping Notes
The exact attribute-level parser should be confirmed against a captured raw sample before coding.
That said, the first-pass mapping is clear enough to define now.

### Source-side concepts we need
From each raw Tenhou replay, the converter must recover:
- source game id
- player names
- ruleset / lobby information when available
- per-kyoku initial state
- all draws and discards in order
- all calls and kan variants
- riichi declaration and acceptance
- dora reveals
- win or draw outcome with point movement

### Expected normalized mapping
Map source concepts into Mortal events as follows:
- player identity and game start -> `start_game`
- kyoku initialization -> `start_kyoku`
- draw -> `tsumo`
- discard -> `dahai`
- chi / pon / kan calls -> `chi`, `pon`, `daiminkan`, `kakan`, `ankan`
- dora reveal -> `dora`
- riichi declaration -> `reach`
- riichi acceptance / stick payment -> `reach_accepted`
- agari -> `hora`
- exhaustive or abortive draw -> `ryukyoku`
- kyoku boundary -> `end_kyoku`
- game boundary -> `end_game`

### Information that should stay in manifests
Do not force these into the event stream in V1:
- Tenhou game id
- source URL or account path
- lobby tier
- acquisition timestamp
- converter version
- raw snapshot identifier

Keep them in manifests keyed by normalized relative path.

## First-Pass Converter Scope
The first actual converter implementation should support only:
- four-player games
- standard Tenhou-style riichi rules close to Mortal's current assumptions
- complete finished games
- sources with enough information to reconstruct `deltas`

The first implementation should reject:
- three-player games
- obviously malformed replays
- incomplete logs
- rule variants that the current loader or game engine does not reliably match

## Normalization Invariants
No normalized file is valid unless all of these are true:
- one file contains exactly one complete game
- the first event is `start_game`
- every kyoku begins with `start_kyoku`
- `hora` and `ryukyoku` include usable `deltas`
- every kyoku ends with `end_kyoku`
- the file ends with `end_game`
- player ordering stays consistent with the source log

## Minimal Converter Outputs
For each run, the converter should emit:
- normalized files
- a success manifest
- a failure manifest
- summary counts by error category

## Suggested First Implementation Sequence
1. Capture one tiny Tenhou sample into `data/raw/tenhou/...`.
2. Write a read-only source inspection note before coding the parser.
3. Implement a parser that converts only one supported ruleset path.
4. Validate against `GameplayLoader` and `Grp`.
5. Only then widen coverage and think about batch ingestion.

## Deferred Source Work
Do not start here yet:
- Mahjong Soul converter implementation
- third-party corpus importers
- large historical backfill
- platform automation
