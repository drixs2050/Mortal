# Step 2 Tenhou Sample Inspection

Date: 2026-03-28

## Purpose
This note records the first real Tenhou source inspection for Step 2.
The goal is to confirm, using one official sample replay, whether raw Tenhou logs expose the metadata and event structure we need for:
- future conversion into Mortal's normalized event format
- filtering for high-level games such as Phoenix-room and high-dan matches

## Sample Captured
- Source: official Tenhou replay sample
- Source game id: `2012060420gm-0009-10011-acfd4b57`
- Source URL: `https://tenhou.net/0/log/?2012060420gm-0009-10011-acfd4b57`
- Local raw snapshot: `data/raw/tenhou/2026-03-28_sample_a/2012060420gm-0009-10011-acfd4b57.xml`
- SHA256: `0b32ddbf739e66167e0d2227fcee2cb9a3146abbdee6039da247b92cde91f4b7`
- Size: `18,813` bytes

## Raw Fields Confirmed Present
From the sample XML header:
- `GO type="9" lobby="10011"`
- `UN ... dan="0,0,18,0" rate="1500.00,1500.00,2232.65,1500.00" sx="F,M,M,F"`
- percent-encoded player names in `n0`..`n3`

Practical meaning:
- raw Tenhou logs do expose lobby metadata
- raw Tenhou logs do expose player-level dan metadata
- raw Tenhou logs do expose player-level rate metadata

This is enough to preserve elite-game eligibility in manifests before conversion.

Important caution:
- this sample confirms the raw fields exist
- it does not, by itself, prove the human-readable mapping from raw lobby or dan codes to labels like "Phoenix" or "10-dan"

For Step 2, the safe rule is:
- preserve raw `lobby`, `type`, `dan`, and `rate` values exactly as observed
- add human-readable labels only after we verify the mapping separately

## Official Verification Follow-Up
Using the official replay viewer's `mjlog2json` response for the same game, we now also confirmed:
- the raw dan ids in this sample render as `新人`, `新人`, `九段`, `新人`
- the inferred dan ladder order matches the official viewer output for this sample
- the rule display for this game is `般南喰赤`

Important correction:
- this sample is not a Phoenix-room ranking-lobby example
- Tenhou's official client logic displays raw lobby `10011` as custom lobby `C0011`
- the official manual says ranking matches live in ranking lobby `0000`

Practical implication:
- Phoenix filtering must not be modeled as a pure lobby-number filter
- we must preserve both raw lobby metadata and separate room/rule metadata

## Event Structure Confirmed Present
The sample includes these important source events:
- `INIT`
- draw and discard tags such as `T*`, `D*`, `U*`, `E*`, `V*`, `F*`, `W*`, `G*`
- `N` call tags
- `REACH`
- `AGARI`
- `RYUUKYOKU`

Practical meaning:
- the raw source contains enough action-level detail to reconstruct gameplay order
- the raw source contains win and draw outcomes with score movement
- the raw source is a plausible first converter target for Mortal

## Fit Against Mortal's Current Loaders
Mortal's current normalized path needs:
- one complete game per file
- line-delimited JSON mjai-style events
- full per-kyoku initialization
- valid end-of-kyoku score deltas

The Tenhou sample confirms the raw source appears to contain:
- game identity
- per-player metadata
- per-kyoku initialization
- action sequence
- enough score movement to derive normalized `hora` and `ryukyoku` deltas

Conclusion:
- the proposed Step 2 data organization fits the current training framework
- the remaining work is source conversion and metadata wiring, not a schema redesign

## What Still Needs Verification
- exact mapping of Tenhou raw room/rule codes for ranking-lobby subsets such as Phoenix
- any additional `GO type` rule bits that should be preserved in manifests beyond the raw integer and official display string

## Step 2 Implication
We should keep these fields in normalized manifests:
- `source = "tenhou"`
- `source_game_id`
- `lobby`
- `lobby_display`
- `type`
- `room`
- `ruleset`
- `player_dan`
- `player_rate`

Then we can generate curated split files for high-level subsets later without changing the core training format.
