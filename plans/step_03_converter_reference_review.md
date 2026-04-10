# Step 3 Converter Reference Review

Date: 2026-03-28

## Purpose
Record a deeper correctness review of the Step 3 Tenhou converter against mature reference implementations and wider reference replay corpora.

This review is about semantic alignment, not only whether the current mini-batch pipeline runs.

## Reference Implementations Compared
- official Tenhou HTML5 replay viewer logic
  - local file: `/tmp/tenhou_1129.js`
  - source: `https://tenhou.net/5/1129.js`
- `tenhou-json`
  - local source: `/tmp/tenhou-json-src/tenhou-json-0.1.2/`
- `riichi`
  - local source: `/tmp/riichi-src/riichi-0.1.0/src/interop/tenhou_log_json/`
- `mjlog2json-core`
  - local source: `/tmp/mjlog2json-core-src/mjlog2json-core-0.1.3/`

## Areas Reviewed
1. Incoming/outgoing Tenhou JSON token model
2. Meld-string parsing for `chi`, `pon`, `daiminkan`, `ankan`, and `kakan`
3. Turn progression after discards and claims
4. Round-result normalization
5. Behavior on wider reference replay JSON corpora

## Comparison Result

### 1. Token model alignment
Our supported outgoing token set matches the mature four-player Tenhou JSON parsers for the core path:
- integer discard
- `60` tsumogiri
- `rNN` and `r60`
- `a` ankan strings
- `k` kakan strings
- `0` daiminkan dummy alignment

This matches the modeled outgoing surface in:
- `tenhou-json` parser/exporter
- `riichi` `TenhouOutgoing`
- `mjlog2json-core` replay conversion

### 2. Meld-string parsing
Our split helpers in `scripts/tenhou_xml.py` are consistent with the marker-position logic in the reference parsers:
- `chi`
- `pon`
- `daiminkan`
- `ankan`
- `kakan`

The most important consistency point is that we are not inventing a different meld alphabet; we are decoding the same one the mature parsers expect.

### 3. Turn progression
Our `choose_next_actor_after_discard()` heuristic is consistent with the official Tenhou viewer logic:
- check pon/daiminkan-style claim strings in the same seat-order pattern as the viewer
- otherwise fall back to the next actor in turn order

This means the basic replay stepping logic is not an ad hoc invention; it is intentionally mirroring the official client behavior.

### 4. Confirmed in-scope gaps found and fixed
The review found one real in-scope result-string gap in our converter:
- `四槓散了`

The review also found two legacy wall-exhaust strings handled by mature parsers but not by us before:
- `全員聴牌`
- `全員不聴`

These are now normalized by `scripts/tenhou_xml.py` and covered by regression tests in `tests/test_tenhou_converter.py`.

### 5. Wider corpus sweep
We ran `official_json_to_mjai_lines()` across the local `riichi` notable-features corpus:
- total files checked: `17`
- converted successfully: `15`
- failed: `2`

The two failures were:
- `abort-almost-nagashi-mangan.json`
- `yakuman-kazoe-17.json`

Both failed with:
- `missing visible dora indicators for kan follow-up`

## Important Finding: Source-Fidelity Boundary
Those two failing JSON files are not evidence that the converter is wrong for official `mjlog2json`.

Instead, they show a source-fidelity boundary:
- the local `riichi` reference replay JSON set appears to omit post-kan visible dora indicators in at least some cases
- our converter currently assumes the replay JSON is authoritative enough to emit mjai `dora` events
- official sample-page cases and our real Phoenix mini-batch still pass under that assumption

Practical implication:
- local third-party replay JSON should not be treated as a fully authoritative substitute for official `mjlog2json`
- especially not for kan-heavy edge cases

## Test/Validation Outcome
Current committed regression suite:
- `11` tests
- all passing

New coverage added in this review:
- `四槓散了` normalization
- legacy `全員聴牌` / `全員不聴` normalization
- explicit failure-path coverage for lossy non-official replay JSON missing post-kan dora

## What This Means For Step 3
Good news:
- the converter is broadly aligned with mature four-player Tenhou JSON models
- the first real Phoenix mini-batch remains valid
- the newly found in-scope result gaps have been fixed

Remaining caution:
- our current mixed-batch staging helper can pair official XML with non-official replay JSON oracles
- that is acceptable for pilot validation
- but it is not strong enough to be the long-term canonical ingestion path for kan-heavy edge cases

## Recommended Next Work
1. Keep using the current staged mini-batch path for pilot four-player validation.
2. Treat official `mjlog2json` access as the preferred oracle path whenever it becomes reliably fetchable.
3. Before calling Step 3 complete, expand the real elite batch and produce the first pilot QA/split outputs.
4. Keep the lossy-reference failure test in place so we do not silently blur the boundary between official and non-official replay JSON behavior.
