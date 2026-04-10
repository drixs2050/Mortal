# Step 2 Source Catalog

Date: 2026-03-28

## Purpose
This is the first concrete Workstream A artifact for Step 2.
It is not the final policy.
It is a working catalog of candidate human-record sources, their apparent usefulness, and the review work still required before any large-scale ingestion starts.

## What The Current Repo Needs
For the current Mortal codebase, a primary training source is only useful if we can eventually turn it into:
- action-level game records
- complete enough replay data to reconstruct legal states and chosen actions
- normalized `libriichi`-compatible `.json.gz` logs

The current trainers do not primarily consume scoreboard summaries.
They consume replay-like records that can be expanded into per-decision training samples.

## Status Labels
- `approved`: acceptable for prototype ingestion work with no obvious policy blocker remaining
- `needs review`: promising, but still blocked by policy, export, provenance, or format uncertainty
- `rejected`: not a good primary training source for the current plan
- `supporting only`: useful for evaluation, metadata, or benchmarking, but not a primary action-level corpus

## Candidate Sources

| Source | Type | Why It Matters | Current Evidence | Main Risks / Gaps | Status |
| --- | --- | --- | --- | --- | --- |
| Tenhou public replay URLs and recent downloadable logs | Official online platform replay source | High player volume, strong-player games exist, replay-style data is already close to common riichi tooling | Tenhou's manual says replay viewing is free, recent logs can be downloaded, replay files are stored as compressed XML, and replay format may change without notice | Copyright and bulk-use policy need review; server-side deletion is possible; replay format stability is not guaranteed | `needs review` |
| Tenhou paid/local replay exports from accounts you control | Official platform export path | Same technical upside as Tenhou public replays, with better access continuity for your own collected logs | Tenhou documents replay saving / analysis support and downloadable recent logs for paid users | Still requires policy review for internal research use and redistribution boundaries; account-ownership and consent rules must be explicit | `needs review` |
| Mahjong Soul self-owned replays / observation features | Official online platform replay source | Large active player base and likely strong-player data availability | Mahjong Soul's official start guide shows both spectating and replay history features in the product UI | We have not yet confirmed bulk export, retention, replay format, or usage rules from official policy text | `needs review` |
| Third-party mirrors or scraped exports of Mahjong Soul / Tenhou logs | Derived public corpora | May offer convenience and existing volume | Common in the ecosystem, but not currently validated here | Provenance, permissions, completeness, and licensing are unclear; high risk of building on data we should not use | `rejected` by default until provenance review |
| M.League official match results and standings | Official pro-league data | Extremely strong players, trusted match context, good benchmark reference | Official site clearly publishes league, rule, and point data | Current quick scan did not find an official action-level replay/export path; likely insufficient as a primary training corpus by itself | `supporting only` |
| User-owned local match logs from private scrims / club games / events with explicit consent | Private controlled source | Strongest rights position if permissions are clear; can be shaped around our schema needs | Not dependent on public platform policy if participants and host agree | Usually much smaller than public platforms; quality and strength tier may vary | `approved` in principle if consent and storage rules are documented |

## Source Notes

### 1. Tenhou
Relevant official signals:
- Tenhou says replay viewing is free and recent logs can be downloaded from the official service.
- Tenhou says replay files are compressed XML.
- Tenhou also says replay logs may be deleted without notice, the format may change without notice, and replay copyright belongs to Tenhou.
- We have now captured one official sample replay XML and confirmed raw `lobby`, `type`, `dan`, and `rate` fields are present.

Practical interpretation:
- Technically very promising.
- Policy-sensitive enough that we should not begin bulk collection until we write down an internal usage rule for it.
- Strong candidate for the first format-study source.

Official references:
- https://tenhou.net/man
- https://cdn.tenhou.net/man/
- https://tenhou.net/mjlog.html

### 2. Mahjong Soul
Relevant official signals:
- Mahjong Soul's official start guide shows built-in spectating.
- The same guide shows built-in replay history for a player's past games.

Practical interpretation:
- The product clearly has replay-like data available to users.
- We still need to confirm whether official policy and tooling allow any export or automated collection path suitable for research.
- Good candidate for product-format study, but not yet approved for collection.

Official references:
- https://mahjongsoul.com/startguide/

### 3. M.League
Relevant official signals:
- The official site publishes league information, rules, and point/standing data.

Practical interpretation:
- Very useful as a high-quality benchmark and metadata source.
- Not currently a primary training source unless we later find an official action-level replay feed.
- Treat as supporting-only for now.

Official references:
- https://m-league.jp/
- https://m-league.jp/about
- https://m-league.jp/points

### 4. Private / controlled logs
Practical interpretation:
- This is the least ambiguous policy path if the logs come from games you directly own or host with explicit participant consent.
- Could be valuable for small clean pilots even if not enough for the full corpus.
- We should define a consent and retention template for this source class later in Step 2.

## Format Readiness Summary

| Source | Action-level replay likely? | Official export documented? | Metadata likely? | Good first format-study candidate? |
| --- | --- | --- | --- | --- |
| Tenhou | Yes | Yes, at least for replay download / save paths | Moderate | Yes |
| Mahjong Soul | Likely yes | Not yet confirmed | Moderate | Yes |
| Third-party mirrors | Unknown / variable | No official path | Variable | No |
| M.League | Not confirmed | Not found in current scan | High for match metadata | No |
| Private controlled logs | Depends on host/platform | Depends on setup | High if we design for it | Yes |

## Recommended Source Priority For Step 2
1. Tenhou as the first official format-study candidate.
2. Mahjong Soul as the second official format-study candidate.
3. Private controlled logs as a fallback / pilot-safe source class.
4. M.League as a benchmark/supporting source, not a primary training corpus.

## Immediate Next Actions
1. Verify Tenhou raw code mappings for lobby labels, dan labels, and relevant `GO type` rule bits.
2. Implement a read-only Tenhou parser for one official sample and preserve raw elite-game metadata in manifests.
3. Use Tenhou as the first converter target for normalized Mortal logs.
4. Keep Mahjong Soul as a secondary research source until its policy and export path are clearer.

## Open Questions
- What exact internal-use rule are we comfortable with for Tenhou replay data?
- Can Mahjong Soul replay data be exported or only viewed inside the client?
- Do we want to build the normalized schema around one source first, or around a source-agnostic event model immediately?
- Do you already have any legally obtained log archive or owned-account replay source that should be treated as priority zero?
