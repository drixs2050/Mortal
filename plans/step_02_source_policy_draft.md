# Step 2 Source Policy Draft

Date: 2026-03-28

## Purpose
This is the first conservative source-policy draft for Mortal's human-record pipeline.
It is intended to unblock prototype ingestion work without accidentally committing us to data-collection behavior we would later regret.

This is not legal advice.
It is an internal engineering policy draft that should remain stricter than "probably okay" whenever source terms are unclear.

## Policy Goals
- Prefer sources with the clearest rights position.
- Avoid collecting data we cannot justify storing or using for internal model training.
- Avoid building the project around scraped or poorly documented corpora.
- Keep the first Step 2 work focused on small format-study samples, not bulk harvesting.

## Allowed Usage Categories

### 1. Approved for prototype ingestion
These sources are acceptable for small pilot ingestion work now:
- User-owned local logs from matches you directly played, hosted, or exported.
- Private scrim, club, or event logs where participant consent and storage expectations are explicit.

Conditions:
- The source of the logs must be documented.
- The storage location must be internal to this workstation or a controlled private backup.
- Redistribution is not allowed unless the participants and platform rules explicitly allow it.

### 2. Reviewable for format-study only
These sources may be used for small format-study samples, schema design, and converter prototyping, but not bulk collection yet:
- Tenhou replay downloads or replay URLs obtained through official product paths.
- Mahjong Soul replay history or replay-view features accessed through official product paths.

Conditions:
- Keep the scope to a tiny sample count while policy questions are still open.
- Do not mirror or publish raw source logs.
- Do not assume that public visibility implies unrestricted training rights.
- Record the exact official page or product path used to obtain the sample.

### 3. Prohibited by default
These sources should not be used unless a later review explicitly reverses the decision:
- Third-party scraped mirrors of Tenhou or Mahjong Soul logs with unclear provenance.
- Repacked community datasets with missing source lineage.
- Any dataset whose license or source terms do not clearly cover our intended internal use.

## Source-Specific Guidance

### Tenhou
Current working rule:
- Allowed for small format-study samples only.
- Not yet approved for bulk internal backfill.
- Not approved for redistribution or public re-hosting.

Reason:
- Tenhou's official pages make replay download and viewing visible, but also say replay copyright belongs to Tenhou and format/availability may change without notice.

Current engineering implication:
- Tenhou is still the best first source to study technically.
- The first Tenhou task should be a tiny sample conversion experiment, not a collection campaign.

Official references:
- https://tenhou.net/mjlog.html
- https://cdn.tenhou.net/man/

### Mahjong Soul
Current working rule:
- Allowed for product-format study only.
- Not approved for ingestion or automation until official replay/export boundaries are clearer.

Reason:
- The product clearly has replay and spectate features, but we have not yet confirmed an official export or training-compatible collection path.

Official reference:
- https://mahjongsoul.com/startguide/

### Private controlled logs
Current working rule:
- Approved in principle when we control the logs or have explicit participant permission.

Minimum requirements:
- Record where the logs came from.
- Record who approved storage and use.
- Keep personal identifiers only when actually needed.

### M.League and similar official result feeds
Current working rule:
- Allowed as metadata or benchmark context only.
- Not a primary training corpus unless an official action-level replay path is later found.

## Storage And Retention Rules
- Keep raw source snapshots in `data/raw/`.
- Keep normalized training logs in `data/normalized/`.
- Keep manifests, provenance notes, and split files in `data/manifests/`.
- Treat raw data as immutable once imported.
- Do not commit raw or normalized game corpora to git.
- Keep any sensitive or personally identifying source notes out of public docs and repos.

## Redistribution Rules
- Do not publish raw platform logs.
- Do not publish converted normalized corpora derived from platform sources unless the source policy is explicitly reviewed and approved for that action.
- Internal manifests may store source identifiers and hashes, but any external release would need a separate review.

## Review Gates Before Bulk Ingestion
All of the following must be true before any large-scale backfill begins:
- The source class is marked approved in the source catalog.
- The usage rule for that source is written down here or in a later revision.
- We have a converter design for that source.
- We have a manifest and QA plan for that source.

## Immediate Step 2 Default
Use this default unless we explicitly revise it:
1. Prototype against a tiny Tenhou format-study sample.
2. Continue treating Mahjong Soul as policy-blocked for ingestion.
3. Accept private controlled logs if you already have them and want them prioritized.
4. Reject third-party scraped mirrors.

## Open Questions
- Are we comfortable approving Tenhou for internal bulk training use, or only for converter prototyping at first?
- Do you already have any owned or consented log archive that should become the safest priority-zero dataset?
- Should player names be preserved in manifests, hashed, or split into public/private manifest layers?
