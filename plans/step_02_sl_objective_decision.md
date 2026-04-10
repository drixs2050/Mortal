# Step 2 Supervised Objective Decision

Date: 2026-03-28

## Purpose
This document resolves the mismatch between the roadmap's planned supervised phase and the repo's current trainer design.

## Current Repo Reality

### GRP trainer
`mortal/train_grp.py` is a supervised model.
It predicts rank outcomes from per-kyoku game progression features.

Useful role:
- supporting model for reward shaping or auxiliary targets

Not sufficient role:
- it is not the main action-selection pretraining path

### Main trainer
`mortal/train.py` is not pure supervised imitation.
It currently trains on replay-derived actions with:
- Monte Carlo style value targets
- CQL regularization
- next-rank auxiliary prediction

Useful role:
- offline RL / offline value-learning baseline

Not accurate to call it:
- "the supervised pretraining phase"

## Decision
For this project, "supervised pretraining" should mean explicit action imitation on human logs.

Recommended first real baseline:
- a separate behavior-cloning training path, or a clearly isolated BC mode, that optimizes legal-action prediction from human records

Do not redefine the existing offline RL trainer as supervised pretraining just because it runs on offline logs.

## Why This Decision Is Better
- It matches the roadmap you want: learn directly from strong human decisions first.
- It gives us clean held-out metrics like action log-likelihood and top-k accuracy.
- It keeps the current offline RL baseline available as a separate comparison instead of mixing objectives too early.
- It makes later self-play finetuning easier to reason about.

## Recommended Training Sequence

### Phase A. GRP support model
Train or refresh GRP on the human corpus.

Purpose:
- preserve the existing reward-model path
- keep auxiliary targets available

### Phase B. Human-log behavior cloning
Build the first true supervised action model.

Suggested loss:
- masked cross-entropy over legal actions

Optional auxiliaries:
- next-rank prediction
- GRP-informed auxiliary targets

### Phase C. Offline RL refinement
Use the existing `train.py` objective, or a refined successor, as a later stage.

Purpose:
- learn value structure and improve action ranking beyond pure imitation

### Phase D. Online self-play RL
Move to controlled self-play only after the earlier phases are stable.

## Implementation Recommendation
Prefer a separate BC entrypoint over overloading `train.py` immediately.

Reason:
- it keeps metrics and checkpoints easier to interpret
- it avoids turning Step 2 into a risky trainer refactor before the data path is ready
- it lets us compare BC and offline RL baselines cleanly on the same corpus

## Minimum Metrics For The BC Baseline
- held-out masked action accuracy
- held-out top-k action accuracy
- held-out negative log-likelihood
- legality correctness
- downstream self-play smoke metrics against fixed baselines

## What This Means For Step 2
Step 2 does not need to implement the BC trainer yet.
But it should treat the data pipeline as serving two future consumers:
- GRP training
- explicit behavior cloning

That means the normalized logs must preserve:
- observations reconstructable from event streams
- chosen human actions
- legal-action reconstruction
- enough metadata to define trustworthy train, val, and test splits

## Immediate Recommendation
Lock this as the project default:
1. Step 2 prepares data for a future BC trainer.
2. The current `train.py` remains the offline RL baseline, not the supervised baseline.
3. Step 3 should begin with a small human-log BC implementation plan before any large-scale training campaign.
