# Step 8 Plan: Evaluation Harness And Regression Suite

Date: 2026-04-09 (revised)

## Purpose

Make every candidate model evaluable by the same harness, with the same seeds,
the same opponents, and the same artifacts, so that "this checkpoint is
better" is a defensible claim instead of a vibe.

Step 8 is not about training. It is about:

1. Preventing fake progress during the BC era (Step 7 and its immediate
   successors).
2. Laying down a stable, programmatic, versioned evaluation surface that
   Step 9 (Self-Play RL Environment Hardening) and Step 10 (RL Algorithm
   Development) can consume without reinventing anything.

The first Step 7 promotable BC checkpoints now exist
(`step7_bc_full_9dan_best.pth`, `step7_bc_full_9dan_full_eval_best.pth`).
They are the first real candidates that need a standardized comparison
surface — and they are the first validation target for the harness itself.

Step 8 follows a **validation-first** philosophy: prove the model actually
plays before building infrastructure around evaluating it. The harness is
built around empirically validated gameplay, not around an unverified
assumption.

## Starting Point

Current evaluation surface at Step 8 handoff.

**Exists:**

- `mortal/eval_bc.py` — offline held-out action-match eval (NLL, top-1, top-k,
  legal rate, per-category accuracy), JSON summary, WandB eval run.
- `libriichi/src/arena/one_vs_three.rs` — `OneVsThree` with `py_vs_py`,
  `ako_vs_py`, `py_vs_ako` entry points, deterministic `(seed_start,
  seed_count)`, mjai log directory output.
- `libriichi/src/arena/two_vs_two.rs` — `TwoVsTwo` variant for 2v2 team play.
- `libriichi/src/stat.rs` — `Stat::from_dir`, ~50 per-player metrics
  (`avg_rank`, `avg_pt`, `rank_X_rate`, `agari_rate`, `houjuu_rate`,
  `riichi_rate`, `tobi_rate`, `yakuman_rate`, ...).
- `libriichi/src/agent/akochan.rs` — `AkochanAgent` pipe-talks to an external
  akochan binary via `AKOCHAN_DIR` / `AKOCHAN_TACTICS`. Integration exists;
  the binary does not.
- `mortal/engine.py` — `MortalEngine` canonical inference wrapper (AMP,
  Boltzmann/top-p, rule-based agari guard, `react_batch` API). Also
  `ExampleMjaiLogEngine` — a passive tsumogiri-only bot that never calls,
  never riichis, never wins.
- `mortal/player.py` — `TestPlayer.test_play` wires `MortalEngine` into
  `OneVsThree` against a `[baseline.test]` champion. Legacy in-training eval
  hook; not a standalone Step 8 harness.
- `tests/` — 20+ pytest files covering BC training, dataloader, runtime,
  campaign, conversion. No inference-correctness, action-legality, or
  self-play harness tests yet.
- Step 1 random-init checkpoints (`step1_baseline.pth`, `step1_best.pth`,
  `step1_main.pth`) — **confirmed to be byte-identical copies of one
  randomly-initialized network** from `scripts/step1_bootstrap.py`. Useless
  as opponents, perfect as frozen regression-test fixtures.

**Does not exist:**

- A single canonical "evaluate this checkpoint" CLI that combines offline and
  self-play metrics into one promoted artifact.
- A standalone self-play ladder runner decoupled from `TestPlayer` / `train.py`.
- A frozen seed and sample-count policy with documented confidence intervals.
- A versioned opponent pool with role semantics.
- A model registry for historical comparison.
- Regression tests for inference correctness, action legality, and mjai log
  parsing.
- A documented promotion gate.
- A Python library surface for RL code to reuse (`load_engine`,
  `load_eval_pool`, `run_match`, `evaluate_checkpoint`).
- A locked-down documentation layer for the action space, mask semantics,
  mjai log schema, and checkpoint keys.
- An inference throughput baseline file for RL replay budgeting.

**Confirmed closed doors:**

- Upstream Mortal does not release pretrained weights (author's 2022-08-19
  gist, anti-cheat reasoning). No third-party learning-based reference is
  available for free download.
- Famous strong AIs (Suphx, NAGA) are not open source.
- This makes akochan the only downloadable strong external reference.
  akochan is therefore **load-bearing** for Step 8, not optional.

## Goals

1. A tiered validation ladder that proves the live Step 7 checkpoint plays
   legal games, pursues winning hands, and competes against an external
   rule-based reference — in that order, with empirical evidence at each
   tier.
2. One canonical `offline_eval` CLI for held-out action-match metrics, with a
   stable versioned JSON schema.
3. One canonical `selfplay` CLI for arena matches, architecture-agnostic,
   deterministic, supporting Mortal-vs-Mortal, Mortal-vs-akochan, and
   Mortal-vs-passive opponents.
4. A versioned opponent pool with `anchor` / `current` / `archive` role
   semantics; anchors that never get replaced.
5. A frozen seed and sample-count policy with concrete numbers and bootstrap
   confidence intervals, not TBDs.
6. A model registry (append-only JSONL) that is already RL-ready:
   accommodates BC checkpoints, RL-finetuned checkpoints, training
   provenance, and historical promotion status.
7. A regression test suite that fails fast on inference numerics drift,
   action legality violations, and mjai log schema drift — and runs in CI
   without needing training artifacts.
8. A written promotion gate that any Step 7+ candidate must clear before it
   is recorded as the new SL baseline or carried into Step 9.
9. An inference throughput baseline file that tells Step 9 how fast replay
   generation can go on this hardware.
10. A stable Python library surface (`mortal/step8_harness.py`) exposing
    `load_engine`, `load_eval_pool`, `run_match`, `evaluate_checkpoint` for
    RL code to reuse.
11. A documentation layer that locks down the action space, mask semantics,
    mjai log schema, checkpoint keys, and deterministic seed reproducibility
    contract.
12. A first-pass evaluation of the live Step 7 best checkpoints under the
    harness, so the harness is proven on a real candidate before Step 7
    finishes.

## Scope

**In scope:**

- Validation-first gameplay ladder (tiers 0–3)
- Offline and self-play evaluation CLIs
- Opponent pool materialization and management
- akochan external integration (**hard requirement**, not optional)
- Seed and sample-count policy
- Model registry
- Regression test suite
- Promotion gate text
- Inference performance baselines
- Python library surface for downstream consumers (RL)
- Documentation of contracts that RL code will depend on
- First-pass evaluation of live Step 7 checkpoints

**Out of scope:**

- New training objectives or loss heads
- Model architecture exploration
- New corpora or ingestion paths
- Training pipeline performance tuning
- Step 9 RL environment or algorithm work itself
- Step 12 platform/Tenhou automation
- Human-in-the-loop eval (viewer, manual hand review)
- Tournament-bracket mode (3+ candidates in one ladder)
- Cross-cohort eval (9dan val vs higher-tier val)
- Non-hanchan modes (sanma, east-only)

## Main Questions To Resolve

1. Does the live Step 7 checkpoint actually play a legal game? Does
   `mortal/player.py`'s `load_engine` path correctly pick up
   `bottleneck_channels` and `hidden_dim` for the 10x-wide model? (The
   current path appears to only pull `conv_channels` and `num_blocks`.)
2. Does `new_py_agent` dispatch on `engine_type`, so `ExampleMjaiLogEngine`
   can be used directly as a tier-1 passive opponent without a wrapper?
3. What is the official offline eval contract — which split, which batch
   size, which JSON schema, which WandB job_type?
4. What is the official primary self-play sample budget? What is the
   promotion ladder budget? How do we bound wall-clock without collapsing
   confidence intervals?
5. Who is in the opponent pool v0, and how is it versioned? Concretely:
   akochan + early Step 7 stage snapshots — anything else?
6. How long does a ladder match against akochan actually take? This sets the
   primary-vs-secondary budget split.
7. How do eval artifacts get stored, named, indexed, and queried?
8. What goes into the promotion gate — offline thresholds, self-play
   thresholds, regression-test pass, written run note?
9. What inference throughput numbers does Step 9 need, and at which batch
   sizes and devices?
10. What is the RL-facing Python library surface? What are its stable
    function signatures?
11. What contracts does RL code need locked down in writing — action space,
    mask semantics, mjai log schema, checkpoint keys, deterministic seed
    reproducibility?
12. How should the harness behave when a checkpoint config disagrees with
    the active config?

## Recommended Work

### 1. Validation-First Gameplay Smoke (Tiers 0–3)

Run this **before** any harness work. The purpose is empirical evidence that
the model plays at all, before we invest in infrastructure for evaluating it.

**Prerequisites (verify before running tier 0):**

- `mortal/player.py` `load_engine` path reads `bottleneck_channels` and
  `hidden_dim` from saved config (fix if not)
- `new_py_agent` dispatches on `engine_type` and accepts `'mjai-log'`
  (write a one-line `'mortal'`-typed wrapper if not)

**Tier 0 — Legal gameplay smoke (minutes of wall)**

- Script: `scripts/run_step8_tier0_smoke.py`
- Challenger: live `step7_bc_full_9dan_full_eval_best.pth`
- Champion: same checkpoint (self-play)
- `seed_start = (10000, 0x2000)`, `seed_count = 1` (4 hanchans total)
- Output: `artifacts/reports/step8/smoke/tier0_<timestamp>/` — mjai logs
  plus one summary JSON
- Pass criteria: no crashes, no illegal actions, all 4 hanchans finish
  cleanly, mjai logs parse back into `Stat` without error
- Failure mode we are hunting: 10x-wide architecture has never been run
  through arena code before; `player.py`'s `load_engine` may silently drop
  `bottleneck_channels` / `hidden_dim`

**Tier 1 — Has it learned to win? (minutes of wall)**

- Script: `scripts/run_step8_tier1_vs_passive.py`
- Challenger: live `step7_bc_full_9dan_full_eval_best.pth`
- Champion: `ExampleMjaiLogEngine` from `mortal/engine.py` — tsumogiri-only,
  never calls, never riichis, never wins
- `seed_start = (10000, 0x2000)`, `seed_count = 25` (100 hanchans)
- Output: `artifacts/reports/step8/smoke/tier1_<timestamp>/` — mjai logs
  plus Stat dump
- Pass criteria: challenger rank-1 rate >= 0.95, no illegal actions over
  100 hanchans
- Failure mode we are hunting: the model has learned to survive (not lose
  on time / not emit illegal moves) but not to win. If rank-1 rate is near
  0.25 (chance), something is badly wrong.

**Tier 2 — Competitive strength vs external reference (hours of wall,
post-akochan-build)**

- Script: `scripts/run_step8_tier2_vs_akochan.py`
- Challenger: live `step7_bc_full_9dan_full_eval_best.pth`
- Champion: akochan via `OneVsThree.ako_vs_py`
- `seed_count`: start with `seed_count = 5` (20 hanchans) as a timing probe,
  then scale based on measured wall-clock per hanchan
- Output: `artifacts/reports/step8/smoke/tier2_<timestamp>/` — mjai logs
  plus Stat dump plus wall-clock timing
- Pass criteria: no crashes, `Stat` output is sensible. Strength result is
  informational, not gate-blocking at the smoke tier.
- Failure mode we are hunting: subtle protocol or seeding issue with the
  akochan subprocess, not strength

**Tier 3 — Internal progress ladder (minutes of wall)**

- Script: `scripts/run_step8_tier3_self_progression.py`
- Challenger: live `step7_bc_full_9dan_full_eval_best.pth`
- Champion: early stage snapshot (e.g.
  `artifacts/checkpoints/step7_bc_full_9dan_stage/stage_step_00008000.pth`,
  which is around the end of epoch 1)
- `seed_count = 125` (500 hanchans)
- Output: `artifacts/reports/step8/smoke/tier3_<timestamp>/`
- Pass criteria: later-step challenger beats earlier-step champion on
  `avg_rank` and `avg_pt`. If not, training has plateaued or regressed.
- This also feeds the internal-progress curve that RL will later consume
  when deciding whether to keep training the SL baseline or freeze and
  start RL.

Tiers 0, 1, 3 use only in-process Python and are runnable today. Tier 2 is
gated on the akochan build (section 4).

### 2. Offline Held-Out Eval Contract

Harden `mortal/eval_bc.py` into a canonical CLI with a stable schema.

**CLI:** `scripts/run_step8_offline_eval.py` (thin wrapper around `eval_bc.py`)

- Arguments: `--checkpoint`, `--splits val,test`, `--output-dir`
- Runs both val and test in one invocation, full splits, `max_batches = 0`
- Writes per-split JSON plus a combined `summary.json`
- Starts a WandB run with `job_type = 'eval'`,
  `name_suffix = '-<split>-offline'`

**JSON schema (`offline_eval.v1`):**

```json
{
  "schema_version": "offline_eval.v1",
  "harness_version": "step8.v0",
  "checkpoint": "...",
  "checkpoint_sha256": "...",
  "checkpoint_stem": "step7_bc_full_9dan_full_eval_best",
  "training_run_id": "h3wfor5r",
  "training_paradigm": "bc",
  "split": "val",
  "file_count": 2000,
  "batch_count": 0,
  "processed_step_count": 0,
  "expected_total_step_count": 0,
  "metrics": {
    "nll": 0.0,
    "accuracy": 0.0,
    "topk_accuracy": 0.0,
    "legal_rate": 0.0,
    "category_accuracy": {}
  },
  "evaluated_at": "ISO8601"
}
```

**Artifact layout:**

```
artifacts/reports/step8/<checkpoint_stem>/
  offline_val.json
  offline_test.json
  offline_summary.json
```

**Config override rules:**

- The CLI may override `batch_size`, `eval_batch_size`, `device`, `split`.
- The CLI must not override anything under `[resnet]`, `[bc.control].version`,
  or `[bc.dataset]` that affects which samples the eval sees, except
  `num_workers` for stability.
- Checkpoint-embedded config is authoritative for model shape.

**Promotion ladder uses full val and full test splits, not sampled.**

**Tests:** `tests/test_step8_offline_eval.py` — tiny fixture split, schema
assertions, mocked WandB.

### 3. Self-Play Ladder Harness (Standalone CLI)

`libriichi.arena.OneVsThree.py_vs_py` is the engine layer. Step 8 wraps it
as a standalone command instead of leaving it entangled with the legacy
training flow.

**CLI:** `scripts/run_step8_selfplay.py`

Arguments:

- `--challenger <ckpt>`
- `--champion <ckpt | akochan | passive>`
- `--seed-start 10000:0x2000`
- `--seed-count N`
- `--log-dir <path>`
- `--output-json <path>`
- `--keep-logs {all, summary, none}`
- `--device cuda:0`

**Champion dispatch:**

- `<ckpt>.pth` → `load_engine_from_checkpoint` → `MortalEngine`
  (deterministic config)
- `akochan` → reads `AKOCHAN_DIR` / `AKOCHAN_TACTICS`, calls
  `OneVsThree.ako_vs_py`
- `passive` → wraps `ExampleMjaiLogEngine`

**Deterministic `MortalEngine` config for evaluation:**

- `enable_amp = True`, `amp_dtype = 'bfloat16'`
- `enable_rule_based_agari_guard = True`
- `enable_quick_eval = True`
- `stochastic_latent = False`
- `boltzmann_epsilon = 0`
- `boltzmann_temp = 1.0`
- `top_p = 1.0`
- `compile` disabled (compile is fragile with inference workloads and adds
  warm-up cost that pollutes throughput measurement)

**Log retention policy:**

- `--keep-logs all` — keep every mjai log (default for promotion runs,
  expensive)
- `--keep-logs summary` — run `Stat` aggregation, keep the aggregated
  summary, delete raw logs (default for primary ladder)
- `--keep-logs none` — delete even the directory after `Stat` runs

**JSON schema (`selfplay.v1`):**

```json
{
  "schema_version": "selfplay.v1",
  "harness_version": "step8.v0",
  "challenger": {"checkpoint": "...", "sha256": "...", "stem": "..."},
  "champion":   {"type": "mortal|akochan|passive", "id": "...", "sha256": "..."},
  "seed_start": [10000, 8192],
  "seed_count": 2000,
  "total_hanchans": 8000,
  "rankings": [0, 0, 0, 0],
  "stat": {
    "avg_rank": 0.0, "avg_pt": 0.0,
    "rank_1_rate": 0.0, "rank_2_rate": 0.0, "rank_3_rate": 0.0, "rank_4_rate": 0.0,
    "agari_rate": 0.0, "houjuu_rate": 0.0, "riichi_rate": 0.0,
    "tobi_rate": 0.0, "yakuman_rate": 0.0
  },
  "confidence_intervals": {
    "avg_rank": {"point": 0.0, "ci_low_95": 0.0, "ci_high_95": 0.0,
                 "method": "bootstrap", "n_resamples": 10000},
    "avg_pt":   {"point": 0.0, "ci_low_95": 0.0, "ci_high_95": 0.0,
                 "method": "bootstrap", "n_resamples": 10000}
  },
  "wall_seconds": 0.0,
  "inference_perf": {
    "challenger_samples_per_sec": 0.0,
    "champion_samples_per_sec": 0.0
  },
  "evaluated_at": "ISO8601"
}
```

**Tests:** `tests/test_step8_selfplay.py` — mocked `OneVsThree` + `Stat`,
asserts CLI wires seeds, log dir, opponent pool entry, and output JSON
correctly without running real games.

### 4. akochan Integration (External Anchor)

akochan is the only downloadable strong rule-based opponent. Integration
code exists in `libriichi/src/agent/akochan.rs`; the binary does not. Since
upstream Mortal does not release pretrained weights, akochan is the single
load-bearing external reference for Step 8 and is a **hard requirement** at
exit.

**External build:**

- Upstream: `critter-mj/akochan`
- Build to a location outside the Mortal tree (e.g. `~/tools/akochan/`)
- Obtain `tactics.json` (search configuration)
- Set env vars in a dedicated file checked out of tree (not committed):
  - `AKOCHAN_DIR=/absolute/path/to/akochan`
  - `AKOCHAN_TACTICS=/absolute/path/to/tactics.json`

**Smoke test:**

- `scripts/run_step8_tier2_vs_akochan.py --seed-count 5` (20 hanchans)
- Verify: akochan subprocess spawns, mjai protocol exchange succeeds, `Stat`
  runs on the output
- Measure: wall-clock per hanchan — this sets the ladder budget policy in
  section 6

**tactics.json versioning:**

- akochan strength depends on its tactics configuration. If `tactics.json`
  is upgraded, that is a new pool entry (`akochan_v1`), not the same
  `akochan_v0`.
- The opponent pool manifest records a `tactics_sha256` field for every
  akochan entry so tactics drift is explicit and comparisons stay
  meaningful across time.

**Runbook documentation:**

- A short `docs/runbooks/akochan_setup.md` with build commands, env var
  setup, smoke-test command, troubleshooting
- Not checked-in binary, not checked-in `tactics.json`

### 5. Opponent Pool v0

Small, versioned, explicit.

**Layout:**

```
artifacts/eval_pool/
  v0/
    manifest.json
    step7_stage_epoch1.pth     # copy or hardlink of stage_step_00008000.pth
    step7_stage_epoch4.pth     # later snapshot for internal progress
    # akochan is not stored here - referenced by env var
  v1/                          # reserved for post-Step-7 winner
    manifest.json
```

**manifest.json schema:**

```json
{
  "schema_version": "eval_pool.v1",
  "pool_version": "v0",
  "created_at": "ISO8601",
  "entries": [
    {
      "id": "akochan_v0",
      "role": "anchor",
      "type": "akochan",
      "env_dir": "AKOCHAN_DIR",
      "env_tactics": "AKOCHAN_TACTICS",
      "tactics_sha256": "...",
      "description": "rule-based reference, critter-mj/akochan build",
      "added_at": "ISO8601"
    },
    {
      "id": "step7_stage_epoch1_v0",
      "role": "anchor",
      "type": "mortal_checkpoint",
      "checkpoint": "artifacts/eval_pool/v0/step7_stage_epoch1.pth",
      "checkpoint_sha256": "...",
      "resnet": {
        "conv_channels": 512, "num_blocks": 48,
        "bottleneck_channels": 64, "hidden_dim": 2048
      },
      "version": 4,
      "source_training_run": "h3wfor5r",
      "source_step": 8000,
      "description": "Step 7 BC checkpoint at ~epoch 1 - internal progress anchor",
      "added_at": "ISO8601"
    }
  ]
}
```

**Role semantics:**

- `anchor` — never replaced, exists so historical comparisons stay
  meaningful across years
- `current` — the SL baseline currently in production, can be replaced by a
  candidate that clears the promotion gate
- `archive` — previous `current` entries, kept for historical queries

**Materialization CLI:** `scripts/materialize_eval_pool.py`.

### 6. Seed And Sample-Count Policy

Concrete numbers, not TBDs.

**Canonical primary seed pair:**

- `seed_start = (10000, 0x2000)` — matches `TestPlayer.test_play` historical
  convention

**Primary ladder (used after every candidate eval):**

- `seed_count = 2000` → 8,000 hanchans per match
- Target wall-clock: under 30 minutes per opponent at Mortal-vs-Mortal speed
- For akochan specifically, `seed_count` may be reduced to `500` (2,000
  hanchans) depending on the timing measurement in section 4. The exact
  reduction is recorded in a timing fixture file under
  `artifacts/reports/step8/timing/` once measured.

**Promotion ladder (used only when a candidate is up for promotion):**

- `seed_count = 8000` → 32,000 hanchans per match
- Target wall-clock: under 3 hours per opponent at Mortal-vs-Mortal speed
- For akochan: `seed_count = 2000` if primary is 500, otherwise 8000

**Confidence interval methodology:**

- Bootstrap over per-game outcomes parsed from mjai logs
- 10,000 bootstrap resamples
- Report 95% CI for `avg_rank` and `avg_pt`
- Implementation: new `mortal/step8_stat.py` that parses logs and computes
  CIs. This does **not** modify the Rust `Stat` aggregator — the aggregator
  stays pure and returns point estimates only.

**Determinism contract:**

- Same challenger checkpoint + same champion checkpoint + same `seed_start`
  + same `seed_count` + same device + same AMP setting →
  **bit-identical mjai logs** at least for the simple Mortal-vs-Mortal case.
- This is asserted by a dedicated regression test (section 8).
- If bit-identical logs prove too flaky due to cudnn nondeterminism or fused
  op atomics, the fallback contract is: identical `Stat` aggregates and
  identical final rankings. Any looser contract than that is a Step
  8-blocking bug.

**Timing fixtures:**

- First primary ladder run records wall-clock per opponent type in
  `artifacts/reports/step8/timing/v0.json`.
- Future runs compare against that baseline and flag regressions.

### 7. Model Registry

Append-only, RL-ready.

**File:** `artifacts/registry/step8_registry.jsonl`

**Schema (`registry.v1`):**

```json
{
  "schema_version": "registry.v1",
  "harness_version": "step8.v0",
  "checkpoint": "artifacts/checkpoints/step7_bc_full_9dan_full_eval_best.pth",
  "checkpoint_sha256": "...",
  "checkpoint_stem": "step7_bc_full_9dan_full_eval_best",
  "model_shape": {
    "conv_channels": 512, "num_blocks": 48,
    "bottleneck_channels": 64, "hidden_dim": 2048, "version": 4
  },
  "training": {
    "paradigm": "bc",
    "base_checkpoint": null,
    "training_run_id": "h3wfor5r",
    "training_step": 21000,
    "config": "configs/step7_bc_full_9dan.toml",
    "training_notes": "..."
  },
  "offline_eval": {
    "val":  {"nll": 0.0, "accuracy": 0.0, "topk_accuracy": 0.0, "legal_rate": 0.0},
    "test": {"nll": 0.0, "accuracy": 0.0, "topk_accuracy": 0.0, "legal_rate": 0.0}
  },
  "selfplay": [
    {
      "opponent_id": "akochan_v0",
      "pool_version": "v0",
      "seed_start": [10000, 8192],
      "seed_count": 500,
      "avg_rank": {"point": 0.0, "ci_low_95": 0.0, "ci_high_95": 0.0},
      "avg_pt":   {"point": 0.0, "ci_low_95": 0.0, "ci_high_95": 0.0},
      "rank_1_rate": 0.0, "rank_4_rate": 0.0,
      "stat_file": "artifacts/reports/step8/.../selfplay_akochan_v0_...json"
    }
  ],
  "regression_tests": {
    "passed": true,
    "suite_version": "step8.v0",
    "tested_at": "ISO8601"
  },
  "promotion": {
    "status": "under_evaluation",
    "gate_results": {},
    "decided_at": null,
    "decided_by": null,
    "notes_file": "artifacts/reports/step8/.../run_notes.md"
  },
  "registered_at": "ISO8601"
}
```

**Key RL-ready fields:**

- `training.paradigm: "bc" | "rl_finetune" | "scratch_rl"` — extensible
  without schema break
- `training.base_checkpoint` — non-null for finetunes, records lineage
- `model_shape` — so RL code can filter "give me all SL baselines at
  10x-wide"
- `selfplay[].pool_version` — so RL comparisons know which pool they were
  measured against

**Query helper:** `scripts/query_step8_registry.py`

- `--list-promoted` — current `promoted` candidates
- `--checkpoint <path>` — full record for a given checkpoint
- `--paradigm rl_finetune` — filter by training paradigm
- `--since YYYY-MM-DD` — time-range filter

### 8. Regression Test Suite

Fast, no training artifacts needed, runs in CI.

**New tests under `tests/`:**

- **`test_inference_numerics.py`**
  - Fixture: a tiny fixed observation tensor committed under
    `tests/fixtures/inference/obs_v1.pt`
  - Model: freshly built small `Brain` + `DQN` with frozen seed, or load
    one of the Step 1 random-init checkpoints (perfect golden-fixture use
    case)
  - Assertion: `MortalEngine.react_batch` output action, top-k indices, and
    masked logits match committed `golden_v1.json`
  - Catches: model graph drift, AMP regressions, mask-handling bugs,
    inference dtype drift
  - Runs on CPU

- **`test_action_legality.py`**
  - Fixture: a small fixture mjai game-state file under
    `tests/fixtures/mjai/midgame_v1.json`
  - Run `MortalEngine.react_batch` on it, assert every emitted action is
    legal under its mask
  - Catches: illegal action emission under new mask layouts

- **`test_mjai_log_roundtrip.py`**
  - Fixture: a small canonical mjai log under
    `tests/fixtures/mjai/hanchan_v1.json.gz`
  - Parse through libriichi, re-serialize, assert canonical-form equality
  - Catches: silent mjai schema drift that would break downstream eval and
    RL replay parsing

- **`test_selfplay_determinism.py`**
  - Runs a `seed_count = 1` (4 hanchan) self-play match of the fixed Step 1
    random-init checkpoint against itself, twice
  - Primary assertion: the two mjai log directories are byte-identical
  - Fallback assertion (if bit-identical proves flaky on CPU-vs-CUDA or
    across cudnn versions): identical `Stat` aggregates and identical final
    rankings
  - Catches: non-determinism creeping into `MortalEngine` or the arena path
  - Fastest possible end-to-end arena smoke

- **`test_step8_offline_eval.py`**
  - Tiny fixture split (5 files), mocked WandB, asserts JSON schema and
    file paths

- **`test_step8_selfplay.py`**
  - Mocked `OneVsThree` + `Stat`, asserts CLI wires seeds, log dir,
    opponent pool entry, and output JSON correctly without running real
    games

- **`test_step8_registry.py`**
  - Asserts append-only semantics, schema validation on write, query helper
    correctness

**Fixtures directory:**

```
tests/fixtures/
  inference/
    obs_v1.pt
    golden_v1.json
  mjai/
    midgame_v1.json
    hanchan_v1.json.gz
  checkpoints/
    step1_fixture.pth   # frozen copy, separate from artifacts/checkpoints/
```

**CI requirements:**

- Total suite wall-clock under 60 seconds
- CPU-only (no CUDA dependency)
- No dataset, no Step 7 checkpoint, no Step 6 artifacts required
- Runs on every PR via the existing GitHub Actions workflow

### 9. Promotion Gate (Written Text)

A candidate is promoted to `current` in the opponent pool when **all** of
the following hold:

1. All regression tests in `tests/` pass.
2. Offline metrics on full `val` and full `test` splits:
   - `accuracy` >= current `current` accuracy
   - `topk_accuracy` >= current `current` topk_accuracy
   - `legal_rate` >= current `current` legal_rate
   - `nll` <= current `current` nll + 0.005 (small regressions tolerated
     if other metrics improve)
3. Self-play primary ladder against every `anchor` opponent:
   - `avg_rank` point estimate is lower (better) than the opponent's
     champion side
   - `avg_pt` point estimate is higher (better) than the opponent's
     champion side
   - 95% bootstrap CI for `avg_rank` does not overlap with 4.0 (rules out
     "lost most games" flukes)
4. Self-play promotion ladder against the current `current`:
   - `avg_rank` and `avg_pt` both strictly better
   - 95% bootstrap CI for the difference excludes zero in the favorable
     direction
5. A written run note under
   `artifacts/reports/step8/<stem>/run_notes.md` describing:
   - Training run ID and config
   - Data cohort
   - Offline and self-play results summary
   - Any caveats or known issues
   - Why this candidate is being considered for promotion

Failing any one of these: the candidate is recorded in the registry with
`promotion.status = "rejected"` and a `gate_results` breakdown of which
criteria failed. No silent rejections.

### 10. Inference Performance Baselines

Step 9 RL will need to budget self-play replay generation. Step 8 provides
the numbers.

**CLI:** `scripts/run_step8_infer_bench.py`

- Loads a given checkpoint
- Measures `MortalEngine.react_batch` throughput at batch sizes
  `[1, 8, 64, 256, 1024, 4096]`
- Measures on `cuda:0`, `cuda:1`, and `cpu`
- Runs a warmup + measurement cycle with wall-clock, samples/s, GPU
  memory, and optional `nvidia-smi` utilization capture
- Also measures the full `OneVsThree` arena loop (4 engines in parallel)
  samples/s — this is what RL actually cares about

**Output:** `artifacts/reports/step8/infer_bench/<checkpoint_stem>.json`

**Schema (`infer_bench.v1`):**

```json
{
  "schema_version": "infer_bench.v1",
  "checkpoint": "...",
  "model_shape": {},
  "device": "cuda:0",
  "amp_dtype": "bfloat16",
  "batch_size_sweep": [
    {"batch_size": 1, "samples_per_sec": 0.0, "p50_latency_ms": 0.0, "gpu_mem_used_mib": 0},
    {"batch_size": 8, "samples_per_sec": 0.0, "p50_latency_ms": 0.0, "gpu_mem_used_mib": 0}
  ],
  "arena_loop_samples_per_sec": 0.0,
  "measured_at": "ISO8601"
}
```

This is consumed by Step 9 to answer: "at my target replay generation
rate, how many ranks do I need, and what batch size should each rank
run?"

### 11. RL Foundation Layer (Python Library Surface + Contracts)

The capstone. This is what Step 9 / 10 will import. Scope for Step 8 is
deliberately minimal: make the existing offline training and eval path
usable from other code, document its contracts, and stop. Broader RL
infrastructure (replay buffers, opponent pool evolution, curriculum,
exploration schedules) is explicitly deferred.

**11a. Python library surface:** a new module at `mortal/step8_harness.py`
with **stable** public functions:

```python
def load_engine(
    checkpoint_path: str,
    *,
    device: torch.device,
    deterministic: bool = True,
    amp: bool = True,
    name: str | None = None,
) -> MortalEngine:
    """Canonical eval-time engine construction. RL code uses this
    instead of reconstructing MortalEngine from scratch.

    Reads checkpoint's embedded config for version, conv_channels,
    num_blocks, bottleneck_channels, hidden_dim. When deterministic=True,
    forces boltzmann_epsilon=0, top_p=1.0, stochastic_latent=False.
    """

def load_eval_pool(pool_version: str) -> EvalPool:
    """Load an opponent pool manifest. Returns an EvalPool object
    with iterable entries, each of which can be instantiated as an
    engine.
    """

def run_match(
    challenger: "MortalEngine | AkochanSpec | PassiveSpec",
    champion:   "MortalEngine | AkochanSpec | PassiveSpec",
    *,
    seed_start: tuple[int, int],
    seed_count: int,
    log_dir: str,
    keep_logs: "Literal['all', 'summary', 'none']" = "summary",
) -> "MatchResult":
    """Run a single ladder match. Returns rankings + Stat + CIs + wall
    time.
    """

def evaluate_checkpoint(
    checkpoint_path: str,
    *,
    pool_version: str,
    ladder: "Literal['primary', 'promotion']" = "primary",
    offline_splits: "list[str]" = ("val", "test"),
    output_dir: str,
) -> "EvaluationResult":
    """Run the full Step 8 eval pipeline on a checkpoint. Offline eval +
    self-play against every pool entry. Writes artifacts, appends to
    registry, returns a structured result. RL training loops call this
    periodically to evaluate intermediate RL checkpoints.
    """

def register_checkpoint(
    result: "EvaluationResult",
    *,
    promotion_decision: str,
) -> None:
    """Append an EvaluationResult to step8_registry.jsonl with a
    promotion decision. Caller is responsible for deciding promotion.
    """
```

**Stability contract:** These function signatures are frozen at Step 8
exit. Future changes are additive only (new keyword args with defaults).
RL code importing these is guaranteed to keep working across Step 9 and
Step 10.

**11b. Documentation contracts:** a new `docs/src/ref/step8_contracts.md`
that locks down:

- **Action space layout** — the full enumeration of action indices, what
  each represents, mask semantics, which actions are per-tile vs
  per-category
- **Observation encoding** — the feature channels, their dimensions, their
  semantics, the version field
- **Mask semantics** — how illegal actions are marked, what "legal" means,
  how `enable_rule_based_agari_guard` modifies the mask
- **Checkpoint key contract** — `mortal`, `current_dqn`, `aux_net`
  (optional), `optimizer` (optional for eval), `config`, `steps`,
  `best_perf`, `tag`, `timestamp` — which are required, which are
  optional, how they are loaded
- **mjai log schema** — the exact event types and fields that downstream
  consumers (RL replay, regression tests) can assume are present, and the
  canonical form for roundtripping
- **Opponent pool manifest schema** (duplicated from section 5 for
  discoverability)
- **Deterministic seed reproducibility contract** — exactly what guarantees
  "same inputs → same outputs" (same device, same AMP dtype, same cudnn
  flags, no compile), plus the fallback contract if bit-identical proves
  flaky
- **Performance baseline file schema** (duplicated from section 10)
- **Registry schema** (duplicated from section 7)

**Why this matters for RL:** Step 9 will want to warm-start from the SL
baseline, which means loading a Step 8-evaluated checkpoint, running RL
updates, and re-evaluating. Every step of that loop depends on the
contracts above. If any of them drift silently, RL results become
uncomparable to SL results and progress measurement breaks.

**11c. Canonical fixtures for RL regression:** a small set of fixture
observations and expected outputs committed under `tests/fixtures/rl_ready/`
that RL code can import to verify its inference path is still aligned with
Step 8's. These are the "smoke test your RL code against the SL contract"
assets.

### 12. First-Pass Evaluation Of Live Step 7 Checkpoints

The capstone proof.

**While Step 7 is still training:**

- Run the full harness (offline eval + primary ladder against both anchors)
  on the live `step7_bc_full_9dan_full_eval_best.pth`
- Register in `step8_registry.jsonl` with
  `promotion.status = "under_evaluation"`
- Repeat once `step7_bc_full_9dan_best.pth` (sampled-val best) diverges
  meaningfully

**After Step 7 finishes (approximately 2 days from 2026-04-09):**

- Run the full harness + promotion ladder on the final
  `step7_bc_full_9dan_full_eval_best.pth`
- Produce a run note
- Record promotion decision in the registry

This is the end-to-end proof that every Step 8 phase is real and not paper.

## Step 7 / Step 8 Coordination

Step 8 does not block Step 7. Step 7 keeps running under its current
production launch. Step 8 work happens in parallel against the checkpoints
Step 7 has already written and against the live `best` / `full_eval_best`
files.

The only piece of Step 8 that depends on Step 7 finishing is section 12's
*post-run* evaluation (against the final checkpoint). Everything else can
and should be exercised against the live checkpoints while Step 7 trains.

## Step 8 / Step 9 Handoff

Step 9 (Self-Play RL Environment Hardening) starts with:

- A stable Python library surface (`load_engine`, `load_eval_pool`,
  `run_match`, `evaluate_checkpoint`, `register_checkpoint`)
- A versioned opponent pool it can consume programmatically
- Documented action space, mask semantics, mjai log schema, checkpoint keys
- Inference throughput baselines for replay budgeting
- A deterministic seed reproducibility contract so RL regression is
  possible
- Regression tests that guarantee the inference path has not drifted
- A working first-pass eval of the SL baseline as the RL starting point
  and anchor

Step 9 does not need to touch `mortal/eval_bc.py`, `libriichi/src/arena/`,
or `libriichi/src/stat.rs` to run evaluations. It imports the library
surface.

Step 9 does not need to reinvent opponents. It extends the opponent pool
to v1 / v2 by adding new entries with the same role semantics.

Step 9 does not need to reinvent promotion. It uses the same written gate,
possibly with additional RL-specific thresholds.

## Exit Criteria

- Tier 0, 1, 2, 3 validation results recorded in
  `artifacts/reports/step8/smoke/` with empirical evidence the live Step 7
  checkpoint plays legal games, crushes the passive opponent, competes
  against akochan, and improves over its own earlier snapshots.
- `scripts/run_step8_offline_eval.py` exists, runs against any BC
  checkpoint, writes the `offline_eval.v1` JSON schema into
  `artifacts/reports/step8/<stem>/`.
- `scripts/run_step8_selfplay.py` exists, handles Mortal / akochan /
  passive champion dispatch, writes the `selfplay.v1` JSON schema.
- `artifacts/eval_pool/v0/manifest.json` exists with at least `akochan_v0`
  and one Step 7 stage-snapshot anchor.
- **akochan external build is complete, smoke-tested, and documented in
  `docs/runbooks/akochan_setup.md`.** This is a hard requirement — Step 8
  does not exit without it.
- Seed and sample-count policy documented with concrete numbers (no TBDs).
- `artifacts/registry/step8_registry.jsonl` exists with at least one real
  Step 7 row.
- All new regression tests pass in CI in under 60 seconds total.
- Promotion gate written as text in this plan and enforced by
  `evaluate_checkpoint`.
- Inference performance baseline file exists for at least one Step 7
  checkpoint.
- Python library surface (`mortal/step8_harness.py`) exists with frozen
  signatures.
- `docs/src/ref/step8_contracts.md` exists, documents action space, masks,
  mjai log schema, checkpoint keys, determinism contract.
- First-pass evaluation of live Step 7 checkpoints recorded in the
  registry.
- Final Step 7 checkpoint evaluated through the full harness with a
  written promotion decision.
- Step 9 can start without having to touch any code or reinvent any
  convention from Steps 1–8.
