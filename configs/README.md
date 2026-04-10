# Configs

`configs/step1_smoke.toml` is the Step 1 local smoke-test profile for this workstation.
`configs/step5_bc_debug.toml` is the fast Step 5 behavior-cloning sanity-check profile on the stable Batch D split.
`configs/step5_bc_medium.toml` is the first serious Step 5 BC baseline profile on the merged Step 4B partial release.
`configs/step5_bc_large.toml` is the next larger bounded Step 5 BC run on the same merged Step 4B partial release.
`configs/step6_bc_debug.toml` through `configs/step6_bc_full_9dan.toml` are the Step 6 hardening profiles and run ladder (superseded by Step 7).
`configs/step7_bc_full_9dan.toml` is the Step 7 production behavior-cloning config (83.4M 10x-wide model, 32 epochs, dual-A100 DDP with torch.compile).
`configs/step7_preflight.toml` is the Step 7 preflight variant (1000 steps, otherwise identical to production).
`configs/step6_bc_large_preflight_full8dan.toml` and the older `*_r3`/`*_r4` variants remain historical Step 6 loader-gate references.
`configs/step6_bc_large_preflight_full8dan_r5.toml` is the current short post-`medium` dual-A100 preflight on the validated raw async CPU-pipe baseline.
`configs/step6_bc_large_bounded_full8dan_8192_r5.toml` is the current 8192-per-GPU large-bounded config used only if that preflight succeeds.
`configs/step6_bc_large_bounded_full8dan_r5.toml` is the matching 4096-per-GPU fallback on that same raw baseline.

Important state note:
- the checked-in Step 5 medium and large configs intentionally still point at the earlier partial `E01` through `E09` release so their recorded metrics remain reproducible
- the full Step 4B release now exists at `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/`
- the stricter Step 6 launch-candidate split now also exists at `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_9dan_actor_any/`
- Step 6 should add new full-corpus profiles rather than silently repointing the historical Step 5 baseline configs

Torch-visible logical GPU order inside the `mortal` Conda environment:
- `cuda:0` = A100
- `cuda:1` = A100
- `cuda:2` = RTX 3070

Step 6 launcher safety rule:
- multi-GPU Step 6 launchers now verify the first `nproc_per_node` torch-visible devices are actually A100s before training starts
- do not substitute `0,2` based on raw `nvidia-smi` ordering; keep the canonical torch-visible A100 pair as `cuda:0` and `cuda:1`

Default Step 1 policy:
- use only `cuda:0`
- keep `cuda:1` reserved until the dual-A100 path is active
- ignore `cuda:2` for training and evaluation

Typical usage from the repo root:

```bash
export MORTAL_CFG=/home/drixs2050/Documents/Mortal/configs/step1_smoke.toml
python scripts/step1_bootstrap.py
```

Once Rust, `torch`, and `libriichi` are available, this profile is enough to:
- generate bootstrap checkpoints
- generate a tiny self-play fixture
- run import/load smoke checks

Typical Step 5 BC usage from the repo root:

```bash
source /home/drixs2050/Documents/Mortal/wandb_key.env
export MORTAL_CFG=/home/drixs2050/Documents/Mortal/configs/step5_bc_debug.toml
python mortal/train_bc.py
python mortal/eval_bc.py --split val
python mortal/eval_bc.py --split test --output-json /home/drixs2050/Documents/Mortal/artifacts/reports/step5_bc_debug_test.json
```

Typical Step 6 one-command campaign usage from the repo root:

```bash
source /home/drixs2050/Documents/Mortal/wandb_key.env
python scripts/run_bc_campaign.py --config /home/drixs2050/Documents/Mortal/configs/step6_bc_full_9dan.toml
```

That command now does:
- distributed or single-GPU training depending on `bc.launch.nproc_per_node`
- deterministic best-checkpoint promotion during training
- final full `val` and `test` offline eval with JSON reports
- campaign-summary JSON emission

Loader-tuning and Step 6 ladder usage now have a separate safe entrypoint:

```bash
python scripts/run_step6_experiment_ladder.py --stop-after phase_a
```

That runner prepares the representative subset, executes the older loader-exploration ladder, records comparison tables, and only moves past loader optimization when later phases are explicitly requested.
The current post-handoff launch path is the raw-baseline `r5` preflight and large-bounded configs above rather than the historical RAM-first ladder.

Step 7 production config summary (`step7_bc_full_9dan.toml`):
- Model: 512ch/48b/bn64/hd2048 (83.4M params, 10x-wide)
- Batch: bs=3072/gpu, ga=1, eff_batch=6,144 (2x A100 DDP)
- LR: peak=4e-4, final=4e-5, 5% warmup, cosine decay
- Pipeline: w4 (num_workers=4, persistent_workers, cpu_batch_pipe='thread')
- Compile: torch.compile + DDP (compile before wrap, optimize_ddp=True)
- Training: 32 epochs, ~235K steps, seed=42
- Logging: train_log_every=1 (full metrics every step), val_log_every=500, save_every=1000, best_eval_every=2000
- Two best checkpoints: val-best (sampled, 500-step), full-eval-best (deterministic, 2000-step)
- DDP: static_graph=false (benchmarked — no benefit over false)
- Data: 9dan acting-seat cohort, min_actor_dan=18

Typical Step 7 launch:

```bash
source /home/drixs2050/Documents/Mortal/wandb_key.env
MORTAL_CFG=configs/step7_bc_full_9dan.toml CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node 2 --master-port 29516 mortal/train_bc.py
```

Current Step 5 split targets:
- `step5_bc_debug.toml` -> stable Batch D Phoenix hanchan split filtered to games with at least one `八段+` player before chronological splitting
- `step5_bc_medium.toml` -> merged partial Step 4B Phoenix hanchan split built from `E01` through `E09`, filtered to games with at least one `八段+` player before chronological splitting
- `step5_bc_large.toml` -> same merged partial Step 4B split as medium, but with a larger bounded train batch; it is now the fresh `500`-step large-run variant with its own checkpoint/tensorboard/WandB names so it does not resume the earlier longer attempt, and it enables a warmup+cosine LR schedule anchored to that bounded run length
- full current Step 4B release available for upcoming Step 6 configs:
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/train.txt`
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/val.txt`
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/test.txt`
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/path_cache_v1.pth`
  - `data/splits/v1/tenhou_phoenix_4y_full_e01_e16_v0/phoenix_hanchan_8dan_actor_any/actor_filter_min17.pth`
  - `data/manifests/normalized/v1/tenhou_phoenix_4y_full_e01_e16_v0.jsonl`

Current Step 5 actor-imitation policy:
- split files are now built from the `八段+ any-player` game cohort first, so train/val/test sizes are defined on the eligible game pool rather than on the broader `七段+` pool
- the Step 5 configs now also point to a precomputed `path_cache` artifact, so train/eval can skip rebuilding normalized absolute file lists on every run
- the Step 5 configs now point to a precomputed `actor_filter_index` artifact, so training/eval no longer need to rescan the full normalized manifest at startup when the index is present
- imitate only acting players with `min_actor_dan = 17` (`八段` and above)
- this is still enforced seat-by-seat via the actor filter index, so mixed tables remain eligible but only `八段+` seats contribute imitation targets

Current Step 6 actor-imitation policy:
- `step6_bc_debug.toml` stays on Batch D `8dan_actor_any`
- `step6_bc_medium_full8dan.toml`, `step6_bc_large_preflight_full8dan.toml`, `step6_bc_large_bounded_full8dan.toml`, `step6_bc_large_bounded_full8dan_8192.toml`, `step6_bc_large_model_probe_width.toml`, and `step6_bc_large_model_probe_depth.toml` use the full merged `8dan_actor_any` split with `min_actor_dan = 17`
- `step6_bc_full_9dan.toml` is the launch-candidate profile and uses the full merged `9dan_actor_any` split with `min_actor_dan = 18`
- official Step 6 configs keep `num_workers = 0` and `eval_num_workers = 0` as the known-safe loader policy
- official distributed Step 6 configs keep `torch.compile` disabled until DDP stability is fully proven
- official Step 6 DDP configs now default to `grad_accum_steps = 1`
- the current Step 6 handoff winner is the raw-loader async CPU-pipe path, not the older RAM-first prepared-data cache path
- the `*_r4` loader-saturation configs remain useful historical artifacts for the prepared-RAM exploration, but they are no longer the recommended launch path
- the checked-in post-`medium` gate now lives on `step6_bc_large_preflight_full8dan_r5.toml`, `step6_bc_large_bounded_full8dan_8192_r5.toml`, and `step6_bc_large_bounded_full8dan_r5.toml`
- the checked-in large-bounded DDP proof configs are now capped at 2 hours, while the preflight stays capped at 10 minutes
- the checked-in full-corpus Step 6 configs already carry concrete `max_steps` values plus wall-clock caps, but those step budgets are currently corpus-scaled estimates until the exact full `step_counts_v1.json` builds finish for the full `8dan` and `9dan` splits

Step 5 derived-artifact naming convention:
- split directories encode the game-level cohort, for example `phoenix_hanchan_8dan_actor_any`
- normalized path caches use `path_cache_v1.pth` inside the split directory
- seat-filter indexes use `actor_filter_min<dan>.pth` and `actor_filter_min<dan>.summary.json` inside the split directory
- optional learnable-step summaries use `step_counts_v1.json` inside the split directory

Step 6 launcher-related config fields:
- `bc.control.seed` and `bc.control.grad_accum_steps` make the effective global batch explicit
- `bc.control.max_runtime_seconds` lets a bounded config stop on wall-clock even when `max_steps` is larger
- `bc.distributed.*` controls DDP startup behavior
- `bc.launch.*` controls the one-command campaign runner outputs and device choices

To rebuild the normalized path cache for a split:

```bash
python scripts/build_bc_path_cache.py \
  --train-list /path/to/train.txt \
  --val-list /path/to/val.txt \
  --test-list /path/to/test.txt \
  --root-dir /home/drixs2050/Documents/Mortal \
  --output /path/to/path_cache_v1.pth \
  --summary /path/to/path_cache_v1.summary.json
```

To rebuild the actor filter index for a split:

```bash
python scripts/build_bc_actor_filter.py \
  --manifest /path/to/normalized_manifest.jsonl \
  --list /path/to/train.txt \
  --list /path/to/val.txt \
  --list /path/to/test.txt \
  --min-actor-dan 17 \
  --output /path/to/actor_filter_min17.pth \
  --summary /path/to/actor_filter_min17.summary.json
```

To run a hardened Step 6 campaign end to end:

```bash
python scripts/run_bc_campaign.py --config /home/drixs2050/Documents/Mortal/configs/step6_bc_large_bounded_full8dan.toml
```

To wait for `medium`, run the 8192-per-GPU preflight, and automatically choose the large-bounded config:

```bash
python scripts/run_step6_large_after_medium.py
```

That helper now defaults to the current raw-baseline `r5` preflight/high-batch/fallback configs rather than the older cache-oriented variants.

Optional: rebuild the learnable-step summary for the active Step 5 config if you want exact eval totals:

```bash
export MORTAL_CFG=/home/drixs2050/Documents/Mortal/configs/step5_bc_medium.toml
python scripts/build_bc_step_counts.py
```

Useful notes:
- the step counter reuses the configured `path_cache` and `actor_filter_index`, so it counts the same seat-filtered BC targets that train/eval will actually see
- `jobs` is optional; by default the counter now auto-scales up to a moderate machine-aware worker count, and `1` keeps it single-process
- if the spawned worker pool crashes on a large split, the counter now automatically retries that split in single-process mode instead of aborting the whole build
- this higher default only applies to the offline step-count preprocessing job; the checked-in BC training/eval worker counts stay conservative because the gameplay loader has shown intermittent native crashes under multiprocessing
- `step_counts_v1.json` is optional and is no longer part of the default Step 5 workflow
- if it exists and you manually point `bc.dataset.step_count_summary` at it, `mortal/eval_bc.py` can show an exact `current/total` batch progress bar for splits present in that summary

Current Step 5 checkpoint/eval policy:
- `val_steps` controls the fast sampled validation pass during training
- `val_batch_size` optionally lets sampled in-training `VAL` use a larger batch than the train step size
- `train_log_every` controls lightweight WandB training-only updates between save/eval windows
- `best_eval_split` and `best_eval_every` control deterministic checkpoint promotion
- BC now also supports the same linear-warmup cosine scheduler used by Mortal’s RL trainer via `[bc.optim.scheduler]`; when `max_steps = 0` in that scheduler block, it inherits `bc.control.max_steps`, and you can make warmup scale with run length via `warm_up_ratio`
- the checked-in CUDA Step 5 profiles now explicitly use `amp_dtype = 'bfloat16'` and `enable_tf32 = true`; BF16 autocast stays on, while grad scaling is only used for FP16 runs
- the checked-in medium/large CUDA profiles now also enable `torch.compile` and fused AdamW; preview keeps compile off to avoid paying compile startup overhead on a tiny sanity run
- `best_state_file` is chosen from the deterministic `BEST-*` eval, not only from the sampled in-training validation pass
- standalone `mortal/eval_bc.py` now also supports config-level defaults for `eval_batch_size` and `eval_max_batches`
- the checked-in medium profile uses a bounded standalone eval by default so val/test sweeps stay similar in scale to the in-training deterministic eval
- the checked-in large profile keeps the same corpus and optimizer but increases train batch size to `8192` while tightening save/eval cadence so checkpoint-selection frequency stays similar in sample terms; it now also runs both sampled `VAL` and deterministic `BEST-VAL` at `32768` batch size
- the current large-profile cadence is acceptable for bounded experimental work, but the observed `50`-step train windows plus `16`-batch validation already cost minutes per cycle, so a later truly large-scale run should validate less frequently or on a smaller sampled window
- use `--max-batches 0` when you explicitly want a full held-out sweep instead of the bounded default
- training logs now include throughput, and CUDA runs also log peak memory usage
- the checked-in Step 5 profiles use `multiprocessing_context = 'spawn'` when worker processes are enabled
- the checked-in Step 5 profiles use `eval_num_workers = 0` so validation and deterministic best-checkpoint evaluation avoid background prefetch crashes and surface loader errors more cleanly
- the current medium baseline profile also uses `num_workers = 0` for training because the BC loader has proven stable in single-process mode on the merged Phoenix corpus while worker-process loading is still showing intermittent native crashes

Current Step 5 WandB policy:
- `bc.wandb.enabled = true` in the checked-in Step 5 profiles
- training mirrors the same loss, accuracy, top-k, legality, throughput, memory, and deterministic `BEST-*` metrics that already go to TensorBoard
- `eval_bc.py` writes a separate WandB eval run by default when `bc.wandb.log_eval_runs = true`
- authenticate first with `source /home/drixs2050/Documents/Mortal/wandb_key.env`
