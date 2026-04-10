# Step 6 Phase 5: Multi-Worker DataLoader Experiment

Date: 2026-04-05

## Motivation

Phases 1-4 exhaustively proved that no single-process rearrangement of the CPU producer path
can close the gap between producer throughput (~2700 samples/s) and GPU consumption (~5000 samples/s).
The GPU idles ~65% of the time waiting for data.

Phase 3 tested `num_workers > 0` but rejected it because the batch stream hash did not match
the `num_workers=0` control. However, the semantic invariant was unnecessarily strict:

- Each training sample is an independent `(obs, action, mask)` tuple
- The model is feedforward (ResNet + linear DQN heads), no recurrence
- The current pipeline already mixes steps from different games freely within batches
- SGD is inherently stochastic — batch ordering does not affect training correctness

The only invariant that matters is **data coverage**: each epoch sees the same set of samples.
Sample ordering within and across batches is irrelevant for training correctness.

## Hypothesis

With `num_workers > 0`, PyTorch DataLoader spawns worker processes that each independently:
1. Read their file shard
2. Run Rust game-state conversion
3. Emit individual samples to the main process

The main process then collates samples into batches. This means:

- **File reading + Rust conversion** is parallelized across N worker processes
- **Collation** still happens in the main process, but is never blocked waiting for conversion
- The async CPU batch pipe thread continuously pulls ready batches from the DataLoader

The module benchmark showed conversion alone runs at 11K samples/s but the integrated producer
sits at ~2.7K. Workers should decouple conversion from collation and raise producer throughput.

## Hardware

- CPU: AMD Threadripper 3990X, 64 cores / 128 threads
- RAM: 256 GB (hard cap at 160 GB usage for this experiment)
- GPU: 2x NVIDIA A100 (cuda:0, cuda:1)
- GPU (reserved): 1x RTX 3070 (cuda:2, not used)

## Code Changes Required

### 1. Relax the `num_workers > 0` + `cpu_batch_pipe_backend='thread'` guard

In `mortal/train_bc.py` around line 2777, the current code raises a hard error when
`num_workers > 0` and `cpu_batch_pipe_backend='thread'`. This guard was a safety measure
for exact semantic matching, not an architectural incompatibility.

Change: convert the error to a logged warning so the async CPU batch pipe works with workers.

### 2. Add Phase 5 experiment helpers

In `mortal/step6_experiments.py`, add:
- `phase5_worker_overrides()`: config overrides for worker experiments
- `phase5_candidate_beats_control()`: promotion decision rule

## Experiment Design

### Phase 5a: Module Benchmark (single GPU, 192 files)

Purpose: Find the optimal worker count for producer throughput on a single rank.

| Experiment | num_workers | persistent_workers | prefetch_factor | cpu_ready_batches |
|---|---|---|---|---|
| control_b4 | 0 | N/A | N/A | 4 |
| w2 | 2 | true | 2 | 4 |
| w4 | 4 | true | 2 | 4 |
| w8 | 8 | true | 2 | 4 |
| w12 | 12 | true | 2 | 4 |
| w16 | 16 | true | 2 | 4 |
| w24 | 24 | true | 2 | 4 |

All variants keep: `pin_memory=true`, `prefetch_chunks=1`, `multiprocessing_context='spawn'`,
`cpu_batch_pipe_backend='thread'`.

### Phase 5b: 200-step Dual-A100 Preflight

Purpose: Validate the top 3 worker configs from Phase 5a on real DDP training.

Each preflight:
- 2 GPUs (A100 + A100)
- 200 steps, 720-second cap
- batch_size=8192 per GPU
- Full metrics collection (loader signature, GPU sampling, RSS tracking)

### Promotion Rule

A worker config is promoted over `control_b4` if:
- `completed_sps >= control_sps * 1.03` (3% throughput improvement), OR
- `(control_wait - candidate_wait) >= 0.05` AND `candidate_sps >= control_sps * 0.99`
- AND `peak_combined_rss_gib <= 150` (safety margin under 160 GB cap)
- AND `preflight_return_code == 0`

### Memory Budget

Per-worker RAM estimate: ~1-2 GB (Rust lib + Python interpreter + data buffers)
Worst case with 24 workers per rank * 2 ranks = 48 workers: ~48-96 GB
Plus 2 training processes: ~20-40 GB
Total estimate: 68-136 GB — within 160 GB cap

### Metrics to Collect

Module benchmark:
- producer SPS (raw_to_cpu_batches)
- training SPS (training_on_produced_batches)
- can_hide_producer flag

Preflight:
- completed-window SPS and wait fraction
- sustained SPS and wait fraction
- steady GPU ratio
- startup seconds
- late-window wait fraction (avg after step 125)
- loader signature: raw_read, collate_or_assemble, cpu_pipe_wait, device_prefetch_wait
- peak combined RSS and peak single-worker RSS

## Exit Criteria

- We have clear evidence whether `num_workers > 0` improves end-to-end training throughput
- We have the optimal worker count for this hardware
- We have memory usage data confirming the 160 GB cap is respected
- Results are logged in `artifacts/reports/step6_phase5_worker/` with full JSON and markdown
