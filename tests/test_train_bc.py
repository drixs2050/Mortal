import sys
import tempfile
import threading
import unittest
import queue
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from model import DQN  # noqa: E402
from lr_scheduler import LinearWarmUpCosineAnnealingLR  # noqa: E402
from train_bc import (  # noqa: E402
    ACTION_CATEGORY_NAMES,
    HandoffStats,
    GpuReadyBatch,
    PageableReadyBatch,
    PendingSlotRelease,
    PinnedReadyBatch,
    PinnedStageWorker,
    DeviceBatchPrefetcher,
    allocate_staging_batch,
    adamw_supports_fused,
    atomic_torch_save,
    action_categories,
    autocast_context_kwargs,
    batch_fits_staging_slot,
    current_learning_rate,
    device_memory_metrics,
    diff_host_mem,
    dqn_policy_outputs,
    grad_scaler_enabled,
    finalize_metric_sums,
    is_better_eval_result,
    loader_window_metrics,
    merge_window_observability,
    masked_logits,
    normalize_best_perf,
    empty_window_observability,
    observe_window_queue_depths,
    preflight_windows_stable,
    record_batch_stream,
    resolve_best_eval_every,
    resolve_amp_dtype,
    resolve_fused_optimizer_enabled,
    resolve_required_stage_splits,
    resolve_scheduler_config,
    handoff_window_metrics,
    stage_batch_into_slot,
    top_k_hits,
    throughput_metrics,
    update_metric_sums,
    empty_metric_sums,
)


class TrainBcHelpersTest(unittest.TestCase):
    def test_atomic_torch_save_replaces_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / 'state.pth'
            atomic_torch_save({'steps': 1}, checkpoint)
            atomic_torch_save({'steps': 2}, checkpoint)

            state = torch.load(checkpoint, map_location='cpu')
            self.assertEqual(state['steps'], 2)
            self.assertEqual(list(Path(tmpdir).glob('.state.pth.*.tmp')), [])

    def test_normalize_best_perf_supports_legacy_fields(self):
        perf = normalize_best_perf({'val_accuracy': 0.4, 'val_nll': 1.2}, 'val')
        self.assertEqual(perf['split'], 'val')
        self.assertAlmostEqual(perf['accuracy'], 0.4)
        self.assertAlmostEqual(perf['nll'], 1.2)
        self.assertEqual(perf['steps'], 0)

    def test_is_better_eval_result_prefers_accuracy_then_nll(self):
        best = {'accuracy': 0.5, 'nll': 1.0}
        self.assertTrue(is_better_eval_result({'accuracy': 0.6, 'nll': 2.0}, best))
        self.assertTrue(is_better_eval_result({'accuracy': 0.5, 'nll': 0.9}, best))
        self.assertFalse(is_better_eval_result({'accuracy': 0.5, 'nll': 1.1}, best))
        self.assertFalse(is_better_eval_result({'accuracy': 0.4, 'nll': 0.5}, best))

    def test_throughput_metrics_and_cpu_memory_metrics(self):
        metrics = throughput_metrics(sample_count=512, step_count=8, elapsed_seconds=2.0)
        self.assertAlmostEqual(metrics['steps_per_second'], 4.0)
        self.assertAlmostEqual(metrics['samples_per_second'], 256.0)
        self.assertAlmostEqual(metrics['elapsed_seconds'], 2.0)
        self.assertEqual(device_memory_metrics(torch.device('cpu')), {})

    def test_loader_window_metrics_reports_wait_fraction_and_queue_bytes(self):
        metrics = loader_window_metrics(
            previous_snapshot={
                'chunk_count_total': 2,
                'chunk_files_total': 10,
                'chunk_samples_total': 100,
                'chunk_bytes_total': 1024,
                'chunk_build_seconds_total': 4.0,
                'chunk_read_seconds_total': 1.0,
                'chunk_rust_convert_seconds_total': 0.4,
                'chunk_sample_materialize_seconds_total': 0.3,
                'collate_seconds_total': 0.2,
                'queued_bytes': 2048,
                'max_queued_bytes': 4096,
                'cpu_consumer_wait_seconds_total': 1.0,
                'cpu_blocked_put_seconds_total': 0.5,
            },
            current_snapshot={
                'chunk_count_total': 5,
                'chunk_files_total': 22,
                'chunk_samples_total': 220,
                'chunk_bytes_total': 5120,
                'chunk_build_seconds_total': 9.0,
                'chunk_read_seconds_total': 2.5,
                'chunk_rust_convert_seconds_total': 1.1,
                'chunk_sample_materialize_seconds_total': 0.9,
                'collate_seconds_total': 0.55,
                'queued_bytes': 3072,
                'max_queued_bytes': 6144,
                'ready_chunks': 3,
                'cpu_ready_batches': 4,
                'cpu_ready_bytes': 8192,
                'max_cpu_ready_batches': 4,
                'max_cpu_ready_bytes': 8192,
                'cpu_consumer_wait_seconds_total': 3.5,
                'cpu_blocked_put_seconds_total': 1.25,
                'last_chunk_files': 6,
                'last_chunk_samples': 60,
                'last_chunk_bytes': 2048,
                'last_chunk_build_seconds': 1.5,
            },
            wait_seconds=2.0,
            elapsed_seconds=10.0,
            cpu_pipe_wait_seconds_override=3.0,
        )
        self.assertAlmostEqual(metrics['wait_fraction'], 0.2)
        self.assertAlmostEqual(metrics['cpu_pipe_wait_fraction'], 0.3)
        self.assertAlmostEqual(metrics['cpu_producer_blocked_put_fraction'], 0.075)
        self.assertEqual(metrics['cpu_ready_batches'], 4)
        self.assertEqual(metrics['chunk_count'], 3)
        self.assertEqual(metrics['chunk_files'], 12)
        self.assertEqual(metrics['chunk_samples'], 120)
        self.assertAlmostEqual(metrics['avg_chunk_build_seconds'], 5.0 / 3.0)
        self.assertAlmostEqual(metrics['raw_read_seconds'], 1.5)
        self.assertAlmostEqual(metrics['rust_convert_seconds'], 0.7)
        self.assertAlmostEqual(metrics['sample_materialize_seconds'], 0.6)
        self.assertAlmostEqual(metrics['collate_or_assemble_seconds'], 0.35)
        self.assertAlmostEqual(metrics['device_prefetch_wait_seconds'], 0.0)
        self.assertAlmostEqual(metrics['cpu_pipe_empty_wait_fraction'], 0.3)

    def test_merge_window_observability_adds_queue_depths_and_rank_skew(self):
        observability = empty_window_observability()
        observability['fw_bw_opt_seconds'] = 3.0
        observability['ddp_sync_wait_seconds'] = 1.0
        observability['save_checkpoint_wait_seconds'] = 0.5
        observability['step_time_seconds_total'] = 4.0
        observability['step_count'] = 4
        observe_window_queue_depths(
            observability,
            loader_snapshot={'cpu_ready_batches': 2, 'ready_chunks': 1},
            queue_snapshot={'gpu_prefetch_depth': 2},
        )
        observe_window_queue_depths(
            observability,
            loader_snapshot={'cpu_ready_batches': 6, 'ready_chunks': 3},
            queue_snapshot={'gpu_prefetch_depth': 4},
        )
        runtime_metrics = {'elapsed_seconds': 10.0}
        loader_metrics = {}
        merge_window_observability(
            runtime_metrics=runtime_metrics,
            loader_metrics=loader_metrics,
            observability=observability,
            rank_step_time_ms_max=1200.0,
            rank_step_time_ms_min=800.0,
        )
        self.assertAlmostEqual(runtime_metrics['fw_bw_opt_fraction'], 0.3)
        self.assertAlmostEqual(runtime_metrics['ddp_sync_wait_fraction'], 0.1)
        self.assertAlmostEqual(runtime_metrics['save_checkpoint_wait_fraction'], 0.05)
        self.assertAlmostEqual(runtime_metrics['rank_step_time_ms'], 1000.0)
        self.assertAlmostEqual(runtime_metrics['rank_step_time_ms_max_minus_min'], 400.0)
        self.assertEqual(loader_metrics['cpu_ready_batches_min'], 2.0)
        self.assertEqual(loader_metrics['cpu_ready_batches_max'], 6.0)
        self.assertAlmostEqual(loader_metrics['cpu_ready_batches_avg'], 4.0)
        self.assertEqual(loader_metrics['loader_ready_chunks_max'], 3.0)
        self.assertEqual(loader_metrics['device_prefetch_depth_max'], 4.0)

    def test_preflight_windows_stable_requires_recent_consistency(self):
        stable = [
            {'samples_per_second': 6000.0, 'wait_fraction': 0.10},
            {'samples_per_second': 6120.0, 'wait_fraction': 0.095},
        ]
        unstable = [
            {'samples_per_second': 6000.0, 'wait_fraction': 0.10},
            {'samples_per_second': 7000.0, 'wait_fraction': 0.30},
        ]
        self.assertTrue(
            preflight_windows_stable(
                windows=stable,
                required_windows=2,
                tolerance=0.05,
            )
        )
        self.assertFalse(
            preflight_windows_stable(
                windows=unstable,
                required_windows=2,
                tolerance=0.05,
            )
        )

    def test_resolve_best_eval_every_preserves_explicit_zero(self):
        self.assertEqual(
            resolve_best_eval_every(control_cfg={'best_eval_every': 0}, save_every=200),
            0,
        )
        self.assertEqual(
            resolve_best_eval_every(control_cfg={}, save_every=200),
            200,
        )

    def test_resolve_required_stage_splits_matches_enabled_eval_paths(self):
        self.assertEqual(
            resolve_required_stage_splits(
                validation_enabled=False,
                best_eval_every=0,
                best_eval_split='val',
            ),
            ['train'],
        )
        self.assertEqual(
            resolve_required_stage_splits(
                validation_enabled=True,
                best_eval_every=0,
                best_eval_split='val',
            ),
            ['train', 'val'],
        )
        self.assertEqual(
            resolve_required_stage_splits(
                validation_enabled=False,
                best_eval_every=400,
                best_eval_split='test',
            ),
            ['train', 'test'],
        )

    def test_device_batch_prefetcher_moves_cpu_batches(self):
        loader_iter = iter([
            (
                torch.tensor([[1, 2]], dtype=torch.int16),
                torch.tensor([3], dtype=torch.int64),
                torch.ones((1, 46), dtype=torch.bool),
            ),
            (
                torch.tensor([[4, 5]], dtype=torch.int16),
                torch.tensor([6], dtype=torch.int64),
                torch.ones((1, 46), dtype=torch.bool),
            ),
        ])
        prefetcher = DeviceBatchPrefetcher(
            loader_iter,
            device=torch.device('cpu'),
            oracle=False,
            queue_depth=2,
        )
        first = next(prefetcher)
        second = next(prefetcher)
        self.assertEqual(first[0].dtype, torch.float32)
        self.assertEqual(first[1].dtype, torch.int64)
        self.assertEqual(first[2].dtype, torch.bool)
        self.assertEqual(second[0].tolist(), [[4.0, 5.0]])

    def test_device_batch_prefetcher_uses_smaller_startup_depth_then_refills(self):
        loader_iter = iter([
            (
                torch.tensor([[1, 2]], dtype=torch.int16),
                torch.tensor([3], dtype=torch.int64),
                torch.ones((1, 46), dtype=torch.bool),
            ),
            (
                torch.tensor([[4, 5]], dtype=torch.int16),
                torch.tensor([6], dtype=torch.int64),
                torch.ones((1, 46), dtype=torch.bool),
            ),
            (
                torch.tensor([[7, 8]], dtype=torch.int16),
                torch.tensor([9], dtype=torch.int64),
                torch.ones((1, 46), dtype=torch.bool),
            ),
        ])
        prefetcher = DeviceBatchPrefetcher(
            loader_iter,
            device=torch.device('cpu'),
            oracle=False,
            queue_depth=3,
            startup_queue_depth=1,
        )
        self.assertEqual(len(prefetcher.ready), 1)
        first = next(prefetcher)
        self.assertEqual(first[0].tolist(), [[1.0, 2.0]])
        self.assertEqual(len(prefetcher.ready), 2)

    def test_device_batch_prefetcher_accepts_handoff_pinning_on_cpu(self):
        loader_iter = iter([
            (
                torch.tensor([[1, 2]], dtype=torch.int16),
                torch.tensor([3], dtype=torch.int64),
                torch.ones((1, 46), dtype=torch.bool),
            ),
        ])
        prefetcher = DeviceBatchPrefetcher(
            loader_iter,
            device=torch.device('cpu'),
            oracle=False,
            queue_depth=1,
            pin_handoff_batches=True,
        )
        batch = next(prefetcher)
        self.assertEqual(batch[0].dtype, torch.float32)
        self.assertEqual(batch[0].tolist(), [[1.0, 2.0]])

    def test_stage_batch_into_slot_supports_smaller_leading_dim(self):
        slot = allocate_staging_batch(
            (
                torch.zeros((3, 2), dtype=torch.int16),
                torch.zeros((3,), dtype=torch.int64),
                torch.zeros((3, 46), dtype=torch.bool),
            ),
            pin_memory=False,
        )
        batch = (
            torch.tensor([[1, 2]], dtype=torch.int16),
            torch.tensor([3], dtype=torch.int64),
            torch.ones((1, 46), dtype=torch.bool),
        )
        self.assertTrue(batch_fits_staging_slot(slot, batch))
        staged = stage_batch_into_slot(slot, batch)
        self.assertEqual(staged[0].shape, torch.Size([1, 2]))
        self.assertEqual(staged[1].shape, torch.Size([1]))
        self.assertEqual(staged[2].shape, torch.Size([1, 46]))
        self.assertEqual(staged[0].tolist(), [[1, 2]])
        self.assertEqual(slot[0][0].tolist(), [1, 2])
        self.assertEqual(slot[1][0].item(), 3)
        self.assertTrue(bool(slot[2][0].all().item()))

    def test_record_batch_stream_is_noop_for_cpu_tensors(self):
        batch = (
            torch.tensor([[1, 2]], dtype=torch.float32),
            torch.tensor([3], dtype=torch.int64),
        )
        record_batch_stream(batch, object())
        self.assertEqual(batch[0].tolist(), [[1.0, 2.0]])

    def test_stage_thread_preserves_order(self):
        batches = [
            PageableReadyBatch(
                batch=(
                    torch.tensor([[idx, idx + 1]], dtype=torch.int16),
                    torch.tensor([idx], dtype=torch.int64),
                    torch.ones((1, 46), dtype=torch.bool),
                ),
                batch_idx=idx,
                nsamples=1,
                nbytes=100,
                t_cpu_ready_s=0.0,
            )
            for idx in range(8)
        ]

        def next_pageable():
            if not batches:
                raise StopIteration
            return batches.pop(0)

        slot_template = allocate_staging_batch(
            (
                torch.zeros((1, 2), dtype=torch.int16),
                torch.zeros((1,), dtype=torch.int64),
                torch.zeros((1, 46), dtype=torch.bool),
            ),
            pin_memory=False,
        )
        slots = [slot_template, allocate_staging_batch(slot_template, pin_memory=False)]
        free_slot_q = queue.Queue(maxsize=2)
        free_slot_q.put(0)
        free_slot_q.put(1)
        pinned_ready_q = queue.Queue(maxsize=2)
        stats = HandoffStats()
        staged = []
        worker = PinnedStageWorker(
            next_pageable=next_pageable,
            slots=slots,
            free_slot_q=free_slot_q,
            pinned_ready_q=pinned_ready_q,
            stop_evt=threading.Event(),
            stats=stats,
        ).start()
        try:
            for _ in range(8):
                item = pinned_ready_q.get(timeout=1.0)
                staged.append(item.batch_idx)
                free_slot_q.put(item.slot_id)
            self.assertEqual(staged, list(range(8)))
        finally:
            worker.close()

    def test_slot_reuse_waits_for_copy_event(self):
        class FakeEvent:
            def __init__(self, ready=False):
                self.ready = ready

            def query(self):
                return self.ready

            def elapsed_time(self, other):
                return 1.25

        prefetcher = DeviceBatchPrefetcher(
            iter(()),
            device=torch.device('cpu'),
            oracle=False,
            queue_depth=1,
            handoff_stage_backend='thread',
        )
        prefetcher.free_slot_q = queue.Queue(maxsize=1)
        prefetcher.slot_release.append(
            PendingSlotRelease(
                slot_id=0,
                nbytes=128,
                copy_end_event=FakeEvent(ready=False),
                copy_start_event=FakeEvent(ready=True),
            )
        )
        prefetcher._occupied_slot_bytes[0] = 128
        prefetcher._reclaim_completed_slots()
        self.assertEqual(prefetcher.free_slot_q.qsize(), 0)
        prefetcher.slot_release[0].copy_end_event.ready = True
        prefetcher._reclaim_completed_slots()
        self.assertEqual(prefetcher.free_slot_q.qsize(), 1)
        self.assertEqual(prefetcher._current_pinned_batch_bytes(), 0)

    def test_inline_reclaim_returns_slot_to_inline_pool(self):
        class FakeEvent:
            def __init__(self, ready=False):
                self.ready = ready

            def query(self):
                return self.ready

            def elapsed_time(self, other):
                return 1.0

        prefetcher = DeviceBatchPrefetcher(
            device=torch.device('cpu'),
            oracle=False,
            queue_depth=1,
            pin_handoff_batches=False,
            handoff_stage_backend='inline',
        )
        prefetcher.pin_handoff_batches = True
        prefetcher.handoff_stage_backend = 'inline'
        prefetcher._free_pinned_slots.clear()
        prefetcher.slot_release.append(
            PendingSlotRelease(
                slot_id=3,
                nbytes=128,
                copy_end_event=FakeEvent(ready=True),
                copy_start_event=FakeEvent(ready=True),
            )
        )
        prefetcher._occupied_slot_bytes[3] = 128
        prefetcher._reclaim_completed_slots()
        self.assertEqual(list(prefetcher._free_pinned_slots), [3])
        self.assertIsNone(prefetcher.free_slot_q)

    def test_inline_handoff_waits_for_slot_progress_before_stopping(self):
        class FakeEvent:
            def __init__(self):
                self.ready = False

            def query(self):
                return self.ready

            def synchronize(self):
                self.ready = True

        prefetcher = DeviceBatchPrefetcher(
            device=torch.device('cpu'),
            oracle=False,
            queue_depth=1,
        )
        prefetcher.pin_handoff_batches = True
        prefetcher.handoff_stage_backend = 'inline'
        prefetcher.free_slot_q = queue.Queue(maxsize=1)
        prefetcher.slot_release.append(
            PendingSlotRelease(
                slot_id=0,
                nbytes=64,
                copy_end_event=FakeEvent(),
                copy_start_event=None,
            )
        )
        prefetcher._occupied_slot_bytes[0] = 64

        fill_calls = {'count': 0}

        def fake_fill_ready_queue(*, target_depth=None):
            fill_calls['count'] += 1
            if not prefetcher.slot_release and not prefetcher.gpu_ready:
                prefetcher.gpu_ready.append(
                    GpuReadyBatch(
                        batch_idx=0,
                        gpu_batch=('ready',),
                        nbytes=64,
                        copy_end_event=None,
                        copy_start_event=None,
                    )
                )

        prefetcher._fill_ready_queue = fake_fill_ready_queue

        batch = next(prefetcher)

        self.assertEqual(batch, ('ready',))
        self.assertGreaterEqual(fill_calls['count'], 2)
        self.assertEqual(list(prefetcher._free_pinned_slots), [0])
        self.assertEqual(prefetcher._current_pinned_batch_bytes(), 0)

    def test_thread_handoff_waits_for_reclaimed_slot_then_receives_ready_batch(self):
        class FakeEvent:
            def __init__(self, ready=False):
                self.ready = ready

            def query(self):
                return self.ready

            def elapsed_time(self, other):
                return 1.0

        prefetcher = DeviceBatchPrefetcher(
            device=torch.device('cpu'),
            oracle=False,
            queue_depth=1,
            pin_handoff_batches=False,
            handoff_stage_backend='thread',
        )
        prefetcher.pin_handoff_batches = True
        prefetcher.handoff_stage_backend = 'thread'
        prefetcher.stream = object()
        prefetcher.stop_evt = threading.Event()
        prefetcher.free_slot_q = queue.Queue(maxsize=1)
        prefetcher.pinned_ready_q = queue.Queue(maxsize=1)
        release_event = FakeEvent(ready=False)
        prefetcher.slot_release.append(
            PendingSlotRelease(
                slot_id=0,
                nbytes=64,
                copy_end_event=release_event,
                copy_start_event=FakeEvent(ready=True),
            )
        )
        prefetcher._occupied_slot_bytes[0] = 64

        ready_item = PinnedReadyBatch(
            slot_id=0,
            batch=('staged',),
            batch_idx=7,
            nsamples=1,
            nbytes=64,
            t_stage_start_s=0.0,
            t_stage_done_s=0.1,
        )

        def producer():
            slot_id = prefetcher.free_slot_q.get(timeout=1.0)
            self.assertEqual(slot_id, 0)
            prefetcher.pinned_ready_q.put(ready_item, timeout=1.0)

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        try:
            release_event.ready = True
            item = prefetcher._get_next_pinned_ready()
        finally:
            producer_thread.join(timeout=1.0)
        self.assertEqual(item.batch_idx, 7)
        self.assertEqual(item.slot_id, 0)

    def test_inline_and_thread_backends_match_batch_indices(self):
        ready_batches = [
            PageableReadyBatch(
                batch=(
                    torch.tensor([[idx, idx + 1]], dtype=torch.int16),
                    torch.tensor([idx], dtype=torch.int64),
                    torch.ones((1, 46), dtype=torch.bool),
                ),
                batch_idx=idx,
                nsamples=1,
                nbytes=100,
                t_cpu_ready_s=0.0,
            )
            for idx in range(4)
        ]
        slot_template = allocate_staging_batch(
            ready_batches[0].batch,
            pin_memory=False,
        )
        inline_slots = [allocate_staging_batch(slot_template, pin_memory=False) for _ in range(2)]
        inline_batch_indices = []
        free_inline = [0, 1]
        for ready in ready_batches:
            slot_id = free_inline.pop(0)
            stage_batch_into_slot(inline_slots[slot_id], ready.batch)
            inline_batch_indices.append(ready.batch_idx)
            free_inline.append(slot_id)

        thread_batches = list(ready_batches)

        def next_pageable():
            if not thread_batches:
                raise StopIteration
            return thread_batches.pop(0)

        thread_slots = [allocate_staging_batch(slot_template, pin_memory=False) for _ in range(2)]
        free_slot_q = queue.Queue(maxsize=2)
        free_slot_q.put(0)
        free_slot_q.put(1)
        pinned_ready_q = queue.Queue(maxsize=2)
        worker = PinnedStageWorker(
            next_pageable=next_pageable,
            slots=thread_slots,
            free_slot_q=free_slot_q,
            pinned_ready_q=pinned_ready_q,
            stop_evt=threading.Event(),
            stats=HandoffStats(),
        ).start()
        thread_batch_indices = []
        try:
            for _ in range(4):
                item = pinned_ready_q.get(timeout=1.0)
                thread_batch_indices.append(item.batch_idx)
                free_slot_q.put(item.slot_id)
        finally:
            worker.close()
        self.assertEqual(thread_batch_indices, inline_batch_indices)

    def test_close_unblocks_stage_worker(self):
        ready_batch = PageableReadyBatch(
            batch=(
                torch.tensor([[1, 2]], dtype=torch.int16),
                torch.tensor([3], dtype=torch.int64),
                torch.ones((1, 46), dtype=torch.bool),
            ),
            batch_idx=0,
            nsamples=1,
            nbytes=100,
            t_cpu_ready_s=0.0,
        )
        batches = [ready_batch]

        def next_pageable():
            if not batches:
                raise StopIteration
            return batches.pop(0)

        pinned_ready_q = queue.Queue(maxsize=1)
        worker = PinnedStageWorker(
            next_pageable=next_pageable,
            slots=[allocate_staging_batch(ready_batch.batch, pin_memory=False)],
            free_slot_q=queue.Queue(maxsize=1),
            pinned_ready_q=pinned_ready_q,
            stop_evt=threading.Event(),
            stats=HandoffStats(),
        ).start()
        worker.close()
        self.assertFalse(worker._thread.is_alive())

    def test_handoff_host_mem_diff_helper_smoke(self):
        self.assertEqual(
            diff_host_mem(None, None),
            {
                'host_num_alloc_delta': 0,
                'host_num_free_delta': 0,
                'host_alloc_time_us_delta': 0,
                'host_free_time_us_delta': 0,
                'host_active_bytes_cur': 0,
                'host_allocated_bytes_cur': 0,
            },
        )
        cur = {
            'num_host_alloc': 10,
            'num_host_free': 4,
            'host_alloc_time.total': 300,
            'host_free_time.total': 120,
            'active_bytes.current': 4096,
            'allocated_bytes.current': 8192,
        }
        prev = {
            'num_host_alloc': 7,
            'num_host_free': 2,
            'host_alloc_time.total': 250,
            'host_free_time.total': 80,
            'active_bytes.current': 2048,
            'allocated_bytes.current': 4096,
        }
        self.assertEqual(
            diff_host_mem(cur, prev),
            {
                'host_num_alloc_delta': 3,
                'host_num_free_delta': 2,
                'host_alloc_time_us_delta': 50,
                'host_free_time_us_delta': 40,
                'host_active_bytes_cur': 4096,
                'host_allocated_bytes_cur': 8192,
            },
        )

    def test_handoff_window_metrics_smoke(self):
        metrics = handoff_window_metrics(
            previous_snapshot={
                'cpu_ready_wait_s_total': 1.0,
                'stage_free_slot_wait_s_total': 0.25,
                'stage_copy_s_total': 0.5,
                'pinned_ready_wait_s_total': 0.1,
                'h2d_submit_s_total': 0.2,
                'copy_ready_on_pop_total': 4,
                'copy_not_ready_on_pop_total': 2,
                'h2d_copy_ms_total': 10.0,
                'h2d_copy_count_total': 4,
                'host_memory_stats': {
                    'num_host_alloc': 3,
                    'num_host_free': 1,
                    'host_alloc_time.total': 40,
                    'host_free_time.total': 20,
                    'active_bytes.current': 100,
                    'allocated_bytes.current': 200,
                },
            },
            current_snapshot={
                'cpu_ready_wait_s_total': 3.0,
                'stage_free_slot_wait_s_total': 0.75,
                'stage_copy_s_total': 1.5,
                'pinned_ready_wait_s_total': 0.4,
                'h2d_submit_s_total': 0.9,
                'copy_ready_on_pop_total': 9,
                'copy_not_ready_on_pop_total': 3,
                'h2d_copy_ms_total': 22.0,
                'h2d_copy_count_total': 7,
                'gpu_prefetch_depth': 2,
                'free_handoff_slots_approx': 1,
                'pinned_ready_q_approx': 2,
                'host_memory_stats': {
                    'num_host_alloc': 4,
                    'num_host_free': 2,
                    'host_alloc_time.total': 55,
                    'host_free_time.total': 25,
                    'active_bytes.current': 120,
                    'allocated_bytes.current': 220,
                },
            },
            elapsed_seconds=4.0,
        )
        self.assertAlmostEqual(metrics['cpu_ready_wait_fraction'], 0.5)
        self.assertAlmostEqual(metrics['stage_copy_fraction'], 0.25)
        self.assertAlmostEqual(metrics['h2d_submit_fraction'], 0.175)
        self.assertAlmostEqual(metrics['copy_ready_on_pop_fraction'], 5 / 6)
        self.assertAlmostEqual(metrics['h2d_copy_ms_avg'], 4.0)
        self.assertEqual(metrics['free_handoff_slots_approx'], 1)
        self.assertEqual(metrics['pinned_ready_q_approx'], 2)

    def test_resolve_amp_dtype_and_scaler_policy(self):
        self.assertEqual(resolve_amp_dtype({'amp_dtype': 'bf16'}), torch.bfloat16)
        self.assertEqual(resolve_amp_dtype({'amp_dtype': 'float16'}), torch.float16)
        self.assertTrue(
            grad_scaler_enabled(
                enable_amp=True,
                amp_dtype=torch.float16,
                device=torch.device('cuda'),
            )
        )
        self.assertFalse(
            grad_scaler_enabled(
                enable_amp=True,
                amp_dtype=torch.bfloat16,
                device=torch.device('cuda'),
            )
        )
        self.assertFalse(
            grad_scaler_enabled(
                enable_amp=False,
                amp_dtype=torch.float16,
                device=torch.device('cuda'),
            )
        )

    def test_autocast_context_kwargs_sets_cuda_dtype_only(self):
        cuda_kwargs = autocast_context_kwargs(
            device=torch.device('cuda'),
            enable_amp=True,
            amp_dtype=torch.bfloat16,
        )
        self.assertEqual(cuda_kwargs['device_type'], 'cuda')
        self.assertEqual(cuda_kwargs['dtype'], torch.bfloat16)
        cpu_kwargs = autocast_context_kwargs(
            device=torch.device('cpu'),
            enable_amp=False,
            amp_dtype=torch.bfloat16,
        )
        self.assertEqual(cpu_kwargs['device_type'], 'cpu')
        self.assertNotIn('dtype', cpu_kwargs)

    def test_resolve_fused_optimizer_enabled_is_device_aware(self):
        self.assertIsInstance(adamw_supports_fused(), bool)
        self.assertEqual(
            resolve_fused_optimizer_enabled(
                optim_cfg={'enable_fused_optimizer': True},
                device=torch.device('cpu'),
            ),
            False,
        )
        self.assertEqual(
            resolve_fused_optimizer_enabled(
                optim_cfg={'enable_fused_optimizer': False},
                device=torch.device('cuda'),
            ),
            False,
        )
        self.assertEqual(
            resolve_fused_optimizer_enabled(
                optim_cfg={'enable_fused_optimizer': True},
                device=torch.device('cuda'),
            ),
            adamw_supports_fused(),
        )

    def test_resolve_scheduler_config_uses_control_max_steps_by_default(self):
        resolved = resolve_scheduler_config(
            optim_cfg={
                'lr': 1e-4,
                'scheduler': {
                    'final': 1e-5,
                    'warm_up_steps': 25,
                    'max_steps': 0,
                    'init': 1e-6,
                },
            },
            max_steps=500,
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved['peak'], 1e-4)
        self.assertEqual(resolved['final'], 1e-5)
        self.assertEqual(resolved['warm_up_steps'], 25)
        self.assertEqual(resolved['max_steps'], 500)
        self.assertEqual(resolved['init'], 1e-6)

    def test_resolve_scheduler_config_supports_dynamic_warmup_ratio(self):
        resolved = resolve_scheduler_config(
            optim_cfg={
                'lr': 1e-4,
                'scheduler': {
                    'peak': 4e-4,
                    'final': 4e-5,
                    'warm_up_ratio': 0.05,
                    'max_steps': 0,
                    'init': 1e-6,
                },
            },
            max_steps=500,
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved['warm_up_steps'], 25)
        self.assertAlmostEqual(resolved['warm_up_ratio'], 0.05)
        self.assertEqual(resolved['max_steps'], 500)

    def test_resolve_scheduler_config_rejects_ratio_and_steps_together(self):
        with self.assertRaises(ValueError):
            resolve_scheduler_config(
                optim_cfg={
                    'lr': 1e-4,
                    'scheduler': {
                        'warm_up_steps': 25,
                        'warm_up_ratio': 0.05,
                    },
                },
                max_steps=500,
            )

    def test_resolve_scheduler_config_requires_bounded_run(self):
        with self.assertRaises(ValueError):
            resolve_scheduler_config(
                optim_cfg={
                    'lr': 1e-4,
                    'scheduler': {
                        'final': 1e-5,
                    },
                },
                max_steps=0,
            )

    def test_action_categories_cover_expected_labels(self):
        labels = torch.tensor([0, 36, 37, 38, 40, 41, 42, 43, 44, 45])
        cats = action_categories(labels)
        self.assertEqual(cats.tolist(), [0, 0, 1, 2, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(ACTION_CATEGORY_NAMES), 8)

    def test_top_k_hits_respects_masked_logits(self):
        logits = torch.tensor([
            [4.0, 3.0, -torch.inf, 0.5],
            [1.0, -torch.inf, 0.1, 0.0],
        ])
        labels = torch.tensor([1, 2])
        hits = top_k_hits(logits, labels, top_k=2)
        self.assertEqual(hits.tolist(), [True, True])

    def test_dqn_policy_outputs_matches_masked_action_logits(self):
        dqn = DQN(version=4)
        phi = torch.randn(3, 1024)
        masks = torch.tensor([
            [True] * 46,
            [True] * 23 + [False] * 23,
            [False, True] * 23,
        ], dtype=torch.bool)
        raw_logits, masked_scores = dqn_policy_outputs(dqn, phi, masks)
        masked_action_logits = masked_logits(raw_logits, masks)
        self.assertTrue(torch.equal(raw_logits, dqn.action_logits(phi)))
        self.assertTrue(torch.equal(masked_scores.argmax(dim=-1), masked_action_logits.argmax(dim=-1)))
        actions = torch.tensor([0, 1, 3], dtype=torch.int64)
        self.assertTrue(
            torch.allclose(
                torch.nn.functional.cross_entropy(masked_scores, actions),
                torch.nn.functional.cross_entropy(masked_action_logits, actions),
            )
        )

    def test_metric_sums_include_category_accuracy(self):
        raw_logits = torch.tensor([
            [3.0, 2.0, 1.0, 0.0],
            [5.0, 1.0, 2.0, 3.0],
        ])
        masks = torch.tensor([
            [True, True, False, False],
            [False, True, True, True],
        ])
        masked = masked_logits(raw_logits, masks)
        actions = torch.tensor([1, 3])
        stats = empty_metric_sums()
        update_metric_sums(
            stats,
            loss=torch.tensor(0.5),
            masked_pred=masked.argmax(dim=-1),
            raw_pred=raw_logits.argmax(dim=-1),
            masked_scores=masked,
            actions=actions,
            masks=masks,
            top_k=2,
        )
        metrics = finalize_metric_sums(stats)
        self.assertAlmostEqual(metrics['nll'], 0.5)
        self.assertAlmostEqual(metrics['accuracy'], 0.5)
        self.assertAlmostEqual(metrics['topk_accuracy'], 1.0)
        self.assertAlmostEqual(metrics['legal_rate'], 0.5)
        self.assertIn('discard', metrics['category_accuracy'])
        self.assertAlmostEqual(metrics['category_accuracy']['discard'], 0.5)

    def test_current_learning_rate_reads_optimizer_group(self):
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([param], lr=3e-4)
        self.assertAlmostEqual(current_learning_rate(optimizer), 3e-4)

    def test_scheduler_accepts_warmup_ratio_metadata(self):
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([param], lr=1.0)
        scheduler = LinearWarmUpCosineAnnealingLR(
            optimizer,
            peak=4e-4,
            final=4e-5,
            warm_up_steps=25,
            warm_up_ratio=0.05,
            max_steps=500,
            init=1e-6,
        )
        self.assertAlmostEqual(scheduler.warm_up_ratio, 0.05)


class DqnPolicyLogitsTest(unittest.TestCase):
    def test_masked_argmax_matches_action_logits_argmax(self):
        dqn = DQN(version=4)
        phi = torch.randn(5, 1024)
        masks = torch.zeros(5, 46, dtype=torch.bool)
        masks[0, :2] = True
        masks[1, 1:3] = True
        masks[2, 2:5] = True
        masks[3, 3:6] = True
        masks[4, 4:7] = True
        q_values = dqn(phi, masks)
        masked_policy = masked_logits(dqn.action_logits(phi), masks)
        self.assertTrue(torch.equal(q_values.argmax(dim=-1), masked_policy.argmax(dim=-1)))


if __name__ == '__main__':
    unittest.main()
