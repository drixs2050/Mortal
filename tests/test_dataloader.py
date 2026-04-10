import os
import sys
import time
import unittest
import gzip
import pickle
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

from dataloader import (  # noqa: E402
    ActionFileDatasetsIter,
    AsyncCpuBatchPipe,
    LoaderStats,
    SyncCpuBatchPipe,
    build_action_file_dataloader,
    resolve_prefetch_budget_bytes,
    suggest_file_batch_size,
)


class _FakeGame:
    def __init__(self, base: int, player_id: int = 0):
        self.base = base
        self.player_id = player_id

    def take_player_id(self):
        return self.player_id

    def take_obs(self):
        return np.array([
            [self.base],
            [self.base + 100],
        ], dtype=np.int16)

    def take_actions(self):
        return np.array([self.base, self.base + 1], dtype=np.int64)

    def take_masks(self):
        return np.ones((2, 46), dtype=np.bool_)


class _FakeGameplayLoader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_gz_log_files(self, file_list):
        files = []
        for filename in file_list:
            idx = int(filename[1:])
            files.append([
                _FakeGame(idx * 10, player_id=idx % 2),
            ])
        return files

    def load_gz_log_blobs(self, gzip_blobs):
        files = []
        for payload in gzip_blobs:
            idx = int(gzip.decompress(payload).decode('utf-8').strip()[1:])
            files.append([
                _FakeGame(idx * 10, player_id=idx % 2),
            ])
        return files


class _FakePackedRawSource:
    def __init__(self, pack_path, index_path):
        del pack_path, index_path

    def read(self, file_id):
        return gzip.compress(str(file_id).encode('utf-8'))

    def read_many(self, file_ids):
        return [self.read(file_id) for file_id in file_ids]

    def close(self):
        return None


class ActionFileDatasetsIterTest(unittest.TestCase):
    def test_loader_stats_is_picklable_for_spawn_workers(self):
        stats = LoaderStats()
        stats.update_queue_state(
            queued_bytes=1,
            ready_chunks=2,
            ready_bytes=3,
            inflight_bytes=4,
        )
        restored = pickle.loads(pickle.dumps(stats))
        self.assertEqual(restored.snapshot()['queued_bytes'], 1)
        self.assertEqual(restored.snapshot()['ready_chunks'], 2)
        self.assertEqual(restored.snapshot()['ready_bytes'], 3)
        self.assertEqual(restored.snapshot()['inflight_bytes'], 4)

    def assert_batch_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
            return
        if isinstance(left, np.ndarray):
            self.assertTrue(np.array_equal(left, right))
            return
        if isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for lhs, rhs in zip(left, right):
                self.assert_batch_tree_equal(lhs, rhs)
            return
        self.assertEqual(left, right)

    def collect_loader_batches(self, file_list, *, shuffle=False):
        loader, _loader_stats = build_action_file_dataloader(
            version=4,
            file_list=file_list,
            oracle=False,
            file_batch_size=2,
            num_epochs=1,
            cycle=False,
            shuffle=shuffle,
            batch_size=3,
            prebatched=False,
            num_workers=0,
            pin_memory=False,
        )
        return loader

    def collect_loader_with_stats(self, file_list):
        return build_action_file_dataloader(
            version=4,
            file_list=file_list,
            oracle=False,
            file_batch_size=2,
            num_epochs=1,
            cycle=False,
            shuffle=False,
            batch_size=3,
            prebatched=False,
            num_workers=0,
            pin_memory=False,
        )

    def collect_prebatched_loader_batches(
        self,
        file_list,
        *,
        prebatch_layout='chunk',
        prebatch_spill_across_chunks=False,
        shuffle=False,
    ):
        loader, _loader_stats = build_action_file_dataloader(
            version=4,
            file_list=file_list,
            oracle=False,
            file_batch_size=2,
            num_epochs=1,
            cycle=False,
            shuffle=shuffle,
            batch_size=3,
            prebatched=True,
            prebatch_layout=prebatch_layout,
            prebatch_spill_across_chunks=prebatch_spill_across_chunks,
            num_workers=0,
            pin_memory=False,
        )
        return list(loader)

    def collect_preassembled_loader_batches(
        self,
        file_list,
        *,
        shuffle=False,
        block_target_samples=65536,
    ):
        loader, _loader_stats = build_action_file_dataloader(
            version=4,
            file_list=file_list,
            oracle=False,
            file_batch_size=2,
            num_epochs=1,
            cycle=False,
            shuffle=shuffle,
            batch_size=3,
            prebatched=False,
            num_workers=0,
            pin_memory=False,
            loader_mode='preassembled_batches',
            loader_block_target_samples=block_target_samples,
        )
        return list(loader)

    def collect_batches(
        self,
        *,
        prefetch_strategy,
        prefetch_budget_bytes,
        allowed_player_ids_by_path=None,
        prebatch_layout='chunk',
        prebatch_shuffle_mode='sample',
        prebatch_spill_across_chunks=False,
        shuffle=False,
        raw_source_backend='files',
    ):
        dataset = ActionFileDatasetsIter(
            version=4,
            file_list=['f0', 'f1', 'f2', 'f3'],
            oracle=False,
            file_batch_size=2,
            num_epochs=1,
            cycle=False,
            shuffle=shuffle,
            allowed_player_ids_by_path=allowed_player_ids_by_path,
            prefetch_strategy=prefetch_strategy,
            prefetch_budget_bytes=prefetch_budget_bytes,
            prefetch_target_chunk_bytes=1024,
            prefetch_threads=2,
            batch_size=3,
            prebatched=True,
            prebatch_layout=prebatch_layout,
            prebatch_shuffle_mode=prebatch_shuffle_mode,
            prebatch_spill_across_chunks=prebatch_spill_across_chunks,
            raw_source_backend=raw_source_backend,
            raw_pack_path='raw.pack',
            raw_pack_index_path='raw.index.json',
        )
        rows = []
        for obs, actions, masks in dataset:
            rows.append((obs.tolist(), actions.tolist(), list(masks.shape)))
        return rows

    def collect_entries(self, *, prefetch_chunks, allowed_player_ids_by_path=None, raw_source_backend='files'):
        dataset = ActionFileDatasetsIter(
            version=4,
            file_list=['f0', 'f1', 'f2', 'f3'],
            oracle=False,
            file_batch_size=2,
            num_epochs=1,
            cycle=False,
            shuffle=False,
            allowed_player_ids_by_path=allowed_player_ids_by_path,
            prefetch_chunks=prefetch_chunks,
            raw_source_backend=raw_source_backend,
            raw_pack_path='raw.pack',
            raw_pack_index_path='raw.index.json',
        )
        return [
            (entry[0].tolist(), int(entry[1]), entry[2].shape)
            for entry in dataset
        ]

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_prefetch_chunks_preserves_entry_order(self):
        baseline = self.collect_entries(prefetch_chunks=0)
        prefetched = self.collect_entries(prefetch_chunks=1)
        self.assertEqual(prefetched, baseline)

    @patch('dataloader.PackedRawSource', _FakePackedRawSource)
    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_raw_pack_prefetch_chunks_preserves_entry_order(self):
        baseline = self.collect_entries(prefetch_chunks=0)
        packed = self.collect_entries(prefetch_chunks=1, raw_source_backend='raw_pack')
        self.assertEqual(packed, baseline)

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_prefetch_chunks_preserves_allowed_player_filter(self):
        allowed = {
            'f0': (0,),
            'f1': (1,),
            'f2': (),
            'f3': (1,),
        }
        filtered = self.collect_entries(
            prefetch_chunks=1,
            allowed_player_ids_by_path=allowed,
        )
        self.assertEqual(
            filtered,
            [
                ([0], 0, (46,)),
                ([100], 1, (46,)),
                ([10], 10, (46,)),
                ([110], 11, (46,)),
                ([30], 30, (46,)),
                ([130], 31, (46,)),
            ],
        )

    @patch('dataloader.PackedRawSource', _FakePackedRawSource)
    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_raw_pack_prefetch_chunks_preserves_allowed_player_filter(self):
        allowed = {
            'f0': (0,),
            'f1': (1,),
            'f2': (),
            'f3': (1,),
        }
        filtered = self.collect_entries(
            prefetch_chunks=1,
            allowed_player_ids_by_path=allowed,
            raw_source_backend='raw_pack',
        )
        self.assertEqual(
            filtered,
            [
                ([0], 0, (46,)),
                ([100], 1, (46,)),
                ([10], 10, (46,)),
                ([110], 11, (46,)),
                ([30], 30, (46,)),
                ([130], 31, (46,)),
            ],
        )

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_prebatched_static_strategy_preserves_batch_order(self):
        batches = self.collect_batches(
            prefetch_strategy='static_chunks',
            prefetch_budget_bytes=0,
        )
        self.assertEqual(
            batches,
            [
                ([[0], [100], [10]], [0, 1, 10], [3, 46]),
                ([[110]], [11], [1, 46]),
                ([[20], [120], [30]], [20, 21, 30], [3, 46]),
                ([[130]], [31], [1, 46]),
            ],
        )

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_loader_stats_capture_stage_and_collate_timings(self):
        loader, loader_stats = self.collect_loader_with_stats(['f0', 'f1', 'f2', 'f3'])
        list(loader)
        snapshot = loader_stats.snapshot()
        self.assertGreater(int(snapshot.get('chunk_count_total', 0)), 0)
        self.assertIn('chunk_rust_convert_seconds_total', snapshot)
        self.assertIn('chunk_sample_materialize_seconds_total', snapshot)
        self.assertIn('collate_seconds_total', snapshot)
        self.assertGreaterEqual(float(snapshot['chunk_rust_convert_seconds_total']), 0.0)
        self.assertGreaterEqual(float(snapshot['chunk_sample_materialize_seconds_total']), 0.0)
        self.assertGreaterEqual(float(snapshot['collate_seconds_total']), 0.0)

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_direct_batch_prebatched_static_strategy_preserves_batch_order(self):
        batches = self.collect_batches(
            prefetch_strategy='static_chunks',
            prefetch_budget_bytes=0,
            prebatch_layout='direct_batches',
            prebatch_shuffle_mode='batch',
        )
        self.assertEqual(
            batches,
            [
                ([[0], [100], [10]], [0, 1, 10], [3, 46]),
                ([[110]], [11], [1, 46]),
                ([[20], [120], [30]], [20, 21, 30], [3, 46]),
                ([[130]], [31], [1, 46]),
            ],
        )

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_direct_batch_prebatched_uses_batch_shuffle_mode(self):
        batches = self.collect_batches(
            prefetch_strategy='static_chunks',
            prefetch_budget_bytes=0,
            prebatch_layout='direct_batches',
            prebatch_shuffle_mode='batch',
            shuffle=True,
        )
        flattened_actions = [action for _obs, actions, _shape in batches for action in actions]
        self.assertCountEqual(flattened_actions, [0, 1, 10, 11, 20, 21, 30, 31])

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_chunk_spill_prebatched_spills_across_chunk_boundaries(self):
        batches = self.collect_batches(
            prefetch_strategy='static_chunks',
            prefetch_budget_bytes=0,
            prebatch_layout='chunk',
            prebatch_spill_across_chunks=True,
        )
        self.assertEqual(
            batches,
            [
                ([[0], [100], [10]], [0, 1, 10], [3, 46]),
                ([[110], [20], [120]], [11, 20, 21], [3, 46]),
                ([[30], [130]], [30, 31], [2, 46]),
            ],
        )

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_chunk_spill_prebatched_matches_non_prebatched_loader_without_shuffle(self):
        baseline_loader = self.collect_loader_batches(['f0', 'f1', 'f2', 'f3'])
        baseline_batches = [tuple(tensor.clone() for tensor in batch) for batch in baseline_loader]
        spill_batches = self.collect_prebatched_loader_batches(
            ['f0', 'f1', 'f2', 'f3'],
            prebatch_layout='chunk',
            prebatch_spill_across_chunks=True,
            shuffle=False,
        )
        self.assertEqual(len(spill_batches), len(baseline_batches))
        for spill_batch, baseline_batch in zip(spill_batches, baseline_batches):
            self.assert_batch_tree_equal(spill_batch, baseline_batch)

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_preassembled_loader_matches_baseline_without_shuffle(self):
        baseline_loader = self.collect_loader_batches(['f0', 'f1', 'f2', 'f3'])
        baseline_batches = [tuple(tensor.clone() for tensor in batch) for batch in baseline_loader]
        preassembled_batches = self.collect_preassembled_loader_batches(
            ['f0', 'f1', 'f2', 'f3'],
            shuffle=False,
            block_target_samples=5,
        )
        self.assertEqual(len(preassembled_batches), len(baseline_batches))
        for preassembled_batch, baseline_batch in zip(preassembled_batches, baseline_batches):
            self.assert_batch_tree_equal(preassembled_batch, baseline_batch)

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_preassembled_loader_matches_baseline_with_shuffle(self):
        random.seed(2026)
        baseline_loader = self.collect_loader_batches(['f0', 'f1', 'f2', 'f3'], shuffle=True)
        baseline_batches = [tuple(tensor.clone() for tensor in batch) for batch in baseline_loader]

        random.seed(2026)
        preassembled_batches = self.collect_preassembled_loader_batches(
            ['f0', 'f1', 'f2', 'f3'],
            shuffle=True,
            block_target_samples=5,
        )

        self.assertEqual(len(preassembled_batches), len(baseline_batches))
        for preassembled_batch, baseline_batch in zip(preassembled_batches, baseline_batches):
            self.assert_batch_tree_equal(preassembled_batch, baseline_batch)

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_dynamic_ram_prebatched_preserves_filter_and_batching(self):
        allowed = {
            'f0': (0,),
            'f1': (1,),
            'f2': (),
            'f3': (1,),
        }
        batches = self.collect_batches(
            prefetch_strategy='dynamic_ram',
            prefetch_budget_bytes=4096,
            allowed_player_ids_by_path=allowed,
        )
        self.assertEqual(
            batches,
            [
                ([[0], [100], [10]], [0, 1, 10], [3, 46]),
                ([[110]], [11], [1, 46]),
                ([[30], [130]], [30, 31], [2, 46]),
            ],
        )

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_decode_threads_preserve_order_when_shuffle_is_false(self):
        dataset = ActionFileDatasetsIter(
            version=4,
            file_list=['f0', 'f1', 'f2', 'f3'],
            oracle=False,
            file_batch_size=4,
            num_epochs=1,
            cycle=False,
            shuffle=False,
            prefetch_strategy='prepared_ram',
            prefetch_budget_bytes=16 * 1024,
            prefetch_target_chunk_bytes=4 * 1024,
            prefetch_threads=1,
            decode_threads=2,
            batch_size=10,
            prebatched=True,
        )
        first_batch = next(iter(dataset))
        obs, actions, _masks = first_batch
        self.assertEqual(obs.tolist(), [[0], [100], [10], [110], [20], [120], [30], [130]])
        self.assertEqual(actions.tolist(), [0, 1, 10, 11, 20, 21, 30, 31])

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_prepared_ram_preserves_order_when_shuffle_is_false(self):
        original = ActionFileDatasetsIter.build_buffer_for_files

        def slow_first_chunk(dataset, file_list, *, augmented):
            if file_list and file_list[0] == 'f0':
                time.sleep(0.05)
            return original(dataset, file_list, augmented=augmented)

        with patch.object(ActionFileDatasetsIter, 'build_buffer_for_files', slow_first_chunk):
            dataset = ActionFileDatasetsIter(
                version=4,
                file_list=['f0', 'f1', 'f2', 'f3'],
                oracle=False,
                file_batch_size=2,
                num_epochs=1,
                cycle=False,
                shuffle=False,
                prefetch_strategy='prepared_ram',
                prefetch_budget_bytes=4096,
                prefetch_target_chunk_bytes=1024,
                prefetch_threads=2,
                prefetch_max_inflight_chunks=2,
                batch_size=10,
                prebatched=True,
                prefetch_out_of_order=True,
            )
            first_batch = next(iter(dataset))

        obs, actions, masks = first_batch
        self.assertEqual(obs.tolist(), [[0], [100], [10], [110]])
        self.assertEqual(actions.tolist(), [0, 1, 10, 11])
        self.assertEqual(list(masks.shape), [4, 46])

    @patch('dataloader.torch.randperm', lambda n: torch.arange(n))
    @patch('dataloader.random.shuffle', lambda items: None)
    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_prepared_ram_can_emit_completion_order_when_shuffle_is_true(self):
        original = ActionFileDatasetsIter.build_buffer_for_files

        def slow_first_chunk(dataset, file_list, *, augmented):
            if file_list and file_list[0] == 'f0':
                time.sleep(0.05)
            return original(dataset, file_list, augmented=augmented)

        with patch.object(ActionFileDatasetsIter, 'build_buffer_for_files', slow_first_chunk):
            dataset = ActionFileDatasetsIter(
                version=4,
                file_list=['f0', 'f1', 'f2', 'f3'],
                oracle=False,
                file_batch_size=2,
                num_epochs=1,
                cycle=False,
                shuffle=True,
                prefetch_strategy='prepared_ram',
                prefetch_budget_bytes=4096,
                prefetch_target_chunk_bytes=1024,
                prefetch_threads=2,
                prefetch_max_inflight_chunks=2,
                batch_size=10,
                prebatched=True,
            )
            first_batch = next(iter(dataset))

        _obs, actions, _masks = first_batch
        self.assertEqual(actions.tolist(), [20, 21, 30, 31])

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_prepared_ram_honors_startup_ready_chunks_before_first_yield(self):
        dataset = ActionFileDatasetsIter(
            version=4,
            file_list=['f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
            oracle=False,
            file_batch_size=2,
            num_epochs=1,
            cycle=False,
            shuffle=False,
            prefetch_strategy='prepared_ram',
            prefetch_budget_bytes=64 * 1024,
            prefetch_target_chunk_bytes=4 * 1024,
            prefetch_threads=2,
            prefetch_max_inflight_chunks=2,
            prefetch_startup_ready_chunks=2,
            batch_size=10,
            prebatched=True,
        )
        iterator = iter(dataset)
        _first_batch = next(iterator)
        snapshot = dataset.loader_stats.snapshot()
        self.assertTrue(snapshot['prefill_complete'])
        self.assertGreaterEqual(snapshot['ready_chunks'], 1)
        self.assertGreater(snapshot['ready_bytes'], 0)
        list(iterator)

    def test_suggest_file_batch_size_uses_observed_bytes(self):
        self.assertEqual(
            suggest_file_batch_size(
                fallback_file_batch_size=48,
                startup_file_batch_size=12,
                remaining_files=10,
                target_chunk_bytes=200,
                observed_bytes_per_file=50.0,
            ),
            4,
        )
        self.assertEqual(
            suggest_file_batch_size(
                fallback_file_batch_size=48,
                startup_file_batch_size=12,
                remaining_files=2,
                target_chunk_bytes=0,
                observed_bytes_per_file=None,
            ),
            2,
        )

    def test_suggest_file_batch_size_uses_startup_size_before_observations_exist(self):
        self.assertEqual(
            suggest_file_batch_size(
                fallback_file_batch_size=48,
                startup_file_batch_size=12,
                remaining_files=20,
                target_chunk_bytes=1024,
                observed_bytes_per_file=None,
            ),
            12,
        )

    def test_resolve_prefetch_budget_bytes_divides_by_world_size(self):
        self.assertEqual(resolve_prefetch_budget_bytes(gib=160, world_size=2), 80 * (1024 ** 3))
        self.assertEqual(resolve_prefetch_budget_bytes(gib=0, world_size=2), 0)

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_sync_cpu_batch_pipe_matches_exact_loader_rank0_slice(self):
        direct_batches = list(self.collect_loader_batches(['f0', 'f2']))
        pipe = SyncCpuBatchPipe(
            lambda: iter(self.collect_loader_batches(['f0', 'f2'])),
        ).start()
        try:
            piped_batches = list(pipe)
        finally:
            pipe.close()
        self.assertEqual(len(piped_batches), len(direct_batches))
        for direct, piped in zip(direct_batches, piped_batches):
            self.assert_batch_tree_equal(direct, piped)

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_async_cpu_batch_pipe_matches_exact_loader_rank1_slice(self):
        direct_batches = list(self.collect_loader_batches(['f1', 'f3']))
        pipe = AsyncCpuBatchPipe(
            make_iter=lambda: iter(self.collect_loader_batches(['f1', 'f3'])),
            max_ready_batches=4,
            poll_timeout_seconds=0.01,
        ).start()
        try:
            piped_batches = list(pipe)
        finally:
            pipe.close()
        self.assertEqual(len(piped_batches), len(direct_batches))
        for direct, piped in zip(direct_batches, piped_batches):
            self.assert_batch_tree_equal(direct, piped)

    @patch('dataloader.GameplayLoader', _FakeGameplayLoader)
    def test_async_cpu_batch_pipe_preserves_actions_without_duplicates(self):
        direct_batches = list(self.collect_loader_batches(['f0', 'f1', 'f2', 'f3']))
        expected_actions = torch.cat([batch[1] for batch in direct_batches]).tolist()
        pipe = AsyncCpuBatchPipe(
            make_iter=lambda: iter(self.collect_loader_batches(['f0', 'f1', 'f2', 'f3'])),
            max_ready_batches=4,
            poll_timeout_seconds=0.01,
        ).start()
        try:
            observed_actions = []
            for batch in pipe:
                observed_actions.extend(batch[1].tolist())
        finally:
            pipe.close()
        self.assertEqual(observed_actions, expected_actions)

    def test_async_cpu_batch_pipe_propagates_producer_exception(self):
        sample_batch = (
            torch.tensor([[1, 2]], dtype=torch.int16),
            torch.tensor([3], dtype=torch.int64),
            torch.ones((1, 46), dtype=torch.bool),
        )

        def make_iter():
            yield sample_batch
            raise RuntimeError('producer failed')

        pipe = AsyncCpuBatchPipe(
            make_iter=make_iter,
            max_ready_batches=2,
            poll_timeout_seconds=0.01,
        ).start()
        try:
            first = next(pipe)
            self.assert_batch_tree_equal(first, sample_batch)
            with self.assertRaisesRegex(RuntimeError, 'producer failed'):
                next(pipe)
        finally:
            pipe.close()

    def test_async_cpu_batch_pipe_close_stops_worker_early(self):
        sample_batch = (
            torch.tensor([[1, 2]], dtype=torch.int16),
            torch.tensor([3], dtype=torch.int64),
            torch.ones((1, 46), dtype=torch.bool),
        )

        def make_iter():
            while True:
                yield sample_batch

        pipe = AsyncCpuBatchPipe(
            make_iter=make_iter,
            max_ready_batches=2,
            poll_timeout_seconds=0.01,
        ).start()
        time.sleep(0.05)
        pipe.close()
        self.assertFalse(pipe._worker.is_alive())

    def test_async_cpu_batch_pipe_backpressure_caps_ready_batches(self):
        sample_batch = (
            torch.tensor([[1, 2]], dtype=torch.int16),
            torch.tensor([3], dtype=torch.int64),
            torch.ones((1, 46), dtype=torch.bool),
        )

        def make_iter():
            for _ in range(8):
                yield sample_batch

        pipe = AsyncCpuBatchPipe(
            make_iter=make_iter,
            max_ready_batches=1,
            poll_timeout_seconds=0.01,
        ).start()
        try:
            time.sleep(0.05)
            snapshot = pipe.snapshot()
            self.assertLessEqual(int(snapshot.get('cpu_ready_batches', 0)), 1)
            self.assertLessEqual(int(snapshot.get('max_cpu_ready_batches', 0)), 1)
        finally:
            pipe.close()

    def test_async_cpu_batch_pipe_backpressure_caps_ready_bytes(self):
        sample_batch = (
            torch.tensor([[1, 2]], dtype=torch.int16),
            torch.tensor([3], dtype=torch.int64),
            torch.ones((1, 46), dtype=torch.bool),
        )
        sample_batch_bytes = sum(
            tensor.element_size() * tensor.numel()
            for tensor in sample_batch
        )

        def make_iter():
            for _ in range(8):
                yield sample_batch

        pipe = AsyncCpuBatchPipe(
            make_iter=make_iter,
            max_ready_batches=8,
            max_ready_bytes=sample_batch_bytes + 1,
            poll_timeout_seconds=0.01,
        ).start()
        try:
            time.sleep(0.05)
            snapshot = pipe.snapshot()
            self.assertLessEqual(int(snapshot.get('cpu_ready_batches', 0)), 1)
            self.assertLessEqual(int(snapshot.get('cpu_ready_bytes', 0)), sample_batch_bytes + 1)
            self.assertLessEqual(int(snapshot.get('max_cpu_ready_bytes', 0)), sample_batch_bytes + 1)
        finally:
            pipe.close()


if __name__ == '__main__':
    unittest.main()
