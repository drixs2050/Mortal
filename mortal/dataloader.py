import logging
import queue
import random
import threading
import time
import gzip
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from config import config
from raw_store import PackedRawSource

class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None

    def build_iter(self):
        # do not put it in __init__, it won't work on Windows
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                for i in range(game_size):
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[at_kyoku[i] + 1],
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    # Cap total Rayon threads across all workers to avoid thread explosion.
    # w4 DDP (8 workers × 128 threads = 1024) is fine; w6+ exceeds the limit
    # and segfaults in native code. Only throttle when total would exceed ~1024.
    import os
    if 'RAYON_NUM_THREADS' not in os.environ:
        worker_info = torch.utils.data.get_worker_info()
        cpu_count = os.cpu_count() or 64
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        total_workers = worker_info.num_workers * world_size
        max_total_threads = 1024
        if total_workers * cpu_count > max_total_threads:
            rayon_threads = max(1, max_total_threads // total_workers)
            os.environ['RAYON_NUM_THREADS'] = str(rayon_threads)

    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]


def _to_tuple_player_ids(values):
    if values is None:
        return None
    return tuple(int(v) for v in values)


def tensor_nbytes(value: torch.Tensor | None) -> int:
    if value is None:
        return 0
    return value.element_size() * value.numel()


def batch_nbytes(batch) -> int:
    return tree_nbytes(batch)


def tree_nbytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return tensor_nbytes(value)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, np.generic):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(tree_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tree_nbytes(item) for item in value)
    return 0


def tree_batch_len(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, np.generic):
        return 1
    if isinstance(value, dict):
        for item in value.values():
            return tree_batch_len(item)
        return 0
    if isinstance(value, (tuple, list)):
        for item in value:
            return tree_batch_len(item)
        return 0
    raise TypeError(f'cannot determine batch length for value of type {type(value)!r}')


def resolve_prefetch_budget_bytes(*, gib: float | int | None, world_size: int) -> int:
    if gib is None:
        return 0
    total_bytes = max(int(float(gib) * (1024 ** 3)), 0)
    if total_bytes == 0:
        return 0
    return max(total_bytes // max(int(world_size), 1), 1)


def load_gz_log_blobs(loader, gzip_blobs):
    if hasattr(loader, 'load_gz_log_blobs'):
        return loader.load_gz_log_blobs(gzip_blobs)
    files = []
    for gzip_blob in gzip_blobs:
        raw_log = gzip.decompress(gzip_blob).decode('utf-8')
        files.append(loader.load_log(raw_log))
    return files


def suggest_file_batch_size(
    *,
    fallback_file_batch_size: int,
    startup_file_batch_size: int,
    remaining_files: int,
    target_chunk_bytes: int,
    observed_bytes_per_file: float | None,
    min_file_batch_size: int = 1,
    max_file_batch_size: int | None = None,
) -> int:
    if remaining_files <= 0:
        return 0
    lower_bound = max(int(min_file_batch_size or 1), 1)
    upper_bound = max_file_batch_size if max_file_batch_size is not None else fallback_file_batch_size
    upper_bound = max(int(upper_bound or fallback_file_batch_size or 1), lower_bound)
    if target_chunk_bytes <= 0 or observed_bytes_per_file is None or observed_bytes_per_file <= 0:
        startup_size = startup_file_batch_size or fallback_file_batch_size
        return min(max(int(startup_size), lower_bound), min(remaining_files, upper_bound))
    suggested = int(round(target_chunk_bytes / observed_bytes_per_file))
    return min(max(suggested, lower_bound), min(remaining_files, upper_bound))


@dataclass
class ActionChunkBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor
    invisible_obs: torch.Tensor | None = None
    file_count: int = 0
    sample_count: int = 0
    size_bytes: int = 0


@dataclass
class SampleBlock:
    obs: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor
    invisible_obs: torch.Tensor | None = None
    file_count: int = 0
    sample_count: int = 0
    size_bytes: int = 0


@dataclass
class ActionSegment:
    obs: np.ndarray | torch.Tensor
    actions: np.ndarray | torch.Tensor
    masks: np.ndarray | torch.Tensor
    invisible_obs: np.ndarray | torch.Tensor | None = None
    sample_count: int = 0
    size_bytes: int = 0


@dataclass
class ActionSegmentedChunkBuffer:
    segments: tuple[ActionSegment, ...]
    segment_ends: np.ndarray
    file_count: int = 0
    sample_count: int = 0
    size_bytes: int = 0


@dataclass
class ActionBatchBuffer:
    batches: tuple
    file_count: int = 0
    sample_count: int = 0
    size_bytes: int = 0


def buffer_file_count(buffer) -> int:
    return int(getattr(buffer, 'file_count', 0))


def buffer_sample_count(buffer) -> int:
    return int(getattr(buffer, 'sample_count', 0))


def buffer_size_bytes(buffer) -> int:
    return int(getattr(buffer, 'size_bytes', 0))


class OrderedBatchAssembler:
    def __init__(self, *, batch_size: int, oracle: bool):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive for OrderedBatchAssembler')
        self.batch_size = int(batch_size)
        self.oracle = bool(oracle)
        self._pending = deque()
        self._pending_rows = 0

    def _slice_block(self, block: SampleBlock, start_idx: int, end_idx: int):
        obs = block.obs[start_idx:end_idx]
        actions = block.actions[start_idx:end_idx]
        masks = block.masks[start_idx:end_idx]
        invisible_obs = block.invisible_obs[start_idx:end_idx] if self.oracle else None
        if self.oracle:
            return obs, invisible_obs, actions, masks
        return obs, actions, masks

    def _materialize_batch(self, target_rows: int):
        obs_parts = []
        actions_parts = []
        masks_parts = []
        invisible_parts = []
        remaining = int(target_rows)

        while remaining > 0:
            current = self._pending[0]
            block = current['block']
            offset = current['offset']
            available = block.sample_count - offset
            take = min(available, remaining)
            batch = self._slice_block(block, offset, offset + take)
            if self.oracle:
                obs, invisible_obs, actions, masks = batch
                invisible_parts.append(invisible_obs)
            else:
                obs, actions, masks = batch
            obs_parts.append(obs)
            actions_parts.append(actions)
            masks_parts.append(masks)
            current['offset'] += take
            if current['offset'] >= block.sample_count:
                self._pending.popleft()
            remaining -= take

        if len(obs_parts) == 1:
            if self.oracle:
                return obs_parts[0], invisible_parts[0], actions_parts[0], masks_parts[0]
            return obs_parts[0], actions_parts[0], masks_parts[0]

        obs = torch.cat(obs_parts, dim=0)
        actions = torch.cat(actions_parts, dim=0)
        masks = torch.cat(masks_parts, dim=0)
        if self.oracle:
            invisible_obs = torch.cat(invisible_parts, dim=0)
            return obs, invisible_obs, actions, masks
        return obs, actions, masks

    def add_block(self, block: SampleBlock):
        if block.sample_count <= 0:
            return []
        self._pending.append({
            'block': block,
            'offset': 0,
        })
        self._pending_rows += block.sample_count
        batches = []
        while self._pending_rows >= self.batch_size:
            batches.append(self._materialize_batch(self.batch_size))
            self._pending_rows -= self.batch_size
        return batches

    def finish(self):
        if self._pending_rows <= 0:
            return []
        batch = self._materialize_batch(self._pending_rows)
        self._pending_rows = 0
        return [batch]


class LoaderStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = {
            'queued_bytes': 0,
            'max_queued_bytes': 0,
            'ready_chunks': 0,
            'ready_bytes': 0,
            'max_ready_bytes': 0,
            'inflight_bytes': 0,
            'max_inflight_bytes': 0,
            'pinned_batch_bytes': 0,
            'max_pinned_batch_bytes': 0,
            'raw_lru_bytes': 0,
            'max_raw_lru_bytes': 0,
            'budget_bytes': 0,
            'discovered_files': 0,
            'submitted_files': 0,
            'prefill_complete': False,
            'producer_blocked_reason': '',
            'chunk_count_total': 0,
            'chunk_files_total': 0,
            'chunk_samples_total': 0,
            'chunk_bytes_total': 0,
            'chunk_build_seconds_total': 0.0,
            'chunk_read_seconds_total': 0.0,
            'chunk_decompress_seconds_total': 0.0,
            'chunk_parse_seconds_total': 0.0,
            'chunk_rust_convert_seconds_total': 0.0,
            'chunk_sample_materialize_seconds_total': 0.0,
            'chunk_assemble_seconds_total': 0.0,
            'last_chunk_files': 0,
            'last_chunk_samples': 0,
            'last_chunk_bytes': 0,
            'last_chunk_build_seconds': 0.0,
            'last_chunk_read_seconds': 0.0,
            'last_chunk_decompress_seconds': 0.0,
            'last_chunk_parse_seconds': 0.0,
            'last_chunk_rust_convert_seconds': 0.0,
            'last_chunk_sample_materialize_seconds': 0.0,
            'last_chunk_assemble_seconds': 0.0,
            'collate_seconds_total': 0.0,
            'cpu_ready_batches': 0,
            'max_cpu_ready_batches': 0,
            'cpu_ready_bytes': 0,
            'max_cpu_ready_bytes': 0,
            'cpu_produced_batches_total': 0,
            'cpu_produced_samples_total': 0,
            'cpu_blocked_put_seconds_total': 0.0,
            'cpu_consumer_wait_seconds_total': 0.0,
        }

    def __getstate__(self):
        return {
            '_values': dict(self._values),
        }

    def __setstate__(self, state):
        self._lock = threading.Lock()
        self._values = dict(state.get('_values') or {})

    def update_queue_state(
        self,
        *,
        queued_bytes: int,
        ready_chunks: int,
        ready_bytes: int | None = None,
        inflight_bytes: int | None = None,
        pinned_batch_bytes: int | None = None,
        raw_lru_bytes: int | None = None,
        budget_bytes: int | None = None,
        discovered_files: int | None = None,
        submitted_files: int | None = None,
        prefill_complete: bool | None = None,
        producer_blocked_reason: str | None = None,
    ) -> None:
        with self._lock:
            self._values['queued_bytes'] = max(int(queued_bytes), 0)
            self._values['ready_chunks'] = max(int(ready_chunks), 0)
            if ready_bytes is not None:
                self._values['ready_bytes'] = max(int(ready_bytes), 0)
                self._values['max_ready_bytes'] = max(
                    self._values['max_ready_bytes'],
                    self._values['ready_bytes'],
                )
            if inflight_bytes is not None:
                self._values['inflight_bytes'] = max(int(inflight_bytes), 0)
                self._values['max_inflight_bytes'] = max(
                    self._values['max_inflight_bytes'],
                    self._values['inflight_bytes'],
                )
            if pinned_batch_bytes is not None:
                self._values['pinned_batch_bytes'] = max(int(pinned_batch_bytes), 0)
                self._values['max_pinned_batch_bytes'] = max(
                    self._values['max_pinned_batch_bytes'],
                    self._values['pinned_batch_bytes'],
                )
            if raw_lru_bytes is not None:
                self._values['raw_lru_bytes'] = max(int(raw_lru_bytes), 0)
                self._values['max_raw_lru_bytes'] = max(
                    self._values['max_raw_lru_bytes'],
                    self._values['raw_lru_bytes'],
                )
            if budget_bytes is not None:
                self._values['budget_bytes'] = max(int(budget_bytes), 0)
            if discovered_files is not None:
                self._values['discovered_files'] = max(int(discovered_files), 0)
            if submitted_files is not None:
                self._values['submitted_files'] = max(int(submitted_files), 0)
            if prefill_complete is not None:
                self._values['prefill_complete'] = bool(prefill_complete)
            if producer_blocked_reason is not None:
                self._values['producer_blocked_reason'] = str(producer_blocked_reason)
            self._values['max_queued_bytes'] = max(
                self._values['max_queued_bytes'],
                self._values['queued_bytes'],
            )

    def record_chunk(
        self,
        *,
        file_count: int,
        sample_count: int,
        size_bytes: int,
        build_seconds: float,
        read_seconds: float = 0.0,
        decompress_seconds: float = 0.0,
        parse_seconds: float = 0.0,
        rust_convert_seconds: float = 0.0,
        sample_materialize_seconds: float = 0.0,
        assemble_seconds: float = 0.0,
    ) -> None:
        with self._lock:
            self._values['chunk_count_total'] += 1
            self._values['chunk_files_total'] += int(file_count)
            self._values['chunk_samples_total'] += int(sample_count)
            self._values['chunk_bytes_total'] += int(size_bytes)
            self._values['chunk_build_seconds_total'] += float(build_seconds)
            self._values['chunk_read_seconds_total'] += float(read_seconds)
            self._values['chunk_decompress_seconds_total'] += float(decompress_seconds)
            self._values['chunk_parse_seconds_total'] += float(parse_seconds)
            self._values['chunk_rust_convert_seconds_total'] += float(rust_convert_seconds)
            self._values['chunk_sample_materialize_seconds_total'] += float(sample_materialize_seconds)
            self._values['chunk_assemble_seconds_total'] += float(assemble_seconds)
            self._values['last_chunk_files'] = int(file_count)
            self._values['last_chunk_samples'] = int(sample_count)
            self._values['last_chunk_bytes'] = int(size_bytes)
            self._values['last_chunk_build_seconds'] = float(build_seconds)
            self._values['last_chunk_read_seconds'] = float(read_seconds)
            self._values['last_chunk_decompress_seconds'] = float(decompress_seconds)
            self._values['last_chunk_parse_seconds'] = float(parse_seconds)
            self._values['last_chunk_rust_convert_seconds'] = float(rust_convert_seconds)
            self._values['last_chunk_sample_materialize_seconds'] = float(sample_materialize_seconds)
            self._values['last_chunk_assemble_seconds'] = float(assemble_seconds)

    def record_collate_seconds(self, collate_seconds: float) -> None:
        with self._lock:
            self._values['collate_seconds_total'] += max(float(collate_seconds), 0.0)

    def update_cpu_pipe_state(
        self,
        *,
        ready_batches: int | None = None,
        ready_bytes: int | None = None,
        produced_batches_total: int | None = None,
        produced_samples_total: int | None = None,
        blocked_put_seconds_total: float | None = None,
        consumer_wait_seconds_total: float | None = None,
    ) -> None:
        with self._lock:
            if ready_batches is not None:
                self._values['cpu_ready_batches'] = max(int(ready_batches), 0)
                self._values['max_cpu_ready_batches'] = max(
                    self._values['max_cpu_ready_batches'],
                    self._values['cpu_ready_batches'],
                )
            if ready_bytes is not None:
                self._values['cpu_ready_bytes'] = max(int(ready_bytes), 0)
                self._values['max_cpu_ready_bytes'] = max(
                    self._values['max_cpu_ready_bytes'],
                    self._values['cpu_ready_bytes'],
                )
            if produced_batches_total is not None:
                self._values['cpu_produced_batches_total'] = max(int(produced_batches_total), 0)
            if produced_samples_total is not None:
                self._values['cpu_produced_samples_total'] = max(int(produced_samples_total), 0)
            if blocked_put_seconds_total is not None:
                self._values['cpu_blocked_put_seconds_total'] = max(float(blocked_put_seconds_total), 0.0)
            if consumer_wait_seconds_total is not None:
                self._values['cpu_consumer_wait_seconds_total'] = max(float(consumer_wait_seconds_total), 0.0)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._values)


class ActionFileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        oracle=False,
        file_batch_size=20,
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        trust_seed=False,
        always_include_kan_select=True,
        cycle=False,
        shuffle=True,
        allowed_player_ids_by_path=None,
        prefetch_chunks=0,
        prefetch_strategy='static_chunks',
        prefetch_budget_bytes=0,
        prefetch_target_chunk_bytes=0,
        prefetch_low_watermark=0.35,
        prefetch_high_watermark=0.85,
        prefetch_threads=1,
        decode_threads=1,
        batch_size=None,
        prebatched=False,
        prebatch_layout='chunk',
        prebatch_shuffle_mode='sample',
        prebatch_spill_across_chunks=False,
        prefetch_out_of_order=False,
        prefetch_startup_file_batch_size=0,
        prefetch_startup_ready_chunks=1,
        prefetch_inflight_budget_bytes=0,
        prefetch_ready_budget_bytes=0,
        prefetch_max_inflight_chunks=0,
        prefetch_min_file_batch_size=1,
        prefetch_raw_lru_budget_bytes=0,
        raw_source_backend='files',
        raw_pack_path='',
        raw_pack_index_path='',
        loader_mode='baseline',
        loader_block_target_samples=65536,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.trust_seed = trust_seed
        self.always_include_kan_select = always_include_kan_select
        self.cycle = cycle
        self.shuffle = shuffle
        self.allowed_player_ids_by_path = allowed_player_ids_by_path
        self.prefetch_chunks = max(int(prefetch_chunks or 0), 0)
        self.prefetch_strategy = str(prefetch_strategy or 'static_chunks')
        self.prefetch_budget_bytes = max(int(prefetch_budget_bytes or 0), 0)
        self.prefetch_target_chunk_bytes = max(int(prefetch_target_chunk_bytes or 0), 0)
        self.prefetch_low_watermark = float(prefetch_low_watermark)
        self.prefetch_high_watermark = float(prefetch_high_watermark)
        self.prefetch_threads = max(int(prefetch_threads or 1), 1)
        self.decode_threads = max(int(decode_threads or 1), 1)
        self.batch_size = int(batch_size or 0)
        self.prebatched = bool(prebatched)
        self.prebatch_layout = str(prebatch_layout or 'chunk')
        self.prebatch_shuffle_mode = str(prebatch_shuffle_mode or 'sample')
        self.prebatch_spill_across_chunks = bool(prebatch_spill_across_chunks)
        self.prefetch_out_of_order = bool(prefetch_out_of_order)
        self.prefetch_startup_file_batch_size = max(int(prefetch_startup_file_batch_size or 0), 0)
        self.prefetch_startup_ready_chunks = max(int(prefetch_startup_ready_chunks or 1), 1)
        self.prefetch_inflight_budget_bytes = max(int(prefetch_inflight_budget_bytes or 0), 0)
        self.prefetch_ready_budget_bytes = max(int(prefetch_ready_budget_bytes or 0), 0)
        self.prefetch_max_inflight_chunks = max(
            int(prefetch_max_inflight_chunks or self.prefetch_threads),
            1,
        )
        self.prefetch_min_file_batch_size = max(int(prefetch_min_file_batch_size or 1), 1)
        self.prefetch_raw_lru_budget_bytes = max(int(prefetch_raw_lru_budget_bytes or 0), 0)
        self.raw_source_backend = str(raw_source_backend or 'files')
        self.raw_pack_path = str(raw_pack_path or '')
        self.raw_pack_index_path = str(raw_pack_index_path or '')
        self.loader_mode = str(loader_mode or 'baseline')
        self.loader_block_target_samples = max(int(loader_block_target_samples or 65536), 1)
        self._raw_source = None
        self._raw_source_lock = threading.Lock()
        self.loader_stats = LoaderStats()
        self.iterator = None

        if not 0 <= self.prefetch_low_watermark <= 1:
            raise ValueError('prefetch_low_watermark must be between 0 and 1')
        if not 0 <= self.prefetch_high_watermark <= 1:
            raise ValueError('prefetch_high_watermark must be between 0 and 1')
        if self.prefetch_low_watermark > self.prefetch_high_watermark:
            raise ValueError('prefetch_low_watermark must be <= prefetch_high_watermark')
        if self.prebatched and self.batch_size <= 0:
            raise ValueError('batch_size must be positive when prebatched=True')
        if self.prebatch_layout not in ('chunk', 'direct_batches'):
            raise ValueError("prebatch_layout must be 'chunk' or 'direct_batches'")
        if self.prebatch_shuffle_mode not in ('sample', 'batch'):
            raise ValueError("prebatch_shuffle_mode must be 'sample' or 'batch'")
        if not self.prebatched and self.prebatch_layout != 'chunk':
            logging.warning(
                'prebatch_layout=%s requires prebatched=True; falling back to prebatch_layout=chunk',
                self.prebatch_layout,
            )
            self.prebatch_layout = 'chunk'
        if not self.prebatched and self.prebatch_spill_across_chunks:
            logging.warning(
                'prebatch_spill_across_chunks=True requires prebatched=True; '
                'falling back to prebatch_spill_across_chunks=False',
            )
            self.prebatch_spill_across_chunks = False
        if self.prebatch_spill_across_chunks and self.prebatch_layout != 'chunk':
            logging.warning(
                'prebatch_spill_across_chunks=True requires prebatch_layout=chunk; '
                'falling back to prebatch_spill_across_chunks=False',
            )
            self.prebatch_spill_across_chunks = False
        if self.prebatch_layout == 'direct_batches' and self.prebatch_shuffle_mode == 'sample':
            logging.warning(
                'prebatch_layout=direct_batches does not support sample-level reshuffle efficiently; '
                'falling back to prebatch_shuffle_mode=batch',
            )
            self.prebatch_shuffle_mode = 'batch'
        if self.prefetch_strategy == 'dynamic_ram' and not self.prebatched:
            logging.warning(
                'prefetch_strategy=dynamic_ram requires prebatched=True; '
                'falling back to static chunk prefetch for this iterator',
            )
            self.prefetch_strategy = 'static_chunks'
        if self.prefetch_strategy == 'prepared_ram' and not self.prebatched:
            logging.warning(
                'prefetch_strategy=prepared_ram requires prebatched=True; '
                'falling back to static chunk prefetch for this iterator',
            )
            self.prefetch_strategy = 'static_chunks'
        if self.raw_source_backend not in ('files', 'raw_pack'):
            raise ValueError("raw_source_backend must be 'files' or 'raw_pack'")
        if self.raw_source_backend == 'raw_pack':
            if not self.raw_pack_path:
                raise ValueError('raw_pack_path is required when raw_source_backend=raw_pack')
            if not self.raw_pack_index_path:
                raise ValueError('raw_pack_index_path is required when raw_source_backend=raw_pack')
        if self.loader_mode not in ('baseline', 'preassembled_batches'):
            raise ValueError("loader_mode must be 'baseline' or 'preassembled_batches'")
        if self.loader_mode == 'preassembled_batches' and self.prebatched:
            raise ValueError('loader_mode=preassembled_batches cannot be combined with prebatched=True')
        if self.loader_mode == 'preassembled_batches' and self.batch_size <= 0:
            raise ValueError('batch_size must be positive when loader_mode=preassembled_batches')
        self.tensor_buffer_mode = bool(self.prebatched or self.loader_mode == 'preassembled_batches')
        self.emits_batches = bool(self.prebatched or self.loader_mode == 'preassembled_batches')
        self.loader_stats.update_queue_state(
            queued_bytes=0,
            ready_chunks=0,
            ready_bytes=0,
            inflight_bytes=0,
            budget_bytes=max(self.prefetch_budget_bytes, self.prefetch_ready_budget_bytes),
            discovered_files=len(self.file_list),
            submitted_files=0,
            prefill_complete=False,
            producer_blocked_reason='idle',
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_raw_source'] = None
        state['_raw_source_lock'] = None
        state['iterator'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._raw_source = None
        self._raw_source_lock = threading.Lock()

    def make_loader(self, *, augmented):
        return GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            trust_seed=self.trust_seed,
            always_include_kan_select=self.always_include_kan_select,
            augmented=augmented,
        )

    def raw_source(self):
        if self.raw_source_backend == 'files':
            return None
        if self._raw_source is None:
            with self._raw_source_lock:
                if self._raw_source is None:
                    self._raw_source = PackedRawSource(
                        self.raw_pack_path,
                        self.raw_pack_index_path,
                    )
        return self._raw_source

    def close_raw_source(self):
        if self._raw_source is None:
            return
        with self._raw_source_lock:
            if self._raw_source is not None:
                self._raw_source.close()
                self._raw_source = None

    def __del__(self):
        try:
            self.close_raw_source()
        except Exception:
            pass

    def load_files_with_fallback(self, file_list, *, loader):
        try:
            data = loader.load_gz_log_files(file_list)
            return list(zip(file_list, data))
        except BaseException as batch_exc:
            if len(file_list) == 1:
                logging.warning(
                    'skipping unreadable BC file %s: %s: %s',
                    file_list[0],
                    type(batch_exc).__name__,
                    batch_exc,
                )
                return []
            logging.warning(
                'BC batch load failed for %s files; retrying individually. error=%s: %s',
                len(file_list),
                type(batch_exc).__name__,
                batch_exc,
            )
            loaded = []
            for filename in file_list:
                try:
                    data = loader.load_gz_log_files([filename])
                except BaseException as single_exc:
                    logging.warning(
                        'skipping unreadable BC file %s: %s: %s',
                        filename,
                        type(single_exc).__name__,
                        single_exc,
                    )
                    continue
                if not data:
                    logging.warning('skipping BC file with empty loader output: %s', filename)
                    continue
                loaded.append((filename, data[0]))
            return loaded

    def load_blobs_with_fallback(self, file_list, gzip_blobs, *, loader):
        try:
            data = load_gz_log_blobs(loader, gzip_blobs)
            return list(zip(file_list, data))
        except BaseException as batch_exc:
            if len(file_list) == 1:
                logging.warning(
                    'skipping unreadable BC raw-pack file %s: %s: %s',
                    file_list[0],
                    type(batch_exc).__name__,
                    batch_exc,
                )
                return []
            logging.warning(
                'BC raw-pack batch load failed for %s files; retrying individually. error=%s: %s',
                len(file_list),
                type(batch_exc).__name__,
                batch_exc,
            )
            loaded = []
            for filename, gzip_blob in zip(file_list, gzip_blobs):
                try:
                    data = load_gz_log_blobs(loader, [gzip_blob])
                except BaseException as single_exc:
                    logging.warning(
                        'skipping unreadable BC raw-pack file %s: %s: %s',
                        filename,
                        type(single_exc).__name__,
                        single_exc,
                    )
                    continue
                if not data:
                    logging.warning('skipping BC raw-pack file with empty loader output: %s', filename)
                    continue
                loaded.append((filename, data[0]))
            return loaded

    def load_files_for_chunk(self, file_list, *, augmented):
        if self.raw_source_backend == 'raw_pack':
            return self.load_files_for_chunk_from_raw_pack(file_list, augmented=augmented)
        if self.decode_threads <= 1 or len(file_list) <= 1:
            loader = self.make_loader(augmented=augmented)
            return self.load_files_with_fallback(file_list, loader=loader)

        stripes = [[] for _ in range(min(self.decode_threads, len(file_list)))]
        for idx, filename in enumerate(file_list):
            stripes[idx % len(stripes)].append((idx, filename))

        def load_stripe(stripe_files):
            loader = self.make_loader(augmented=augmented)
            loaded = self.load_files_with_fallback(
                [filename for _idx, filename in stripe_files],
                loader=loader,
            )
            indexed = []
            original_indices = {
                filename: original_idx
                for original_idx, filename in stripe_files
            }
            for filename, payload in loaded:
                indexed.append((original_indices[filename], filename, payload))
            return indexed

        loaded_files = []
        with ThreadPoolExecutor(
            max_workers=len(stripes),
            thread_name_prefix='bc-decode',
        ) as executor:
            futures = [
                executor.submit(load_stripe, stripe_files)
                for stripe_files in stripes
                if stripe_files
            ]
            for future in futures:
                loaded_files.extend(future.result())
        loaded_files.sort(key=lambda row: row[0])
        return [(filename, payload) for _idx, filename, payload in loaded_files]

    def load_files_for_chunk_from_raw_pack(self, file_list, *, augmented):
        if self.decode_threads <= 1 or len(file_list) <= 1:
            loader = self.make_loader(augmented=augmented)
            gzip_blobs = self.raw_source().read_many(file_list)
            return self.load_blobs_with_fallback(file_list, gzip_blobs, loader=loader)

        stripes = [[] for _ in range(min(self.decode_threads, len(file_list)))]
        for idx, filename in enumerate(file_list):
            stripes[idx % len(stripes)].append((idx, filename))

        def load_stripe(stripe_files):
            loader = self.make_loader(augmented=augmented)
            stripe_filenames = [filename for _idx, filename in stripe_files]
            gzip_blobs = self.raw_source().read_many(stripe_filenames)
            loaded = self.load_blobs_with_fallback(
                stripe_filenames,
                gzip_blobs,
                loader=loader,
            )
            indexed = []
            original_indices = {
                filename: original_idx
                for original_idx, filename in stripe_files
            }
            for filename, payload in loaded:
                indexed.append((original_indices[filename], filename, payload))
            return indexed

        loaded_files = []
        with ThreadPoolExecutor(
            max_workers=len(stripes),
            thread_name_prefix='bc-raw-pack-decode',
        ) as executor:
            futures = [
                executor.submit(load_stripe, stripe_files)
                for stripe_files in stripes
                if stripe_files
            ]
            for future in futures:
                loaded_files.extend(future.result())
        loaded_files.sort(key=lambda row: row[0])
        return [(filename, payload) for _idx, filename, payload in loaded_files]

    def build_buffer_for_files(self, file_list, *, augmented):
        started_at = time.perf_counter()
        load_started_at = time.perf_counter()
        with torch.profiler.record_function('bc.raw_loader'):
            loaded_files = self.load_files_for_chunk(file_list, augmented=augmented)
        read_seconds = time.perf_counter() - load_started_at
        segmented_chunk_mode = self.loader_mode == 'preassembled_batches'
        if self.prebatched and self.prebatch_layout == 'direct_batches':
            batches = []
            current_obs_parts = []
            current_actions_parts = []
            current_masks_parts = []
            current_invisible_parts = [] if self.oracle else None
            current_sample_count = 0
        elif segmented_chunk_mode:
            segments = []
            segment_ends = []
            total_segment_samples = 0
            total_segment_bytes = 0
        elif self.tensor_buffer_mode:
            obs_parts = []
            actions_parts = []
            masks_parts = []
            invisible_parts = [] if self.oracle else None
        else:
            buffer = []
            buffer_size_bytes = 0
        rust_convert_seconds = 0.0
        sample_materialize_seconds = 0.0
        parse_started_at = time.perf_counter()

        def finalize_direct_batch():
            nonlocal current_obs_parts, current_actions_parts, current_masks_parts
            nonlocal current_invisible_parts, current_sample_count
            if current_sample_count <= 0:
                return
            obs_tensor = torch.from_numpy(np.concatenate(current_obs_parts, axis=0))
            actions_tensor = torch.from_numpy(np.concatenate(current_actions_parts, axis=0))
            masks_tensor = torch.from_numpy(np.concatenate(current_masks_parts, axis=0))
            invisible_tensor = None
            if self.oracle:
                invisible_tensor = torch.from_numpy(np.concatenate(current_invisible_parts, axis=0))
            batch = (
                (obs_tensor, invisible_tensor, actions_tensor, masks_tensor)
                if self.oracle
                else (obs_tensor, actions_tensor, masks_tensor)
            )
            batches.append(batch)
            current_obs_parts = []
            current_actions_parts = []
            current_masks_parts = []
            current_invisible_parts = [] if self.oracle else None
            current_sample_count = 0

        with torch.profiler.record_function('bc.sample_emit'):
            for filename, file in loaded_files:
                allowed_player_ids = None
                if self.allowed_player_ids_by_path is not None:
                    allowed_player_ids = _to_tuple_player_ids(self.allowed_player_ids_by_path.get(filename))
                for game in file:
                    convert_started_at = time.perf_counter()
                    if allowed_player_ids is not None:
                        player_id = game.take_player_id()
                        if player_id not in allowed_player_ids:
                            rust_convert_seconds += time.perf_counter() - convert_started_at
                            continue
                    obs = game.take_obs()
                    if self.oracle:
                        invisible_obs = game.take_invisible_obs()
                    actions = game.take_actions()
                    masks = game.take_masks()
                    rust_convert_seconds += time.perf_counter() - convert_started_at

                    materialize_started_at = time.perf_counter()
                    if self.prebatched and self.prebatch_layout == 'direct_batches':
                        obs = np.asarray(obs)
                        actions = np.asarray(actions)
                        masks = np.asarray(masks)
                        if self.oracle:
                            invisible_obs = np.asarray(invisible_obs)
                        offset = 0
                        game_sample_count = int(actions.shape[0])
                        while offset < game_sample_count:
                            remaining = self.batch_size - current_sample_count
                            take = min(remaining, game_sample_count - offset)
                            end_idx = offset + take
                            current_obs_parts.append(obs[offset:end_idx])
                            current_actions_parts.append(actions[offset:end_idx])
                            current_masks_parts.append(masks[offset:end_idx])
                            if self.oracle:
                                current_invisible_parts.append(invisible_obs[offset:end_idx])
                            current_sample_count += take
                            offset = end_idx
                            if current_sample_count == self.batch_size:
                                finalize_direct_batch()
                        sample_materialize_seconds += time.perf_counter() - materialize_started_at
                        continue

                    if segmented_chunk_mode:
                        obs_array = np.asarray(obs)
                        actions_array = np.asarray(actions)
                        masks_array = np.asarray(masks)
                        invisible_array = (
                            np.asarray(invisible_obs)
                            if self.oracle
                            else None
                        )
                        segment_sample_count = int(actions_array.shape[0])
                        if segment_sample_count > 0:
                            segment_size_bytes = (
                                tree_nbytes(obs_array)
                                + tree_nbytes(actions_array)
                                + tree_nbytes(masks_array)
                                + tree_nbytes(invisible_array)
                            )
                            segments.append(
                                ActionSegment(
                                    obs=obs_array,
                                    actions=actions_array,
                                    masks=masks_array,
                                    invisible_obs=invisible_array,
                                    sample_count=segment_sample_count,
                                    size_bytes=segment_size_bytes,
                                ),
                            )
                            total_segment_samples += segment_sample_count
                            total_segment_bytes += segment_size_bytes
                            segment_ends.append(total_segment_samples)
                        sample_materialize_seconds += time.perf_counter() - materialize_started_at
                        continue

                    if self.tensor_buffer_mode:
                        obs_parts.append(obs)
                        actions_parts.append(actions)
                        masks_parts.append(masks)
                        if self.oracle:
                            invisible_parts.append(invisible_obs)
                        sample_materialize_seconds += time.perf_counter() - materialize_started_at
                        continue

                    for idx, action in enumerate(actions):
                        entry = [
                            obs[idx],
                            action,
                            masks[idx],
                        ]
                        if self.oracle:
                            entry.insert(1, invisible_obs[idx])
                        buffer.append(entry)
                        buffer_size_bytes += batch_nbytes(entry)
                    sample_materialize_seconds += time.perf_counter() - materialize_started_at
        parse_seconds = time.perf_counter() - parse_started_at
        assemble_started_at = time.perf_counter()

        if not self.tensor_buffer_mode:
            build_seconds = time.perf_counter() - started_at
            self.loader_stats.record_chunk(
                file_count=len(file_list),
                sample_count=len(buffer),
                size_bytes=buffer_size_bytes,
                build_seconds=build_seconds,
                read_seconds=read_seconds,
                decompress_seconds=0.0,
                parse_seconds=parse_seconds,
                rust_convert_seconds=rust_convert_seconds,
                sample_materialize_seconds=sample_materialize_seconds,
                assemble_seconds=0.0,
            )
            return buffer

        if self.prebatch_layout == 'direct_batches':
            with torch.profiler.record_function('bc.collate_or_assemble'):
                finalize_direct_batch()
                if self.shuffle and self.prebatch_shuffle_mode == 'batch' and len(batches) > 1:
                    random.shuffle(batches)
                size_bytes = sum(batch_nbytes(batch) for batch in batches)
                sample_count = sum(int(batch[2].shape[0] if self.oracle else batch[1].shape[0]) for batch in batches)
                assemble_seconds = time.perf_counter() - assemble_started_at
            build_seconds = time.perf_counter() - started_at
            batch_buffer = ActionBatchBuffer(
                batches=tuple(batches),
                file_count=len(file_list),
                sample_count=sample_count,
                size_bytes=size_bytes,
            )
            self.loader_stats.record_chunk(
                file_count=len(file_list),
                sample_count=sample_count,
                size_bytes=size_bytes,
                build_seconds=build_seconds,
                read_seconds=read_seconds,
                decompress_seconds=0.0,
                parse_seconds=parse_seconds,
                rust_convert_seconds=rust_convert_seconds,
                sample_materialize_seconds=sample_materialize_seconds,
                assemble_seconds=assemble_seconds,
            )
            return batch_buffer

        if segmented_chunk_mode:
            with torch.profiler.record_function('bc.chunk_finalize'):
                segment_ends_array = (
                    np.asarray(segment_ends, dtype=np.int64)
                    if segment_ends
                    else np.empty((0,), dtype=np.int64)
                )
                assemble_seconds = time.perf_counter() - assemble_started_at
            build_seconds = time.perf_counter() - started_at
            chunk = ActionSegmentedChunkBuffer(
                segments=tuple(segments),
                segment_ends=segment_ends_array,
                file_count=len(file_list),
                sample_count=total_segment_samples,
                size_bytes=total_segment_bytes + int(segment_ends_array.nbytes),
            )
            self.loader_stats.record_chunk(
                file_count=len(file_list),
                sample_count=total_segment_samples,
                size_bytes=chunk.size_bytes,
                build_seconds=build_seconds,
                read_seconds=read_seconds,
                decompress_seconds=0.0,
                parse_seconds=parse_seconds,
                rust_convert_seconds=rust_convert_seconds,
                sample_materialize_seconds=sample_materialize_seconds,
                assemble_seconds=assemble_seconds,
            )
            return chunk

        with torch.profiler.record_function('bc.collate_or_assemble'):
            if obs_parts:
                obs_tensor = torch.from_numpy(np.concatenate(obs_parts, axis=0))
                actions_tensor = torch.from_numpy(np.concatenate(actions_parts, axis=0))
                masks_tensor = torch.from_numpy(np.concatenate(masks_parts, axis=0))
                invisible_tensor = None
                if self.oracle:
                    invisible_tensor = torch.from_numpy(np.concatenate(invisible_parts, axis=0))
            else:
                obs_tensor = torch.empty((0,), dtype=torch.int16)
                actions_tensor = torch.empty((0,), dtype=torch.int64)
                masks_tensor = torch.empty((0, 46), dtype=torch.bool)
                invisible_tensor = (
                    torch.empty((0,), dtype=torch.int16)
                    if self.oracle
                    else None
                )
            assemble_seconds = time.perf_counter() - assemble_started_at
        build_seconds = time.perf_counter() - started_at
        size_bytes = (
            tensor_nbytes(obs_tensor)
            + tensor_nbytes(actions_tensor)
            + tensor_nbytes(masks_tensor)
            + tensor_nbytes(invisible_tensor)
        )
        sample_count = int(actions_tensor.shape[0])
        chunk = ActionChunkBuffer(
            obs=obs_tensor,
            actions=actions_tensor,
            masks=masks_tensor,
            invisible_obs=invisible_tensor,
            file_count=len(file_list),
            sample_count=sample_count,
            size_bytes=size_bytes,
        )
        self.loader_stats.record_chunk(
            file_count=len(file_list),
            sample_count=sample_count,
            size_bytes=size_bytes,
            build_seconds=build_seconds,
            read_seconds=read_seconds,
            decompress_seconds=0.0,
            parse_seconds=parse_seconds,
            rust_convert_seconds=rust_convert_seconds,
            sample_materialize_seconds=sample_materialize_seconds,
            assemble_seconds=assemble_seconds,
        )
        return chunk

    def iter_chunk_buffers(self, file_list, *, augmented):
        chunks = [
            file_list[start_idx:start_idx + self.file_batch_size]
            for start_idx in range(0, len(file_list), self.file_batch_size)
        ]
        if not chunks:
            return

        if self.prefetch_chunks <= 0:
            for chunk in chunks:
                yield self.build_buffer_for_files(chunk, augmented=augmented)
            return

        chunk_iter = iter(chunks)
        max_prefetch = max(1, self.prefetch_chunks)
        out_of_order = self.prefetch_out_of_order and max_prefetch > 1 and len(chunks) > 1
        pending = [] if out_of_order else deque()
        logged_initial_submit = False
        logged_first_ready = False

        with ThreadPoolExecutor(max_workers=max_prefetch, thread_name_prefix='bc-prefetch') as executor:
            def submit_next():
                nonlocal logged_initial_submit
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    return False
                if not logged_initial_submit:
                    logging.info(
                        'loader warmup: submitting first static chunk with %s file(s) '
                        '(prefetch_chunks=%s out_of_order=%s)',
                        len(chunk),
                        max_prefetch,
                        out_of_order,
                    )
                    logged_initial_submit = True
                pending.append({
                    'future': executor.submit(self.build_buffer_for_files, chunk, augmented=augmented),
                })
                return True

            while len(pending) < max_prefetch and submit_next():
                pass

            while pending:
                if out_of_order:
                    done, _ = wait(
                        [item['future'] for item in pending],
                        return_when=FIRST_COMPLETED,
                    )
                    completed_future = next(iter(done))
                    current_idx = next(
                        idx
                        for idx, item in enumerate(pending)
                        if item['future'] is completed_future
                    )
                    current = pending[current_idx]
                    del pending[current_idx]
                else:
                    current = pending.popleft()
                buffer = current['future'].result()
                if not logged_first_ready:
                    if hasattr(buffer, 'sample_count') and hasattr(buffer, 'size_bytes'):
                        logging.info(
                            'loader warmup: first static chunk ready files=%s samples=%s size_gib=%.2f',
                            buffer_file_count(buffer),
                            buffer_sample_count(buffer),
                            buffer_size_bytes(buffer) / (1024 ** 3),
                        )
                    else:
                        logging.info(
                            'loader warmup: first static chunk ready entries=%s',
                            len(buffer),
                        )
                    logged_first_ready = True
                while len(pending) < max_prefetch and submit_next():
                    pass
                yield buffer

    def iter_chunk_buffers_dynamic(self, file_list, *, augmented):
        if not file_list:
            return

        total_budget_bytes = max(self.prefetch_budget_bytes, 0)
        ready_budget_bytes = max(
            self.prefetch_ready_budget_bytes or (
                total_budget_bytes
                - self.prefetch_inflight_budget_bytes
                - self.prefetch_raw_lru_budget_bytes
            ),
            0,
        )
        inflight_budget_bytes = max(self.prefetch_inflight_budget_bytes, 0)
        low_watermark_bytes = int(total_budget_bytes * self.prefetch_low_watermark)
        high_watermark_bytes = int(total_budget_bytes * self.prefetch_high_watermark)
        if total_budget_bytes > 0:
            if high_watermark_bytes <= 0:
                high_watermark_bytes = total_budget_bytes
            if low_watermark_bytes > high_watermark_bytes:
                low_watermark_bytes = high_watermark_bytes
        startup_ready_chunks = max(int(self.prefetch_startup_ready_chunks or 1), 1)
        next_file_idx = 0
        next_chunk_idx = 0
        next_emit_idx = 0
        refill_active = True
        observed_bytes_per_file = None
        ready_bytes = 0
        inflight_bytes = 0
        raw_lru_bytes = 0
        out_of_order = self.shuffle and self.prefetch_max_inflight_chunks > 1
        pending = {}
        ready = deque()
        completed = {}
        logged_initial_submit = False
        logged_first_ready = False
        started_at = time.perf_counter()
        last_progress_log_at = started_at
        prefill_complete = False
        exhausted = False
        stop_requested = False
        producer_error = None
        blocked_reason = 'startup_prefill'
        condition = threading.Condition()

        def queue_state_bytes() -> int:
            return ready_bytes + inflight_bytes + raw_lru_bytes

        def update_loader_state_locked() -> None:
            self.loader_stats.update_queue_state(
                queued_bytes=queue_state_bytes(),
                ready_chunks=len(ready),
                ready_bytes=ready_bytes,
                inflight_bytes=inflight_bytes,
                raw_lru_bytes=raw_lru_bytes,
                budget_bytes=total_budget_bytes,
                discovered_files=len(file_list),
                submitted_files=next_file_idx,
                prefill_complete=prefill_complete,
                producer_blocked_reason=blocked_reason,
            )

        def maybe_mark_prefill_complete_locked() -> None:
            nonlocal prefill_complete
            if prefill_complete:
                return
            if len(ready) >= startup_ready_chunks:
                prefill_complete = True
                return
            if high_watermark_bytes > 0 and ready_bytes >= high_watermark_bytes:
                prefill_complete = True
                return
            if next_file_idx >= len(file_list) and not pending and (ready or completed):
                prefill_complete = True
                return

        def move_completed_to_ready_locked() -> None:
            nonlocal ready_bytes, next_emit_idx, observed_bytes_per_file, logged_first_ready
            if out_of_order:
                ready_indices = sorted(completed.keys())
            else:
                ready_indices = []
                while next_emit_idx in completed:
                    ready_indices.append(next_emit_idx)
                    next_emit_idx += 1
            for emit_idx in ready_indices:
                chunk = completed.pop(emit_idx)
                if self.tensor_buffer_mode and hasattr(chunk, 'size_bytes'):
                    ready_bytes += buffer_size_bytes(chunk)
                ready.append(chunk)
                if (
                    self.tensor_buffer_mode
                    and hasattr(chunk, 'size_bytes')
                    and buffer_file_count(chunk) > 0
                    and buffer_size_bytes(chunk) > 0
                ):
                    observed = buffer_size_bytes(chunk) / buffer_file_count(chunk)
                    if observed_bytes_per_file is None:
                        observed_bytes_per_file = observed
                    else:
                        observed_bytes_per_file = (observed_bytes_per_file * 0.7) + (observed * 0.3)
                if not logged_first_ready:
                    if hasattr(chunk, 'sample_count') and hasattr(chunk, 'size_bytes'):
                        logging.info(
                            'loader warmup: first dynamic chunk ready files=%s samples=%s size_gib=%.2f '
                            'ready_gib=%.2f inflight_gib=%.2f ready_chunks=%s',
                            buffer_file_count(chunk),
                            buffer_sample_count(chunk),
                            buffer_size_bytes(chunk) / (1024 ** 3),
                            ready_bytes / (1024 ** 3),
                            inflight_bytes / (1024 ** 3),
                            len(ready),
                        )
                    else:
                        logging.info(
                            'loader warmup: first dynamic chunk ready entries=%s ready_gib=%.2f '
                            'inflight_gib=%.2f ready_chunks=%s',
                            len(chunk),
                            ready_bytes / (1024 ** 3),
                            inflight_bytes / (1024 ** 3),
                            len(ready),
                        )
                    logged_first_ready = True

        with ThreadPoolExecutor(
            max_workers=self.prefetch_max_inflight_chunks,
            thread_name_prefix='bc-dynamic-prefetch',
        ) as executor:
            def submit_next_locked() -> bool:
                nonlocal next_file_idx, next_chunk_idx, inflight_bytes, logged_initial_submit, blocked_reason
                remaining_files = len(file_list) - next_file_idx
                if remaining_files <= 0:
                    blocked_reason = 'input_exhausted'
                    return False
                if len(pending) >= self.prefetch_max_inflight_chunks:
                    blocked_reason = 'max_inflight_chunks'
                    return False
                if inflight_budget_bytes > 0 and inflight_bytes >= inflight_budget_bytes:
                    blocked_reason = 'inflight_budget'
                    return False
                if ready_budget_bytes > 0 and ready_bytes >= ready_budget_bytes:
                    blocked_reason = 'ready_budget'
                    return False
                if total_budget_bytes > 0 and prefill_complete and queue_state_bytes() >= high_watermark_bytes:
                    blocked_reason = 'high_watermark'
                    return False

                chunk_file_count = suggest_file_batch_size(
                    fallback_file_batch_size=self.file_batch_size,
                    startup_file_batch_size=self.prefetch_startup_file_batch_size,
                    remaining_files=remaining_files,
                    target_chunk_bytes=self.prefetch_target_chunk_bytes,
                    observed_bytes_per_file=observed_bytes_per_file,
                    min_file_batch_size=self.prefetch_min_file_batch_size,
                    max_file_batch_size=self.file_batch_size,
                )
                chunk_files = file_list[next_file_idx:next_file_idx + chunk_file_count]
                if not chunk_files:
                    blocked_reason = 'empty_chunk'
                    return False

                reserved_bytes = (
                    int(round(observed_bytes_per_file * len(chunk_files)))
                    if observed_bytes_per_file is not None and observed_bytes_per_file > 0
                    else max(self.prefetch_target_chunk_bytes, 0)
                )
                reserved_bytes = max(reserved_bytes, 1)
                if inflight_budget_bytes > 0:
                    remaining_inflight = max(inflight_budget_bytes - inflight_bytes, 0)
                    if remaining_inflight <= 0:
                        blocked_reason = 'inflight_budget'
                        return False
                    reserved_bytes = min(reserved_bytes, remaining_inflight)
                if total_budget_bytes > 0:
                    available_total_bytes = max(total_budget_bytes - queue_state_bytes(), 0)
                    if available_total_bytes <= 0:
                        blocked_reason = 'total_budget'
                        return False
                    reserved_bytes = min(reserved_bytes, available_total_bytes)

                next_file_idx += len(chunk_files)
                if not logged_initial_submit:
                    logging.info(
                        'loader warmup: submitting first dynamic chunk with %s file(s) '
                        '(target_chunk_gib=%.2f prefetch_budget_gib=%.2f builder_workers=%s decode_threads=%s out_of_order=%s startup_ready_chunks=%s)',
                        len(chunk_files),
                        self.prefetch_target_chunk_bytes / (1024 ** 3),
                        self.prefetch_budget_bytes / (1024 ** 3),
                        self.prefetch_max_inflight_chunks,
                        self.decode_threads,
                        out_of_order,
                        startup_ready_chunks,
                    )
                    logged_initial_submit = True
                pending[next_chunk_idx] = {
                    'future': executor.submit(self.build_buffer_for_files, chunk_files, augmented=augmented),
                    'reserved_bytes': reserved_bytes,
                }
                inflight_bytes += reserved_bytes
                next_chunk_idx += 1
                blocked_reason = 'producing'
                update_loader_state_locked()
                return True

            def producer_loop() -> None:
                nonlocal refill_active, blocked_reason, exhausted, producer_error, inflight_bytes, last_progress_log_at
                try:
                    while True:
                        with condition:
                            if stop_requested:
                                blocked_reason = 'stopped'
                                update_loader_state_locked()
                                condition.notify_all()
                                return
                            total_live_bytes = queue_state_bytes()
                            if total_budget_bytes > 0:
                                if total_live_bytes >= high_watermark_bytes and prefill_complete:
                                    refill_active = False
                                elif total_live_bytes <= low_watermark_bytes:
                                    refill_active = True

                            while (
                                next_file_idx < len(file_list)
                                and len(pending) < self.prefetch_max_inflight_chunks
                                and (not prefill_complete or refill_active or len(ready) < startup_ready_chunks)
                            ):
                                if not submit_next_locked():
                                    break

                            move_completed_to_ready_locked()
                            maybe_mark_prefill_complete_locked()
                            update_loader_state_locked()
                            condition.notify_all()

                            if next_file_idx >= len(file_list) and not pending and not completed:
                                exhausted = True
                                blocked_reason = 'input_exhausted'
                                update_loader_state_locked()
                                condition.notify_all()
                                return

                            futures = [item['future'] for item in pending.values()]

                        if not futures:
                            time.sleep(0.05)
                            continue

                        done, _ = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                        if not done:
                            now = time.perf_counter()
                            if now - last_progress_log_at >= 2.0:
                                with condition:
                                    logging.info(
                                        'loader warmup: elapsed=%.1fs discovered_files=%s submitted_files=%s ready_chunks=%s '
                                        'ready_gib=%.2f inflight_gib=%.2f budget_gib=%.2f blocked=%s prefill_complete=%s',
                                        now - started_at,
                                        len(file_list),
                                        next_file_idx,
                                        len(ready),
                                        ready_bytes / (1024 ** 3),
                                        inflight_bytes / (1024 ** 3),
                                        total_budget_bytes / (1024 ** 3),
                                        blocked_reason,
                                        prefill_complete,
                                    )
                                last_progress_log_at = now
                            continue

                        with condition:
                            for chunk_idx, item in list(pending.items()):
                                if item['future'] not in done:
                                    continue
                                del pending[chunk_idx]
                                chunk = item['future'].result()
                                inflight_bytes = max(inflight_bytes - item['reserved_bytes'], 0)
                                completed[chunk_idx] = chunk
                            move_completed_to_ready_locked()
                            maybe_mark_prefill_complete_locked()
                            blocked_reason = 'ready_queue_full' if ready and not refill_active else blocked_reason
                            update_loader_state_locked()
                            condition.notify_all()
                except BaseException as exc:
                    with condition:
                        producer_error = exc
                        exhausted = True
                        blocked_reason = f'error:{type(exc).__name__}'
                        update_loader_state_locked()
                        condition.notify_all()

            producer_thread = threading.Thread(
                target=producer_loop,
                name='bc-runtime-cache-producer',
                daemon=True,
            )
            producer_thread.start()
            try:
                first_yield = True
                while True:
                    with condition:
                        while True:
                            if producer_error is not None:
                                raise producer_error
                            startup_satisfied = prefill_complete or len(ready) >= startup_ready_chunks
                            if ready and (not first_yield or startup_satisfied):
                                break
                            if exhausted and not ready:
                                return
                            condition.wait(timeout=0.25)

                        chunk = ready.popleft()
                        if hasattr(chunk, 'size_bytes'):
                            ready_bytes = max(ready_bytes - buffer_size_bytes(chunk), 0)
                        refill_active = True if total_budget_bytes <= 0 else queue_state_bytes() <= low_watermark_bytes
                        blocked_reason = 'consumer_active'
                        update_loader_state_locked()
                        condition.notify_all()
                    first_yield = False
                    yield chunk
            finally:
                with condition:
                    stop_requested = True
                    condition.notify_all()
                producer_thread.join()

    def iter_batches_from_chunk(self, chunk: ActionChunkBuffer):
        if chunk.sample_count == 0:
            return
        order = self.build_sample_order(chunk.sample_count)

        for start_idx in range(0, chunk.sample_count, self.batch_size):
            end_idx = min(start_idx + self.batch_size, chunk.sample_count)
            if order is None:
                obs = chunk.obs[start_idx:end_idx]
                actions = chunk.actions[start_idx:end_idx]
                masks = chunk.masks[start_idx:end_idx]
                invisible_obs = (
                    chunk.invisible_obs[start_idx:end_idx]
                    if self.oracle
                    else None
                )
            else:
                batch_indices = order[start_idx:end_idx]
                obs = chunk.obs.index_select(0, batch_indices)
                actions = chunk.actions.index_select(0, batch_indices)
                masks = chunk.masks.index_select(0, batch_indices)
                invisible_obs = (
                    chunk.invisible_obs.index_select(0, batch_indices)
                    if self.oracle
                    else None
                )
            if self.oracle:
                yield obs, invisible_obs, actions, masks
            else:
                yield obs, actions, masks

    def slice_batch_from_chunk(self, chunk: ActionChunkBuffer, start_idx: int, end_idx: int, order=None):
        if order is None:
            obs = chunk.obs[start_idx:end_idx]
            actions = chunk.actions[start_idx:end_idx]
            masks = chunk.masks[start_idx:end_idx]
            invisible_obs = (
                chunk.invisible_obs[start_idx:end_idx]
                if self.oracle
                else None
            )
        else:
            batch_indices = order[start_idx:end_idx]
            obs = chunk.obs.index_select(0, batch_indices)
            actions = chunk.actions.index_select(0, batch_indices)
            masks = chunk.masks.index_select(0, batch_indices)
            invisible_obs = (
                chunk.invisible_obs.index_select(0, batch_indices)
                if self.oracle
                else None
            )
        if self.oracle:
            return obs, invisible_obs, actions, masks
        return obs, actions, masks

    def _assemble_batch_parts(self, obs_parts, actions_parts, masks_parts, invisible_parts):
        if not obs_parts:
            obs = torch.empty((0,), dtype=torch.int16)
            actions = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((0, 46), dtype=torch.bool)
            if self.oracle:
                invisible_obs = torch.empty((0,), dtype=torch.int16)
                return obs, invisible_obs, actions, masks
            return obs, actions, masks
        if len(obs_parts) == 1:
            if self.oracle:
                return (
                    obs_parts[0],
                    invisible_parts[0],
                    actions_parts[0],
                    masks_parts[0],
                )
            return obs_parts[0], actions_parts[0], masks_parts[0]

        obs = torch.cat(obs_parts, dim=0)
        actions = torch.cat(actions_parts, dim=0)
        masks = torch.cat(masks_parts, dim=0)
        if self.oracle:
            invisible_obs = torch.cat(invisible_parts, dim=0)
            return obs, invisible_obs, actions, masks
        return obs, actions, masks

    def _torch_dtype_for_source(self, source) -> torch.dtype:
        if isinstance(source, torch.Tensor):
            return source.dtype
        return torch.as_tensor(source).dtype

    def _allocate_batch_tensor(self, source, rows: int) -> torch.Tensor:
        shape = (int(rows),) + tuple(source.shape[1:])
        return torch.empty(shape, dtype=self._torch_dtype_for_source(source))

    def _allocate_output_batch(self, *, chunk, rows: int):
        rows = int(rows)
        if rows <= 0:
            return self._assemble_batch_parts([], [], [], [])
        if isinstance(chunk, ActionSegmentedChunkBuffer):
            if not chunk.segments:
                return self._assemble_batch_parts([], [], [], [])
            first = chunk.segments[0]
            obs_source = first.obs
            actions_source = first.actions
            masks_source = first.masks
            invisible_source = first.invisible_obs
        else:
            obs_source = chunk.obs
            actions_source = chunk.actions
            masks_source = chunk.masks
            invisible_source = chunk.invisible_obs

        obs = self._allocate_batch_tensor(obs_source, rows)
        actions = self._allocate_batch_tensor(actions_source, rows)
        masks = self._allocate_batch_tensor(masks_source, rows)
        if self.oracle:
            invisible_obs = self._allocate_batch_tensor(invisible_source, rows)
            return obs, invisible_obs, actions, masks
        return obs, actions, masks

    def _selector_length(self, selector) -> int:
        if isinstance(selector, slice):
            return max(int(selector.stop) - int(selector.start), 0)
        if isinstance(selector, torch.Tensor):
            return int(selector.numel())
        return int(len(selector))

    def _normalize_selector(self, selector):
        if isinstance(selector, slice):
            return selector
        if isinstance(selector, torch.Tensor):
            selector = selector.detach().cpu().numpy()
        elif not isinstance(selector, np.ndarray):
            selector = np.asarray(selector, dtype=np.int64)
        if selector.size == 0:
            return slice(0, 0)
        selector = selector.astype(np.int64, copy=False)
        if selector.size == 1:
            start = int(selector[0])
            return slice(start, start + 1)
        diffs = np.diff(selector)
        if np.all(diffs == 1):
            return slice(int(selector[0]), int(selector[-1]) + 1)
        return selector

    def _copy_rows_into_tensor(self, dest: torch.Tensor, dest_start: int, source, selector) -> int:
        selector = self._normalize_selector(selector)
        row_count = self._selector_length(selector)
        dest_view = dest[dest_start:dest_start + row_count]

        if isinstance(source, torch.Tensor):
            if isinstance(selector, slice):
                dest_view.copy_(source[selector])
                return 0
            index_tensor = torch.as_tensor(selector, dtype=torch.long)
            dest_view.copy_(source.index_select(0, index_tensor))
            return 1

        if isinstance(selector, slice):
            dest_view.copy_(torch.from_numpy(source[selector]))
            return 1
        dest_view.copy_(torch.from_numpy(source[selector]))
        return 1

    def _fill_output_from_sources(
        self,
        output_batch,
        *,
        dest_start: int,
        obs_source,
        actions_source,
        masks_source,
        invisible_source,
        selector,
    ) -> int:
        temp_tensors = 0
        if self.oracle:
            obs_out, invisible_out, actions_out, masks_out = output_batch
            temp_tensors += self._copy_rows_into_tensor(
                invisible_out,
                dest_start,
                invisible_source,
                selector,
            )
        else:
            obs_out, actions_out, masks_out = output_batch

        temp_tensors += self._copy_rows_into_tensor(obs_out, dest_start, obs_source, selector)
        temp_tensors += self._copy_rows_into_tensor(actions_out, dest_start, actions_source, selector)
        temp_tensors += self._copy_rows_into_tensor(masks_out, dest_start, masks_source, selector)
        return temp_tensors

    def _fill_dense_chunk_into_output(
        self,
        output_batch,
        *,
        dest_start: int,
        chunk: ActionChunkBuffer,
        start_idx: int,
        end_idx: int,
        order=None,
    ) -> tuple[int, int]:
        selector = slice(start_idx, end_idx)
        if order is not None:
            selector = order[start_idx:end_idx]
        temp_tensors = self._fill_output_from_sources(
            output_batch,
            dest_start=dest_start,
            obs_source=chunk.obs,
            actions_source=chunk.actions,
            masks_source=chunk.masks,
            invisible_source=chunk.invisible_obs,
            selector=selector,
        )
        return 1, temp_tensors

    def _fill_segmented_chunk_into_output(
        self,
        output_batch,
        *,
        dest_start: int,
        chunk: ActionSegmentedChunkBuffer,
        start_idx: int,
        end_idx: int,
        order=None,
    ) -> tuple[int, int]:
        source_runs = 0
        temp_tensors = 0
        write_offset = int(dest_start)

        if order is None:
            remaining_start = int(start_idx)
            remaining_end = int(end_idx)
            previous_end = 0
            for segment, segment_end in zip(chunk.segments, chunk.segment_ends):
                if remaining_end <= previous_end:
                    break
                if remaining_start >= segment_end:
                    previous_end = int(segment_end)
                    continue
                local_start = max(remaining_start, previous_end) - previous_end
                local_end = min(remaining_end, int(segment_end)) - previous_end
                if local_end > local_start:
                    temp_tensors += self._fill_output_from_sources(
                        output_batch,
                        dest_start=write_offset,
                        obs_source=segment.obs,
                        actions_source=segment.actions,
                        masks_source=segment.masks,
                        invisible_source=segment.invisible_obs,
                        selector=slice(local_start, local_end),
                    )
                    write_offset += local_end - local_start
                    source_runs += 1
                previous_end = int(segment_end)
            return source_runs, temp_tensors

        batch_indices = order[start_idx:end_idx]
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.detach().cpu().numpy()
        elif not isinstance(batch_indices, np.ndarray):
            batch_indices = np.asarray(batch_indices, dtype=np.int64)
        if batch_indices.size == 0:
            return 0, 0
        batch_indices = batch_indices.astype(np.int64, copy=False)
        segment_ids = np.searchsorted(chunk.segment_ends, batch_indices, side='right')
        total_rows = int(batch_indices.shape[0])
        run_start = 0
        while run_start < total_rows:
            segment_id = int(segment_ids[run_start])
            run_end = run_start + 1
            while run_end < total_rows and int(segment_ids[run_end]) == segment_id:
                run_end += 1
            segment = chunk.segments[segment_id]
            segment_start = 0 if segment_id == 0 else int(chunk.segment_ends[segment_id - 1])
            selector = batch_indices[run_start:run_end] - segment_start
            temp_tensors += self._fill_output_from_sources(
                output_batch,
                dest_start=write_offset,
                obs_source=segment.obs,
                actions_source=segment.actions,
                masks_source=segment.masks,
                invisible_source=segment.invisible_obs,
                selector=selector,
            )
            write_offset += run_end - run_start
            source_runs += 1
            run_start = run_end
        return source_runs, temp_tensors

    def slice_batch_from_segmented_chunk(
        self,
        chunk: ActionSegmentedChunkBuffer,
        start_idx: int,
        end_idx: int,
        order=None,
    ):
        row_count = max(int(end_idx) - int(start_idx), 0)
        output_batch = self._allocate_output_batch(chunk=chunk, rows=row_count)
        self._fill_segmented_chunk_into_output(
            output_batch,
            dest_start=0,
            chunk=chunk,
            start_idx=start_idx,
            end_idx=end_idx,
            order=order,
        )
        return output_batch

    def build_sample_order(self, sample_count: int, *, exact_shuffle: bool = False):
        if not self.shuffle or sample_count <= 0:
            return None
        if exact_shuffle:
            return np.asarray(
                random.sample(range(sample_count), sample_count),
                dtype=np.int64,
            )
        return torch.randperm(sample_count)

    def materialize_spilled_batch(self, pending: deque, batch_size: int):
        started_at = time.perf_counter()
        with torch.profiler.record_function('bc.collate_or_assemble'):
            remaining = batch_size
            output_batch = None

            while remaining > 0:
                current = pending[0]
                chunk = current['chunk']
                offset = current['offset']
                order = current['order']
                available = chunk.sample_count - offset
                take = min(available, remaining)
                if output_batch is None:
                    output_batch = self._allocate_output_batch(chunk=chunk, rows=batch_size)

                dest_start = batch_size - remaining
                if isinstance(chunk, ActionSegmentedChunkBuffer):
                    self._fill_segmented_chunk_into_output(
                        output_batch,
                        dest_start=dest_start,
                        chunk=chunk,
                        start_idx=offset,
                        end_idx=offset + take,
                        order=order,
                    )
                else:
                    self._fill_dense_chunk_into_output(
                        output_batch,
                        dest_start=dest_start,
                        chunk=chunk,
                        start_idx=offset,
                        end_idx=offset + take,
                        order=order,
                    )

                current['offset'] += take
                if current['offset'] >= chunk.sample_count:
                    pending.popleft()
                remaining -= take

            result = output_batch
        self.loader_stats.record_collate_seconds(time.perf_counter() - started_at)
        return result

    def iter_batches_from_buffers_spill(self, buffers, *, exact_shuffle: bool = False):
        pending = deque()
        pending_rows = 0
        for buffer in buffers:
            if isinstance(buffer, ActionBatchBuffer):
                yield from buffer.batches
                continue
            if not isinstance(buffer, (ActionChunkBuffer, ActionSegmentedChunkBuffer)):
                raise TypeError(
                    'prebatch_spill_across_chunks requires ActionChunkBuffer-compatible buffers; '
                    f'got {type(buffer)!r}'
                )
            if buffer.sample_count <= 0:
                continue
            order = self.build_sample_order(
                buffer.sample_count,
                exact_shuffle=exact_shuffle,
            )
            pending.append({
                'chunk': buffer,
                'offset': 0,
                'order': order,
            })
            pending_rows += buffer.sample_count
            while pending_rows >= self.batch_size:
                yield self.materialize_spilled_batch(pending, self.batch_size)
                pending_rows -= self.batch_size
        if pending_rows > 0:
            yield self.materialize_spilled_batch(pending, pending_rows)

    def iter_batches_from_buffer(self, buffer):
        if isinstance(buffer, ActionBatchBuffer):
            yield from buffer.batches
            return
        yield from self.iter_batches_from_chunk(buffer)

    def build_chunk_iter(self, file_list, *, augmented):
        if self.prefetch_strategy in ('dynamic_ram', 'prepared_ram') and self.prefetch_budget_bytes > 0:
            return self.iter_chunk_buffers_dynamic(file_list, augmented=augmented)
        return self.iter_chunk_buffers(file_list, augmented=augmented)

    def ordered_sample_block_from_chunk(self, chunk: ActionChunkBuffer) -> SampleBlock:
        started_at = time.perf_counter()
        with torch.profiler.record_function('bc.collate_or_assemble'):
            if chunk.sample_count <= 0:
                block = SampleBlock(
                    obs=chunk.obs,
                    actions=chunk.actions,
                    masks=chunk.masks,
                    invisible_obs=chunk.invisible_obs,
                    file_count=chunk.file_count,
                    sample_count=chunk.sample_count,
                    size_bytes=chunk.size_bytes,
                )
            elif not self.shuffle:
                block = SampleBlock(
                    obs=chunk.obs,
                    actions=chunk.actions,
                    masks=chunk.masks,
                    invisible_obs=chunk.invisible_obs,
                    file_count=chunk.file_count,
                    sample_count=chunk.sample_count,
                    size_bytes=chunk.size_bytes,
                )
            else:
                order = torch.tensor(
                    random.sample(range(chunk.sample_count), chunk.sample_count),
                    dtype=torch.long,
                )
                obs = chunk.obs.index_select(0, order)
                actions = chunk.actions.index_select(0, order)
                masks = chunk.masks.index_select(0, order)
                invisible_obs = (
                    chunk.invisible_obs.index_select(0, order)
                    if self.oracle
                    else None
                )
                block = SampleBlock(
                    obs=obs,
                    actions=actions,
                    masks=masks,
                    invisible_obs=invisible_obs,
                    file_count=chunk.file_count,
                    sample_count=chunk.sample_count,
                    size_bytes=(
                        tensor_nbytes(obs)
                        + tensor_nbytes(actions)
                        + tensor_nbytes(masks)
                        + tensor_nbytes(invisible_obs)
                    ),
                )
        self.loader_stats.record_collate_seconds(time.perf_counter() - started_at)
        return block

    def merge_sample_blocks(self, blocks: list[SampleBlock]) -> SampleBlock:
        if not blocks:
            empty_masks = torch.empty((0, 46), dtype=torch.bool)
            return SampleBlock(
                obs=torch.empty((0,), dtype=torch.int16),
                actions=torch.empty((0,), dtype=torch.int64),
                masks=empty_masks,
                invisible_obs=torch.empty((0,), dtype=torch.int16) if self.oracle else None,
                file_count=0,
                sample_count=0,
                size_bytes=tensor_nbytes(empty_masks),
            )
        if len(blocks) == 1:
            return blocks[0]

        started_at = time.perf_counter()
        with torch.profiler.record_function('bc.collate_or_assemble'):
            obs = torch.cat([block.obs for block in blocks], dim=0)
            actions = torch.cat([block.actions for block in blocks], dim=0)
            masks = torch.cat([block.masks for block in blocks], dim=0)
            invisible_obs = (
                torch.cat([block.invisible_obs for block in blocks], dim=0)
                if self.oracle
                else None
            )
        self.loader_stats.record_collate_seconds(time.perf_counter() - started_at)
        return SampleBlock(
            obs=obs,
            actions=actions,
            masks=masks,
            invisible_obs=invisible_obs,
            file_count=sum(block.file_count for block in blocks),
            sample_count=sum(block.sample_count for block in blocks),
            size_bytes=(
                tensor_nbytes(obs)
                + tensor_nbytes(actions)
                + tensor_nbytes(masks)
                + tensor_nbytes(invisible_obs)
            ),
        )

    def iter_sample_blocks(self, file_list, *, augmented):
        pending_blocks = []
        pending_samples = 0
        for chunk in self.build_chunk_iter(file_list, augmented=augmented):
            if not isinstance(chunk, ActionChunkBuffer):
                raise TypeError(
                    'loader_mode=preassembled_batches requires ActionChunkBuffer chunks; '
                    f'got {type(chunk)!r}'
                )
            block = self.ordered_sample_block_from_chunk(chunk)
            if block.sample_count <= 0:
                continue
            pending_blocks.append(block)
            pending_samples += block.sample_count
            if pending_samples >= self.loader_block_target_samples:
                yield self.merge_sample_blocks(pending_blocks)
                pending_blocks = []
                pending_samples = 0
        if pending_blocks:
            yield self.merge_sample_blocks(pending_blocks)

    def load_preassembled_batches(self, augmented):
        file_list = list(self.file_list)
        if self.shuffle:
            random.shuffle(file_list)
        chunk_iter = self.build_chunk_iter(file_list, augmented=augmented)
        yield from self.iter_batches_from_buffers_spill(
            chunk_iter,
            exact_shuffle=True,
        )

    def build_iter(self):
        try:
            while True:
                for _ in range(self.num_epochs):
                    yield from self.load_files(self.augmented_first)
                    if self.enable_augmentation:
                        yield from self.load_files(not self.augmented_first)
                if not self.cycle:
                    break
        finally:
            self.close_raw_source()

    def load_files(self, augmented):
        if self.loader_mode == 'preassembled_batches':
            yield from self.load_preassembled_batches(augmented)
            return

        file_list = list(self.file_list)
        if self.shuffle:
            random.shuffle(file_list)

        chunk_iter = self.build_chunk_iter(file_list, augmented=augmented)

        if self.prebatched and self.prebatch_spill_across_chunks:
            yield from self.iter_batches_from_buffers_spill(chunk_iter)
            return

        for buffer in chunk_iter:
            if self.prebatched:
                yield from self.iter_batches_from_buffer(buffer)
                continue
            buffer_size = len(buffer)
            if self.shuffle:
                order = random.sample(range(buffer_size), buffer_size)
            else:
                order = range(buffer_size)
            for i in order:
                yield buffer[i]

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def build_action_file_dataloader(
    *,
    version,
    file_list,
    oracle=False,
    file_batch_size=20,
    player_names=None,
    excludes=None,
    num_epochs=1,
    enable_augmentation=False,
    augmented_first=False,
    trust_seed=False,
    always_include_kan_select=True,
    cycle=False,
    shuffle=True,
    allowed_player_ids_by_path=None,
    prefetch_chunks=0,
    prefetch_strategy='static_chunks',
    prefetch_budget_bytes=0,
    prefetch_target_chunk_bytes=0,
    prefetch_low_watermark=0.35,
    prefetch_high_watermark=0.85,
    prefetch_threads=1,
    decode_threads=1,
    batch_size=None,
    prebatched=False,
    prebatch_layout='chunk',
    prebatch_shuffle_mode='sample',
    prebatch_spill_across_chunks=False,
    prefetch_out_of_order=False,
    prefetch_startup_file_batch_size=0,
    prefetch_startup_ready_chunks=1,
    prefetch_inflight_budget_bytes=0,
    prefetch_ready_budget_bytes=0,
    prefetch_max_inflight_chunks=0,
    prefetch_min_file_batch_size=1,
    prefetch_raw_lru_budget_bytes=0,
    num_workers=0,
    pin_memory=False,
    multiprocessing_context='',
    persistent_workers=False,
    prefetch_factor=None,
    in_order=True,
    raw_source_backend='files',
    raw_pack_path='',
    raw_pack_index_path='',
    loader_mode='baseline',
    loader_block_target_samples=65536,
):
    data = ActionFileDatasetsIter(
        version=version,
        file_list=file_list,
        oracle=oracle,
        file_batch_size=file_batch_size,
        player_names=player_names,
        excludes=excludes,
        num_epochs=num_epochs,
        enable_augmentation=enable_augmentation,
        augmented_first=augmented_first,
        trust_seed=trust_seed,
        always_include_kan_select=always_include_kan_select,
        cycle=cycle,
        shuffle=shuffle,
        allowed_player_ids_by_path=allowed_player_ids_by_path,
        prefetch_chunks=prefetch_chunks,
        prefetch_strategy=prefetch_strategy,
        prefetch_budget_bytes=prefetch_budget_bytes,
        prefetch_target_chunk_bytes=prefetch_target_chunk_bytes,
        prefetch_low_watermark=prefetch_low_watermark,
        prefetch_high_watermark=prefetch_high_watermark,
        prefetch_threads=prefetch_threads,
        decode_threads=decode_threads,
        batch_size=batch_size,
        prebatched=prebatched,
        prebatch_layout=prebatch_layout,
        prebatch_shuffle_mode=prebatch_shuffle_mode,
        prebatch_spill_across_chunks=prebatch_spill_across_chunks,
        prefetch_out_of_order=prefetch_out_of_order,
        prefetch_startup_file_batch_size=prefetch_startup_file_batch_size,
        prefetch_startup_ready_chunks=prefetch_startup_ready_chunks,
        prefetch_inflight_budget_bytes=prefetch_inflight_budget_bytes,
        prefetch_ready_budget_bytes=prefetch_ready_budget_bytes,
        prefetch_max_inflight_chunks=prefetch_max_inflight_chunks,
        prefetch_min_file_batch_size=prefetch_min_file_batch_size,
        prefetch_raw_lru_budget_bytes=prefetch_raw_lru_budget_bytes,
        raw_source_backend=raw_source_backend,
        raw_pack_path=raw_pack_path,
        raw_pack_index_path=raw_pack_index_path,
        loader_mode=loader_mode,
        loader_block_target_samples=loader_block_target_samples,
    )
    loader_kwargs = dict(
        dataset=data,
        batch_size=None if data.emits_batches else batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        in_order=bool(in_order),
    )
    if not data.emits_batches and batch_size is not None and num_workers == 0:
        def timed_collate(batch):
            started_at = time.perf_counter()
            with torch.profiler.record_function('bc.collate_or_assemble'):
                collated = default_collate(batch)
            data.loader_stats.record_collate_seconds(time.perf_counter() - started_at)
            return collated

        loader_kwargs['collate_fn'] = timed_collate
    if num_workers > 0 and multiprocessing_context:
        loader_kwargs['multiprocessing_context'] = multiprocessing_context
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(prefetch_factor)
    loader = DataLoader(**loader_kwargs)
    loader.loader_stats = data.loader_stats
    return loader, data.loader_stats


_CPU_PIPE_END = object()


@dataclass
class ReadyBatch:
    batch: object
    nbytes: int
    nsamples: int
    batch_idx: int


class SyncCpuBatchPipe:
    def __init__(self, make_iter, *, loader_stats: LoaderStats | None = None):
        self._make_iter = make_iter
        self._it = None
        self._loader_stats = loader_stats
        self._consumer_wait_seconds = 0.0
        self._closed = False
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        if self._loader_stats is None:
            return
        self._loader_stats.update_cpu_pipe_state(
            ready_batches=0,
            ready_bytes=0,
            consumer_wait_seconds_total=self._consumer_wait_seconds,
        )

    def start(self):
        if self._it is None:
            self._it = self._make_iter()
        return self

    def record_consumer_wait(self, wait_seconds: float) -> None:
        self._consumer_wait_seconds += max(float(wait_seconds), 0.0)
        self._refresh_stats()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_fn = getattr(self._it, 'close', None)
        if callable(close_fn):
            close_fn()

    def snapshot(self) -> dict:
        return (
            self._loader_stats.snapshot()
            if self._loader_stats is not None
            else {
                'cpu_ready_batches': 0,
                'cpu_ready_bytes': 0,
                'cpu_consumer_wait_seconds_total': self._consumer_wait_seconds,
            }
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self.start()
        return next(self._it)


class AsyncCpuBatchPipe:
    def __init__(
        self,
        *,
        make_iter,
        max_ready_batches: int,
        max_ready_bytes: int = 0,
        loader_stats: LoaderStats | None = None,
        poll_timeout_seconds: float = 0.1,
    ):
        self._make_iter = make_iter
        self._loader_stats = loader_stats
        self._ready = queue.Queue(maxsize=max(int(max_ready_batches), 1))
        self._max_ready_bytes_limit = max(int(max_ready_bytes or 0), 0)
        self._poll_timeout_seconds = max(float(poll_timeout_seconds), 1e-3)
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        self._exc = None
        self._ready_batches = 0
        self._ready_bytes = 0
        self._max_ready_batches = 0
        self._max_ready_bytes = 0
        self._produced_batches = 0
        self._produced_samples = 0
        self._blocked_put_seconds = 0.0
        self._consumer_wait_seconds = 0.0
        self._worker = threading.Thread(
            target=self._worker_main,
            name='bc-cpu-producer',
            daemon=False,
        )
        self._closed = False
        self._refresh_stats_locked()

    def _byte_capacity_available_locked(self, item_nbytes: int) -> bool:
        if self._max_ready_bytes_limit <= 0:
            return True
        if self._ready_batches <= 0:
            return True
        return (self._ready_bytes + max(int(item_nbytes), 0)) <= self._max_ready_bytes_limit

    def _refresh_stats_locked(self) -> None:
        if self._loader_stats is None:
            return
        self._loader_stats.update_cpu_pipe_state(
            ready_batches=self._ready_batches,
            ready_bytes=self._ready_bytes,
            produced_batches_total=self._produced_batches,
            produced_samples_total=self._produced_samples,
            blocked_put_seconds_total=self._blocked_put_seconds,
            consumer_wait_seconds_total=self._consumer_wait_seconds,
        )

    def start(self):
        self._worker.start()
        return self

    def _worker_main(self):
        iterator = None
        try:
            iterator = self._make_iter()
            for batch_idx, batch in enumerate(iterator):
                if self._stop.is_set():
                    break
                item = ReadyBatch(
                    batch=batch,
                    nbytes=tree_nbytes(batch),
                    nsamples=tree_batch_len(batch),
                    batch_idx=batch_idx,
                )
                while not self._stop.is_set():
                    t0 = time.perf_counter()
                    with self._stats_lock:
                        byte_capacity_available = self._byte_capacity_available_locked(item.nbytes)
                    if not byte_capacity_available:
                        time.sleep(self._poll_timeout_seconds)
                        dt = time.perf_counter() - t0
                        with self._stats_lock:
                            self._blocked_put_seconds += dt
                            self._refresh_stats_locked()
                        continue
                    try:
                        with torch.profiler.record_function('bc.cpu_queue_put'):
                            self._ready.put(item, timeout=self._poll_timeout_seconds)
                    except queue.Full:
                        dt = time.perf_counter() - t0
                        with self._stats_lock:
                            self._blocked_put_seconds += dt
                            self._refresh_stats_locked()
                        continue
                    dt = time.perf_counter() - t0
                    with self._stats_lock:
                        self._blocked_put_seconds += dt
                        self._produced_batches += 1
                        self._produced_samples += item.nsamples
                        self._ready_batches += 1
                        self._ready_bytes += item.nbytes
                        self._max_ready_batches = max(self._max_ready_batches, self._ready_batches)
                        self._max_ready_bytes = max(self._max_ready_bytes, self._ready_bytes)
                        self._refresh_stats_locked()
                    break
        except BaseException as exc:
            self._exc = exc
        finally:
            close_fn = getattr(iterator, 'close', None)
            if callable(close_fn):
                close_fn()
            while True:
                try:
                    with torch.profiler.record_function('bc.cpu_queue_put'):
                        self._ready.put(_CPU_PIPE_END, timeout=self._poll_timeout_seconds)
                    break
                except queue.Full:
                    if self._stop.is_set():
                        continue

    def record_consumer_wait(self, wait_seconds: float) -> None:
        with self._stats_lock:
            self._consumer_wait_seconds += max(float(wait_seconds), 0.0)
            self._refresh_stats_locked()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        while True:
            try:
                item = self._ready.get_nowait()
            except queue.Empty:
                break
            if item is _CPU_PIPE_END:
                continue
            with self._stats_lock:
                self._ready_batches = max(self._ready_batches - 1, 0)
                self._ready_bytes = max(self._ready_bytes - int(item.nbytes), 0)
                self._refresh_stats_locked()
        self._worker.join(timeout=10.0)

    def snapshot(self) -> dict:
        if self._loader_stats is not None:
            return self._loader_stats.snapshot()
        with self._stats_lock:
            return {
                'cpu_ready_batches': self._ready_batches,
                'cpu_ready_bytes': self._ready_bytes,
                'max_cpu_ready_batches': self._max_ready_batches,
                'max_cpu_ready_bytes': self._max_ready_bytes,
                'cpu_produced_batches_total': self._produced_batches,
                'cpu_produced_samples_total': self._produced_samples,
                'cpu_blocked_put_seconds_total': self._blocked_put_seconds,
                'cpu_consumer_wait_seconds_total': self._consumer_wait_seconds,
            }

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                with torch.profiler.record_function('bc.cpu_queue_get'):
                    item = self._ready.get(timeout=self._poll_timeout_seconds)
            except queue.Empty:
                if self._stop.is_set() and not self._worker.is_alive():
                    if self._exc is not None:
                        raise self._exc
                    raise StopIteration
                continue
            if item is _CPU_PIPE_END:
                if self._exc is not None:
                    raise self._exc
                raise StopIteration
            with self._stats_lock:
                self._ready_batches = max(self._ready_batches - 1, 0)
                self._ready_bytes = max(self._ready_bytes - int(item.nbytes), 0)
                self._refresh_stats_locked()
            return item.batch
