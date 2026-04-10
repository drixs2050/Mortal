import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from math import ceil
from os import path
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from bc_dataset import normalize_metadata_path


_WORKER_LOADER = None


def batch_count_for_steps(step_count: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if step_count <= 0:
        return 0
    return ceil(step_count / batch_size)


def save_step_count_summary(output_path: str, summary: dict) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def load_step_count_summary(summary_path: str) -> dict:
    with open(summary_path, encoding='utf-8') as f:
        payload = json.load(f)
    if payload.get('format') != 'bc_step_counts_v1':
        raise ValueError(f'unsupported step-count format in {summary_path}')
    return payload


def step_count_config_summary(
    *,
    path_cache: str,
    actor_filter_index: str,
    actor_filter_manifest: str,
    min_actor_dan: int | None,
    version: int,
    oracle: bool,
    trust_seed: bool,
    always_include_kan_select: bool,
    file_batch_size: int,
    batch_size_reference: int,
    jobs: int,
    chunk_size: int,
) -> dict:
    return {
        'path_cache': normalize_metadata_path(path_cache) if path_cache else '',
        'actor_filter_index': normalize_metadata_path(actor_filter_index) if actor_filter_index else '',
        'actor_filter_manifest': normalize_metadata_path(actor_filter_manifest) if actor_filter_manifest else '',
        'min_actor_dan': min_actor_dan,
        'version': version,
        'oracle': oracle,
        'trust_seed': trust_seed,
        'always_include_kan_select': always_include_kan_select,
        'file_batch_size': file_batch_size,
        'batch_size_reference': batch_size_reference,
        'jobs': jobs,
        'chunk_size': chunk_size,
    }


def expected_batches_from_summary(
    payload: dict,
    *,
    split: str,
    batch_size: int,
    file_count: int | None = None,
    max_batches: int = 0,
) -> tuple[int, int] | None:
    split_summary = (payload.get('splits') or {}).get(split)
    if not split_summary:
        return None
    expected_file_count = split_summary.get('requested_file_count')
    if file_count is not None and expected_file_count not in (None, file_count):
        return None
    step_count = int(split_summary.get('step_count', 0))
    batch_count = batch_count_for_steps(step_count, batch_size)
    if max_batches > 0:
        batch_count = min(batch_count, max_batches)
    return step_count, batch_count


def _chunked(items: list[str], chunk_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _load_files_with_fallback(loader, file_list: list[str]) -> list[tuple[str, object]]:
    try:
        data = loader.load_gz_log_files(file_list)
        return list(zip(file_list, data))
    except BaseException as batch_exc:
        if len(file_list) == 1:
            logging.warning(
                'skipping unreadable BC count file %s: %s: %s',
                file_list[0],
                type(batch_exc).__name__,
                batch_exc,
            )
            return []
        logging.warning(
            'BC count batch load failed for %s files; retrying individually. error=%s: %s',
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
                    'skipping unreadable BC count file %s: %s: %s',
                    filename,
                    type(single_exc).__name__,
                    single_exc,
                )
                continue
            if not data:
                logging.warning('skipping BC count file with empty loader output: %s', filename)
                continue
            loaded.append((filename, data[0]))
        return loaded


def _count_loaded_files(
    loaded_files: list[tuple[str, object]],
    *,
    allowed_player_ids_by_path: dict[str, tuple[int, ...]] | None,
) -> dict:
    requested_file_count = len(loaded_files)
    nonempty_file_count = 0
    trajectory_count = 0
    step_count = 0

    for filename, file_data in loaded_files:
        allowed_player_ids = None
        if allowed_player_ids_by_path is not None:
            allowed_player_ids = allowed_player_ids_by_path.get(filename)
        file_steps = 0
        for game in file_data:
            if allowed_player_ids is not None:
                player_id = game.take_player_id()
                if player_id not in allowed_player_ids:
                    continue
            trajectory_count += 1
            file_steps += len(game.take_actions())
        if file_steps > 0:
            nonempty_file_count += 1
        step_count += file_steps

    return {
        'requested_file_count': requested_file_count,
        'loaded_file_count': requested_file_count,
        'nonempty_file_count': nonempty_file_count,
        'trajectory_count': trajectory_count,
        'step_count': step_count,
        'skipped_file_count': 0,
    }


def _count_chunk(
    *,
    loader,
    file_list: list[str],
    file_batch_size: int,
    allowed_player_ids_by_path: dict[str, tuple[int, ...]] | None,
) -> dict:
    totals = {
        'requested_file_count': len(file_list),
        'loaded_file_count': 0,
        'nonempty_file_count': 0,
        'trajectory_count': 0,
        'step_count': 0,
        'skipped_file_count': 0,
    }
    for subchunk in _chunked(file_list, file_batch_size):
        loaded_files = _load_files_with_fallback(loader, subchunk)
        counts = _count_loaded_files(
            loaded_files,
            allowed_player_ids_by_path=allowed_player_ids_by_path,
        )
        totals['loaded_file_count'] += counts['loaded_file_count']
        totals['nonempty_file_count'] += counts['nonempty_file_count']
        totals['trajectory_count'] += counts['trajectory_count']
        totals['step_count'] += counts['step_count']
        totals['skipped_file_count'] += len(subchunk) - counts['loaded_file_count']
    return totals


def _init_step_counter_worker(
    version: int,
    oracle: bool,
    player_names: list[str] | None,
    excludes: list[str] | None,
    trust_seed: bool,
    always_include_kan_select: bool,
) -> None:
    global _WORKER_LOADER

    from libriichi.dataset import GameplayLoader

    _WORKER_LOADER = GameplayLoader(
        version=version,
        oracle=oracle,
        player_names=player_names,
        excludes=excludes,
        trust_seed=trust_seed,
        always_include_kan_select=always_include_kan_select,
        augmented=False,
    )


def _count_chunk_worker(task: tuple[list[str], dict[str, tuple[int, ...]] | None, int]) -> dict:
    file_list, allowed_player_ids_by_path, file_batch_size = task
    if _WORKER_LOADER is None:
        raise RuntimeError('step-count worker loader was not initialized')
    return _count_chunk(
        loader=_WORKER_LOADER,
        file_list=file_list,
        file_batch_size=file_batch_size,
        allowed_player_ids_by_path=allowed_player_ids_by_path,
    )


def _merge_count_totals(accum: dict, update: dict) -> None:
    for key in (
        'requested_file_count',
        'loaded_file_count',
        'nonempty_file_count',
        'trajectory_count',
        'step_count',
        'skipped_file_count',
    ):
        accum[key] += int(update.get(key, 0))


def _empty_count_totals() -> dict:
    return {
        'requested_file_count': 0,
        'loaded_file_count': 0,
        'nonempty_file_count': 0,
        'trajectory_count': 0,
        'step_count': 0,
        'skipped_file_count': 0,
    }


def _finalize_count_totals(
    totals: dict,
    *,
    batch_size_reference: int,
    effective_jobs: int,
    fell_back_to_single_process: bool,
) -> dict:
    totals = dict(totals)
    totals['batch_size_reference'] = batch_size_reference
    totals['batch_count_reference'] = batch_count_for_steps(
        totals['step_count'],
        batch_size_reference,
    )
    totals['effective_jobs'] = effective_jobs
    totals['fell_back_to_single_process'] = fell_back_to_single_process
    return totals


def _count_split_steps_serial(
    *,
    split_name: str,
    file_list: list[str],
    version: int,
    oracle: bool,
    file_batch_size: int,
    player_names: list[str] | None,
    excludes: list[str] | None,
    trust_seed: bool,
    always_include_kan_select: bool,
    allowed_player_ids_by_path: dict[str, tuple[int, ...]] | None,
    chunk_size: int,
) -> dict:
    from libriichi.dataset import GameplayLoader

    totals = _empty_count_totals()
    loader = GameplayLoader(
        version=version,
        oracle=oracle,
        player_names=player_names,
        excludes=excludes,
        trust_seed=trust_seed,
        always_include_kan_select=always_include_kan_select,
        augmented=False,
    )

    with tqdm(
        total=len(file_list),
        desc=f'COUNT-{split_name.upper()}',
        unit='file',
        dynamic_ncols=True,
        ascii=True,
    ) as pb:
        for chunk_files in _chunked(file_list, chunk_size):
            allowed_chunk = None
            if allowed_player_ids_by_path is not None:
                allowed_chunk = {
                    filename: allowed_player_ids_by_path[filename]
                    for filename in chunk_files
                }
            counts = _count_chunk(
                loader=loader,
                file_list=chunk_files,
                file_batch_size=file_batch_size,
                allowed_player_ids_by_path=allowed_chunk,
            )
            _merge_count_totals(totals, counts)
            pb.update(len(chunk_files))
    return totals


def _count_split_steps_parallel(
    *,
    split_name: str,
    file_list: list[str],
    version: int,
    oracle: bool,
    file_batch_size: int,
    player_names: list[str] | None,
    excludes: list[str] | None,
    trust_seed: bool,
    always_include_kan_select: bool,
    allowed_player_ids_by_path: dict[str, tuple[int, ...]] | None,
    jobs: int,
    chunk_size: int,
) -> dict:
    totals = _empty_count_totals()
    tasks = []
    for chunk_files in _chunked(file_list, chunk_size):
        allowed_chunk = None
        if allowed_player_ids_by_path is not None:
            allowed_chunk = {
                filename: allowed_player_ids_by_path[filename]
                for filename in chunk_files
            }
        tasks.append((chunk_files, allowed_chunk, file_batch_size))

    mp_context = multiprocessing.get_context('spawn')
    with tqdm(
        total=len(file_list),
        desc=f'COUNT-{split_name.upper()}',
        unit='file',
        dynamic_ncols=True,
        ascii=True,
    ) as pb:
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=mp_context,
            initializer=_init_step_counter_worker,
            initargs=(
                version,
                oracle,
                player_names,
                excludes,
                trust_seed,
                always_include_kan_select,
            ),
        ) as pool:
            futures = {
                pool.submit(_count_chunk_worker, task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                chunk_files = futures[future]
                counts = future.result()
                _merge_count_totals(totals, counts)
                pb.update(len(chunk_files))
    return totals


def _count_split_steps(
    *,
    split_name: str,
    file_list: list[str],
    version: int,
    oracle: bool,
    file_batch_size: int,
    player_names: list[str] | None,
    excludes: list[str] | None,
    trust_seed: bool,
    always_include_kan_select: bool,
    allowed_player_ids_by_path: dict[str, tuple[int, ...]] | None,
    batch_size_reference: int,
    jobs: int,
    chunk_size: int,
) -> dict:
    if jobs <= 1:
        totals = _count_split_steps_serial(
            split_name=split_name,
            file_list=file_list,
            version=version,
            oracle=oracle,
            file_batch_size=file_batch_size,
            player_names=player_names,
            excludes=excludes,
            trust_seed=trust_seed,
            always_include_kan_select=always_include_kan_select,
            allowed_player_ids_by_path=allowed_player_ids_by_path,
            chunk_size=chunk_size,
        )
        return _finalize_count_totals(
            totals,
            batch_size_reference=batch_size_reference,
            effective_jobs=1,
            fell_back_to_single_process=False,
        )

    try:
        totals = _count_split_steps_parallel(
            split_name=split_name,
            file_list=file_list,
            version=version,
            oracle=oracle,
            file_batch_size=file_batch_size,
            player_names=player_names,
            excludes=excludes,
            trust_seed=trust_seed,
            always_include_kan_select=always_include_kan_select,
            allowed_player_ids_by_path=allowed_player_ids_by_path,
            jobs=jobs,
            chunk_size=chunk_size,
        )
        return _finalize_count_totals(
            totals,
            batch_size_reference=batch_size_reference,
            effective_jobs=jobs,
            fell_back_to_single_process=False,
        )
    except (BrokenProcessPool, EOFError, RuntimeError) as exc:
        logging.warning(
            'concurrent BC step counting failed for split=%s with jobs=%s; '
            'falling back to single-process counting. error=%s: %s',
            split_name,
            jobs,
            type(exc).__name__,
            exc,
        )
        totals = _count_split_steps_serial(
            split_name=split_name,
            file_list=file_list,
            version=version,
            oracle=oracle,
            file_batch_size=file_batch_size,
            player_names=player_names,
            excludes=excludes,
            trust_seed=trust_seed,
            always_include_kan_select=always_include_kan_select,
            allowed_player_ids_by_path=allowed_player_ids_by_path,
            chunk_size=chunk_size,
        )
        return _finalize_count_totals(
            totals,
            batch_size_reference=batch_size_reference,
            effective_jobs=1,
            fell_back_to_single_process=True,
        )


def build_step_count_summary(
    *,
    split_lists: dict[str, list[str]],
    version: int,
    oracle: bool,
    file_batch_size: int,
    player_names: list[str] | None,
    excludes: list[str] | None,
    trust_seed: bool,
    always_include_kan_select: bool,
    actor_filter_map: dict[str, tuple[int, ...]] | None,
    batch_size_reference: int,
    jobs: int,
    chunk_size: int,
    config_summary: dict,
) -> dict:
    splits = {}
    for split_name, file_list in split_lists.items():
        splits[split_name] = _count_split_steps(
            split_name=split_name,
            file_list=file_list,
            version=version,
            oracle=oracle,
            file_batch_size=file_batch_size,
            player_names=player_names,
            excludes=excludes,
            trust_seed=trust_seed,
            always_include_kan_select=always_include_kan_select,
            allowed_player_ids_by_path=actor_filter_map,
            batch_size_reference=batch_size_reference,
            jobs=jobs,
            chunk_size=chunk_size,
        )

    return {
        'format': 'bc_step_counts_v1',
        'config': config_summary,
        'splits': splits,
    }
