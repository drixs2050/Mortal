import json
import os
import time
from argparse import ArgumentParser
from glob import glob
from os import path

import torch
from torch.utils.data import DataLoader
from wandb_utils import default_wandb_run_name, maybe_init_wandb_run


def parse_args():
    parser = ArgumentParser(
        description='Evaluate a saved behavior-cloning checkpoint on a held-out split.',
    )
    parser.add_argument(
        '--checkpoint',
        default='',
        help='Checkpoint to load. Defaults to bc.control.best_state_file from the active config.',
    )
    parser.add_argument(
        '--split',
        choices=('train', 'val', 'test'),
        default='val',
        help='Configured dataset split to evaluate when --list is not provided.',
    )
    parser.add_argument(
        '--list',
        default='',
        help='Optional explicit file list to evaluate instead of a configured split file.',
    )
    parser.add_argument(
        '--device',
        default='',
        help='Optional device override. Defaults to bc.control.device from the active config.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=0,
        help='Optional batch-size override. Defaults to bc.control.batch_size from the active config.',
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Optional limit on evaluation batches. Defaults to bc.control.eval_max_batches; 0 means full split.',
    )
    parser.add_argument(
        '--output-json',
        default='',
        help='Optional path to write a JSON summary.',
    )
    return parser.parse_args()


def split_sources(dataset_cfg: dict, split: str) -> tuple[str, list[str]]:
    if split == 'train':
        return dataset_cfg.get('train_list', ''), dataset_cfg.get('train_globs', [])
    if split == 'val':
        return dataset_cfg.get('val_list', ''), dataset_cfg.get('val_globs', [])
    if split == 'test':
        return dataset_cfg.get('test_list', ''), dataset_cfg.get('test_globs', [])
    raise ValueError(f'unexpected split: {split}')


def filtered_trimmed_lines(lines):
    return [line.strip() for line in lines if line.strip()]


def load_path_list(filename: str, root_dir: str = '') -> list[str]:
    with open(filename, encoding='utf-8') as f:
        paths = filtered_trimmed_lines(f)
    if root_dir:
        return [
            p if path.isabs(p) else path.join(root_dir, p)
            for p in paths
        ]
    return paths


def resolve_eval_file_list(dataset_cfg: dict, split: str, root_dir: str) -> list[str]:
    list_file, globs = split_sources(dataset_cfg, split)
    if list_file:
        return load_path_list(list_file, root_dir)
    if globs:
        file_list = []
        for pattern in globs:
            file_list.extend(glob(pattern, recursive=True))
        return sorted(file_list, reverse=True)
    raise ValueError(
        f'bc.dataset has no configured {split} split. '
        f'Expected {split}_list or {split}_globs.'
    )


def make_summary(*, checkpoint, split, file_count, batch_count, max_batches, metrics, state) -> dict:
    summary = {
        'checkpoint': checkpoint,
        'trainer': state.get('trainer'),
        'steps': state.get('steps'),
        'best_perf': state.get('best_perf'),
        'split': split,
        'file_count': file_count,
        'batch_count': batch_count,
        'max_batches': max_batches,
        'is_capped_eval': max_batches > 0,
        'processed_step_count': metrics.get('count'),
        'metrics': metrics,
    }
    expected_total_step_count = metrics.get('expected_total_step_count')
    if expected_total_step_count is not None:
        summary['expected_total_step_count'] = expected_total_step_count
    expected_total_batch_count = metrics.get('expected_total_batch_count')
    if expected_total_batch_count is not None:
        summary['expected_total_batch_count'] = expected_total_batch_count
    return summary


def wandb_eval_payload(*, split: str, batch_count: int, file_count: int, metrics: dict) -> dict:
    payload = {
        'eval/split': split,
        'eval/file_count': file_count,
        'eval/batch_count': batch_count,
        'eval/processed_step_count': metrics['count'],
        f'eval/{split}/nll': metrics['nll'],
        f'eval/{split}/top1': metrics['accuracy'],
        f'eval/{split}/topk': metrics['topk_accuracy'],
        f'eval/{split}/legal_rate': metrics['legal_rate'],
    }
    if metrics.get('expected_total_step_count') is not None:
        payload['eval/expected_total_step_count'] = metrics['expected_total_step_count']
    if metrics.get('expected_total_batch_count') is not None:
        payload['eval/expected_total_batch_count'] = metrics['expected_total_batch_count']
    for name, value in metrics.get('category_accuracy', {}).items():
        payload[f'eval/{split}/category_accuracy/{name}'] = value
    return payload


def evaluate():
    import prelude  # noqa: F401

    import logging

    from common import tqdm
    from bc_dataset import load_path_cache, normalize_file_list, resolve_actor_filter_map
    from bc_ram_cache import (
        resolve_runtime_cache_settings,
        runtime_cache_enabled,
        runtime_cache_split_settings,
    )
    from bc_stage import (
        StagedShardIterableDataset,
        resolve_stage_settings,
        stage_enabled,
        stage_manifest_paths,
        stage_preload_budget_bytes,
    )
    from bc_step_counts import expected_batches_from_summary, load_step_count_summary
    from dataloader import ActionFileDatasetsIter, resolve_prefetch_budget_bytes, worker_init_fn
    from model import Brain, DQN
    from config import config
    from train_bc import (
        apply_cuda_precision_settings,
        autocast_context_kwargs,
        empty_metric_sums,
        extract_policy_features,
        finalize_metric_sums,
        masked_logits,
        DeviceBatchPrefetcher,
        resolve_amp_dtype,
        update_metric_sums,
    )

    args = parse_args()
    cfg = config['bc']
    control_cfg = cfg['control']
    dataset_cfg = cfg['dataset']
    runtime_cache_cfg = resolve_runtime_cache_settings(config)
    use_runtime_cache = runtime_cache_enabled(config) and not bool(args.list)
    stage_cfg = resolve_stage_settings(config)
    wandb_cfg = cfg.get('wandb', {})
    use_staged_cache = stage_enabled(config) and not use_runtime_cache and not bool(args.list)
    checkpoint = args.checkpoint or control_cfg['best_state_file']
    device = torch.device(args.device or control_cfg['device'])
    default_eval_batch_size = control_cfg.get('eval_batch_size', 0) or control_cfg['batch_size']
    batch_size = args.batch_size or default_eval_batch_size
    default_eval_max_batches = control_cfg.get('eval_max_batches', 0)
    max_batches = default_eval_max_batches if args.max_batches is None else args.max_batches
    eval_num_workers = dataset_cfg.get('eval_num_workers', dataset_cfg.get('num_workers', 0))
    prefetch_chunks = dataset_cfg.get('prefetch_chunks', 0)
    eval_prefetch_chunks = dataset_cfg.get('eval_prefetch_chunks', prefetch_chunks)
    prefetch_strategy = str(dataset_cfg.get('prefetch_strategy', 'static_chunks') or 'static_chunks')
    prefetch_ram_budget_gib = float(dataset_cfg.get('prefetch_ram_budget_gib', 0) or 0)
    eval_prefetch_ram_budget_gib = float(
        dataset_cfg.get('eval_prefetch_ram_budget_gib', prefetch_ram_budget_gib) or 0
    )
    prefetch_target_chunk_gib = float(dataset_cfg.get('prefetch_target_chunk_gib', 0) or 0)
    prefetch_low_watermark = float(dataset_cfg.get('prefetch_low_watermark', 0.35))
    prefetch_high_watermark = float(dataset_cfg.get('prefetch_high_watermark', 0.85))
    prefetch_threads = int(dataset_cfg.get('prefetch_threads', 1) or 1)
    prebatched = bool(dataset_cfg.get('prebatched', False))
    eval_prefetch_startup_file_batch_size = int(
        dataset_cfg.get(
            'eval_prefetch_startup_file_batch_size',
            dataset_cfg.get('prefetch_startup_file_batch_size', 0),
        ) or 0
    )
    eval_prefetch_out_of_order = bool(
        dataset_cfg.get('eval_prefetch_out_of_order', dataset_cfg.get('prefetch_out_of_order', False))
    )
    eval_device_prefetch_batches = int(
        dataset_cfg.get('eval_device_prefetch_batches', dataset_cfg.get('device_prefetch_batches', 2)) or 2
    )
    eval_device_prefetch_startup_batches = int(
        dataset_cfg.get('eval_device_prefetch_startup_batches', min(eval_device_prefetch_batches, 1)) or 1
    )
    eval_pin_memory = dataset_cfg.get(
        'eval_pin_memory',
        dataset_cfg.get('pin_memory', device.type == 'cuda'),
    )
    multiprocessing_context = dataset_cfg.get('multiprocessing_context', 'spawn')
    step_count_summary_path = dataset_cfg.get('step_count_summary', '')
    runtime_eval_cache = (
        runtime_cache_split_settings(config, split_name=args.split, world_size=1)
        if use_runtime_cache
        else {}
    )
    enable_amp = control_cfg.get('enable_amp', False)
    amp_dtype = resolve_amp_dtype(control_cfg)
    torch.backends.cudnn.benchmark = control_cfg.get('enable_cudnn_benchmark', False)
    enable_tf32 = apply_cuda_precision_settings(control_cfg=control_cfg, device=device)
    eval_prefetch_budget_bytes = resolve_prefetch_budget_bytes(
        gib=eval_prefetch_ram_budget_gib,
        world_size=1,
    )
    prefetch_target_chunk_bytes = max(int(prefetch_target_chunk_gib * (1024 ** 3)), 0)
    stage_preload_budget = (
        stage_preload_budget_bytes(full_config=config, world_size=1)
        if use_staged_cache
        else 0
    )
    stage_preload_low_watermark = float(stage_cfg.get('preload_low_watermark', 0.65))
    stage_preload_high_watermark = float(stage_cfg.get('preload_high_watermark', 0.90))
    stage_preload_threads = int(stage_cfg.get('preload_threads', 4) or 4)

    state = torch.load(checkpoint, weights_only=True, map_location=device)
    state_cfg = state.get('config', config)
    state_bc = state_cfg.get('bc', {})
    state_control_cfg = state_bc.get('control', {})
    state_dataset_cfg = state_bc.get('dataset', {})
    version = state_control_cfg.get('version', state_cfg.get('control', {}).get('version', 4))
    resnet_cfg = state_bc.get('resnet', state_cfg.get('resnet', config['resnet']))
    oracle = state_dataset_cfg.get('oracle', dataset_cfg.get('oracle', False))
    trust_seed = dataset_cfg.get('trust_seed', state_dataset_cfg.get('trust_seed', False))
    always_include_kan_select = dataset_cfg.get(
        'always_include_kan_select',
        state_dataset_cfg.get('always_include_kan_select', True),
    )
    root_dir = dataset_cfg.get('root_dir', '')
    path_cache = dataset_cfg.get('path_cache', '')
    if args.list:
        file_list = load_path_list(args.list, root_dir)
        file_list = normalize_file_list(file_list, desc='PATHS-EVAL')
    else:
        if path_cache and path.exists(path_cache):
            cached_split_lists = load_path_cache(
                path_cache,
                expected_splits=[args.split],
                expected_sources={
                    split_name: dataset_cfg.get(f'{split_name}_list', '')
                    for split_name in ('train', 'val', 'test')
                    if dataset_cfg.get(f'{split_name}_list', '')
                },
            )
            file_list = cached_split_lists[args.split]
        else:
            file_list = resolve_eval_file_list(dataset_cfg, args.split, root_dir)
            file_list = normalize_file_list(file_list, desc='PATHS-EVAL')

    player_names = []
    exclude_names = []
    for filename in dataset_cfg.get('player_names_files', []):
        with open(filename, encoding='utf-8') as f:
            player_names.extend(filtered_trimmed_lines(f))
    for filename in dataset_cfg.get('exclude_names_files', []):
        with open(filename, encoding='utf-8') as f:
            exclude_names.extend(filtered_trimmed_lines(f))

    mortal = Brain(version=version, is_oracle=oracle, **resnet_cfg).to(device)
    dqn = DQN(version=version, hidden_dim=mortal.hidden_dim).to(device)
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    mortal.eval()
    dqn.eval()
    logging.info(
        'precision: amp=%s amp_dtype=%s tf32=%s',
        enable_amp,
        str(amp_dtype).replace('torch.', ''),
        enable_tf32,
    )
    logging.info(
        'eval loader: file_batch_size=%s eval_num_workers=%s eval_prefetch_chunks=%s prefetch_strategy=%s '
        'eval_prefetch_budget_gib=%s target_chunk_gib=%s prefetch_threads=%s prebatched=%s '
        'eval_startup_file_batch_size=%s eval_prefetch_out_of_order=%s eval_pin_memory=%s '
        'eval_device_prefetch_batches=%s eval_device_prefetch_startup_batches=%s runtime_cache_enabled=%s stage_enabled=%s',
        runtime_eval_cache.get('max_files_per_chunk', dataset_cfg['file_batch_size']) if use_runtime_cache else dataset_cfg['file_batch_size'],
        eval_num_workers,
        0 if use_runtime_cache else eval_prefetch_chunks,
        runtime_cache_cfg['mode'] if use_runtime_cache else prefetch_strategy,
        runtime_eval_cache.get('data_budget_bytes', 0) / (1024 ** 3) if use_runtime_cache else eval_prefetch_ram_budget_gib,
        runtime_eval_cache.get('target_chunk_bytes', 0) / (1024 ** 3) if use_runtime_cache else prefetch_target_chunk_gib,
        runtime_eval_cache.get('max_inflight_chunk_builders', 1) if use_runtime_cache else prefetch_threads,
        prebatched,
        runtime_eval_cache.get('min_files_per_chunk', eval_prefetch_startup_file_batch_size) if use_runtime_cache else eval_prefetch_startup_file_batch_size,
        eval_prefetch_out_of_order,
        eval_pin_memory,
        eval_device_prefetch_batches,
        eval_device_prefetch_startup_batches,
        use_runtime_cache,
        use_staged_cache,
    )
    if use_runtime_cache and stage_enabled(config):
        logging.warning('bc.runtime_cache.enabled=true takes precedence over bc.stage.enabled=true during eval')
    if use_runtime_cache:
        logging.info(
            'runtime cache: mode=%s node_ram_budget_gib=%s node_pinned_budget_gib=%s node_inflight_budget_gib=%s '
            'target_chunk_gib=%.2f decode_threads=%s max_inflight_chunk_builders=%s min_files_per_chunk=%s max_files_per_chunk=%s',
            runtime_cache_cfg['mode'],
            runtime_cache_cfg['node_ram_budget_gib'],
            runtime_cache_cfg['node_pinned_budget_gib'],
            runtime_cache_cfg['node_inflight_budget_gib'],
            runtime_eval_cache.get('target_chunk_bytes', 0) / (1024 ** 3),
            runtime_eval_cache.get('decode_threads', 1),
            runtime_eval_cache.get('max_inflight_chunk_builders', 1),
            runtime_eval_cache.get('min_files_per_chunk', 0),
            runtime_eval_cache.get('max_files_per_chunk', 0),
        )
    if use_staged_cache:
        logging.info(
            'stage cache: backend=%s cache_root=%s preload_ram_budget_gib=%s preload_low_watermark=%.2f '
            'preload_high_watermark=%.2f preload_threads=%s',
            stage_cfg['backend'],
            stage_cfg['cache_root'],
            stage_cfg['preload_ram_budget_gib'],
            stage_preload_low_watermark,
            stage_preload_high_watermark,
            stage_preload_threads,
        )

    if device.type == 'cuda' and batch_size >= 32768:
        logging.warning(
            'very large eval batch_size=%s requested; this may be unstable even on large GPUs. '
            'Prefer building step counts first, then try 8192 or 16384 before pushing higher.',
            f'{batch_size:,}',
        )

    actor_filter_map = None
    min_actor_dan = dataset_cfg.get('min_actor_dan')
    actor_filter_manifest = dataset_cfg.get('actor_filter_manifest', '')
    actor_filter_index = dataset_cfg.get('actor_filter_index', '')
    if min_actor_dan is not None and not use_staged_cache:
        actor_filter_map, actor_filter_summary = resolve_actor_filter_map(
            file_lists=[file_list],
            min_actor_dan=min_actor_dan,
            actor_filter_manifest=actor_filter_manifest,
            actor_filter_index=actor_filter_index,
            inputs_are_normalized=True,
        )
        logging.info(
            'actor dan filter enabled for eval: source=%s min_actor_dan=%s matched_files=%s eligible_files=%s filtered_out_files=%s',
            actor_filter_summary.get('source', 'unknown'),
            actor_filter_summary['min_actor_dan'],
            f"{actor_filter_summary['matched_row_count']:,}",
            f"{actor_filter_summary['eligible_file_count']:,}",
            f"{actor_filter_summary['filtered_out_file_count']:,}",
        )
    elif min_actor_dan is not None and use_staged_cache:
        logging.info(
            'actor dan filter is embedded in the staged cache: min_actor_dan=%s manifest=%s index=%s',
            min_actor_dan,
            actor_filter_manifest or 'n/a',
            actor_filter_index or 'n/a',
        )

    expected_total_step_count = None
    expected_total_batch_count = None
    if not args.list and step_count_summary_path and path.exists(step_count_summary_path):
        try:
            step_count_summary = load_step_count_summary(step_count_summary_path)
            step_count_info = expected_batches_from_summary(
                step_count_summary,
                split=args.split,
                batch_size=batch_size,
                file_count=len(file_list),
                max_batches=max_batches,
            )
        except Exception as exc:
            logging.warning(
                'failed to load BC step-count summary %s: %s: %s',
                step_count_summary_path,
                type(exc).__name__,
                exc,
            )
            step_count_info = None
        if step_count_info is None:
            logging.warning(
                'step-count summary %s did not match eval split=%s file_count=%s; continuing without total batch estimate',
                step_count_summary_path,
                args.split,
                f'{len(file_list):,}',
            )
        else:
            expected_total_step_count, expected_total_batch_count = step_count_info
            logging.info(
                'loaded step-count summary from %s for split=%s: total_steps=%s expected_batches=%s batch_size=%s',
                step_count_summary_path,
                args.split,
                f'{expected_total_step_count:,}',
                f'{expected_total_batch_count:,}',
                f'{batch_size:,}',
            )

    if use_staged_cache:
        manifest_path = stage_manifest_paths(config, splits=[args.split])[args.split]
        if not manifest_path.exists():
            raise FileNotFoundError(
                'missing staged BC shard manifest; run scripts/stage_bc_tensor_shards.py first: '
                f'{manifest_path}'
            )
        eval_data = StagedShardIterableDataset(
            manifest_path=manifest_path,
            batch_size=batch_size,
            shuffle=False,
            cycle=False,
            num_epochs=1,
            preload_budget_bytes=stage_preload_budget,
            preload_low_watermark=stage_preload_low_watermark,
            preload_high_watermark=stage_preload_high_watermark,
            preload_threads=stage_preload_threads,
            rank=0,
            world_size=1,
        )
        loader = DataLoader(
            dataset=eval_data,
            batch_size=None,
            drop_last=False,
            num_workers=0,
            pin_memory=eval_pin_memory,
        )
    else:
        active_file_batch_size = runtime_eval_cache.get('max_files_per_chunk', dataset_cfg['file_batch_size']) if use_runtime_cache else dataset_cfg['file_batch_size']
        active_prefetch_strategy = runtime_cache_cfg['mode'] if use_runtime_cache else prefetch_strategy
        active_prefetch_budget_bytes = int(runtime_eval_cache.get('data_budget_bytes', 0)) if use_runtime_cache else eval_prefetch_budget_bytes
        active_prefetch_target_chunk_bytes = int(runtime_eval_cache.get('target_chunk_bytes', 0)) if use_runtime_cache else prefetch_target_chunk_bytes
        active_prefetch_low_watermark = float(runtime_eval_cache.get('low_watermark', prefetch_low_watermark)) if use_runtime_cache else prefetch_low_watermark
        active_prefetch_high_watermark = float(runtime_eval_cache.get('high_watermark', prefetch_high_watermark)) if use_runtime_cache else prefetch_high_watermark
        active_prefetch_threads = int(runtime_eval_cache.get('max_inflight_chunk_builders', 1)) if use_runtime_cache else prefetch_threads
        active_decode_threads = int(runtime_eval_cache.get('decode_threads', 1)) if use_runtime_cache else 1
        active_prefetch_startup_file_batch_size = int(runtime_eval_cache.get('min_files_per_chunk', eval_prefetch_startup_file_batch_size)) if use_runtime_cache else eval_prefetch_startup_file_batch_size
        eval_data = ActionFileDatasetsIter(
            version=version,
            file_list=file_list,
            oracle=oracle,
            file_batch_size=active_file_batch_size,
            player_names=sorted(set(player_names)) or None,
            excludes=sorted(set(exclude_names)) or None,
            num_epochs=1,
            enable_augmentation=False,
            augmented_first=False,
            trust_seed=trust_seed,
            always_include_kan_select=always_include_kan_select,
            cycle=False,
            shuffle=False,
            allowed_player_ids_by_path=actor_filter_map,
            prefetch_chunks=0 if use_runtime_cache else eval_prefetch_chunks,
            prefetch_strategy=active_prefetch_strategy,
            prefetch_budget_bytes=active_prefetch_budget_bytes,
            prefetch_target_chunk_bytes=active_prefetch_target_chunk_bytes,
            prefetch_low_watermark=active_prefetch_low_watermark,
            prefetch_high_watermark=active_prefetch_high_watermark,
            prefetch_threads=active_prefetch_threads,
            decode_threads=active_decode_threads,
            batch_size=batch_size,
            prebatched=prebatched,
            prefetch_out_of_order=eval_prefetch_out_of_order,
            prefetch_startup_file_batch_size=active_prefetch_startup_file_batch_size,
            prefetch_inflight_budget_bytes=int(runtime_eval_cache.get('inflight_budget_bytes', 0)) if use_runtime_cache else 0,
            prefetch_ready_budget_bytes=int(runtime_eval_cache.get('ready_budget_bytes', 0)) if use_runtime_cache else 0,
            prefetch_max_inflight_chunks=int(runtime_eval_cache.get('max_inflight_chunk_builders', 1)) if use_runtime_cache else max(int(prefetch_threads or 1), 1),
            prefetch_min_file_batch_size=int(runtime_eval_cache.get('min_files_per_chunk', 1)) if use_runtime_cache else 1,
            prefetch_raw_lru_budget_bytes=int(runtime_eval_cache.get('raw_lru_budget_bytes', 0)) if use_runtime_cache else 0,
        )
        loader_kwargs = dict(
            dataset=eval_data,
            batch_size=None if prebatched else batch_size,
            drop_last=False,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
            worker_init_fn=worker_init_fn,
        )
        if eval_num_workers > 0 and multiprocessing_context:
            loader_kwargs['multiprocessing_context'] = multiprocessing_context
        loader = DataLoader(**loader_kwargs)

    stats = empty_metric_sums()
    batch_count = 0
    with torch.inference_mode():
        pb_total = expected_total_batch_count
        if pb_total is None and max_batches > 0:
            pb_total = max_batches
        pb = tqdm(total=pb_total, desc='EVAL')
        logging.info(
            'loader priming: building initial %s batches on %s '
            '(startup_queue_depth=%s full_queue_depth=%s strategy=%s startup_files=%s target_chunk_gib=%.2f '
            'runtime_cache_enabled=%s stage_enabled=%s)',
            args.split.upper(),
            device,
            eval_device_prefetch_startup_batches,
            eval_device_prefetch_batches,
            runtime_cache_cfg['mode'] if use_runtime_cache else prefetch_strategy,
            runtime_eval_cache.get('min_files_per_chunk', eval_prefetch_startup_file_batch_size) if use_runtime_cache else eval_prefetch_startup_file_batch_size,
            runtime_eval_cache.get('target_chunk_bytes', 0) / (1024 ** 3) if use_runtime_cache else prefetch_target_chunk_gib,
            use_runtime_cache,
            use_staged_cache,
        )
        eval_loader_prime_started_at = time.perf_counter()
        eval_iter = DeviceBatchPrefetcher(
            iter(loader),
            device=device,
            oracle=oracle,
            queue_depth=eval_device_prefetch_batches,
            startup_queue_depth=eval_device_prefetch_startup_batches,
        )
        eval_loader_snapshot = getattr(eval_data, 'loader_stats', None)
        eval_loader_snapshot = eval_loader_snapshot.snapshot() if eval_loader_snapshot is not None else {}
        logging.info(
            'loader priming: %s batches ready in %.2fs queued_gib=%.2f ready_gib=%.2f inflight_gib=%.2f '
            'ready_chunks=%s discovered_files=%s submitted_files=%s last_chunk_files=%s last_chunk_samples=%s',
            args.split.upper(),
            time.perf_counter() - eval_loader_prime_started_at,
            float(eval_loader_snapshot.get('queued_bytes', 0)) / (1024 ** 3),
            float(eval_loader_snapshot.get('ready_bytes', 0)) / (1024 ** 3),
            float(eval_loader_snapshot.get('inflight_bytes', 0)) / (1024 ** 3),
            int(eval_loader_snapshot.get('ready_chunks', 0)),
            int(eval_loader_snapshot.get('discovered_files', 0)),
            int(eval_loader_snapshot.get('submitted_files', 0)),
            int(eval_loader_snapshot.get('last_chunk_files', 0)),
            int(eval_loader_snapshot.get('last_chunk_samples', 0)),
        )
        for batch in eval_iter:
            if max_batches > 0 and batch_count >= max_batches:
                break
            if oracle:
                obs, invisible_obs, actions, masks = batch
            else:
                obs, actions, masks = batch
                invisible_obs = None
            with torch.autocast(**autocast_context_kwargs(device=device, enable_amp=enable_amp, amp_dtype=amp_dtype)):
                brain_out = mortal(obs, invisible_obs)
                phi = extract_policy_features(brain_out)
                raw_logits = dqn.action_logits(phi)
                masked_scores = masked_logits(raw_logits, masks)
                loss = torch.nn.functional.cross_entropy(masked_scores, actions)
            update_metric_sums(
                stats,
                loss=loss,
                masked_pred=masked_scores.argmax(dim=-1),
                raw_pred=raw_logits.argmax(dim=-1),
                masked_scores=masked_scores,
                actions=actions,
                masks=masks,
                top_k=control_cfg.get('top_k', 3),
            )
            batch_count += 1
            pb.update(1)
        pb.close()

    metrics = finalize_metric_sums(stats)
    metrics['expected_total_step_count'] = expected_total_step_count
    metrics['expected_total_batch_count'] = expected_total_batch_count
    summary = make_summary(
        checkpoint=checkpoint,
        split=args.split if not args.list else 'custom',
        file_count=len(file_list),
        batch_count=batch_count,
        max_batches=max_batches,
        metrics=metrics,
        state=state,
    )
    if (
        max_batches == 0
        and expected_total_step_count is not None
        and metrics['count'] != expected_total_step_count
    ):
        logging.warning(
            'eval step count mismatch for split=%s: processed=%s expected=%s',
            summary['split'],
            f"{metrics['count']:,}",
            f'{expected_total_step_count:,}',
        )

    if wandb_cfg.get('enabled', False) and wandb_cfg.get('log_eval_runs', True):
        eval_run = maybe_init_wandb_run(
            full_config=config,
            wandb_cfg={**wandb_cfg, 'job_type': wandb_cfg.get('eval_job_type', 'eval')},
            fallback_name=default_wandb_run_name(),
            job_type='eval',
            name_suffix=f"-{summary['split']}-eval",
        )
        eval_run.summary['checkpoint'] = checkpoint
        eval_run.summary['eval_split'] = summary['split']
        eval_run.summary['file_count'] = summary['file_count']
        eval_run.summary['batch_count'] = summary['batch_count']
        eval_run.summary['steps'] = summary['steps']
        eval_run.log(
            wandb_eval_payload(
                split=summary['split'],
                batch_count=batch_count,
                file_count=len(file_list),
                metrics=metrics,
            ),
            step=state.get('steps', 0),
        )
        eval_run.finish()

    logging.info(
        'eval split=%s batches=%s processed_steps=%s nll=%.6f acc=%.6f top%d=%.6f legal=%.6f',
        summary['split'],
        f'{batch_count:,}',
        f"{metrics['count']:,}",
        metrics['nll'],
        metrics['accuracy'],
        control_cfg.get('top_k', 3),
        metrics['topk_accuracy'],
        metrics['legal_rate'],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        output_path = path.abspath(args.output_json)
        path_parent = path.dirname(output_path)
        if path_parent:
            os.makedirs(path_parent, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write('\n')


if __name__ == '__main__':
    evaluate()
