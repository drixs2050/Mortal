import json
from os import path
from pathlib import Path
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]


def normalize_file_path(filename: str) -> str:
    candidate = Path(filename).expanduser()
    if candidate.is_absolute():
        return path.abspath(str(candidate))
    return path.abspath(str(ROOT / candidate))


def normalize_metadata_path(filename: str) -> str:
    return path.abspath(str(Path(filename).expanduser()))


def normalize_file_list(file_list: list[str], *, desc: str = 'PATHS') -> list[str]:
    show_progress = len(file_list) >= 10_000
    iterator = tqdm(
        file_list,
        total=len(file_list),
        desc=desc,
        unit='file',
        dynamic_ncols=True,
        ascii=True,
    ) if show_progress else file_list
    return [normalize_file_path(filename) for filename in iterator]


def save_path_cache(
    output_path: str,
    *,
    split_lists: dict[str, list[str]],
    source_files: dict[str, str] | None = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'bc_path_cache_v1',
        'split_lists': split_lists,
        'source_files': {
            split_name: normalize_metadata_path(filename)
            for split_name, filename in (source_files or {}).items()
        },
    }
    torch.save(payload, output_path)


def load_path_cache(
    cache_path: str,
    *,
    expected_splits: list[str] | None = None,
    expected_sources: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    payload = torch.load(cache_path, weights_only=False, map_location='cpu')
    if payload.get('format') != 'bc_path_cache_v1':
        raise ValueError(f'unsupported path cache format in {cache_path}')
    split_lists = payload.get('split_lists') or {}
    source_files = {
        split_name: normalize_metadata_path(filename)
        for split_name, filename in (payload.get('source_files') or {}).items()
    }
    for split_name, expected in (expected_sources or {}).items():
        normalized_expected = normalize_metadata_path(expected)
        if expected and source_files.get(split_name, normalized_expected) != normalized_expected:
            raise ValueError(
                f'path cache {cache_path} source mismatch for {split_name}: '
                f'expected {normalized_expected}, got {source_files.get(split_name)}'
            )
    required = expected_splits or list(split_lists.keys())
    missing = [split_name for split_name in required if split_name not in split_lists]
    if missing:
        raise ValueError(
            f'path cache {cache_path} is missing required splits: {", ".join(sorted(missing))}'
        )
    return {
        split_name: [normalize_file_path(filename) for filename in split_lists[split_name]]
        for split_name in required
    }


def allowed_player_ids_for_row(row: dict, min_actor_dan: int) -> tuple[int, ...]:
    player_dan = row.get('player_dan') or []
    return tuple(
        idx
        for idx, dan in enumerate(player_dan)
        if dan is not None and dan >= min_actor_dan
    )


def wanted_file_set(
    file_lists: list[list[str]],
    *,
    inputs_are_normalized: bool = False,
    desc: str = 'ACTOR-INDEX',
) -> tuple[set[str], int]:
    flat_files = [
        filename
        for file_list in file_lists
        for filename in file_list
    ]
    show_index_progress = len(flat_files) >= 10_000
    index_iter = tqdm(
        flat_files,
        total=len(flat_files),
        desc=desc,
        unit='file',
        dynamic_ncols=True,
        ascii=True,
    ) if show_index_progress else flat_files
    if inputs_are_normalized:
        wanted_files = {filename for filename in index_iter}
    else:
        wanted_files = {normalize_file_path(filename) for filename in index_iter}
    return wanted_files, len(flat_files)


def actor_filter_summary(
    *,
    source: str,
    min_actor_dan: int,
    indexed_file_count: int,
    requested_file_count: int,
    matched_row_count: int,
    eligible_file_count: int,
    scanned_row_count: int | None = None,
    manifest_path: str = '',
    index_path: str = '',
    stored_file_count: int | None = None,
) -> dict:
    summary = {
        'source': source,
        'min_actor_dan': min_actor_dan,
        'indexed_file_count': indexed_file_count,
        'requested_file_count': requested_file_count,
        'matched_row_count': matched_row_count,
        'eligible_file_count': eligible_file_count,
        'filtered_out_file_count': matched_row_count - eligible_file_count,
    }
    if scanned_row_count is not None:
        summary['scanned_row_count'] = scanned_row_count
    if manifest_path:
        summary['manifest_path'] = manifest_path
    if index_path:
        summary['index_path'] = index_path
    if stored_file_count is not None:
        summary['stored_file_count'] = stored_file_count
    return summary


def validate_actor_filter_map(
    actor_filter_map: dict[str, tuple[int, ...]],
    wanted_files: set[str],
    *,
    error_prefix: str,
) -> None:
    missing_files = sorted(wanted_files - actor_filter_map.keys())
    if missing_files:
        examples = ', '.join(missing_files[:5])
        raise ValueError(
            f'{error_prefix}; first examples: {examples}'
        )


def build_actor_filter_map(
    *,
    manifest_path: str,
    file_lists: list[list[str]],
    min_actor_dan: int,
    inputs_are_normalized: bool = False,
) -> tuple[dict[str, tuple[int, ...]], dict]:
    wanted_files, indexed_file_count = wanted_file_set(
        file_lists,
        inputs_are_normalized=inputs_are_normalized,
    )
    actor_filter_map: dict[str, tuple[int, ...]] = {}
    matched_rows = 0
    eligible_rows = 0
    scanned_rows = 0
    manifest_size = path.getsize(manifest_path)

    with open(manifest_path, 'rb') as f, tqdm(
        total=manifest_size,
        desc='ACTOR-FILTER',
        unit='B',
        unit_scale=True,
        dynamic_ncols=True,
        ascii=True,
    ) as pb:
        for raw_line in f:
            pb.update(len(raw_line))
            line = raw_line.strip()
            if not line:
                continue
            scanned_rows += 1
            row = json.loads(line)
            relative_path = row.get('relative_path')
            if not relative_path:
                continue
            normalized_path = normalize_file_path(relative_path)
            if normalized_path not in wanted_files:
                continue
            matched_rows += 1
            allowed_player_ids = allowed_player_ids_for_row(row, min_actor_dan)
            actor_filter_map[normalized_path] = allowed_player_ids
            if allowed_player_ids:
                eligible_rows += 1
            if scanned_rows % 1000 == 0:
                pb.set_postfix(
                    scanned=f'{scanned_rows:,}',
                    matched=f'{matched_rows:,}',
                    eligible=f'{eligible_rows:,}',
                    refresh=False,
                )

    validate_actor_filter_map(
        actor_filter_map,
        wanted_files,
        error_prefix='actor filter manifest is missing file metadata for split files',
    )

    summary = actor_filter_summary(
        source='manifest',
        manifest_path=manifest_path,
        min_actor_dan=min_actor_dan,
        indexed_file_count=indexed_file_count,
        requested_file_count=len(wanted_files),
        matched_row_count=matched_rows,
        eligible_file_count=eligible_rows,
        scanned_row_count=scanned_rows,
    )
    return actor_filter_map, summary


def save_actor_filter_index(
    output_path: str,
    *,
    actor_filter_map: dict[str, tuple[int, ...]],
    summary: dict,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'bc_actor_filter_v1',
        'summary': summary,
        'actor_filter_map': {
            filename: list(player_ids)
            for filename, player_ids in actor_filter_map.items()
        },
    }
    torch.save(payload, output_path)


def load_actor_filter_index(
    *,
    index_path: str,
    file_lists: list[list[str]],
    min_actor_dan: int,
    inputs_are_normalized: bool = False,
) -> tuple[dict[str, tuple[int, ...]], dict]:
    wanted_files, indexed_file_count = wanted_file_set(
        file_lists,
        inputs_are_normalized=inputs_are_normalized,
    )
    payload = torch.load(index_path, weights_only=False, map_location='cpu')
    if payload.get('format') != 'bc_actor_filter_v1':
        raise ValueError(f'unsupported actor filter index format in {index_path}')
    payload_summary = payload.get('summary') or {}
    payload_min_actor_dan = payload_summary.get('min_actor_dan')
    if payload_min_actor_dan is not None and payload_min_actor_dan != min_actor_dan:
        raise ValueError(
            f'actor filter index min_actor_dan={payload_min_actor_dan} does not match requested {min_actor_dan}'
        )

    stored_map = {
        normalize_file_path(filename): tuple(player_ids)
        for filename, player_ids in (payload.get('actor_filter_map') or {}).items()
    }
    actor_filter_map = {
        filename: stored_map[filename]
        for filename in wanted_files
        if filename in stored_map
    }
    validate_actor_filter_map(
        actor_filter_map,
        wanted_files,
        error_prefix='actor filter index is missing file metadata for split files',
    )
    eligible_rows = sum(bool(player_ids) for player_ids in actor_filter_map.values())
    summary = actor_filter_summary(
        source='index',
        index_path=index_path,
        manifest_path=payload_summary.get('manifest_path', ''),
        min_actor_dan=min_actor_dan,
        indexed_file_count=indexed_file_count,
        requested_file_count=len(wanted_files),
        matched_row_count=len(actor_filter_map),
        eligible_file_count=eligible_rows,
        stored_file_count=len(stored_map),
    )
    return actor_filter_map, summary


def resolve_actor_filter_map(
    *,
    file_lists: list[list[str]],
    min_actor_dan: int,
    actor_filter_manifest: str = '',
    actor_filter_index: str = '',
    inputs_are_normalized: bool = False,
) -> tuple[dict[str, tuple[int, ...]], dict]:
    if actor_filter_index:
        return load_actor_filter_index(
            index_path=actor_filter_index,
            file_lists=file_lists,
            min_actor_dan=min_actor_dan,
            inputs_are_normalized=inputs_are_normalized,
        )
    if not actor_filter_manifest:
        raise ValueError(
            'bc.dataset.actor_filter_index or bc.dataset.actor_filter_manifest is required '
            'when bc.dataset.min_actor_dan is set'
        )
    return build_actor_filter_map(
        manifest_path=actor_filter_manifest,
        file_lists=file_lists,
        min_actor_dan=min_actor_dan,
        inputs_are_normalized=inputs_are_normalized,
    )
