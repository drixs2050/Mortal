from __future__ import annotations

import hashlib
import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RAW_PACK_FORMAT = 'raw_pack_v1'


def normalize_raw_source_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


@dataclass(frozen=True)
class RawPackEntry:
    key: str
    source_path: str
    offset: int
    size: int
    sha256: str

    @property
    def end_offset(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class RawPackIndex:
    format: str
    entry_count: int
    entries: dict[str, RawPackEntry]
    pack_bytes: int = 0


class FileRawSource:
    def read(self, file_id: str | Path) -> bytes:
        return Path(file_id).read_bytes()

    def read_many(self, file_ids: Iterable[str | Path]) -> list[bytes]:
        return [self.read(file_id) for file_id in file_ids]

    def close(self) -> None:
        return None


class PackedRawSource:
    def __init__(self, pack_path: str | Path, index_path: str | Path):
        self.pack_path = Path(pack_path).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self.index = load_raw_pack_index(self.index_path)
        self._pack_file = self.pack_path.open('rb')
        self._mmap = mmap.mmap(self._pack_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._pack_size = self._mmap.size()
        for entry in self.index.entries.values():
            if entry.end_offset > self._pack_size:
                raise ValueError(
                    f'raw pack entry {entry.source_path} exceeds pack size: '
                    f'{entry.end_offset} > {self._pack_size}'
                )

    def read(self, file_id: str | Path) -> bytes:
        key = normalize_raw_source_key(file_id)
        try:
            entry = self.index.entries[key]
        except KeyError as exc:
            raise KeyError(f'raw pack index missing entry for {key}') from exc
        return bytes(self._mmap[entry.offset:entry.end_offset])

    def read_many(self, file_ids: Iterable[str | Path]) -> list[bytes]:
        return [self.read(file_id) for file_id in file_ids]

    def close(self) -> None:
        try:
            if getattr(self, '_mmap', None) is not None:
                self._mmap.close()
        finally:
            self._mmap = None
            if getattr(self, '_pack_file', None) is not None:
                self._pack_file.close()
                self._pack_file = None

    def __enter__(self) -> 'PackedRawSource':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_raw_pack_index(index_path: str | Path) -> RawPackIndex:
    path = Path(index_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    pack_format = str(payload.get('format') or '')
    if pack_format != RAW_PACK_FORMAT:
        raise ValueError(
            f'unsupported raw pack format in {path}: {pack_format!r}'
        )
    raw_entries = payload.get('entries')
    if not isinstance(raw_entries, list):
        raise ValueError(f'raw pack index {path} must contain a list of entries')
    entries = {}
    for item in raw_entries:
        entry = RawPackEntry(
            key=str(item['key']),
            source_path=str(item['source_path']),
            offset=int(item['offset']),
            size=int(item['size']),
            sha256=str(item['sha256']),
        )
        entries[entry.key] = entry
    return RawPackIndex(
        format=pack_format,
        entry_count=int(payload.get('entry_count', len(entries))),
        entries=entries,
        pack_bytes=int(payload.get('pack_bytes', 0) or 0),
    )


def build_raw_pack(
    file_list: Iterable[str | Path],
    *,
    pack_path: str | Path,
    index_path: str | Path,
) -> dict:
    resolved_files = [normalize_raw_source_key(path) for path in file_list]
    pack_output = Path(pack_path).expanduser().resolve()
    index_output = Path(index_path).expanduser().resolve()
    pack_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    total_bytes = 0
    with pack_output.open('wb') as pack_file:
        for source_path in resolved_files:
            payload = Path(source_path).read_bytes()
            offset = pack_file.tell()
            pack_file.write(payload)
            total_bytes += len(payload)
            entries.append(
                {
                    'key': source_path,
                    'source_path': source_path,
                    'offset': offset,
                    'size': len(payload),
                    'sha256': hashlib.sha256(payload).hexdigest(),
                }
            )

    metadata = {
        'format': RAW_PACK_FORMAT,
        'entry_count': len(entries),
        'pack_bytes': total_bytes,
        'entries': entries,
    }
    index_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return {
        'format': RAW_PACK_FORMAT,
        'entry_count': len(entries),
        'pack_bytes': total_bytes,
        'pack_path': str(pack_output),
        'index_path': str(index_output),
    }


def verify_raw_pack(
    *,
    pack_path: str | Path,
    index_path: str | Path,
    file_list: Iterable[str | Path] | None = None,
) -> dict:
    index = load_raw_pack_index(index_path)
    keys = (
        [normalize_raw_source_key(path) for path in file_list]
        if file_list is not None
        else sorted(index.entries.keys())
    )
    missing_entries = []
    missing_source_files = []
    mismatched_entries = []
    checked_count = 0

    with PackedRawSource(pack_path, index_path) as source:
        for key in keys:
            entry = index.entries.get(key)
            if entry is None:
                missing_entries.append(key)
                continue
            source_path = Path(entry.source_path)
            if not source_path.exists():
                missing_source_files.append(entry.source_path)
                continue
            expected = source_path.read_bytes()
            observed = source.read(entry.source_path)
            if expected != observed:
                mismatched_entries.append(
                    {
                        'source_path': entry.source_path,
                        'expected_sha256': hashlib.sha256(expected).hexdigest(),
                        'observed_sha256': hashlib.sha256(observed).hexdigest(),
                        'expected_size': len(expected),
                        'observed_size': len(observed),
                    }
                )
                continue
            checked_count += 1

    return {
        'ok': not missing_entries and not missing_source_files and not mismatched_entries,
        'checked_count': checked_count,
        'missing_entries': missing_entries,
        'missing_source_files': missing_source_files,
        'mismatched_entries': mismatched_entries,
    }
