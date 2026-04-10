from __future__ import annotations

import sys
from pathlib import Path

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


def _default_report_every(total: int) -> int:
    if total <= 0:
        return 1
    return max(1, total // 25)


def count_lines(path: str | Path) -> int:
    with Path(path).open(encoding='utf-8') as f:
        return sum(1 for _ in f)


class ProgressReporter:
    def __init__(
        self,
        *,
        total: int,
        desc: str,
        unit: str,
        report_every: int | None = None,
        stream=None,
    ) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.stream = stream or sys.stderr
        self.report_every = report_every or _default_report_every(total)
        self.count = 0
        self.bar = None

        if _tqdm is not None and getattr(self.stream, 'isatty', lambda: False)():
            self.bar = _tqdm(
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                ascii=True,
                file=self.stream,
            )
        else:
            self._write(f'{self.desc}: starting {self.total} {self.unit}')

    def _write(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def update(self, n: int = 1, *, status: str = '') -> None:
        self.count += n
        if self.bar is not None:
            self.bar.update(n)
            if status:
                self.bar.set_postfix_str(status, refresh=False)
            return

        if (
            self.count == 1
            or self.count == self.total
            or self.count % self.report_every == 0
        ):
            percent = (self.count / self.total * 100.0) if self.total else 100.0
            suffix = f' | {status}' if status else ''
            self._write(
                f'{self.desc}: {self.count}/{self.total} {self.unit} ({percent:5.1f}%)'
                f'{suffix}'
            )

    def close(self, *, status: str = '') -> None:
        if self.bar is not None:
            if status:
                self.bar.set_postfix_str(status, refresh=False)
            self.bar.close()
            return

        suffix = f' | {status}' if status else ''
        self._write(
            f'{self.desc}: finished {self.count}/{self.total} {self.unit}{suffix}'
        )
