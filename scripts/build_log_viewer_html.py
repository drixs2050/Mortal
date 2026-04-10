#!/usr/bin/env python3
"""Generate browser-viewable HTML files for a directory of mjai log archives.

For each unique game in the input directory, write a self-contained HTML
file inside log-viewer/<prefix>/ that uses the existing log-viewer assets
(log-viewer/files/css, log-viewer/files/js) via ../files/... relative
paths.

The per-run subdirectory layout keeps things manageable when you have
many tiers / many runs:

    log-viewer/
      files/                    <- shared assets (css, js, images)
      index.example.html        <- template
      tier0_n10/
        10000_8192_a.html       <- one file per unique game
        10001_8192_a.html
        ...
      tier1_n25/
        10000_8192_a.html
        ...

The <title> element inside each HTML file still contains the full prefix
so browser tabs remain self-identifying even though filenames are short.

By default, functionally-equivalent logs are deduplicated by content
hash. OneVsThree's challenger-rotation produces 4 mjai logs per wall
(one per seat the challenger occupies). When challenger and champion
are the same model (self-play), all 4 rotations play out identically
and hash to the same value, so only one is kept. When they differ
(any non-self-play tier), each rotation is a genuinely different game
and all 4 are kept.

The dedup hash strips:
  - 'meta' blocks (q_values, eval_time_ns -- vary with bf16 noise)
  - 'names' from start_game (vary with rotation labels)
Everything else (actions, seeds, tehais, scores) is hashed.

Usage:
  scripts/build_log_viewer_html.py <log_dir> --prefix tier1_n25
  scripts/build_log_viewer_html.py <log_dir> --prefix tier0_n10 --no-dedupe
  scripts/build_log_viewer_html.py <log_dir> --prefix foo --flat
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = ROOT / "log-viewer"
TEMPLATE_PATH = VIEWER_DIR / "index.example.html"

# Boundaries of the embedded allActions block in index.example.html.
# Line 9: `    allActions = ``    (opening backtick)
# Line 265: `    `.trim().split('\n').map(s => JSON.parse(s))`   (closing backtick)
OPEN_LINE_IDX = 9   # 0-indexed, line *after* the opening-backtick line
CLOSE_LINE_IDX = 265  # 0-indexed, the closing-backtick line itself


def parse_args():
    p = argparse.ArgumentParser(description="Generate viewer HTML for mjai logs")
    p.add_argument("log_dir", help="directory containing *.json.gz mjai logs")
    p.add_argument(
        "--prefix",
        required=True,
        help="subdirectory name under log-viewer/ and browser-tab title "
        "prefix. Example: 'tier1_n25'.",
    )
    p.add_argument(
        "--out-dir",
        default=str(VIEWER_DIR),
        help="root output directory (default: log-viewer/). Files land in "
        "<out-dir>/<prefix>/ unless --flat is given.",
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help="write files directly to <out-dir>/ with filenames prefixed "
        "(legacy layout). Default is per-prefix subdirectories.",
    )
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="generate one HTML per log file even if multiple logs are "
        "functionally the same game (default: dedupe by content hash)",
    )
    return p.parse_args()


def rewrite_asset_paths_for_subdir(html_text: str) -> str:
    """Rewrite files/... asset references to ../files/... for a viewer
    HTML written one directory below log-viewer/.
    """
    return (
        html_text
        .replace('href="files/', 'href="../files/')
        .replace('src="files/', 'src="../files/')
        .replace('resourceDir = "files"', 'resourceDir = "../files"')
    )


def normalize_for_dedupe(events_text: str) -> str:
    """Return a canonical form of an mjai log for content-hash dedup.

    Strips 'meta' blocks (q_values / eval_time_ns drift with bf16 noise)
    and 'names' from start_game (vary across challenger seat rotations).
    Everything else - actions, seeds, tehais, scores - is preserved.
    """
    out = []
    for line in events_text.strip().split("\n"):
        if not line:
            continue
        ev = json.loads(line)
        ev.pop("meta", None)
        if ev.get("type") == "start_game":
            ev.pop("names", None)
        out.append(json.dumps(ev, sort_keys=True, separators=(",", ":")))
    return "\n".join(out)


def content_hash(events_text: str) -> str:
    return hashlib.sha256(normalize_for_dedupe(events_text).encode("utf-8")).hexdigest()


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_root = Path(args.out_dir)
    if args.flat:
        out_dir = out_root
    else:
        out_dir = out_root / args.prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"template not found: {TEMPLATE_PATH}")

    template_lines = TEMPLATE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    prefix_text = "".join(template_lines[:OPEN_LINE_IDX])
    suffix_text = "".join(template_lines[CLOSE_LINE_IDX - 1:])

    # When writing to a subdirectory, the shared files/ assets live one
    # level up, so rewrite the relative asset paths in the template.
    if not args.flat:
        prefix_text = rewrite_asset_paths_for_subdir(prefix_text)
        suffix_text = rewrite_asset_paths_for_subdir(suffix_text)

    log_files = sorted(log_dir.glob("*.json.gz"))
    if not log_files:
        raise SystemExit(f"no .json.gz files found in {log_dir}")

    # Read every log into memory once.
    log_records = []
    for log_path in log_files:
        with gzip.open(log_path, "rt", encoding="utf-8") as f:
            events = f.read().rstrip("\n")
        log_records.append((log_path, events))

    # Dedupe by content hash (default) or pass through everything.
    if args.no_dedupe:
        unique_records = log_records
        skipped = []
    else:
        seen_hashes = {}
        unique_records = []
        skipped = []
        for log_path, events in log_records:
            h = content_hash(events)
            if h in seen_hashes:
                skipped.append((log_path, seen_hashes[h]))
            else:
                seen_hashes[h] = log_path
                unique_records.append((log_path, events))

    written = []
    for log_path, events in unique_records:
        stem = log_path.stem.replace('.json', '')
        if args.flat:
            out_name = f"{args.prefix}_{stem}.html"
        else:
            out_name = f"{stem}.html"
        out_path = out_dir / out_name
        title_line = f"  <title>{args.prefix} - {log_path.name}</title>\n"
        body = prefix_text + events + "\n    " + suffix_text
        body = body.replace(
            "  <title>Mahjong Archive Player</title>\n", title_line
        )
        out_path.write_text(body, encoding="utf-8")
        written.append((log_path.name, out_path, len(events.splitlines())))

    for name, path, n in written:
        print(f"{name}: {n} events -> {path}")
    if skipped:
        print()
        print(f"deduped {len(skipped)} functionally-equivalent log(s):")
        for dup_path, kept_path in skipped:
            print(f"  {dup_path.name} == {kept_path.name} (kept)")
    print()
    print(f"wrote {len(written)} unique viewer HTML file(s) "
          f"({len(log_records)} input logs, {len(skipped)} duplicates skipped)")


if __name__ == "__main__":
    main()
