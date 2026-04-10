#!/usr/bin/env python3
"""Convert a tenhou.net/6 JSON paifu file into NAGA "custom game record"
format.

NAGA's submission form at https://naga.dmv.nico/naga_report/order_form/
accepts a custom game record as a sequence of `https://tenhou.net/5/#json=...`
lines (one per kyoku) pasted into the middle tab. This format was reverse
engineered from
https://github.com/honvl/Majsoul-to-NAGA/blob/master/index.js, which is
the canonical Mahjong-Soul-to-NAGA converter and which produces the same
line layout from its source data.

Key facts about the NAGA format:

  - One line per kyoku (round). For an N-kyoku hanchan you produce N
    lines.
  - Each line is `https://tenhou.net/5/#json=<RAW_JSON>` -- the JSON is
    in the URL hash fragment and is NOT URL-encoded. Hash fragments are
    client-side only, so length and special characters are unconstrained.
  - The JSON structure is `{title, name, rule, log}` where `log` is an
    array containing exactly ONE kyoku (the rest of the original game's
    kyokus go on other lines).
  - The `rule` field uses `{"disp": "<DISPLAY_STRING>", "aka": 1}` --
    single `aka` key, NOT the per-tile `aka51`/`aka52`/`aka53` form that
    mjai.ekyu.moe needs. We rewrite the rule on output to use this form.

Usage:
    scripts/tenhou_json_to_naga.py <input.tenhou.json>
    scripts/tenhou_json_to_naga.py <input.tenhou.json> --output naga.txt
    scripts/tenhou_json_to_naga.py --input-dir <dir> --output-dir <dir>

Output: one line per kyoku, suitable for pasting into the NAGA custom
game analysis form. If both --input-dir and --output-dir are given, each
input file produces a sibling .naga.txt file with the lines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def normalize_rule_for_naga(rule: dict) -> dict:
    """Convert a tenhou rule dict to NAGA-friendly form.

    NAGA examples use a single `aka` field. We collapse any per-tile
    `aka51`/`aka52`/`aka53` flags down to one. Unknown extra keys are
    preserved as-is.
    """
    out = dict(rule)
    has_per_tile = any(k in out for k in ('aka51', 'aka52', 'aka53'))
    if has_per_tile:
        any_red = any(out.get(k) for k in ('aka51', 'aka52', 'aka53'))
        for k in ('aka51', 'aka52', 'aka53'):
            out.pop(k, None)
        out['aka'] = 1 if any_red else 0
    elif 'aka' not in out:
        out['aka'] = 1
    return out


def tenhou_json_to_naga_lines(data: dict) -> list[str]:
    """Convert one tenhou.net/6 paifu dict into NAGA-format URL lines.

    Returns a list with one URL per kyoku (i.e. one URL per entry in
    data['log']).
    """
    title = data.get('title', ['', ''])
    name = data.get('name', ['', '', '', ''])
    rule = normalize_rule_for_naga(data.get('rule', {}))
    rounds = data.get('log', [])

    lines: list[str] = []
    for kyoku in rounds:
        per_kyoku = {
            'title': title,
            'name': name,
            'rule': rule,
            'log': [kyoku],
        }
        # NAGA's example output is compact (no spaces between separators)
        # except for one cosmetic space after the rule object that
        # appears to be a JS artifact. We use the strict compact form
        # because that's what JSON.stringify(obj) produces in JS, which
        # is what the canonical Majsoul-to-NAGA converter does.
        as_json = json.dumps(per_kyoku, ensure_ascii=False, separators=(',', ':'))
        lines.append(f'https://tenhou.net/5/#json={as_json}')
    return lines


def convert_one(input_path: Path, output_path: Path | None) -> list[str]:
    with input_path.open(encoding='utf-8') as f:
        data = json.load(f)
    lines = tenhou_json_to_naga_lines(data)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    return lines


def parse_args():
    p = argparse.ArgumentParser(description='Convert tenhou JSON paifu to NAGA custom game record format')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('input', nargs='?', help='single tenhou.net/6 JSON file')
    g.add_argument('--input-dir', help='directory of *.tenhou.json files')
    p.add_argument('--output', help='output .txt path for single-file mode (default: stdout)')
    p.add_argument('--output-dir', help='output dir for batch mode (default: <input-dir>/../naga)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.input:
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output else None
        lines = convert_one(in_path, out_path)
        if out_path is None:
            for line in lines:
                print(line)
        else:
            print(f'wrote {len(lines)} kyoku lines to {out_path}', file=sys.stderr)
        return

    in_dir = Path(args.input_dir)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = in_dir.parent / 'naga'
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob('*.tenhou.json'))
    if not files:
        sys.exit(f'no *.tenhou.json files in {in_dir}')
    total_lines = 0
    for f in files:
        stem = f.name
        for suffix in ('.tenhou.json', '.json'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        out_path = out_dir / f'{stem}.naga.txt'
        lines = convert_one(f, out_path)
        total_lines += len(lines)
    print(f'wrote {len(files)} NAGA files ({total_lines} kyoku lines total) -> {out_dir}', file=sys.stderr)


if __name__ == '__main__':
    main()
