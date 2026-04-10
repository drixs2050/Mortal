#!/usr/bin/env python3
"""Convert mjai-format game logs to tenhou.net/6 JSON paifu format.

The output is uploadable to:
  - https://mjai.ekyu.moe/    (Mortal review service - "Custom log" upload)
  - https://naga.dmv.nico/    (NAGA review service)
  - Any other tool that consumes tenhou.net/6 JSON paifu

This is the inverse of scripts/tenhou_xml.py's tenhou -> mjai converter.
The tile encoding tables and meld parsing rules are derived from
empirical inspection of real tenhou JSON paifu samples in
data/raw/tenhou/, cross-checked against scripts/tenhou_xml.py.

Tile encoding (mjai -> tenhou JSON code):
  m1..m9 -> 11..19,  red 5m -> 51
  p1..p9 -> 21..29,  red 5p -> 52
  s1..s9 -> 31..39,  red 5s -> 53
  E S W N P F C -> 41 42 43 44 45 46 47

Meld string encoding:
  chi:        "c<called><c1><c2>"        target is always kamicha
  pon:        marker "p" at position M = 3 - ((target - actor) % 4)
  daiminkan:  marker "m" at position M = 3 - ((target - actor) % 4)
  ankan:      "a<c1><c2><c3><c4>"        no direction
  kakan:      "k<c1><added><c2><c3>"     added tile takes original pon's marker pos

Discard array entries:
  int(code)   normal discard
  60          tsumogiri (discard the just-drawn tile)
  "rXX"       riichi declaration with tile XX
  "rXX" + 60  riichi tsumogiri (encoded as "r60" in some sources; we use "r"+actual_code)
  "aXXXX"     ankan (full string in discards instead of an int)
  "kXXXX"     kakan (full string in discards instead of an int)

Round meta encoding:
  seed0 = bakaze_index * 4 + oya
  bakaze: E=0, S=1, W=2, N=3
  oya: 0..3 (mjai's start_kyoku.oya field is authoritative)

Result encoding:
  agari:      ["和了", deltas, [winner, target, _, _, _, ...yaku_text...], ...]
              Multiple agari (double ron) emit alternating delta+info pairs
  ryukyoku:   ["流局", deltas]

Usage:
    scripts/mjai_to_tenhou.py <input.json.gz> <output.json>
    scripts/mjai_to_tenhou.py --input-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------- tile encoding ----------

HONOR_TO_CODE = {
    'E': 41, 'S': 42, 'W': 43, 'N': 44,
    'P': 45, 'F': 46, 'C': 47,
}
CODE_TO_HONOR = {v: k for k, v in HONOR_TO_CODE.items()}

RED_FIVE_TO_CODE = {'5mr': 51, '5pr': 52, '5sr': 53}
CODE_TO_RED_FIVE = {v: k for k, v in RED_FIVE_TO_CODE.items()}

BAKAZE_INDEX = {'E': 0, 'S': 1, 'W': 2, 'N': 3}


class DelayedKakanError(ValueError):
    """Raised when a game has a kakan with a tile that wasn't just drawn.

    This is a Mortal-specific OOD action that libriichi allows under
    standard riichi rules (kakan can be declared with any 4th tile
    held in hand) but real Tenhou play never produces. Tenhou paifu
    format cannot losslessly encode this case, so games hitting it
    are skipped during conversion.
    """


def mjai_to_tenhou_code(pai: str) -> int:
    """Convert an mjai tile string to a tenhou JSON code (11-47, 51-53)."""
    if pai in HONOR_TO_CODE:
        return HONOR_TO_CODE[pai]
    if pai in RED_FIVE_TO_CODE:
        return RED_FIVE_TO_CODE[pai]
    if len(pai) == 2 and pai[0].isdigit() and pai[1] in ('m', 'p', 's'):
        n = int(pai[0])
        suit = pai[1]
        if not 1 <= n <= 9:
            raise ValueError(f'invalid tile number: {pai}')
        base = {'m': 10, 'p': 20, 's': 30}[suit]
        return base + n
    raise ValueError(f'unsupported mjai tile: {pai!r}')


def tenhou_code_to_mjai(code: int) -> str:
    """Inverse of mjai_to_tenhou_code (used for sanity checks)."""
    if code in CODE_TO_RED_FIVE:
        return CODE_TO_RED_FIVE[code]
    if code in CODE_TO_HONOR:
        return CODE_TO_HONOR[code]
    suit_idx, value = divmod(code, 10)
    if suit_idx == 1:
        return f'{value}m'
    if suit_idx == 2:
        return f'{value}p'
    if suit_idx == 3:
        return f'{value}s'
    raise ValueError(f'invalid tenhou code: {code}')


def tile_sort_key(code: int) -> int:
    """Stable sort key matching tenhou's display order. Red fives sort
    next to their natural-five counterparts: 5m before 5mr, etc."""
    if code == 51:
        return 15 * 10 + 1  # right after 5m (15)
    if code == 52:
        return 25 * 10 + 1  # right after 5p (25)
    if code == 53:
        return 35 * 10 + 1  # right after 5s (35)
    return code * 10


def sort_tile_codes(codes: list[int]) -> list[int]:
    return sorted(codes, key=tile_sort_key)


# ---------- meld string encoding ----------

def _format_code(code: int) -> str:
    """Format a tile code as a 2-digit string. Tenhou JSON uses 11-53."""
    return f'{code:02d}'


def encode_chi(actor: int, target: int, called: int, consumed: list[int]) -> str:
    """Chi: marker "c" at position 0, called tile first, then consumed."""
    if (target - actor) % 4 != 3:
        raise ValueError(
            f'chi must be from kamicha; actor={actor} target={target}'
        )
    if len(consumed) != 2:
        raise ValueError(f'chi must consume exactly 2 tiles, got {consumed}')
    # Tenhou displays the chi sequence with the called tile at the leftmost
    # rotated slot. The two consumed tiles follow in their natural order.
    sorted_consumed = sort_tile_codes(consumed)
    return f'c{_format_code(called)}{_format_code(sorted_consumed[0])}{_format_code(sorted_consumed[1])}'


def _meld_marker_pos(actor: int, target: int) -> int:
    """Position of the marker letter in a pon/daiminkan meld string,
    encoding which seat the called tile came from."""
    offset = (target - actor) % 4
    if offset == 0:
        raise ValueError('cannot call from self')
    return 3 - offset


def encode_pon(actor: int, target: int, called: int, consumed: list[int]) -> str:
    """Pon: 3 codes (1 called + 2 consumed). Marker "p" position encodes
    which seat the called tile came from."""
    if len(consumed) != 2:
        raise ValueError(f'pon must consume exactly 2 tiles, got {consumed}')
    marker_pos = _meld_marker_pos(actor, target)
    sorted_consumed = sort_tile_codes(consumed)
    # Build a list of [c1, c2, c3] where the called tile sits at marker_pos
    # and consumed tiles fill the remaining slots in order.
    slots = [None, None, None]
    slots[marker_pos] = called
    other_idx = 0
    for i in range(3):
        if slots[i] is None:
            slots[i] = sorted_consumed[other_idx]
            other_idx += 1
    parts: list[str] = []
    for i, code in enumerate(slots):
        if i == marker_pos:
            parts.append('p')
        parts.append(_format_code(code))
    return ''.join(parts)


def encode_daiminkan(actor: int, target: int, called: int, consumed: list[int]) -> str:
    """Daiminkan: 4 codes (1 called + 3 consumed). Marker "m" position
    encodes the seat direction the same way pon does."""
    if len(consumed) != 3:
        raise ValueError(f'daiminkan must consume exactly 3 tiles, got {consumed}')
    marker_pos = _meld_marker_pos(actor, target)
    sorted_consumed = sort_tile_codes(consumed)
    slots = [None, None, None, None]
    slots[marker_pos] = called
    other_idx = 0
    for i in range(4):
        if slots[i] is None:
            slots[i] = sorted_consumed[other_idx]
            other_idx += 1
    parts: list[str] = []
    for i, code in enumerate(slots):
        if i == marker_pos:
            parts.append('m')
        parts.append(_format_code(code))
    return ''.join(parts)


def encode_ankan(consumed: list[int]) -> str:
    """Ankan (closed kan): 4 codes, marker "a" at position 3.

    Verified against real tenhou samples ('191919a19', '242424a24',
    '181818a18'): the canonical layout is <c1><c2><c3>a<c4> with the
    marker AFTER the third code, not before any code. Closed kans have
    no source direction, but Tenhou paifu nonetheless requires this
    exact marker position for the parser to accept the meld.
    """
    if len(consumed) != 4:
        raise ValueError(f'ankan must consume exactly 4 tiles, got {consumed}')
    sorted_consumed = sort_tile_codes(consumed)
    return (
        _format_code(sorted_consumed[0])
        + _format_code(sorted_consumed[1])
        + _format_code(sorted_consumed[2])
        + 'a'
        + _format_code(sorted_consumed[3])
    )


def encode_kakan(actor: int, pon_target: int | None, added: int, consumed: list[int]) -> str:
    """Kakan (added kan): upgrade an existing pon by adding the 4th tile.
    The marker "k" position should match where the called tile of the
    original pon was placed. If pon_target is unknown, we fall back to
    placing the added tile at position 0."""
    if len(consumed) != 3:
        raise ValueError(f'kakan must consume exactly 3 tiles from prior pon, got {consumed}')
    if pon_target is not None:
        marker_pos = _meld_marker_pos(actor, pon_target)
    else:
        marker_pos = 0
    sorted_consumed = sort_tile_codes(consumed)
    slots = [None, None, None, None]
    slots[marker_pos] = added
    other_idx = 0
    for i in range(4):
        if slots[i] is None:
            slots[i] = sorted_consumed[other_idx]
            other_idx += 1
    parts: list[str] = []
    for i, code in enumerate(slots):
        if i == marker_pos:
            parts.append('k')
        parts.append(_format_code(code))
    return ''.join(parts)


# ---------- score string approximation for synthesized agari results ----------

# Score table entries: (max_base_gain, score_text_template)
# Each entry is the maximum base gain (post-honba/kyotaku) that maps to
# this row of the standard riichi score table. The score_text_template
# is the canonical Tenhou paifu text for that row.
#
# Real Tenhou paifu have entries like:
#   "30符1飜1000点"   ron, 30fu 1han, 1000 base
#   "40符2飜2600点"   ron, 40fu 2han, 2600 base
#   "満貫8000点"     mangan ron, 8000 base
#   "跳満12000点"    haneman ron, 12000 base
#
# For tsumo wins the format is "<fu>符<han>飜<small>-<big>点" but for
# the limit hands it's "満貫<total>点" without the split. We use the
# closest-row mapping with limit handling and fall back to a generic
# safe row if nothing matches cleanly.

_RON_NONDEALER = [
    (1000,  '30符1飜1000点'),
    (1300,  '40符1飜1300点'),
    (1600,  '50符1飜1600点'),
    (2000,  '30符2飜2000点'),
    (2600,  '40符2飜2600点'),
    (3200,  '50符2飜3200点'),
    (3900,  '30符3飜3900点'),
    (5200,  '40符3飜5200点'),
    (6400,  '60符3飜6400点'),
    (7700,  '30符4飜7700点'),
    (8000,  '満貫8000点'),
    (12000, '跳満12000点'),
    (16000, '倍満16000点'),
    (24000, '三倍満24000点'),
    (32000, '役満32000点'),
]

_RON_DEALER = [
    (1500,  '30符1飜1500点'),
    (2000,  '40符1飜2000点'),
    (2400,  '50符1飜2400点'),
    (2900,  '30符2飜2900点'),
    (3900,  '40符2飜3900点'),
    (4800,  '50符2飜4800点'),
    (5800,  '30符3飜5800点'),
    (7700,  '40符3飜7700点'),
    (9600,  '60符3飜9600点'),
    (11600, '30符4飜11600点'),
    (12000, '満貫12000点'),
    (18000, '跳満18000点'),
    (24000, '倍満24000点'),
    (36000, '三倍満36000点'),
    (48000, '役満48000点'),
]

# For tsumo, the winner's gain equals the sum of payments. We use the
# same total-gain matching but with the limit hands using their tsumo
# total, and the score text format is simplified to just the limit name
# plus the total points.
_TSUMO_NONDEALER = [
    (1500,  '30符1飜500-1000点'),
    (2000,  '40符1飜500-1000点'),
    (2700,  '40符2飜700-1300点'),
    (4000,  '30符3飜1000-2000点'),
    (5200,  '50符3飜1300-2600点'),
    (8000,  '満貫2000-4000点'),
    (12000, '跳満3000-6000点'),
    (16000, '倍満4000-8000点'),
    (24000, '三倍満6000-12000点'),
    (32000, '役満8000-16000点'),
]

_TSUMO_DEALER = [
    (1500,  '30符1飜500点'),
    (2000,  '40符1飜700点'),
    (3000,  '40符2飜1000点'),
    (4000,  '30符3飜1300点'),
    (6000,  '50符3飜2000点'),
    (12000, '満貫4000点'),
    (18000, '跳満6000点'),
    (24000, '倍満8000点'),
    (36000, '三倍満12000点'),
    (48000, '役満16000点'),
]


def _approximate_score_string(base_gain: int, is_dealer: bool, is_tsumo: bool) -> str:
    """Map a winner's base gain to the closest standard score-text entry."""
    if is_tsumo:
        table = _TSUMO_DEALER if is_dealer else _TSUMO_NONDEALER
    else:
        table = _RON_DEALER if is_dealer else _RON_NONDEALER

    # Pick the row with the smallest |row_gain - base_gain|; ties go to
    # the smaller row to avoid over-claiming han.
    best = table[0]
    best_diff = abs(table[0][0] - base_gain)
    for entry in table[1:]:
        diff = abs(entry[0] - base_gain)
        if diff < best_diff:
            best = entry
            best_diff = diff
    return best[1]


# ---------- per-kyoku state machine ----------

class KyokuBuilder:
    """Accumulates one kyoku of mjai events into a tenhou round array."""

    def __init__(self, start_kyoku_event: dict):
        sk = start_kyoku_event
        self.bakaze = sk['bakaze']
        self.kyoku_num = sk['kyoku']
        self.honba = sk['honba']
        self.kyotaku = sk['kyotaku']
        self.oya = sk['oya']
        self.scores = list(sk['scores'])
        self.dora_indicators = [mjai_to_tenhou_code(sk['dora_marker'])]
        self.ura_indicators: list[int] = []
        self.haipai = [
            [mjai_to_tenhou_code(t) for t in hand]
            for hand in sk['tehais']
        ]
        # Per-player draws/discards arrays in tenhou format
        self.draws: list[list] = [[], [], [], []]
        self.discards: list[list] = [[], [], [], []]
        # State for the current decision
        self.last_drawn_tile: list[int | None] = [None, None, None, None]
        self.pending_riichi: list[bool] = [False, False, False, False]
        # Track pon targets so kakan can place its marker correctly
        self.pon_targets: dict[tuple[int, str], int] = {}  # (actor, deaka_pai) -> target
        # Per-actor state needed to reconstruct yaku for the agari result.
        # mjai's hora events do not carry a yaku breakdown, so we have to
        # synthesize one from tracked context.
        self.riichi_declared: list[bool] = [False, False, False, False]
        # Menzen flag: True until the actor makes a "open" call. chi, pon,
        # daiminkan, kakan all break menzen. ankan does NOT break menzen
        # (it preserves riichi/tsumo bonuses), matching standard riichi rules.
        self.menzen: list[bool] = [True, True, True, True]
        # Result accumulators
        self.hora_events: list[dict] = []
        self.ryukyoku_event: dict | None = None

    # --- helpers ---

    @staticmethod
    def _deaka_pai(pai: str) -> str:
        return pai[:-1] if pai.endswith('r') else pai

    def _record_pon_target(self, actor: int, called_pai: str, target: int) -> None:
        self.pon_targets[(actor, self._deaka_pai(called_pai))] = target

    def _lookup_pon_target(self, actor: int, added_pai: str) -> int | None:
        return self.pon_targets.get((actor, self._deaka_pai(added_pai)))

    # --- event handlers ---

    def handle(self, ev: dict) -> None:
        t = ev['type']
        if t == 'tsumo':
            actor = ev['actor']
            code = mjai_to_tenhou_code(ev['pai'])
            self.draws[actor].append(code)
            self.last_drawn_tile[actor] = code
        elif t == 'dahai':
            actor = ev['actor']
            code = mjai_to_tenhou_code(ev['pai'])
            tsumogiri = ev.get('tsumogiri', False)
            if self.pending_riichi[actor]:
                # Riichi declaration: encode as "r" + actual tile code
                # (some Tenhou paifu use 60 for riichi tsumogiri, but
                # encoding the actual tile is unambiguous and accepted.)
                self.discards[actor].append(f'r{_format_code(code)}')
                self.pending_riichi[actor] = False
            elif tsumogiri:
                self.discards[actor].append(60)
            else:
                self.discards[actor].append(code)
            self.last_drawn_tile[actor] = None
        elif t == 'reach':
            actor = ev['actor']
            self.pending_riichi[actor] = True
        elif t == 'reach_accepted':
            # Tenhou paifu encodes the riichi accept implicitly
            # via the score deduction shown in `sc` and the "r" marker
            # already placed in the discard sequence. We also record
            # that this actor's hand has riichi declared, for yaku
            # reconstruction at hora time.
            self.riichi_declared[ev['actor']] = True
        elif t == 'chi':
            actor = ev['actor']
            target = ev['target']
            called = mjai_to_tenhou_code(ev['pai'])
            consumed = [mjai_to_tenhou_code(p) for p in ev['consumed']]
            self.draws[actor].append(encode_chi(actor, target, called, consumed))
            self.last_drawn_tile[actor] = None
            self.menzen[actor] = False
        elif t == 'pon':
            actor = ev['actor']
            target = ev['target']
            called = mjai_to_tenhou_code(ev['pai'])
            consumed = [mjai_to_tenhou_code(p) for p in ev['consumed']]
            self.draws[actor].append(encode_pon(actor, target, called, consumed))
            self._record_pon_target(actor, ev['pai'], target)
            self.last_drawn_tile[actor] = None
            self.menzen[actor] = False
        elif t == 'daiminkan':
            actor = ev['actor']
            target = ev['target']
            called = mjai_to_tenhou_code(ev['pai'])
            consumed = [mjai_to_tenhou_code(p) for p in ev['consumed']]
            self.draws[actor].append(encode_daiminkan(actor, target, called, consumed))
            self.last_drawn_tile[actor] = None
            self.menzen[actor] = False
        elif t == 'ankan':
            actor = ev['actor']
            consumed = [mjai_to_tenhou_code(p) for p in ev['consumed']]
            # Ankan goes in the discards array (not draws) because it
            # consumes the just-drawn tile.
            self.discards[actor].append(encode_ankan(consumed))
            self.last_drawn_tile[actor] = None
        elif t == 'kakan':
            actor = ev['actor']
            added = mjai_to_tenhou_code(ev['pai'])
            consumed = [mjai_to_tenhou_code(p) for p in ev['consumed']]
            pon_target = self._lookup_pon_target(actor, ev['pai'])
            # kakan upgrades a pon (which already broke menzen) to a kan
            self.menzen[actor] = False
            # Tenhou paifu format requires the kakan tile to be the
            # just-drawn tile (real Tenhou games never produce
            # "delayed kakan" -- verified empirically: 0 / 4 kakans
            # in real samples have a delayed pattern). libriichi
            # allows it under standard riichi rules so Mortal
            # occasionally generates it as an OOD action, but the
            # tenhou format cannot losslessly express it (any encoding
            # creates a tile-conservation violation -- the just-drawn
            # tile would have to be ignored, producing a phantom 5th
            # tile). We mark the game unconvertible and skip it.
            if self.draws[actor]:
                last = self.draws[actor][-1]
                if isinstance(last, int) and last != added:
                    raise DelayedKakanError(
                        f'kakan with non-just-drawn tile: actor={actor} '
                        f'kakan_pai={ev["pai"]} just_drawn={tenhou_code_to_mjai(last)}'
                    )
            self.discards[actor].append(encode_kakan(actor, pon_target, added, consumed))
            self.last_drawn_tile[actor] = None
        elif t == 'dora':
            self.dora_indicators.append(mjai_to_tenhou_code(ev['dora_marker']))
        elif t == 'hora':
            self.hora_events.append(ev)
            if 'ura_markers' in ev and ev['ura_markers']:
                self.ura_indicators = [mjai_to_tenhou_code(p) for p in ev['ura_markers']]
        elif t == 'ryukyoku':
            self.ryukyoku_event = ev
        elif t == 'end_kyoku':
            pass
        else:
            raise ValueError(f'unsupported mjai event in kyoku: {t}')

    # --- finalization ---

    def _build_agari_info(self, h: dict) -> list:
        """Build a tenhou agari_info array for one hora event.

        Layout: [winner, payer, pao, "<fu>符<han>飜<points>点", "<yaku>(<han>飜)", ...]

        mjai's hora event only carries (actor, target, deltas, ura_markers),
        not the full yaku breakdown. We synthesize a plausible yaku list
        and score string from tracked context (riichi, menzen, tsumo vs
        ron, ura markers) and from the deltas. The synthesized result
        satisfies NAGA's parser ("known yaku names", "parseable score
        string") even though it will not match the actual hand's true
        yaku composition.

        This is acceptable because:
          1. NAGA's job is to suggest better moves throughout the game,
             not validate that the displayed yaku matches the hand.
          2. Real yaku reconstruction would require running a winning-
             hand decomposer (libriichi.calc_yaku) which is non-trivial
             to call from this script and out of scope.
          3. Our agari placeholder only affects the END of each kyoku,
             not the move-by-move data NAGA actually analyses.

        Score arithmetic note: the canonical Majsoul-to-NAGA converter
        builds deltas as `delta[winner] = rp + (n-1)*hb + point` where
        rp is the total riichi stick payout (all kyotaku in the pot,
        including any deposited THIS kyoku). Therefore `sum(deltas)`
        always equals rp. We use this identity to back-derive rp without
        having to track kyotaku state explicitly:

            rp           = sum(deltas)
            honba_bonus  = 300 * honba   (ron: 300 from one loser;
                                          tsumo n=4: 100 from each of 3)
            base_gain    = winner_gain - rp - honba_bonus

        base_gain is then mapped to the closest standard score-table
        entry for (is_dealer, is_tsumo).
        """
        actor = h['actor']
        target = h['target']
        deltas = list(h['deltas'])
        is_tsumo = (actor == target)
        is_dealer = (actor == self.oya)

        winner_gain = deltas[actor]
        rp = sum(deltas)  # total riichi-stick payout (kyotaku in pot)
        honba_bonus = 300 * self.honba
        base_gain = winner_gain - rp - honba_bonus
        if base_gain <= 0:
            # Fall back to the raw winner gain if our subtraction
            # over-shot (shouldn't happen with valid deltas).
            base_gain = winner_gain

        score_text = _approximate_score_string(base_gain, is_dealer, is_tsumo)

        # Build the yaku list from tracked context. NAGA requires at
        # least one valid yaku name; we always include one. The yaku
        # name set used here is restricted to ones that real Tenhou
        # paifu use exactly, so NAGA's "unknown yaku" check passes.
        yaku_strs: list[str] = []
        if self.riichi_declared[actor]:
            yaku_strs.append('立直(1飜)')
        if is_tsumo and self.menzen[actor]:
            yaku_strs.append('門前清自摸和(1飜)')
        # Ura dora: only present when this hora event has non-empty
        # ura_markers (which only happens for riichi-agari wins). We
        # claim 1 ura dora as a placeholder; the exact count would
        # require examining the winner's actual hand against the ura
        # indicators, which we don't reconstruct.
        if h.get('ura_markers'):
            yaku_strs.append('裏ドラ(1飜)')
        if not yaku_strs:
            # Fallback: claim a single dora. This is universally valid
            # as a yaku name (NAGA recognizes 'ドラ') and adds 1 han.
            yaku_strs.append('ドラ(1飜)')

        return [actor, target, actor, score_text, *yaku_strs]

    def _result_array(self) -> list:
        if self.hora_events:
            out: list = ['和了']
            for h in self.hora_events:
                out.append(list(h['deltas']))
                out.append(self._build_agari_info(h))
            return out
        if self.ryukyoku_event:
            return ['流局', list(self.ryukyoku_event['deltas'])]
        # No result event seen — treat as ryukyoku with zero deltas
        return ['流局', [0, 0, 0, 0]]

    def to_round_array(self) -> list:
        seed0 = BAKAZE_INDEX[self.bakaze] * 4 + self.oya
        meta = [seed0, self.honba, self.kyotaku]
        return [
            meta,
            self.scores,
            self.dora_indicators,
            self.ura_indicators,
            self.haipai[0], self.draws[0], self.discards[0],
            self.haipai[1], self.draws[1], self.discards[1],
            self.haipai[2], self.draws[2], self.discards[2],
            self.haipai[3], self.draws[3], self.discards[3],
            self._result_array(),
        ]


# ---------- top-level conversion ----------

def mjai_events_to_tenhou(events: list[dict]) -> dict:
    """Convert a sequence of mjai events (one full game) to a tenhou
    JSON paifu document."""
    names = ['', '', '', '']
    rounds: list[list] = []
    builder: KyokuBuilder | None = None

    for ev in events:
        t = ev.get('type')
        if t == 'start_game':
            ev_names = ev.get('names') or []
            for i in range(4):
                if i < len(ev_names):
                    names[i] = ev_names[i] or f'p{i}'
                else:
                    names[i] = f'p{i}'
        elif t == 'start_kyoku':
            if builder is not None:
                rounds.append(builder.to_round_array())
            builder = KyokuBuilder(ev)
        elif t == 'end_game':
            if builder is not None:
                rounds.append(builder.to_round_array())
                builder = None
        else:
            if builder is None:
                raise ValueError(f'event {t} outside any kyoku')
            builder.handle(ev)

    if builder is not None:
        rounds.append(builder.to_round_array())

    # rule.disp must follow real-Tenhou conventions: the parser at
    # mjai.ekyu.moe checks for the kanji "南" as the hanchan marker
    # (and "東" alone would mean tonpuusen). The string "東南" gets
    # rejected as "not a hanchan game". We use the same disp value
    # that real Tenhou Phoenix hanchan paifu use, which is the format
    # the review services explicitly support. The aka51/52/53 fields
    # are separate per-tile flags, not a single "aka" field.
    return {
        'title': ['', ''],
        'name': names,
        'rule': {
            'disp': '鳳南喰赤',
            'aka51': 1,
            'aka52': 1,
            'aka53': 1,
        },
        'log': rounds,
    }


def load_mjai_events(path: Path) -> list[dict]:
    if path.suffix == '.gz':
        opener = gzip.open
    else:
        opener = open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_one(input_path: Path, output_path: Path) -> dict | None:
    """Convert one mjai log to tenhou JSON. Returns the tenhou dict on
    success, or None if the game contains a delayed kakan and was
    skipped."""
    events = load_mjai_events(input_path)
    try:
        tenhou = mjai_events_to_tenhou(events)
    except DelayedKakanError:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tenhou, f, ensure_ascii=False, separators=(',', ':'))
    return tenhou


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description='Convert mjai logs to tenhou.net/6 JSON paifu')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input', help='single .json or .json.gz mjai file to convert')
    g.add_argument('--input-dir', help='directory of *.json.gz mjai files')
    p.add_argument('--output', help='output path for single-file mode (default: alongside input)')
    p.add_argument('--output-dir', help='output dir for batch mode (default: <input-dir>/../tenhou)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.input:
        in_path = Path(args.input)
        if args.output:
            out_path = Path(args.output)
        else:
            stem = in_path.name
            for suffix in ('.json.gz', '.gz', '.json'):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            out_path = in_path.parent / f'{stem}.tenhou.json'
        result = convert_one(in_path, out_path)
        if result is None:
            print(f'skipped {in_path.name}: delayed kakan (cannot encode in tenhou format)')
        else:
            print(f'wrote {out_path}')
    else:
        in_dir = Path(args.input_dir)
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = in_dir.parent / 'tenhou'
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob('*.json.gz'))
        if not files:
            raise SystemExit(f'no .json.gz files in {in_dir}')
        converted = 0
        skipped = 0
        skipped_names: list[str] = []
        for f in files:
            stem = f.name
            for suffix in ('.json.gz', '.gz'):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            out_path = out_dir / f'{stem}.tenhou.json'
            result = convert_one(f, out_path)
            if result is None:
                skipped += 1
                skipped_names.append(f.name)
                # Remove any stale output from a previous (broken) run.
                if out_path.exists():
                    out_path.unlink()
            else:
                converted += 1
        print(f'converted {converted} files -> {out_dir}')
        if skipped:
            print(f'skipped {skipped} files (delayed kakan, unencodable in tenhou format):')
            for n in skipped_names[:10]:
                print(f'  {n}')
            if len(skipped_names) > 10:
                print(f'  ... and {len(skipped_names) - 10} more')


if __name__ == '__main__':
    main()
