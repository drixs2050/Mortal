from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

SOURCE_GAME_ID_RE = re.compile(
    r'^(?P<ts>\d{10})gm-(?P<rule>[0-9a-f]{4})-(?P<lobby>\d{4,5})-(?P<game>x?[0-9a-f]{8,16})$',
)

DRAW_TAGS = {'T': 0, 'U': 1, 'V': 2, 'W': 3}
DISCARD_TAGS = {'D': 0, 'E': 1, 'F': 2, 'G': 3}
RED_FIVE_IDS = {
    16: '5mr',
    52: '5pr',
    88: '5sr',
}
HONOR_TILES = ('E', 'S', 'W', 'N', 'P', 'F', 'C')

# The order follows Tenhou's official 段級位制 ladder in the manual.
# It is cross-checked against the official replay JSON for the saved sample
# where raw dan 0 renders as 新人 and raw dan 18 renders as 九段.
DAN_LABELS = (
    '新人',
    '9級',
    '8級',
    '7級',
    '6級',
    '5級',
    '4級',
    '3級',
    '2級',
    '1級',
    '初段',
    '二段',
    '三段',
    '四段',
    '五段',
    '六段',
    '七段',
    '八段',
    '九段',
    '十段',
    '天鳳位',
)


def parse_csv(raw, cast=int):
    if raw in (None, ''):
        return []
    return [cast(part) for part in raw.split(',')]


def decode_name(raw):
    if raw is None:
        return ''
    return unquote(raw)


def infer_dan_label(dan_id):
    if 0 <= dan_id < len(DAN_LABELS):
        return DAN_LABELS[dan_id]
    return None


def tile136_to_mjai(tile_id):
    tile_id = int(tile_id)
    if tile_id in RED_FIVE_IDS:
        return RED_FIVE_IDS[tile_id]

    kind = tile_id // 4
    if kind < 9:
        return f'{kind + 1}m'
    if kind < 18:
        return f'{kind - 8}p'
    if kind < 27:
        return f'{kind - 17}s'
    if kind < 34:
        return HONOR_TILES[kind - 27]
    raise ValueError(f'invalid Tenhou tile id: {tile_id}')


def tile136_to_tenhou_json_code(tile_id):
    tile_id = int(tile_id)
    if tile_id == 16:
        return 51
    if tile_id == 52:
        return 52
    if tile_id == 88:
        return 53

    kind = tile_id // 4
    if kind < 9:
        return 11 + kind
    if kind < 18:
        return 21 + (kind - 9)
    if kind < 27:
        return 31 + (kind - 18)
    if kind < 34:
        return 41 + (kind - 27)
    raise ValueError(f'invalid Tenhou tile id: {tile_id}')


def initial_hand_order_key(tile_id):
    code = tile136_to_tenhou_json_code(tile_id)
    if code == 51:
        return 151
    if code == 52:
        return 251
    if code == 53:
        return 351
    return code * 10


def meld_code_sort_key(tile_id):
    code = tile136_to_tenhou_json_code(tile_id)
    if code == 51:
        return 151
    if code == 52:
        return 251
    if code == 53:
        return 351
    return code


def sort_meld_codes(codes):
    return sorted(codes, key=meld_code_sort_key)


def source_game_id_parts(source_game_id):
    match = SOURCE_GAME_ID_RE.match(source_game_id)
    if not match:
        return None
    ts = datetime.strptime(match.group('ts'), '%Y%m%d%H')
    return {
        'source_game_id': source_game_id,
        'game_date': ts.strftime('%Y-%m-%dT%H:00:00'),
        'year': ts.year,
        'month': ts.month,
        'rule_code': match.group('rule'),
        'lobby_code': match.group('lobby'),
        'game_code': match.group('game'),
    }


def classify_lobby(lobby_id):
    lobby_id = int(lobby_id)
    if lobby_id > 10000:
        return {
            'lobby': lobby_id,
            'lobby_kind': 'custom',
            'lobby_display': f'C{lobby_id % 10000:04d}',
            'ranking_lobby': False,
        }
    return {
        'lobby': lobby_id,
        'lobby_kind': 'numbered',
        'lobby_display': f'L{lobby_id:04d}',
        'ranking_lobby': lobby_id == 0,
    }


def infer_rule_display_from_go_type(go_type):
    go_type = int(go_type)
    room_labels = ('般', '上', '特', '鳳')
    room_index = ((1 if go_type & 0x80 else 0) << 1) | (1 if go_type & 0x20 else 0)
    room = room_labels[room_index]
    hanchan = bool(go_type & 0x08)
    no_red = bool(go_type & 0x02)
    no_kuitan = bool(go_type & 0x04)
    soku = bool(go_type & 0x40)
    sanma = bool(go_type & 0x10)

    return {
        'room': room,
        'ruleset': (
            f"{room}"
            f"{'南' if hanchan else '東'}"
            f"{'' if no_kuitan else '喰'}"
            f"{'' if no_red else '赤'}"
            f"{'速' if soku else ''}"
        ),
        'table_size': 3 if sanma else 4,
        'is_sanma': sanma,
        'is_hanchan': hanchan,
        'is_soku': soku,
        'aka_enabled': not no_red,
        'kuitan_enabled': not no_kuitan,
    }


def round_from_seed(seed0):
    winds = ('E', 'S', 'W')
    bakaze_idx, kyoku_idx = divmod(seed0, 4)
    bakaze = winds[bakaze_idx] if bakaze_idx < len(winds) else f'X{bakaze_idx}'
    return bakaze, kyoku_idx + 1


def parse_scores(raw_scores):
    return [score * 100 for score in parse_csv(raw_scores)]


def parse_sc(raw_sc):
    values = parse_csv(raw_sc)
    scores_before = [value * 100 for value in values[0::2]]
    deltas = [value * 100 for value in values[1::2]]
    scores_after = [score + delta for score, delta in zip(scores_before, deltas)]
    return {
        'scores_before': scores_before,
        'deltas': deltas,
        'scores_after': scores_after,
    }


def parse_players(un_elem, oracle=None):
    names = [decode_name(un_elem.attrib.get(f'n{i}')) for i in range(4)]
    dan_ids = parse_csv(un_elem.attrib.get('dan'))
    rates = parse_csv(un_elem.attrib.get('rate'), cast=float)
    sexes = un_elem.attrib.get('sx', '').split(',') if un_elem.attrib.get('sx') else []
    inferred_dan = [infer_dan_label(value) for value in dan_ids]
    oracle_dan = oracle.get('dan') if oracle else None

    return {
        'names': names,
        'dan_ids': dan_ids,
        'dan_labels_inferred': inferred_dan,
        'dan_labels_oracle': oracle_dan,
        'dan_labels_match_oracle': oracle_dan == inferred_dan if oracle_dan else None,
        'rates': rates,
        'sexes': sexes,
    }


def parse_init(elem, round_index):
    seed = parse_csv(elem.attrib['seed'])
    bakaze, kyoku = round_from_seed(seed[0])
    tehais = [
        [
            tile136_to_mjai(tile_id)
            for tile_id in sorted(
                parse_csv(elem.attrib[f'hai{i}']),
                key=initial_hand_order_key,
            )
        ]
        for i in range(4)
    ]
    return {
        'round_index': round_index,
        'seed': seed,
        'bakaze': bakaze,
        'kyoku': kyoku,
        'honba': seed[1],
        'kyotaku': seed[2],
        'dice': seed[3:5],
        'dora_indicator': tile136_to_mjai(seed[5]),
        'oya': int(elem.attrib['oya']),
        'start_scores': parse_scores(elem.attrib['ten']),
        'start_hands': tehais,
        'events': [],
    }


def parse_round_event(elem):
    tag = elem.tag
    if tag == 'DORA':
        return {
            'kind': 'dora',
            'tile': tile136_to_mjai(elem.attrib['hai']),
        }
    if tag == 'N':
        return {
            'kind': 'call',
            'actor': int(elem.attrib['who']),
            'meld_code': int(elem.attrib['m']),
        }
    if tag == 'REACH':
        event = {
            'kind': 'reach',
            'actor': int(elem.attrib['who']),
            'step': int(elem.attrib['step']),
        }
        if 'ten' in elem.attrib:
            event['scores_after_step'] = parse_scores(elem.attrib['ten'])
        return event
    if tag == 'AGARI':
        event = {
            'kind': 'agari',
            'actor': int(elem.attrib['who']),
            'target': int(elem.attrib['fromWho']),
            'result': parse_sc(elem.attrib['sc']),
        }
        if 'machi' in elem.attrib:
            event['machi'] = tile136_to_mjai(elem.attrib['machi'])
        if 'doraHaiUra' in elem.attrib:
            event['ura_markers'] = [
                tile136_to_mjai(tile_id)
                for tile_id in parse_csv(elem.attrib['doraHaiUra'])
            ]
        return event
    if tag == 'RYUUKYOKU':
        event = {
            'kind': 'ryuukyoku',
            'result': parse_sc(elem.attrib['sc']),
        }
        if 'type' in elem.attrib:
            event['draw_type'] = elem.attrib['type']
        return event
    if tag and tag[0] in DRAW_TAGS and tag[1:].isdigit():
        return {
            'kind': 'draw',
            'actor': DRAW_TAGS[tag[0]],
            'tile': tile136_to_mjai(tag[1:]),
        }
    if tag and tag[0] in DISCARD_TAGS and tag[1:].isdigit():
        return {
            'kind': 'discard',
            'actor': DISCARD_TAGS[tag[0]],
            'tile': tile136_to_mjai(tag[1:]),
        }
    return None


def summarize_round(round_data):
    counts = Counter(event['kind'] for event in round_data['events'])
    terminal_events = [
        event for event in round_data['events']
        if event['kind'] in ('agari', 'ryuukyoku')
    ]
    summary = {
        'round_index': round_data['round_index'],
        'bakaze': round_data['bakaze'],
        'kyoku': round_data['kyoku'],
        'honba': round_data['honba'],
        'kyotaku': round_data['kyotaku'],
        'oya': round_data['oya'],
        'dora_indicator': round_data['dora_indicator'],
        'start_scores': round_data['start_scores'],
        'event_count': len(round_data['events']),
        'event_counts': dict(sorted(counts.items())),
        'contains_calls': counts['call'] > 0,
        'contains_reach': counts['reach'] > 0,
        'terminal_events': terminal_events,
    }
    return summary


def parse_official_json_oracle(filename):
    with open(filename, encoding='utf-8') as f:
        obj = json.load(f)

    rule = obj.get('rule') or {}
    rule_display = rule.get('disp')
    room = rule_display[0] if rule_display else None
    return {
        'title': obj.get('title'),
        'rule_display': rule_display,
        'room': room,
        'lobby': obj.get('lobby'),
        'dan': obj.get('dan'),
        'rate': obj.get('rate'),
        'sx': obj.get('sx'),
        'aka53': rule.get('aka53'),
        'aka52': rule.get('aka52'),
        'aka51': rule.get('aka51'),
    }


def parse_tenhou_xml(xml_filename, official_json_filename=None, include_round_events=False):
    xml_filename = Path(xml_filename)
    source_game_id = xml_filename.stem.split('.')[0]
    source_info = source_game_id_parts(source_game_id)

    root = ET.parse(xml_filename).getroot()
    go_elem = root.find('GO')
    un_elem = root.find('UN')
    if go_elem is None or un_elem is None:
        raise ValueError('invalid Tenhou XML: missing GO or UN tag')

    oracle = parse_official_json_oracle(official_json_filename) if official_json_filename else None
    lobby = classify_lobby(go_elem.attrib.get('lobby', 0))
    inferred_rule = infer_rule_display_from_go_type(go_elem.attrib['type'])
    players = parse_players(un_elem, oracle)

    rounds = []
    current_round = None
    for elem in root:
        if elem.tag == 'INIT':
            current_round = parse_init(elem, len(rounds))
            rounds.append(current_round)
            continue
        if current_round is None:
            continue
        event = parse_round_event(elem)
        if event is not None:
            current_round['events'].append(event)

    round_summaries = [summarize_round(round_data) for round_data in rounds]
    event_counts = Counter()
    for round_data in round_summaries:
        event_counts.update(round_data['event_counts'])

    parsed = {
        'source': 'tenhou',
        'source_game_id': source_game_id,
        'raw_xml_version': root.attrib.get('ver'),
        'go_type': int(go_elem.attrib['type']),
        'go_type_bits': [
            bit for bit in range(32)
            if int(go_elem.attrib['type']) & (1 << bit)
        ],
        'inferred_rule_display': inferred_rule['ruleset'],
        'inferred_room_code': inferred_rule['room'],
        'inferred_table_size': inferred_rule['table_size'],
        'inferred_is_sanma': inferred_rule['is_sanma'],
        'inferred_is_hanchan': inferred_rule['is_hanchan'],
        'inferred_is_soku': inferred_rule['is_soku'],
        'inferred_aka_enabled': inferred_rule['aka_enabled'],
        'inferred_kuitan_enabled': inferred_rule['kuitan_enabled'],
        **lobby,
        'players': players,
        'summary': {
            'round_count': len(rounds),
            'event_count': sum(len(round_data['events']) for round_data in rounds),
            'event_counts': dict(sorted(event_counts.items())),
        },
        'rounds': rounds if include_round_events else round_summaries,
    }
    if source_info is not None:
        parsed.update(source_info)
    if oracle is not None:
        parsed.update({
            'official_rule_display': oracle['rule_display'],
            'official_room_code': oracle['room'],
            'official_title': oracle['title'],
            'official_aka53': oracle['aka53'],
            'official_aka52': oracle['aka52'],
            'official_aka51': oracle['aka51'],
        })
    return parsed


def build_normalized_manifest_row(
    parsed,
    *,
    raw_snapshot_id,
    relative_path=None,
    dataset_id='pending',
    converter_version='tenhou-xml-readonly-v0',
    validation_status='raw_inspected',
):
    if relative_path is None:
        relative_path = (
            f"data/normalized/v1/source=tenhou/year={parsed['year']:04d}/"
            f"month={parsed['month']:02d}/{parsed['source_game_id']}.json.gz"
        )

    row = {
        'dataset_id': dataset_id,
        'relative_path': relative_path,
        'source': 'tenhou',
        'source_game_id': parsed['source_game_id'],
        'raw_snapshot_id': raw_snapshot_id,
        'game_date': parsed.get('game_date'),
        'year': parsed.get('year'),
        'month': parsed.get('month'),
        'ruleset': parsed.get('official_rule_display') or parsed.get('inferred_rule_display'),
        'room': parsed.get('official_room_code') or parsed.get('inferred_room_code'),
        'lobby': str(parsed['lobby']),
        'lobby_display': parsed['lobby_display'],
        'lobby_kind': parsed['lobby_kind'],
        'ranking_lobby': parsed['ranking_lobby'],
        'go_type': parsed['go_type'],
        'player_names_present': any(parsed['players']['names']),
        'player_ids_hashed': False,
        'player_dan': parsed['players']['dan_ids'],
        'player_dan_label': parsed['players']['dan_labels_oracle'] or parsed['players']['dan_labels_inferred'],
        'player_rate': parsed['players']['rates'],
        'player_sex': parsed['players']['sexes'],
        'converter_version': converter_version,
        'validation_status': validation_status,
        'table_size': parsed.get('inferred_table_size', 4),
        'byte_size': None,
        'file_sha256': None,
        'event_count': parsed['summary']['event_count'],
        'kyoku_count': parsed['summary']['round_count'],
        'duplicate_group': f"tenhou:{parsed['source_game_id']}",
        'split': None,
    }
    if parsed.get('official_title'):
        row['title'] = parsed['official_title']
    return row


JSON_RED_FIVES = {
    51: '5mr',
    52: '5pr',
    53: '5sr',
}


def tenhou_json_tile_to_mjai(tile_id):
    tile_id = int(tile_id)
    if tile_id in JSON_RED_FIVES:
        return JSON_RED_FIVES[tile_id]

    suit = tile_id // 10
    value = tile_id % 10
    if suit == 1:
        return f'{value}m'
    if suit == 2:
        return f'{value}p'
    if suit == 3:
        return f'{value}s'
    if suit == 4:
        return HONOR_TILES[value - 1]
    raise ValueError(f'invalid mjlog2json tile id: {tile_id}')


def parse_official_mjlog2json(filename):
    with open(filename, encoding='utf-8') as f:
        return json.load(f)


def parse_meld_string(raw):
    marker = None
    marker_pos = None
    codes = []
    idx = 0
    while idx < len(raw):
        ch = raw[idx]
        if ch.isdigit():
            if idx + 1 >= len(raw) or not raw[idx + 1].isdigit():
                raise ValueError(f'invalid meld string: {raw}')
            codes.append(int(raw[idx:idx + 2]))
            idx += 2
            continue
        if marker is not None:
            raise ValueError(f'unsupported meld string with multiple markers: {raw}')
        marker = ch
        marker_pos = len(codes)
        idx += 1

    if marker is None or marker_pos is None:
        raise ValueError(f'missing meld marker: {raw}')
    return marker, marker_pos, codes


def deaka_mjai(pai):
    return pai[:-1] if pai.endswith('r') else pai


def same_json_tile_kind(lhs, rhs):
    return deaka_mjai(tenhou_json_tile_to_mjai(lhs)) == deaka_mjai(tenhou_json_tile_to_mjai(rhs))


def choose_matching_code_index(codes, preferred_code):
    for idx, code in enumerate(codes):
        if code == preferred_code:
            return idx
    for idx, code in enumerate(codes):
        if same_json_tile_kind(code, preferred_code):
            return idx
    raise ValueError(f'could not match tile {preferred_code} inside {codes}')


def parse_chi_string(raw):
    if not raw.startswith('c') or len(raw) != 7:
        raise ValueError(f'unsupported chi string: {raw}')
    called = int(raw[1:3])
    consumed = [int(raw[3:5]), int(raw[5:7])]
    return called, consumed


def parse_pon_string(raw):
    marker, marker_pos, codes = parse_meld_string(raw)
    if marker != 'p' or len(codes) != 3:
        raise ValueError(f'unsupported pon string: {raw}')
    if marker_pos == 0:
        called = codes[0]
        consumed = codes[1:]
    elif marker_pos == 1:
        called = codes[1]
        consumed = [codes[0], codes[2]]
    elif marker_pos == 2:
        called = codes[2]
        consumed = codes[:2]
    else:
        raise ValueError(f'unsupported pon string: {raw}')
    return called, consumed


def parse_daiminkan_string(raw, discarded_code):
    marker, _, codes = parse_meld_string(raw)
    if marker != 'm' or len(codes) != 4:
        raise ValueError(f'unsupported daiminkan string: {raw}')
    called_idx = choose_matching_code_index(codes, discarded_code)
    called = codes[called_idx]
    consumed = codes[:called_idx] + codes[called_idx + 1:]
    return called, consumed


def parse_ankan_string(raw):
    marker, _, codes = parse_meld_string(raw)
    if marker != 'a' or len(codes) != 4:
        raise ValueError(f'unsupported ankan string: {raw}')
    return codes


def parse_kakan_string(raw, added_code):
    marker, _, codes = parse_meld_string(raw)
    if marker != 'k' or len(codes) != 4:
        raise ValueError(f'unsupported kakan string: {raw}')
    added_idx = choose_matching_code_index(codes, added_code)
    added = codes[added_idx]
    consumed = codes[:added_idx] + codes[added_idx + 1:]
    return added, consumed


def classify_draw_item(item):
    if item in (None, 0):
        return None
    if isinstance(item, int):
        return 'draw'
    if isinstance(item, str):
        if item.startswith('c'):
            return 'chi'
        if 'p' in item:
            return 'pon'
        if 'm' in item:
            return 'daiminkan'
    raise ValueError(f'unsupported draw item: {item!r}')


def classify_discard_item(item):
    if item in (None, 0):
        return None
    if isinstance(item, int):
        return 'discard'
    if isinstance(item, str):
        if item.startswith('r'):
            return 'reach'
        if 'a' in item:
            return 'ankan'
        if 'k' in item:
            return 'kakan'
        if 'f' in item:
            return 'fspecial'
    raise ValueError(f'unsupported discard item: {item!r}')


def claim_matches(raw, discarded_code, offset):
    if not isinstance(raw, str) or ('p' not in raw and 'm' not in raw):
        return False
    if offset == 1:
        match = re.match(r'^[pm](\d\d)', raw)
        return match is not None and same_json_tile_kind(int(match.group(1)), discarded_code)
    if offset == 2:
        match = re.match(r'^\d\d[pm](\d\d)', raw)
        return match is not None and same_json_tile_kind(int(match.group(1)), discarded_code)
    if offset == 3:
        match = re.search(r'[pm](\d\d)$', raw)
        return match is not None and same_json_tile_kind(int(match.group(1)), discarded_code)
    raise ValueError(f'invalid offset: {offset}')


def advance_next_player(current_actor, draws, indices):
    for offset in range(1, 5):
        actor = (current_actor + offset) & 3
        if indices[actor] < len(draws[actor]):
            return actor
    return None


def choose_next_actor_after_discard(current_actor, discarded_code, draws, indices):
    for offset in (3, 2, 1):
        actor = (current_actor + offset) & 3
        if indices[actor] >= len(draws[actor]):
            continue
        raw = draws[actor][indices[actor]]
        if claim_matches(raw, discarded_code, offset):
            return actor
    return advance_next_player(current_actor, draws, indices)


def emit_dora_events(events, dora_indicators, dora_index, pending_count):
    for _ in range(pending_count):
        if dora_index >= len(dora_indicators):
            raise ValueError('missing visible dora indicators for kan follow-up')
        events.append({
            'type': 'dora',
            'dora_marker': tenhou_json_tile_to_mjai(dora_indicators[dora_index]),
        })
        dora_index += 1
    return dora_index


def result_is_immediate_hora_on_actor(result, actor):
    if not result or result[0] != '和了':
        return False
    if len(result) < 3 or len(result) % 2 == 0:
        return False
    found_winner = False
    for idx in range(1, len(result), 2):
        agari_info = result[idx + 1]
        winner, target, _ = agari_info[:3]
        if winner == actor:
            return False
        if target != actor:
            return False
        found_winner = True
    return found_winner


def has_remaining_round_actions(draws, discards, indices):
    for actor in range(4):
        idx = indices[actor]
        if idx < len(draws[actor]):
            return True
        if idx < len(discards[actor]):
            return True
    return False


def round_arrays_to_mjai_events(round_data):
    meta = round_data[0]
    scores = round_data[1]
    dora_indicators = round_data[2]
    ura_indicators = round_data[3]
    if not dora_indicators:
        raise ValueError('round is missing its initial visible dora indicator')

    seed0, honba, kyotaku = meta
    bakaze, kyoku = round_from_seed(seed0)
    oya = seed0 & 3
    hands = [round_data[4], round_data[7], round_data[10], round_data[13]]
    draws = [round_data[5], round_data[8], round_data[11], round_data[14]]
    discards = [round_data[6], round_data[9], round_data[12], round_data[15]]
    result = round_data[16]

    events = [{
        'type': 'start_kyoku',
        'bakaze': bakaze,
        'dora_marker': tenhou_json_tile_to_mjai(dora_indicators[0]),
        'kyoku': kyoku,
        'honba': honba,
        'kyotaku': kyotaku,
        'oya': oya,
        'scores': scores,
        'tehais': [
            [tenhou_json_tile_to_mjai(tile_id) for tile_id in hand]
            for hand in hands
        ],
    }]

    indices = [0, 0, 0, 0]
    actor = oya
    last_discard_actor = None
    last_discard_code = None
    dora_index = 1
    pending_kan_dora = 0

    while any(indices[i] < len(draws[i]) for i in range(4)):
        idx = indices[actor]
        draw_item = draws[actor][idx] if idx < len(draws[actor]) else None
        discard_item = discards[actor][idx] if idx < len(discards[actor]) else None
        draw_kind = classify_draw_item(draw_item)
        draw_code = None

        if draw_kind is None:
            next_actor = advance_next_player(actor, draws, indices)
            if next_actor is None:
                break
            actor = next_actor
            continue

        if draw_kind == 'draw':
            last_discard_actor = None
            last_discard_code = None
            draw_code = int(draw_item)
            events.append({
                'type': 'tsumo',
                'actor': actor,
                'pai': tenhou_json_tile_to_mjai(draw_code),
            })
            if pending_kan_dora:
                dora_index = emit_dora_events(events, dora_indicators, dora_index, pending_kan_dora)
                pending_kan_dora = 0
        elif draw_kind == 'chi':
            if last_discard_actor is None or last_discard_code is None:
                raise ValueError('chi encountered without a preceding discard to claim')
            called_code, consumed_codes = parse_chi_string(draw_item)
            if called_code != last_discard_code:
                raise ValueError(f'chi string {draw_item} does not match discarded tile {last_discard_code}')
            events.append({
                'type': 'chi',
                'actor': actor,
                'target': last_discard_actor,
                'pai': tenhou_json_tile_to_mjai(called_code),
                'consumed': [tenhou_json_tile_to_mjai(code) for code in consumed_codes],
            })
            last_discard_actor = None
            last_discard_code = None
        elif draw_kind == 'pon':
            if last_discard_actor is None or last_discard_code is None:
                raise ValueError('pon encountered without a preceding discard to claim')
            called_code, consumed_codes = parse_pon_string(draw_item)
            if called_code != last_discard_code:
                raise ValueError(f'pon string {draw_item} does not match discarded tile {last_discard_code}')
            events.append({
                'type': 'pon',
                'actor': actor,
                'target': last_discard_actor,
                'pai': tenhou_json_tile_to_mjai(called_code),
                'consumed': [tenhou_json_tile_to_mjai(code) for code in consumed_codes],
            })
            last_discard_actor = None
            last_discard_code = None
        elif draw_kind == 'daiminkan':
            if last_discard_actor is None or last_discard_code is None:
                raise ValueError('daiminkan encountered without a preceding discard to claim')
            called_code, consumed_codes = parse_daiminkan_string(draw_item, last_discard_code)
            events.append({
                'type': 'daiminkan',
                'actor': actor,
                'target': last_discard_actor,
                'pai': tenhou_json_tile_to_mjai(called_code),
                'consumed': [tenhou_json_tile_to_mjai(code) for code in consumed_codes],
            })
            last_discard_actor = None
            last_discard_code = None
            pending_kan_dora += 1
        else:
            raise ValueError(f'unsupported draw kind in minimal converter: {draw_kind}')

        indices[actor] += 1
        if draw_kind == 'daiminkan':
            continue

        discard_kind = classify_discard_item(discard_item)
        if discard_kind is None:
            next_actor = advance_next_player(actor, draws, indices)
            if next_actor is None:
                break
            actor = next_actor
            continue

        if discard_kind == 'ankan':
            if draw_code is None:
                raise ValueError('ankan without a regular draw is not supported')
            consumed_codes = parse_ankan_string(discard_item)
            events.append({
                'type': 'ankan',
                'actor': actor,
                'consumed': [tenhou_json_tile_to_mjai(code) for code in consumed_codes],
            })
            last_discard_actor = None
            last_discard_code = None
            pending_kan_dora += 1
            if (
                result_is_immediate_hora_on_actor(result, actor)
                and not has_remaining_round_actions(draws, discards, indices)
            ):
                break
            continue

        if discard_kind == 'kakan':
            if draw_code is None:
                raise ValueError('kakan without a regular draw is not supported')
            pai_code, consumed_codes = parse_kakan_string(discard_item, draw_code)
            events.append({
                'type': 'kakan',
                'actor': actor,
                'pai': tenhou_json_tile_to_mjai(pai_code),
                'consumed': [tenhou_json_tile_to_mjai(code) for code in consumed_codes],
            })
            last_discard_actor = None
            last_discard_code = None
            pending_kan_dora += 1
            if (
                result_is_immediate_hora_on_actor(result, actor)
                and not has_remaining_round_actions(draws, discards, indices)
            ):
                break
            continue

        if discard_kind == 'fspecial':
            raise ValueError(
                'minimal converter does not support f-encoded special actions yet '
                '(likely sanma North extraction, outside the current 4-player scope)'
            )

        if discard_kind == 'reach':
            if draw_code is None:
                raise ValueError('reach without a regular draw is not supported')
            actual_code = int(discard_item[1:])
            if actual_code == 60:
                actual_code = draw_code
                tsumogiri = True
            else:
                tsumogiri = actual_code == draw_code
            events.append({
                'type': 'reach',
                'actor': actor,
            })
            events.append({
                'type': 'dahai',
                'actor': actor,
                'pai': tenhou_json_tile_to_mjai(actual_code),
                'tsumogiri': tsumogiri,
            })
            events.append({
                'type': 'reach_accepted',
                'actor': actor,
            })
        else:
            if discard_item == 60:
                if draw_code is None:
                    raise ValueError('tsumogiri marker without a draw tile')
                actual_code = draw_code
                tsumogiri = True
            else:
                actual_code = int(discard_item)
                tsumogiri = draw_code is not None and actual_code == draw_code
            events.append({
                'type': 'dahai',
                'actor': actor,
                'pai': tenhou_json_tile_to_mjai(actual_code),
                'tsumogiri': tsumogiri,
            })

        last_discard_actor = actor
        last_discard_code = actual_code
        next_actor = choose_next_actor_after_discard(actor, actual_code, draws, indices)
        if next_actor is None:
            break
        actor = next_actor

    if result[0] == '和了':
        if len(result) < 3 or len(result) % 2 == 0:
            raise ValueError('invalid agari result layout in official replay JSON')
        for idx in range(1, len(result), 2):
            deltas = result[idx]
            agari_info = result[idx + 1]
            actor_id, target_id, _ = agari_info[:3]
            event = {
                'type': 'hora',
                'actor': actor_id,
                'target': target_id,
                'deltas': deltas,
            }
            # Ura-dora only matters for riichi winners, so keep it tied to the
            # yaku list when multiple hora outcomes share the same round.
            if ura_indicators and any('立直' in yaku for yaku in agari_info[4:]):
                event['ura_markers'] = [
                    tenhou_json_tile_to_mjai(tile_id)
                    for tile_id in ura_indicators
                ]
            events.append(event)
    elif result[0] in {'流局', '全員聴牌', '全員不聴'}:
        events.append({
            'type': 'ryukyoku',
            'deltas': result[1] if len(result) > 1 else [0, 0, 0, 0],
        })
    elif result[0] == '流し満貫':
        events.append({
            'type': 'ryukyoku',
            'deltas': result[1],
        })
    elif result[0] in {'三家和了', '四風連打', '四家立直', '九種九牌', '四槓散了'}:
        events.append({
            'type': 'ryukyoku',
            'deltas': [0, 0, 0, 0],
        })
    else:
        raise ValueError(f'unsupported round result: {result[0]!r}')

    events.append({'type': 'end_kyoku'})
    return events


ABORTIVE_XML_RYUUKYOKU_TYPES = {
    'yao9',
    'reach4',
    'ron3',
    'kan4',
    'kaze4',
}


def decode_xml_meld(meld_code, actor):
    from_who = meld_code & 0x3
    target = (actor + from_who) & 3

    if meld_code & 0x4:
        base_called = meld_code >> 10
        called = base_called % 3
        base = base_called // 3
        base = (base // 7) * 9 + (base % 7)
        offsets = [
            (meld_code >> 3) & 0x3,
            (meld_code >> 5) & 0x3,
            (meld_code >> 7) & 0x3,
        ]
        codes = [(base + i) * 4 + offsets[i] for i in range(3)]
        return {
            'type': 'chi',
            'actor': actor,
            'target': target,
            'pai': tile136_to_mjai(codes[called]),
            'consumed': [
                tile136_to_mjai(code)
                for idx, code in enumerate(codes)
                if idx != called
            ],
        }

    if meld_code & 0x18:
        base_called = meld_code >> 9
        called = base_called % 3
        base = base_called // 3
        tile4 = (meld_code >> 5) & 0x3
        all_codes = [base * 4 + i for i in range(4)]
        used_codes = [code for idx, code in enumerate(all_codes) if idx != tile4]
        called_code = used_codes[called]
        consumed_codes = [
            code for idx, code in enumerate(used_codes)
            if idx != called
        ]
        if meld_code & 0x8:
            return {
                'type': 'pon',
                'actor': actor,
                'target': target,
                'pai': tile136_to_mjai(called_code),
                'consumed': [
                    tile136_to_mjai(code)
                    for code in sort_meld_codes(consumed_codes)
                ],
            }
        if meld_code & 0x10:
            return {
                'type': 'kakan',
                'actor': actor,
                'pai': tile136_to_mjai(all_codes[tile4]),
                'consumed': [
                    tile136_to_mjai(code)
                    for code in sort_meld_codes(
                        [code for idx, code in enumerate(all_codes) if idx != tile4]
                    )
                ],
            }
        raise ValueError(f'unsupported XML pon/kakan meld code: {meld_code}')

    if meld_code & 0x20:
        raise ValueError(
            'xml converter does not support 0x20 special meld codes yet '
            '(likely sanma North extraction, outside the current 4-player scope)'
        )

    base_called = meld_code >> 8
    called = base_called % 4
    base = base_called // 4
    codes = [base * 4 + i for i in range(4)]
    if from_who == 0:
        return {
            'type': 'ankan',
            'actor': actor,
            'consumed': [
                tile136_to_mjai(code)
                for code in sort_meld_codes(codes)
            ],
        }
    return {
        'type': 'daiminkan',
        'actor': actor,
        'target': target,
        'pai': tile136_to_mjai(codes[called]),
        'consumed': [
            tile136_to_mjai(code)
            for code in sort_meld_codes([
                code for idx, code in enumerate(codes)
                if idx != called
            ])
        ],
    }


def xml_terminal_event_to_mjai(event):
    if event['kind'] == 'agari':
        mjai_event = {
            'type': 'hora',
            'actor': event['actor'],
            'target': event['target'],
            'deltas': event['result']['deltas'],
        }
        if event.get('ura_markers'):
            mjai_event['ura_markers'] = event['ura_markers']
        return mjai_event

    if event['kind'] != 'ryuukyoku':
        raise ValueError(f'unsupported XML terminal event: {event["kind"]!r}')

    draw_type = event.get('draw_type')
    deltas = event['result']['deltas']
    if draw_type == 'nm':
        return {'type': 'ryukyoku', 'deltas': deltas}
    if draw_type in ABORTIVE_XML_RYUUKYOKU_TYPES:
        return {'type': 'ryukyoku', 'deltas': [0, 0, 0, 0]}

    return {
        'type': 'ryukyoku',
        'deltas': deltas,
    }


def xml_round_events_to_mjai(round_data):
    events = [{
        'type': 'start_kyoku',
        'bakaze': round_data['bakaze'],
        'dora_marker': round_data['dora_indicator'],
        'kyoku': round_data['kyoku'],
        'honba': round_data['honba'],
        'kyotaku': round_data['kyotaku'],
        'oya': round_data['oya'],
        'scores': round_data['start_scores'],
        'tehais': round_data['start_hands'],
    }]

    last_draw_tile = [None, None, None, None]
    reach_declared_actor = None
    reach_accept_actor = None
    pending_kan_actor = None
    pending_kan_dora_markers = []

    for event in round_data['events']:
        kind = event['kind']

        if kind == 'draw':
            actor = event['actor']
            events.append({
                'type': 'tsumo',
                'actor': actor,
                'pai': event['tile'],
            })
            last_draw_tile[actor] = event['tile']

            if pending_kan_actor == actor and pending_kan_dora_markers:
                for marker in pending_kan_dora_markers:
                    events.append({'type': 'dora', 'dora_marker': marker})
                pending_kan_dora_markers.clear()
                pending_kan_actor = None
            continue

        if kind == 'discard':
            actor = event['actor']
            if reach_declared_actor == actor:
                events.append({'type': 'reach', 'actor': actor})
                reach_accept_actor = actor
                reach_declared_actor = None

            pai = event['tile']
            events.append({
                'type': 'dahai',
                'actor': actor,
                'pai': pai,
                'tsumogiri': last_draw_tile[actor] == pai,
            })
            last_draw_tile[actor] = None
            continue

        if kind == 'reach':
            if event['step'] == 1:
                reach_declared_actor = event['actor']
            elif event['step'] == 2:
                actor = event['actor']
                events.append({'type': 'reach_accepted', 'actor': actor})
                if reach_accept_actor == actor:
                    reach_accept_actor = None
            else:
                raise ValueError(f'unsupported reach step: {event["step"]}')
            continue

        if kind == 'call':
            meld = decode_xml_meld(event['meld_code'], event['actor'])
            events.append(meld)
            last_draw_tile[event['actor']] = None
            if meld['type'] in {'ankan', 'kakan', 'daiminkan'}:
                pending_kan_actor = event['actor']
            continue

        if kind == 'dora':
            marker = event['tile']
            if pending_kan_actor is None:
                events.append({'type': 'dora', 'dora_marker': marker})
            elif events and events[-1]['type'] == 'tsumo' and events[-1]['actor'] == pending_kan_actor:
                events.append({'type': 'dora', 'dora_marker': marker})
                pending_kan_actor = None
            else:
                pending_kan_dora_markers.append(marker)
            continue

        if kind in {'agari', 'ryuukyoku'}:
            if reach_accept_actor is not None:
                events.append({'type': 'reach_accepted', 'actor': reach_accept_actor})
                reach_accept_actor = None
            events.append(xml_terminal_event_to_mjai(event))
            continue

        raise ValueError(f'unsupported XML round event kind: {kind!r}')

    events.append({'type': 'end_kyoku'})
    return events


def xml_to_mjai_lines(xml_filename, official_json_filename=None):
    parsed = parse_tenhou_xml(
        xml_filename,
        official_json_filename=official_json_filename,
        include_round_events=True,
    )
    lines = [{
        'type': 'start_game',
        'names': parsed['players']['names'],
    }]
    for round_data in parsed['rounds']:
        lines.extend(xml_round_events_to_mjai(round_data))
    lines.append({'type': 'end_game'})
    return lines


def official_json_to_mjai_lines(official_json_filename):
    obj = parse_official_mjlog2json(official_json_filename)
    lines = [{
        'type': 'start_game',
        'names': obj['name'],
    }]
    for round_data in obj['log']:
        lines.extend(round_arrays_to_mjai_events(round_data))
    lines.append({'type': 'end_game'})
    return lines
