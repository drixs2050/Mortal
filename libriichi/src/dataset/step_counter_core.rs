//! Core step counting logic — no PyO3 dependency.
//!
//! Replicates the decision-point logic from [`Gameplay`] but skips
//! `encode_obs` entirely. This makes counting ~400x faster.

use crate::chi_type::ChiType;
use crate::mjai::Event;
use crate::state::PlayerState;

use anyhow::{Context, Result, bail};
use serde_json as json;

/// Action categories for the distribution histogram.
pub const CAT_DISCARD: usize = 0;
pub const CAT_RIICHI: usize = 1;
pub const CAT_CHI: usize = 2;
pub const CAT_PON: usize = 3;
pub const CAT_KAN: usize = 4;
pub const CAT_AGARI: usize = 5;
pub const CAT_RYUKYOKU: usize = 6;
pub const CAT_PASS: usize = 7;
pub const CAT_KAN_SELECT: usize = 8;
pub const NUM_CATEGORIES: usize = 9;

pub fn label_to_category(label: usize, is_kan_select: bool) -> usize {
    if is_kan_select {
        return CAT_KAN_SELECT;
    }
    match label {
        0..=36 => CAT_DISCARD,
        37 => CAT_RIICHI,
        38..=40 => CAT_CHI,
        41 => CAT_PON,
        42 => CAT_KAN,
        43 => CAT_AGARI,
        44 => CAT_RYUKYOKU,
        45 => CAT_PASS,
        _ => CAT_DISCARD,
    }
}

/// Per-player statistics from counting a single game.
#[derive(Clone, Debug)]
pub struct PlayerStats {
    pub player_id: u8,
    pub step_count: u64,
    pub kyoku_count: u32,
    pub action_dist: [u64; NUM_CATEGORIES],
    pub shanten_hist: [u64; 8],
    pub turn_max: u8,
    pub agari_count: u8,
    pub riichi_count: u8,
}

impl Default for PlayerStats {
    fn default() -> Self {
        Self {
            player_id: 0,
            step_count: 0,
            kyoku_count: 0,
            action_dist: [0; NUM_CATEGORIES],
            shanten_hist: [0; 8],
            turn_max: 0,
            agari_count: 0,
            riichi_count: 0,
        }
    }
}

/// Count steps for a single player in a parsed game.
pub fn count_player_steps(
    events: &[Event],
    player_id: u8,
    _version: u32,
    always_include_kan_select: bool,
) -> Result<PlayerStats> {
    let mut stats = PlayerStats {
        player_id,
        ..Default::default()
    };

    let mut state = PlayerState::new(player_id);
    let mut kyoku_idx: u32 = 0;

    if events.len() < 4 {
        return Ok(stats);
    }

    for wnd in events.windows(4) {
        let cur = &wnd[0];
        let next = if matches!(wnd[1], Event::ReachAccepted { .. } | Event::Dora { .. }) {
            &wnd[2]
        } else {
            &wnd[1]
        };

        if let Event::EndKyoku = cur {
            kyoku_idx += 1;
        }

        let cans = state.update(cur)?;
        if !cans.can_act() {
            continue;
        }

        let mut kan_select = None;
        let label_opt = match *next {
            Event::Dahai { pai, .. } => Some(pai.as_usize()),
            Event::Reach { .. } => Some(37),
            Event::Chi {
                actor,
                pai,
                consumed,
                ..
            } if actor == player_id => match ChiType::new(consumed, pai) {
                ChiType::Low => Some(38),
                ChiType::Mid => Some(39),
                ChiType::High => Some(40),
            },
            Event::Pon { actor, .. } if actor == player_id => Some(41),
            Event::Daiminkan { actor, pai, .. } if actor == player_id => {
                if always_include_kan_select {
                    kan_select = Some(pai.deaka().as_usize());
                }
                Some(42)
            }
            Event::Kakan { pai, .. } => {
                if always_include_kan_select || state.kakan_candidates().len() > 1 {
                    kan_select = Some(pai.deaka().as_usize());
                }
                Some(42)
            }
            Event::Ankan { consumed, .. } => {
                if always_include_kan_select || state.ankan_candidates().len() > 1 {
                    kan_select = Some(consumed[0].deaka().as_usize());
                }
                Some(42)
            }
            Event::Ryukyoku { .. } if cans.can_ryukyoku => Some(44),
            _ => {
                let mut ret = None;
                let has_any_ron = matches!(wnd[1], Event::Hora { .. });
                if has_any_ron {
                    for ev in &wnd[1..] {
                        match *ev {
                            Event::EndKyoku => break,
                            Event::Hora { actor, .. } if actor == player_id => {
                                ret = Some(43);
                                break;
                            }
                            _ => (),
                        };
                    }
                }
                if ret.is_none()
                    && (cans.can_chi() && matches!(next, Event::Tsumo { .. })
                        || (cans.can_pon || cans.can_daiminkan || cans.can_ron_agari)
                            && !has_any_ron)
                {
                    ret = Some(45);
                }
                ret
            }
        };

        if let Some(label) = label_opt {
            stats.step_count += 1;
            let cat = label_to_category(label, false);
            stats.action_dist[cat] += 1;

            if cat == CAT_AGARI {
                stats.agari_count = stats.agari_count.saturating_add(1);
            }
            if cat == CAT_RIICHI {
                stats.riichi_count = stats.riichi_count.saturating_add(1);
            }

            let shanten = state.shanten();
            let idx = shanten.clamp(0, 7) as usize;
            stats.shanten_hist[idx] += 1;

            let turn = state.at_turn();
            if turn > stats.turn_max {
                stats.turn_max = turn;
            }

            if kan_select.is_some() {
                stats.step_count += 1;
                stats.action_dist[CAT_KAN_SELECT] += 1;
            }
        }
    }

    stats.kyoku_count = kyoku_idx;
    Ok(stats)
}

/// Count steps for a single file. Returns per-player stats + event count.
pub fn count_file_steps(
    raw_log: &str,
    version: u32,
    always_include_kan_select: bool,
    allowed_player_ids: Option<&[u8]>,
) -> Result<(Vec<PlayerStats>, usize)> {
    let events: Vec<Event> = raw_log
        .lines()
        .map(json::from_str)
        .collect::<Result<_, _>>()
        .context("failed to parse log")?;

    let event_count = events.len();

    let [Event::StartGame { names, .. }, ..] = events.as_slice() else {
        bail!("empty or invalid game log");
    };

    let player_ids: Vec<u8> = names
        .iter()
        .enumerate()
        .filter(|&(i, _)| {
            if let Some(allowed) = allowed_player_ids {
                allowed.contains(&(i as u8))
            } else {
                true
            }
        })
        .map(|(i, _)| i as u8)
        .collect();

    let stats: Vec<PlayerStats> = player_ids
        .iter()
        .map(|&pid| count_player_steps(&events, pid, version, always_include_kan_select))
        .collect::<Result<_>>()?;

    Ok((stats, event_count))
}
