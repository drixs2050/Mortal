use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use riichi::dataset::step_counter_core;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct SplitResult {
    file_count: u64,
    trajectory_count: u64,
    step_count: u64,
    event_count: u64,
    kyoku_count: u64,
    action_distribution: ActionDist,
    shanten_histogram: [u64; 8],
    avg_steps_per_trajectory: f64,
    avg_steps_per_file: f64,
    avg_trajectories_per_file: f64,
    elapsed_seconds: f64,
    error_count: usize,
}

#[derive(Debug, Default, Serialize)]
struct ActionDist {
    discard: u64,
    riichi: u64,
    chi: u64,
    pon: u64,
    kan: u64,
    agari: u64,
    ryukyoku: u64,
    pass: u64,
    kan_select: u64,
}

#[derive(Debug, Serialize)]
struct FileStats {
    step_count: u64,
    trajectory_count: u32,
}

#[derive(Debug, Serialize)]
struct TrajectoryRecord {
    file_idx: u32,
    player_id: u8,
    step_count: u32,
    kyoku_count: u16,
    action_dist: [u16; 9],
    shanten_hist: [u16; 8],
    turn_max: u8,
    agari_count: u8,
    riichi_count: u8,
}

#[derive(Debug, Serialize)]
struct OutputRoot {
    format: String,
    version: u32,
    always_include_kan_select: bool,
    splits: HashMap<String, SplitResult>,
    /// per-file stats: filename -> {step_count, trajectory_count}
    file_stats: HashMap<String, FileStats>,
}

fn read_file_list(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut files = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim().to_string();
        if !line.is_empty() {
            files.push(line);
        }
    }
    Ok(files)
}

fn read_actor_filter(path: &Path) -> Result<HashMap<String, Vec<u8>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut filter = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let filepath = parts.next().unwrap().to_string();
        let ids_str = parts.next().unwrap_or("");
        let ids: Vec<u8> = ids_str
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().parse::<u8>())
            .collect::<std::result::Result<_, _>>()
            .with_context(|| format!("parsing player IDs for {filepath}"))?;
        filter.insert(filepath, ids);
    }
    Ok(filter)
}

struct FileResult {
    filename: String,
    file_idx: u32,
    trajectories: Vec<step_counter_core::PlayerStats>,
    event_count: usize,
    error: Option<String>,
}

fn count_split(
    split_name: &str,
    file_list: &[String],
    actor_filter: &Option<HashMap<String, Vec<u8>>>,
    version: u32,
    always_include_kan_select: bool,
    file_stats_out: &mut HashMap<String, FileStats>,
    traj_records_out: &mut Vec<TrajectoryRecord>,
) -> SplitResult {
    let start = Instant::now();

    let results: Vec<FileResult> = file_list
        .par_iter()
        .enumerate()
        .map(|(idx, filename)| {
            let inner = || -> Result<(Vec<step_counter_core::PlayerStats>, usize)> {
                let file = File::open(filename)?;
                let gz = GzDecoder::new(file);
                let raw = io::read_to_string(gz)?;

                let allowed_ids: Option<Vec<u8>> = actor_filter
                    .as_ref()
                    .and_then(|af| af.get(filename.as_str()))
                    .cloned();

                step_counter_core::count_file_steps(
                    &raw,
                    version,
                    always_include_kan_select,
                    allowed_ids.as_deref(),
                )
            };
            match inner() {
                Ok((stats, event_count)) => FileResult {
                    filename: filename.clone(),
                    file_idx: idx as u32,
                    trajectories: stats,
                    event_count,
                    error: None,
                },
                Err(e) => FileResult {
                    filename: filename.clone(),
                    file_idx: idx as u32,
                    trajectories: Vec::new(),
                    event_count: 0,
                    error: Some(format!("{e:#}")),
                },
            }
        })
        .collect();

    let mut total_files: u64 = 0;
    let mut total_trajs: u64 = 0;
    let mut total_steps: u64 = 0;
    let mut total_events: u64 = 0;
    let mut total_kyoku: u64 = 0;
    let mut action_dist = [0u64; 9];
    let mut shanten_hist = [0u64; 8];
    let mut errors = Vec::new();

    for fr in &results {
        if let Some(ref err) = fr.error {
            errors.push(err.clone());
            continue;
        }

        total_files += 1;
        total_events += fr.event_count as u64;
        let mut file_steps: u64 = 0;
        let mut file_trajs: u32 = 0;

        for ps in &fr.trajectories {
            total_trajs += 1;
            total_steps += ps.step_count;
            total_kyoku += ps.kyoku_count as u64;
            file_steps += ps.step_count;
            file_trajs += 1;

            for i in 0..9 {
                action_dist[i] += ps.action_dist[i];
            }
            for i in 0..8 {
                shanten_hist[i] += ps.shanten_hist[i];
            }

            // Write per-trajectory record
            traj_records_out.push(TrajectoryRecord {
                file_idx: fr.file_idx,
                player_id: ps.player_id,
                step_count: ps.step_count as u32,
                kyoku_count: ps.kyoku_count as u16,
                action_dist: [
                    ps.action_dist[0] as u16,
                    ps.action_dist[1] as u16,
                    ps.action_dist[2] as u16,
                    ps.action_dist[3] as u16,
                    ps.action_dist[4] as u16,
                    ps.action_dist[5] as u16,
                    ps.action_dist[6] as u16,
                    ps.action_dist[7] as u16,
                    ps.action_dist[8] as u16,
                ],
                shanten_hist: [
                    ps.shanten_hist[0] as u16,
                    ps.shanten_hist[1] as u16,
                    ps.shanten_hist[2] as u16,
                    ps.shanten_hist[3] as u16,
                    ps.shanten_hist[4] as u16,
                    ps.shanten_hist[5] as u16,
                    ps.shanten_hist[6] as u16,
                    ps.shanten_hist[7] as u16,
                ],
                turn_max: ps.turn_max,
                agari_count: ps.agari_count,
                riichi_count: ps.riichi_count,
            });
        }

        file_stats_out.insert(
            fr.filename.clone(),
            FileStats {
                step_count: file_steps,
                trajectory_count: file_trajs,
            },
        );
    }

    let elapsed = start.elapsed().as_secs_f64();

    if !errors.is_empty() {
        eprintln!(
            "[{split_name}] {}/{} files had errors (showing first 5):",
            errors.len(),
            file_list.len()
        );
        for e in errors.iter().take(5) {
            eprintln!("  {e}");
        }
    }

    eprintln!(
        "[{split_name}] {total_files} files, {total_trajs} trajectories, \
         {total_steps} steps in {elapsed:.1}s",
    );

    SplitResult {
        file_count: total_files,
        trajectory_count: total_trajs,
        step_count: total_steps,
        event_count: total_events,
        kyoku_count: total_kyoku,
        action_distribution: ActionDist {
            discard: action_dist[0],
            riichi: action_dist[1],
            chi: action_dist[2],
            pon: action_dist[3],
            kan: action_dist[4],
            agari: action_dist[5],
            ryukyoku: action_dist[6],
            pass: action_dist[7],
            kan_select: action_dist[8],
        },
        shanten_histogram: shanten_hist,
        avg_steps_per_trajectory: if total_trajs > 0 {
            total_steps as f64 / total_trajs as f64
        } else {
            0.0
        },
        avg_steps_per_file: if total_files > 0 {
            total_steps as f64 / total_files as f64
        } else {
            0.0
        },
        avg_trajectories_per_file: if total_files > 0 {
            total_trajs as f64 / total_files as f64
        } else {
            0.0
        },
        elapsed_seconds: elapsed,
        error_count: errors.len(),
    }
}

fn write_trajectory_binary(path: &Path, records: &[TrajectoryRecord]) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header: magic + version + record count
    w.write_all(b"TRAJ")?; // magic
    w.write_all(&1u32.to_le_bytes())?; // format version
    w.write_all(&(records.len() as u64).to_le_bytes())?; // record count

    // Fixed-size records: 4 + 1 + 4 + 2 + 18 + 16 + 1 + 1 + 1 = 48 bytes each
    for r in records {
        w.write_all(&r.file_idx.to_le_bytes())?;
        w.write_all(&[r.player_id])?;
        w.write_all(&r.step_count.to_le_bytes())?;
        w.write_all(&r.kyoku_count.to_le_bytes())?;
        for &v in &r.action_dist {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in &r.shanten_hist {
            w.write_all(&v.to_le_bytes())?;
        }
        w.write_all(&[r.turn_max])?;
        w.write_all(&[r.agari_count])?;
        w.write_all(&[r.riichi_count])?;
    }
    w.flush()?;
    Ok(())
}

const USAGE: &str = "\
Usage: count_steps --split-dir <DIR> --output <PATH> [OPTIONS]

Options:
  --split-dir <DIR>          Directory containing train.txt, val.txt, test.txt
  --actor-filter <FILE>      TSV file: filepath<TAB>player_id1,player_id2,...
  --output <PATH>            Output JSON path (trajectory binary goes to <PATH>.bin)
  --version <N>              Game version (default: 4)
  --no-kan-select            Disable always_include_kan_select
  --splits <LIST>            Comma-separated split names (default: train,val,test)
";

fn main() -> Result<()> {
    // PlayerState is large — increase rayon worker stack to 8 MB
    ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .ok();

    let args: Vec<String> = std::env::args().collect();

    let mut split_dir: Option<PathBuf> = None;
    let mut actor_filter_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut version: u32 = 4;
    let mut always_include_kan_select = true;
    let mut split_names: Vec<String> = vec!["train".into(), "val".into(), "test".into()];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--split-dir" => {
                i += 1;
                split_dir = Some(PathBuf::from(&args[i]));
            }
            "--actor-filter" => {
                i += 1;
                actor_filter_path = Some(PathBuf::from(&args[i]));
            }
            "--output" => {
                i += 1;
                output_path = Some(PathBuf::from(&args[i]));
            }
            "--version" => {
                i += 1;
                version = args[i].parse()?;
            }
            "--no-kan-select" => {
                always_include_kan_select = false;
            }
            "--splits" => {
                i += 1;
                split_names = args[i].split(',').map(|s| s.trim().to_string()).collect();
            }
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown argument: {other}\n{USAGE}"),
        }
        i += 1;
    }

    let split_dir = split_dir.context("--split-dir is required")?;
    let output_path = output_path.context("--output is required")?;

    // Load actor filter if provided
    let actor_filter = if let Some(ref af_path) = actor_filter_path {
        eprintln!("Loading actor filter from {}", af_path.display());
        let af = read_actor_filter(af_path)?;
        eprintln!("  {} entries", af.len());
        Some(af)
    } else {
        None
    };

    let mut splits = HashMap::new();
    let mut file_stats = HashMap::new();
    let mut all_traj_records = Vec::new();

    let total_start = Instant::now();

    for split_name in &split_names {
        let list_path = split_dir.join(format!("{split_name}.txt"));
        if !list_path.exists() {
            eprintln!("[{split_name}] {list_path:?} not found, skipping");
            continue;
        }

        let file_list = read_file_list(&list_path)?;
        eprintln!("[{split_name}] {} files", file_list.len());

        let result = count_split(
            split_name,
            &file_list,
            &actor_filter,
            version,
            always_include_kan_select,
            &mut file_stats,
            &mut all_traj_records,
        );

        splits.insert(split_name.clone(), result);
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    eprintln!("Total: {total_elapsed:.1}s, {} trajectory records", all_traj_records.len());

    // Write JSON summary
    let output = OutputRoot {
        format: "dataset_step_stats_v1".into(),
        version,
        always_include_kan_select,
        splits,
        file_stats,
    };

    let json_file = File::create(&output_path)?;
    serde_json::to_writer_pretty(BufWriter::new(json_file), &output)?;
    eprintln!("Wrote JSON: {}", output_path.display());

    // Write trajectory binary
    let bin_path = output_path.with_extension("bin");
    write_trajectory_binary(&bin_path, &all_traj_records)?;
    let bin_size = fs::metadata(&bin_path)?.len();
    eprintln!(
        "Wrote trajectory binary: {} ({:.1} MB, {} records)",
        bin_path.display(),
        bin_size as f64 / 1e6,
        all_traj_records.len(),
    );

    Ok(())
}
