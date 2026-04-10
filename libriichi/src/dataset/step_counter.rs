//! PyO3 wrapper for the step counter.

use super::step_counter_core::{self, NUM_CATEGORIES};
use std::collections::HashMap;
use std::fs::File;
use std::io;

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Python-exposed step counter.
#[pyclass]
#[derive(Debug)]
pub struct StepCounter {
    #[pyo3(get)]
    version: u32,
    #[pyo3(get)]
    always_include_kan_select: bool,
}

#[pymethods]
impl StepCounter {
    #[new]
    #[pyo3(signature = (version = 4, *, always_include_kan_select = true))]
    fn new(version: u32, always_include_kan_select: bool) -> Self {
        Self {
            version,
            always_include_kan_select,
        }
    }

    /// Count steps across multiple gzipped files in parallel.
    #[pyo3(signature = (gzip_filenames, player_filter = None))]
    fn count_gz_files(
        &self,
        gzip_filenames: Vec<String>,
        player_filter: Option<HashMap<String, Vec<u8>>>,
    ) -> Result<PyObject> {
        let version = self.version;
        let always_include_kan_select = self.always_include_kan_select;

        // Aggregate in Rust first
        let mut total_file_count: u64 = 0;
        let mut total_trajectory_count: u64 = 0;
        let mut total_step_count: u64 = 0;
        let mut total_event_count: u64 = 0;
        let mut total_kyoku_count: u64 = 0;
        let mut total_action_dist = [0u64; NUM_CATEGORIES];
        let mut total_shanten_hist = [0u64; 8];
        let mut errors: Vec<String> = Vec::new();

        let file_results: Vec<Result<(u64, u64, u64, u64, [u64; NUM_CATEGORIES], [u64; 8])>> =
            gzip_filenames
                .par_iter()
                .map(|filename| {
                    let inner = || -> Result<(u64, u64, u64, u64, [u64; NUM_CATEGORIES], [u64; 8])>
                    {
                        let file = File::open(filename)?;
                        let gz = GzDecoder::new(file);
                        let raw = io::read_to_string(gz)?;

                        let allowed_ids: Option<Vec<u8>> = player_filter
                            .as_ref()
                            .and_then(|pf| pf.get(filename.as_str()))
                            .cloned();

                        let (player_stats, event_count) = step_counter_core::count_file_steps(
                            &raw,
                            version,
                            always_include_kan_select,
                            allowed_ids.as_deref(),
                        )?;

                        let mut steps = 0u64;
                        let mut trajs = 0u64;
                        let mut kyoku = 0u64;
                        let mut adist = [0u64; NUM_CATEGORIES];
                        let mut shist = [0u64; 8];

                        for ps in &player_stats {
                            trajs += 1;
                            steps += ps.step_count;
                            kyoku += ps.kyoku_count as u64;
                            for i in 0..NUM_CATEGORIES {
                                adist[i] += ps.action_dist[i];
                            }
                            for i in 0..8 {
                                shist[i] += ps.shanten_hist[i];
                            }
                        }

                        Ok((steps, trajs, event_count as u64, kyoku, adist, shist))
                    };
                    inner().with_context(|| format!("error counting {filename}"))
                })
                .collect();

        for res in file_results {
            match res {
                Ok((steps, trajs, events, kyoku, adist, shist)) => {
                    total_file_count += 1;
                    total_step_count += steps;
                    total_trajectory_count += trajs;
                    total_event_count += events;
                    total_kyoku_count += kyoku;
                    for i in 0..NUM_CATEGORIES {
                        total_action_dist[i] += adist[i];
                    }
                    for i in 0..8 {
                        total_shanten_hist[i] += shist[i];
                    }
                }
                Err(e) => errors.push(format!("{e:#}")),
            }
        }

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("file_count", total_file_count)?;
            dict.set_item("trajectory_count", total_trajectory_count)?;
            dict.set_item("step_count", total_step_count)?;
            dict.set_item("event_count", total_event_count)?;
            dict.set_item("kyoku_count", total_kyoku_count)?;

            let action_names = [
                "discard", "riichi", "chi", "pon", "kan",
                "agari", "ryukyoku", "pass", "kan_select",
            ];
            let action_dict = pyo3::types::PyDict::new(py);
            for (i, name) in action_names.iter().enumerate() {
                action_dict.set_item(*name, total_action_dist[i])?;
            }
            dict.set_item("action_distribution", action_dict)?;

            let shanten_dict = pyo3::types::PyDict::new(py);
            for i in 0..8usize {
                shanten_dict.set_item(i as i8, total_shanten_hist[i])?;
            }
            dict.set_item("shanten_histogram", shanten_dict)?;

            if total_trajectory_count > 0 {
                dict.set_item(
                    "avg_steps_per_trajectory",
                    total_step_count as f64 / total_trajectory_count as f64,
                )?;
            }
            if total_file_count > 0 {
                dict.set_item(
                    "avg_steps_per_file",
                    total_step_count as f64 / total_file_count as f64,
                )?;
                dict.set_item(
                    "avg_trajectories_per_file",
                    total_trajectory_count as f64 / total_file_count as f64,
                )?;
                dict.set_item(
                    "avg_events_per_file",
                    total_event_count as f64 / total_file_count as f64,
                )?;
            }

            if !errors.is_empty() {
                dict.set_item("error_count", errors.len())?;
                let max_show = errors.len().min(10);
                let error_list = pyo3::types::PyList::new(py, &errors[..max_show])?;
                dict.set_item("sample_errors", error_list)?;
            }

            Ok(dict.into())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "StepCounter(version={}, always_include_kan_select={})",
            self.version, self.always_include_kan_select,
        )
    }
}
