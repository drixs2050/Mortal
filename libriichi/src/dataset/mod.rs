//! Sample extractions.

mod gameplay;
mod grp;
mod invisible;
pub mod step_counter_core;
mod step_counter;

use crate::py_helper::add_submodule;
pub use gameplay::{Gameplay, GameplayLoader};
pub use grp::Grp;
pub use invisible::Invisible;
pub use step_counter::StepCounter;

use pyo3::prelude::*;

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "dataset")?;
    m.add_class::<Gameplay>()?;
    m.add_class::<GameplayLoader>()?;
    m.add_class::<Grp>()?;
    m.add_class::<StepCounter>()?;
    add_submodule(py, prefix, super_mod, &m)
}
