//! Surface extraction with the Transvoxel algorithm (`rendering.md`
//! section 2): the published tables ([`tables`]), the corner lattice of a
//! cell ([`lattice`], sections 2.1 and 2.3), and regular cells
//! ([`regular`], sections 2.2 and 2.4).
//!
//! This module stands alone: nothing in the world draws with it yet.
//! Transition cells between depths come later; until then a cell is
//! extracted only from same-depth inputs.

pub mod lattice;
pub mod regular;
pub mod tables;

pub use lattice::{CornerLattice, LatticeError, Neighborhood, Slot};
pub use regular::{extract_regular, SURFACE_OCCUPANCY};
