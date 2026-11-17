//! Surface extraction with the Transvoxel algorithm (`rendering.md`
//! section 2): the published tables ([`tables`]), the corner lattice of a
//! cell ([`lattice`], sections 2.1 and 2.3), regular cells ([`regular`],
//! sections 2.2 and 2.4), and cells between depths with transition cells
//! ([`mixed`], sections 2.2, 2.3, and 2.5).
//!
//! The world draws every cell with it through [`crate::extract`].

pub mod lattice;
pub mod mixed;
pub mod regular;
pub mod tables;

pub use lattice::{CornerLattice, LatticeError, Neighborhood, Slot};
pub use mixed::{extract_mixed, lattice_value, CellAt, MixedField, MixedMesh, TRANSITION_WIDTH};
pub use regular::{extract_regular, SURFACE_OCCUPANCY};
