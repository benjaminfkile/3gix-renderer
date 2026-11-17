//! The corner lattice of one cell at one depth (`rendering.md` sections 2.1
//! and 2.3, the same-depth part).
//!
//! A section of resolution `n` holds values at its sample centers. The
//! lattice holds values at the `(n + 1)^3` corners `origin + h * (i, j, k)`,
//! `h = e / n`, `0 <= i, j, k <= n`. A corner's occupancy is the mean of the
//! eight samples whose centers surround it. Corners on a face, edge, or
//! corner of the cell take the samples beyond it from the same-depth
//! neighbors across that face, edge, or corner ([`Neighborhood`]). Solid and
//! fluid samples count with their occupancy; gas, plasma, and vacuum count
//! as occupancy 0, because only solid and fluid matter has a surface
//! (`matter-format.md` section 3.3).
//!
//! Two cells that share a face compute every corner on it from the same
//! eight samples, summed in the same order, so both get the same value bit
//! for bit (section 2.3).
//!
//! # Neighbors
//!
//! Each of the 26 neighbor slots is one of [`Slot`]:
//!
//! - **Present:** the neighbor's section, which must have the cell's
//!   resolution.
//! - **Empty:** the neighbor is known to hold no matter: the hub returned an
//!   empty section for it, or it lies outside the frame's root cell. Its
//!   samples are vacuum.
//! - **Missing:** nothing is known yet. A cell with a missing neighbor is
//!   not extracted at all ([`LatticeError::MissingNeighbor`]): its boundary
//!   corners would have no defined value. The scheduling that calls this
//!   module waits until every slot is present or empty.
//!
//! A present neighbor of a different resolution is refused
//! ([`LatticeError::ResolutionMismatch`]): its sample centers do not
//! surround this cell's corners, and the contract has no rule for it.
//!
//! # The margin
//!
//! Normals are central differences on the lattice ([`CornerLattice::gradient`]),
//! so the lattice also holds one ring of corners beyond the cell (indices
//! `-1` and `n + 1`), whose samples lie up to two layers into the
//! neighbors. That needs `n >= 2`; for `n = 1` there is no margin and the
//! gradient is one-sided at the cell boundary.

use gx_core::matter::{Section, State};
use std::sync::Arc;

/// One neighbor slot of a [`Neighborhood`].
#[derive(Clone, Debug)]
pub enum Slot {
    /// The neighbor's section, at the same depth and resolution.
    Present(Arc<Section>),
    /// The neighbor is known to hold no matter: an empty section, or a
    /// region outside the frame's root cell.
    Empty,
    /// Nothing is known about the neighbor yet.
    Missing,
}

/// A cell and its 26 same-depth neighbors.
#[derive(Clone, Debug)]
pub struct Neighborhood {
    /// The cell itself, which must not be empty.
    pub center: Arc<Section>,
    /// The 27 slots indexed by [`Neighborhood::slot_index`]; index 13 (the
    /// center) is ignored.
    pub slots: [Slot; 27],
}

impl Neighborhood {
    /// The slot index of the neighbor at offset `(dx, dy, dz)`, each in
    /// `-1..=1`: `(dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)`.
    pub fn slot_index(dx: i32, dy: i32, dz: i32) -> usize {
        ((dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)) as usize
    }

    /// A neighborhood whose slots are all [`Slot::Missing`].
    pub fn new(center: Arc<Section>) -> Self {
        Neighborhood {
            center,
            slots: std::array::from_fn(|_| Slot::Missing),
        }
    }
}

/// Why a lattice could not be built.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LatticeError {
    /// The center section is empty: it has no lattice.
    EmptyCenter,
    /// The slot at this index is [`Slot::Missing`].
    MissingNeighbor(usize),
    /// The section in the slot at this index has another resolution.
    ResolutionMismatch(usize),
}

impl std::fmt::Display for LatticeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatticeError::EmptyCenter => write!(f, "the cell is empty"),
            LatticeError::MissingNeighbor(i) => write!(f, "neighbor slot {i} is missing"),
            LatticeError::ResolutionMismatch(i) => {
                write!(f, "neighbor slot {i} has another resolution")
            }
        }
    }
}

impl std::error::Error for LatticeError {}

/// Surface attributes at one corner (`rendering.md` section 2.4,
/// `matter-format.md` section 3.5): the density-weighted means over the
/// solid and fluid samples among the eight around the corner.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CornerAttributes {
    /// Mean density of the solid and fluid matter over the eight samples,
    /// kilograms per cubic meter.
    pub density: f64,
    /// Density-weighted reflectance in the three bands.
    pub albedo: [f64; 3],
    /// Density-weighted roughness.
    pub roughness: f64,
    /// Density-weighted radiating temperature, kelvin.
    pub temperature: f64,
    /// State of the densest solid or fluid sample, as its format integer;
    /// 0 when there is none.
    pub state: u8,
}

/// One sample as the lattice reads it.
#[derive(Copy, Clone, Debug, Default)]
struct Cell {
    /// Occupancy if solid or fluid, else 0.
    occupancy: f64,
    /// Density if solid or fluid, else 0.
    density: f64,
    albedo: [f64; 3],
    roughness: f64,
    temperature: f64,
    state: u8,
}

/// The corner lattice of one cell.
#[derive(Clone, Debug)]
pub struct CornerLattice {
    /// Resolution `n` of the cell.
    pub resolution: usize,
    /// Minimum corner of the cell, meters, frame coordinates.
    pub origin: [f64; 3],
    /// Lattice spacing `e / n`, meters.
    pub spacing: f64,
    /// Corners stored beyond the cell on every side: 1, or 0 when `n = 1`.
    margin: usize,
    /// Corners per axis in storage: `n + 1 + 2 * margin`.
    dim: usize,
    /// Occupancy per stored corner, x fastest.
    values: Vec<f64>,
    /// Attributes per corner of the cell itself (`(n + 1)^3`, x fastest).
    attributes: Vec<CornerAttributes>,
}

impl CornerLattice {
    /// Builds the lattice of `hood.center` (module docs).
    pub fn build(hood: &Neighborhood) -> Result<CornerLattice, LatticeError> {
        let center = &hood.center;
        let Some(_) = center.samples() else {
            return Err(LatticeError::EmptyCenter);
        };
        let n = usize::from(center.resolution());
        for (i, slot) in hood.slots.iter().enumerate() {
            if i == 13 {
                continue;
            }
            match slot {
                Slot::Missing => return Err(LatticeError::MissingNeighbor(i)),
                Slot::Present(s) if !s.is_empty() && usize::from(s.resolution()) != n => {
                    return Err(LatticeError::ResolutionMismatch(i))
                }
                _ => {}
            }
        }
        let margin = usize::from(n >= 2);
        // Samples needed per axis: global indices -1 - margin ..= n + margin.
        let lo = -1 - margin as i64;
        let sdim = n + 2 + 2 * margin;
        let mut samples = vec![Cell::default(); sdim * sdim * sdim];
        let ni = n as i64;
        for sz in 0..sdim {
            for sy in 0..sdim {
                for sx in 0..sdim {
                    let g = [sx as i64 + lo, sy as i64 + lo, sz as i64 + lo];
                    let d = g.map(|v| v.div_euclid(ni) as i32);
                    let l = g.map(|v| v.rem_euclid(ni) as usize);
                    let section = if d == [0, 0, 0] {
                        Some(center)
                    } else {
                        match &hood.slots[Neighborhood::slot_index(d[0], d[1], d[2])] {
                            Slot::Present(s) => Some(s),
                            _ => None,
                        }
                    };
                    let Some(samples_of) = section.and_then(|s| s.samples()) else {
                        continue;
                    };
                    let s = samples_of
                        .get(l[0] + n * (l[1] + n * l[2]))
                        .expect("index inside the section");
                    if !matches!(s.state, State::Solid | State::Fluid) {
                        continue;
                    }
                    samples[sx + sdim * (sy + sdim * sz)] = Cell {
                        occupancy: s.occupancy.value(),
                        density: s.density.value(),
                        albedo: s.albedo.map(|a| a.value()),
                        roughness: s.roughness.value(),
                        temperature: s.radiating_temperature.value(),
                        state: s.state.as_u8(),
                    };
                }
            }
        }
        let dim = n + 1 + 2 * margin;
        let mut values = vec![0.0; dim * dim * dim];
        let mut attributes = vec![CornerAttributes::default(); (n + 1) * (n + 1) * (n + 1)];
        for cz in 0..dim {
            for cy in 0..dim {
                for cx in 0..dim {
                    // Corner (cx, cy, cz) in storage is surrounded by the
                    // samples (cx + a, cy + b, cz + c) in sample storage,
                    // a, b, c in 0..2, always summed in this order.
                    let mut occ = 0.0;
                    let mut around = [Cell::default(); 8];
                    for (o, slot) in around.iter_mut().enumerate() {
                        let (a, b, c) = (o & 1, (o >> 1) & 1, (o >> 2) & 1);
                        *slot = samples[(cx + a) + sdim * ((cy + b) + sdim * (cz + c))];
                        occ += slot.occupancy;
                    }
                    values[cx + dim * (cy + dim * cz)] = occ / 8.0;
                    let inside = |v: usize| v >= margin && v <= n + margin;
                    if inside(cx) && inside(cy) && inside(cz) {
                        let (ix, iy, iz) = (cx - margin, cy - margin, cz - margin);
                        attributes[ix + (n + 1) * (iy + (n + 1) * iz)] = corner_attributes(&around);
                    }
                }
            }
        }
        let o = center.origin();
        Ok(CornerLattice {
            resolution: n,
            origin: [o.x, o.y, o.z],
            spacing: center.edge().value() / n as f64,
            margin,
            dim,
            values,
            attributes,
        })
    }

    /// Occupancy at corner `(i, j, k)`, each in `-m..=n + m` where `m` is
    /// the margin (module docs).
    pub fn value(&self, i: i64, j: i64, k: i64) -> f64 {
        let m = self.margin as i64;
        let d = self.dim;
        self.values[(i + m) as usize + d * ((j + m) as usize + d * (k + m) as usize)]
    }

    /// Attributes at corner `(i, j, k)`, each in `0..=n`.
    pub fn attributes(&self, i: usize, j: usize, k: usize) -> CornerAttributes {
        let n1 = self.resolution + 1;
        self.attributes[i + n1 * (j + n1 * k)]
    }

    /// Position of corner `(i, j, k)`, meters, frame coordinates.
    pub fn position(&self, c: [usize; 3]) -> [f64; 3] {
        std::array::from_fn(|a| self.origin[a] + self.spacing * c[a] as f64)
    }

    /// Gradient of occupancy at corner `(i, j, k)` (each in `0..=n`), per
    /// meter: central differences, one-sided where the lattice has no
    /// margin.
    pub fn gradient(&self, c: [usize; 3]) -> [f64; 3] {
        let n = self.resolution as i64;
        let m = self.margin as i64;
        let p = c.map(|v| v as i64);
        std::array::from_fn(|a| {
            let mut lo = p;
            let mut hi = p;
            lo[a] = (p[a] - 1).max(-m);
            hi[a] = (p[a] + 1).min(n + m);
            let dv = self.value(hi[0], hi[1], hi[2]) - self.value(lo[0], lo[1], lo[2]);
            dv / (self.spacing * (hi[a] - lo[a]) as f64)
        })
    }
}

/// Density-weighted attributes over the solid and fluid samples among
/// `around`.
fn corner_attributes(around: &[Cell; 8]) -> CornerAttributes {
    let mut out = CornerAttributes::default();
    let mut mass = 0.0;
    let mut densest = 0.0;
    for s in around {
        if s.density <= 0.0 {
            continue;
        }
        mass += s.density;
        for b in 0..3 {
            out.albedo[b] += s.density * s.albedo[b];
        }
        out.roughness += s.density * s.roughness;
        out.temperature += s.density * s.temperature;
        if s.density > densest {
            densest = s.density;
            out.state = s.state;
        }
    }
    if mass > 0.0 {
        out.albedo = out.albedo.map(|v| v / mass);
        out.roughness /= mass;
        out.temperature /= mass;
        out.density = mass / 8.0;
    }
    out
}
