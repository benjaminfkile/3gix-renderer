//! The walking observer's matter probe: what the loaded cells of one frame
//! say about the matter at a point.
//!
//! Implements the [`MatterProbe`] of [`gx_core::observer`] for walk mode
//! (`space-model.md` sections 3, rule 3, and 6.3) over the composited
//! sections of the cell cache ([`crate::stream`], `matter-format.md`
//! section 3.5). The probe knows quantities only: densities, states, and
//! the velocity of the frame's rotation.
//!
//! # The finest loaded depth
//!
//! A point is answered by the deepest cell containing it that is loaded:
//! from the frame's deepest depth (the deepest any layer offers,
//! [`crate::protocol::frame_system_from_registries`]) up to depth 0, the
//! first cell that is `Ready` or `Empty` answers. An `Empty` cell is
//! vacuum. A point no loaded cell contains is unknown (`None`), which the
//! core reads as vacuum; the observer is held before that matters
//! ([`waiting_for_matter`]).
//!
//! # One sample
//!
//! Inside a `Ready` cell of resolution `n` the sample values sit at the
//! sample centers. The density is trilinear over the eight centers around
//! the point. A center beyond the cell's faces is read from the cell at the
//! same depth that holds it when that cell is loaded (its nearest sample
//! when `Ready`, vacuum when `Empty`), and is the cell's own nearest sample
//! otherwise, the same padding rule as surface extraction
//! ([`crate::extract`]).
//!
//! The state is nearest-sample, split at the drawn surface so the observer
//! stands on what is drawn:
//!
//! - where the trilinear density of the solid and fluid centers alone is
//!   above the cell's surface isovalue
//!   ([`crate::stream::ReadyCell::surface_isovalue`], the isovalue of the
//!   drawn mesh), the state is that of the nearest solid or fluid center,
//!   and the density is the full trilinear density,
//! - else, where gas or plasma centers are among the eight, the state is
//!   that of the nearest of them and the density the trilinear density of
//!   those centers alone, so a solid surface does not bleed buoyancy into
//!   the gas above it,
//! - else the point is vacuum.
//!
//! The velocity is the frame's rotation at the point, `w x p`, with `w`
//! the frame's angular velocity about its own axes: matter is at rest in
//! its frame.
//!
//! # Determinism
//!
//! Lookups go through the cache's ordered map and use only `+`, `-`, `*`,
//! `/`, and `floor`, so a probe gives the same answer for the same point
//! and the same loaded cells on every machine.

use crate::stream::{CellCache, CellState, ReadyCell};
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::matter::{Section, State};
use gx_core::observer::{MatterProbe, Sample};
use gx_core::units::{Density, Vec3};
use std::sync::Arc;

/// Most steps of [`CacheProbe::outermost_along`] before it gives up.
pub const MAX_MARCH_STEPS: usize = 1 << 20;

/// Halvings of the bracket around the outermost matter on a line.
const MARCH_HALVINGS: u32 = 48;

/// What a loaded cell holds.
#[derive(Clone, Copy)]
enum Loaded<'a> {
    Ready(&'a Arc<ReadyCell>),
    Empty,
}

/// The probe answer at a point with the cell that gave it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Probed {
    /// The matter at the point.
    pub sample: Sample,
    /// The depth of the cell that answered.
    pub depth: u8,
    /// The edge of one sample of that cell, meters (the cell edge for an
    /// `Empty` cell).
    pub sample_edge: f64,
}

/// The matter of one frame as the cell cache has it loaded (module docs).
#[derive(Clone, Copy)]
pub struct CacheProbe<'a> {
    cache: &'a CellCache,
    frame_id: u64,
    root_extent: f64,
    max_depth: u8,
    angular_velocity: Vec3,
}

/// The key of the cell at `depth` of a frame of edge `root_extent` that
/// contains `p` (frame coordinates), or `None` outside the frame's cube.
pub fn key_at(frame_id: u64, root_extent: f64, depth: u8, p: Vec3) -> Option<CellKey> {
    let count = (1u64 << depth) as f64;
    let edge = root_extent / count;
    let index = |v: f64| -> Option<u32> {
        let i = ((v + root_extent / 2.0) / edge).floor();
        (i >= 0.0 && i < count).then_some(i as u32)
    };
    CellKey::new(frame_id, depth, index(p.x)?, index(p.y)?, index(p.z)?).ok()
}

/// The cells at `depth` around `p` (frame coordinates): the up to eight
/// cells whose centers are nearest, the ones the probe's trilinear lookup
/// reads at that depth, in key order. Cells outside the frame's cube are
/// left out.
pub fn cells_around(frame_id: u64, root_extent: f64, depth: u8, p: Vec3) -> Vec<CellKey> {
    let count = 1u64 << depth;
    let edge = root_extent / count as f64;
    let low = |v: f64| ((v + root_extent / 2.0) / edge - 0.5).floor();
    let (lx, ly, lz) = (low(p.x), low(p.y), low(p.z));
    let mut out = Vec::with_capacity(8);
    for c in 0..8u32 {
        let pick = |base: f64, bit: u32| -> Option<u32> {
            let i = base + f64::from((c >> bit) & 1);
            (i >= 0.0 && i < count as f64).then_some(i as u32)
        };
        if let (Some(x), Some(y), Some(z)) = (pick(lx, 0), pick(ly, 1), pick(lz, 2)) {
            if let Ok(k) = CellKey::new(frame_id, depth, x, y, z) {
                out.push(k);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The streaming priority of a walking observer whose feet are at `feet`
/// (frame coordinates) in frame `frame_id`: [`cells_around`] the feet at
/// every depth from 0 to the frame's deepest, so contact uses the finest
/// matter any layer offers there. Empty for a frame not in the system.
pub fn feet_cells(system: &FrameSystem, frame_id: u64, feet: Vec3) -> Vec<CellKey> {
    let Some(frame) = system.tree().get(frame_id) else {
        return Vec::new();
    };
    (0..=frame.max_depth)
        .flat_map(|d| cells_around(frame_id, frame.root_extent.value(), d, feet))
        .collect()
}

/// Returns `true` while matter under the observer is not loaded yet: some
/// cell around the feet at the frame's deepest depth ([`cells_around`]) is
/// not fetched (`Missing`, `Requested`, or not in the cache). A cell that
/// turned out `Gone` does not hold the observer; it will never arrive.
pub fn waiting_for_matter(
    cache: &CellCache,
    system: &FrameSystem,
    frame_id: u64,
    feet: Vec3,
) -> bool {
    let Some(frame) = system.tree().get(frame_id) else {
        return false;
    };
    cells_around(frame_id, frame.root_extent.value(), frame.max_depth, feet)
        .iter()
        .any(|k| {
            matches!(
                cache.state(k),
                None | Some(CellState::Missing) | Some(CellState::Requested)
            )
        })
}

impl<'a> CacheProbe<'a> {
    /// The probe of frame `frame_id` over `cache`, with the frame's angular
    /// velocity from `system`. `None` if the frame is not in the system.
    pub fn new(
        cache: &'a CellCache,
        system: &FrameSystem,
        frame_id: u64,
    ) -> Option<CacheProbe<'a>> {
        let frame = system.tree().get(frame_id)?;
        let w = system.state(frame_id)?.angular_velocity;
        Some(CacheProbe {
            cache,
            frame_id,
            root_extent: frame.root_extent.value(),
            max_depth: frame.max_depth,
            angular_velocity: w,
        })
    }

    /// The deepest loaded cell containing `p`.
    fn finest(&self, p: Vec3) -> Option<(CellKey, Loaded<'a>)> {
        for depth in (0..=self.max_depth).rev() {
            let key = key_at(self.frame_id, self.root_extent, depth, p)?;
            match self.cache.state(&key) {
                Some(CellState::Ready(cell)) => return Some((key, Loaded::Ready(cell))),
                Some(CellState::Empty) => return Some((key, Loaded::Empty)),
                _ => {}
            }
        }
        None
    }

    /// The density and state of the sample of `section` nearest `p`.
    fn nearest_in(section: &Section, p: Vec3) -> (f64, State) {
        let Some(samples) = section.samples() else {
            return (0.0, State::Vacuum);
        };
        let n = usize::from(section.resolution());
        let step = section.edge().value() / n as f64;
        let o = section.origin();
        let index = |v: f64, o: f64| (((v - o) / step).floor().max(0.0) as usize).min(n - 1);
        let i = index(p.x, o.x) + n * (index(p.y, o.y) + n * index(p.z, o.z));
        (samples.density(i).value(), samples.state(i))
    }

    /// The density and state at the sample center `c`, which lies outside
    /// `own` (at depth `depth`): from the loaded cell at the same depth that
    /// holds it, else `own`'s nearest sample.
    fn padding(&self, own: &Section, depth: u8, c: Vec3) -> (f64, State) {
        let across = key_at(self.frame_id, self.root_extent, depth, c);
        match across.and_then(|k| self.cache.state(&k)) {
            Some(CellState::Ready(cell)) => Self::nearest_in(&cell.section, c),
            Some(CellState::Empty) => (0.0, State::Vacuum),
            _ => Self::nearest_in(own, c),
        }
    }

    /// The probe answer at `p` with the depth and sample size of the cell
    /// that gave it, or `None` where no cell is loaded (module docs).
    pub fn lookup(&self, p: Vec3) -> Option<Probed> {
        let (key, loaded) = self.finest(p)?;
        let velocity = self.angular_velocity.cross(p);
        let vacuum = Sample {
            density: Density::new(0.0),
            state: State::Vacuum,
            velocity,
        };
        let cell = match loaded {
            Loaded::Empty => {
                return Some(Probed {
                    sample: vacuum,
                    depth: key.depth,
                    sample_edge: self.root_extent / (1u64 << key.depth) as f64,
                })
            }
            Loaded::Ready(cell) => cell,
        };
        let section = &cell.section;
        let n = section.resolution();
        let step = section.edge().value() / f64::from(n);
        let probed = |sample| Probed {
            sample,
            depth: key.depth,
            sample_edge: step,
        };
        let Some(samples) = section.samples() else {
            return Some(probed(vacuum));
        };
        let o = section.origin();
        let n = i64::from(n);
        let u = |v: f64, o: f64| (v - o) / step - 0.5;
        let (ux, uy, uz) = (u(p.x, o.x), u(p.y, o.y), u(p.z, o.z));
        let (bx, by, bz) = (ux.floor(), uy.floor(), uz.floor());
        let (fx, fy, fz) = (ux - bx, uy - by, uz - bz);
        // Eight corners: weight, density, state.
        let mut corners = [(0.0, 0.0, State::Vacuum); 8];
        for (c, corner) in corners.iter_mut().enumerate() {
            let bit = |b: usize| ((c >> b) & 1) as i64;
            let (ix, iy, iz) = (bx as i64 + bit(0), by as i64 + bit(1), bz as i64 + bit(2));
            let w = |f: f64, b: i64| if b == 1 { f } else { 1.0 - f };
            let weight = w(fx, bit(0)) * w(fy, bit(1)) * w(fz, bit(2));
            let inside = |i: i64| (0..n).contains(&i);
            let (d, s) = if inside(ix) && inside(iy) && inside(iz) {
                let i = (ix + n * (iy + n * iz)) as usize;
                (samples.density(i).value(), samples.state(i))
            } else {
                let center = |o: f64, i: i64| o + (i as f64 + 0.5) * step;
                let c = Vec3::new(center(o.x, ix), center(o.y, iy), center(o.z, iz));
                self.padding(section, key.depth, c)
            };
            *corner = (weight, d, s);
        }
        let surface = |s: State| matches!(s, State::Solid | State::Fluid);
        let volume = |s: State| matches!(s, State::Gas | State::Plasma);
        let nearest = |keep: &dyn Fn(State) -> bool| {
            corners
                .iter()
                .filter(|(_, _, s)| keep(*s))
                .fold(None, |best: Option<(f64, State)>, &(w, _, s)| match best {
                    Some((bw, _)) if bw >= w => best,
                    _ => Some((w, s)),
                })
                .map(|(_, s)| s)
        };
        let sum = |keep: &dyn Fn(State) -> bool| {
            corners
                .iter()
                .filter(|(_, _, s)| keep(*s))
                .map(|(w, d, _)| w * d)
                .sum::<f64>()
        };
        let iso = cell.surface_isovalue;
        if let (Some(iso), Some(state)) = (iso, nearest(&surface)) {
            if sum(&surface) > iso {
                let density = sum(&|_| true);
                return Some(probed(Sample {
                    density: Density::new(density),
                    state,
                    velocity,
                }));
            }
        }
        let thin = sum(&volume);
        match nearest(&volume) {
            Some(state) if thin > 0.0 => Some(probed(Sample {
                density: Density::new(thin),
                state,
                velocity,
            })),
            _ => Some(probed(vacuum)),
        }
    }

    /// The distance from the frame origin along the unit vector `dir`
    /// (frame axes) to the outermost point that is not vacuum, meters, or
    /// `None` if the line meets no loaded matter. Marches inward from the
    /// corner distance of the frame's cube in steps of half the answering
    /// cell's sample edge, then bisects the last step.
    pub fn outermost_along(&self, dir: Vec3) -> Option<f64> {
        let solid = |r: f64| {
            self.lookup(dir.scale(r))
                .map(|p| (p.sample.state != State::Vacuum, p.sample_edge))
        };
        let coarse = self.root_extent / 64.0;
        let mut r = self.root_extent * 0.5 * 3f64.sqrt();
        let mut outside = None;
        for _ in 0..MAX_MARCH_STEPS {
            if r <= 0.0 {
                return None;
            }
            match solid(r) {
                Some((true, _)) => {
                    let Some(mut hi) = outside else {
                        return Some(r);
                    };
                    let mut lo = r;
                    for _ in 0..MARCH_HALVINGS {
                        let mid = 0.5 * (lo + hi);
                        if solid(mid).is_some_and(|(m, _)| m) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    return Some(lo);
                }
                Some((false, edge)) => {
                    outside = Some(r);
                    r -= 0.5 * edge;
                }
                None => {
                    outside = Some(r);
                    r -= coarse;
                }
            }
        }
        None
    }
}

impl MatterProbe for CacheProbe<'_> {
    fn sample(&self, point_in_frame: Vec3) -> Option<Sample> {
        self.lookup(point_in_frame).map(|p| p.sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{decode_cell, FetchOutcome};
    use gx_core::matter::{encode, Compression, Sample as Matter, Samples};
    use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
    use gx_core::units::{Attenuation, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds};

    const EXTENT: f64 = 16.0;

    fn system(max_depth: u8) -> FrameSystem {
        let frame = Frame {
            frame_id: 1,
            parent_frame_id: ROOT_PARENT,
            root_extent: Meters::new(EXTENT),
            max_depth,
            mass: Kilograms::new(0.0),
            position: Vec3::zero(),
            velocity: Vec3::zero(),
            orientation: Quat::identity(),
            angular_velocity: Vec3::new(0.0, 0.0, 0.5),
            j2: 0.0,
            reference_radius: Meters::new(0.0),
            principal_moments: Vec3::zero(),
        };
        let r = Registry::new(Seconds::new(0.0), vec![frame]).unwrap();
        FrameSystem::from_tree(FrameTree::from_registries(&[r]).unwrap())
    }

    fn matter(density: f64, state: State) -> Matter {
        if density == 0.0 {
            return Matter::VACUUM;
        }
        Matter {
            density: Density::new(density),
            state,
            temperature: Kelvin::new(300.0),
            albedo: [Ratio::new(0.5); 3],
            roughness: Ratio::new(0.5),
            attenuation: Attenuation::new(0.0),
            radiating_temperature: Kelvin::new(300.0),
        }
    }

    /// Loads `key` with samples from `f` of the sample center.
    fn load(cache: &mut CellCache, key: CellKey, res: u8, f: impl Fn(Vec3) -> Matter) {
        let g = key.geometry(Meters::new(EXTENT));
        let step = g.edge.value() / f64::from(res);
        let samples = Samples::from_fn(res, |x, y, z| {
            let c = |o: f64, i: u32| o + (f64::from(i) + 0.5) * step;
            f(Vec3::new(
                c(g.origin.x, x),
                c(g.origin.y, y),
                c(g.origin.z, z),
            ))
        });
        let s = Section::new(key, g.origin, g.edge, res, samples).unwrap();
        let bytes = encode(&s, Compression::None);
        let out = decode_cell(&key, &[bytes], Meters::new(EXTENT));
        cache.complete(key, out, 0.0);
    }

    #[test]
    fn the_finest_loaded_depth_answers_with_the_frame_rotation() {
        let sys = system(1);
        let mut cache = CellCache::new();
        let probe = CacheProbe::new(&cache, &sys, 1).unwrap();
        assert_eq!(probe.lookup(Vec3::zero()), None);
        // Depth 0: solid below z = 0. Depth 1 at the top half: vacuum.
        load(&mut cache, CellKey::new(1, 0, 0, 0, 0).unwrap(), 8, |c| {
            matter(if c.z < 0.0 { 2000.0 } else { 0.0 }, State::Solid)
        });
        cache.complete(
            CellKey::new(1, 1, 1, 1, 1).unwrap(),
            FetchOutcome::Decoded(None),
            0.0,
        );
        let probe = CacheProbe::new(&cache, &sys, 1).unwrap();
        let low = probe.lookup(Vec3::new(1.0, 2.0, -3.0)).unwrap();
        assert_eq!(low.depth, 0);
        assert_eq!(low.sample.state, State::Solid);
        assert_eq!(low.sample.density.value(), 2000.0);
        assert_eq!(low.sample.velocity, Vec3::new(-1.0, 0.5, 0.0));
        let high = probe.lookup(Vec3::new(1.0, 1.0, 1.0)).unwrap();
        assert_eq!(high.depth, 1);
        assert_eq!(high.sample.state, State::Vacuum);
        // The trilinear surface sits halfway between the sample centers
        // either side of z = 0 (at -1 and +1).
        let edge = probe.lookup(Vec3::new(-3.0, -3.0, -0.01)).unwrap();
        assert_eq!(edge.sample.state, State::Solid);
        let above = probe.lookup(Vec3::new(-3.0, -3.0, 0.01)).unwrap();
        assert_eq!(above.sample.state, State::Vacuum);
        // Straight down the matter reaches the face of the cube; straight
        // up the line meets none.
        let r = probe.outermost_along(Vec3::new(0.0, 0.0, -1.0)).unwrap();
        assert!((r - 8.0).abs() < 1e-9, "{r}");
        assert_eq!(probe.outermost_along(Vec3::new(0.0, 0.0, 1.0)), None);
    }

    #[test]
    fn gas_above_a_solid_keeps_its_own_density() {
        let sys = system(0);
        let mut cache = CellCache::new();
        load(&mut cache, CellKey::new(1, 0, 0, 0, 0).unwrap(), 8, |c| {
            if c.z < 0.0 {
                matter(3000.0, State::Solid)
            } else {
                matter(1.0, State::Gas)
            }
        });
        let probe = CacheProbe::new(&cache, &sys, 1).unwrap();
        let s = probe.lookup(Vec3::new(0.0, 0.0, 0.5)).unwrap().sample;
        assert_eq!(s.state, State::Gas);
        // Only the gas corners count, at their weights.
        assert!((s.density.value() - 0.75).abs() < 1e-12, "{s:?}");
        assert_eq!(
            probe
                .lookup(Vec3::new(0.0, 0.0, -0.5))
                .unwrap()
                .sample
                .state,
            State::Solid
        );
    }

    #[test]
    fn feet_cells_cover_every_depth_and_hold_until_loaded() {
        let sys = system(2);
        let p = Vec3::new(0.1, 0.1, 0.1);
        let keys = feet_cells(&sys, 1, p);
        // One at depth 0, eight at depths 1 and 2.
        assert_eq!(keys.len(), 1 + 8 + 8);
        let mut cache = CellCache::new();
        assert!(waiting_for_matter(&cache, &sys, 1, p));
        for k in keys.iter().filter(|k| k.depth == 2) {
            cache.complete(*k, FetchOutcome::Decoded(None), 0.0);
        }
        assert!(!waiting_for_matter(&cache, &sys, 1, p));
        assert!(!waiting_for_matter(
            &cache,
            &sys,
            1,
            Vec3::new(100.0, 0.0, 0.0)
        ));
    }
}
