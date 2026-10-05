//! Volumetric matter: gas and plasma drawn by ray marching the density
//! field.
//!
//! Implements the renderer's step "extracts surfaces or ray marches the
//! field" of `space-model.md` section 2 for matter with no hard surface,
//! with the derived quantities of `matter-format.md` section 3.3: emission
//! is blackbody radiance from `temperature` scaled by `1 - albedo` per band,
//! and extinction per meter is `density * attenuation`. The extinction model
//! is exactly the one of `gx_core::extinction`: Beer-Lambert transmittance
//! `exp(-k L)` with `k` looked up from the nearest sample, the lookup
//! `gx_core::extinction::optical_depth_along` uses.
//!
//! # Which samples
//!
//! Gas and plasma samples form volumes ([`forms_volume`]); solid and fluid
//! samples form surfaces ([`crate::extract::forms_surface`]). A composited
//! section whose densest sample is gas or plasma is therefore drawn as a
//! volume, and a cell that holds both kinds draws both: the mesh from the
//! solid and fluid samples and the volume from the rest. [`VolumeGrid`]
//! holds the section's density, temperature, albedo, and attenuation with
//! every solid and fluid sample set to vacuum.
//!
//! # The march
//!
//! For every pixel the cell covers, the ray from the camera is clipped to
//! the box around the cell's gas and plasma samples ([`VolumeGrid::bounds_min`],
//! [`VolumeGrid::bounds_max`]; everything outside it is vacuum and adds
//! nothing) and to the opaque surface depth, and the remaining segment is
//! split into [`VOLUME_STEPS`] equal steps of length `ds`. At the midpoint of
//! each step, front to back, with `k = density * attenuation` of the sample
//! containing the point and `E` its emitted band radiance:
//!
//! ```text
//! L += T * E * k * ds
//! T *= exp(-k * ds)
//! ```
//!
//! starting from `L = 0`, `T = 1`. The pixel becomes `L + T * behind`: the
//! volume's own glow plus what is behind it, dimmed by its transmittance.
//! For uniform matter the sum tends to `E * (1 - exp(-k D))` over a path of
//! length `D`, the analytic result ([`uniform_radiance`]). The GPU version is
//! `shaders/volume.wgsl`; [`VolumeGrid::march`] is the same loop on the CPU
//! in `f64`, used as the reference in tests.
//!
//! Emission comes from the same 1 K temperature buckets as surfaces
//! ([`crate::light::EmissionTable`]), so the shader looks band radiance up in
//! the emission table by index and multiplies by `1 - albedo`.

use crate::extract::forms_surface;
use crate::light::temperature_bucket;
use gx_core::extinction::transmittance;
use gx_core::key::CellKey;
use gx_core::matter::{Section, State};
use gx_core::radiance::band_radiance;
use gx_core::units::{Attenuation, Density, Kelvin, Meters, Vec3};

/// Steps of the march through one cell, whatever its size.
pub const VOLUME_STEPS: u32 = 64;

/// Returns `true` for the states drawn as a volume: gas and plasma.
pub fn forms_volume(state: State) -> bool {
    matches!(state, State::Gas | State::Plasma)
}

/// The gas and plasma of one composited section, ready to upload as 3D
/// textures and to march.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeGrid {
    /// The cell.
    pub key: CellKey,
    /// Minimum corner of the cell, meters, frame coordinates.
    pub origin: Vec3,
    /// Cell edge, meters.
    pub edge: f64,
    /// Samples per axis, 1 to 64.
    pub resolution: u8,
    /// Density per sample, kilograms per cubic meter, 0 for samples that
    /// are not gas or plasma. Index `x + n * (y + n * z)`.
    pub density: Vec<f32>,
    /// Temperature per sample, kelvin.
    pub temperature: Vec<f32>,
    /// Albedo per sample, three bands, long wavelength first.
    pub albedo: Vec<[f32; 3]>,
    /// Mass attenuation coefficient per sample, square meters per kilogram.
    pub attenuation: Vec<f32>,
    /// Smallest corner of the box around every gas or plasma sample,
    /// meters, frame coordinates.
    pub bounds_min: [f64; 3],
    /// Largest corner of that box, meters, frame coordinates.
    pub bounds_max: [f64; 3],
}

/// The result of marching one ray.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct March {
    /// Radiance the volume adds along the ray, W m^-2 sr^-1 per band.
    pub radiance: [f64; 3],
    /// Fraction of the light from behind the volume that passes it.
    pub transmittance: f64,
}

impl VolumeGrid {
    /// The gas and plasma of `section`, or `None` when it holds none.
    pub fn from_section(section: &Section) -> Option<VolumeGrid> {
        let samples = section.samples()?;
        let n = usize::from(section.resolution());
        let count = n * n * n;
        let edge = section.edge().value();
        let step = edge / n as f64;
        let origin = section.origin();
        let o = [origin.x, origin.y, origin.z];
        let mut grid = VolumeGrid {
            key: section.key(),
            origin,
            edge,
            resolution: section.resolution(),
            density: vec![0.0; count],
            temperature: vec![0.0; count],
            albedo: vec![[0.0; 3]; count],
            attenuation: vec![0.0; count],
            bounds_min: [f64::INFINITY; 3],
            bounds_max: [f64::NEG_INFINITY; 3],
        };
        let mut any = false;
        for (i, s) in samples.iter().enumerate() {
            if !forms_volume(s.state) {
                continue;
            }
            any = true;
            grid.density[i] = s.density.value() as f32;
            grid.temperature[i] = s.temperature.value() as f32;
            grid.albedo[i] = s.albedo.map(|a| a.value() as f32);
            grid.attenuation[i] = s.attenuation.value() as f32;
            let idx = [i % n, (i / n) % n, i / (n * n)];
            for a in 0..3 {
                let lo = o[a] + idx[a] as f64 * step;
                grid.bounds_min[a] = grid.bounds_min[a].min(lo);
                grid.bounds_max[a] = grid.bounds_max[a].max(lo + step);
            }
        }
        any.then_some(grid)
    }

    /// Samples per axis as a `usize`.
    pub fn n(&self) -> usize {
        usize::from(self.resolution)
    }

    /// Distance from `point` (frame coordinates) to the box around the gas
    /// and plasma samples, 0 inside.
    pub fn distance_to(&self, point: Vec3) -> f64 {
        let p = [point.x, point.y, point.z];
        let gap = |a: usize| {
            (self.bounds_min[a] - p[a])
                .max(p[a] - self.bounds_max[a])
                .max(0.0)
        };
        Vec3::new(gap(0), gap(1), gap(2)).length()
    }

    /// The two 3D textures of the GPU, one texel per sample in sample
    /// order: `(density, attenuation, emission index, temperature)` and
    /// `(albedo, 0)`. `emission_index` turns a temperature into its
    /// emission table index ([`crate::light::EmissionTable::index_for`]);
    /// it is asked only for samples with matter.
    pub fn texels(
        &self,
        mut emission_index: impl FnMut(f64) -> u32,
    ) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
        let mut params = Vec::with_capacity(self.density.len());
        let mut albedo = Vec::with_capacity(self.density.len());
        for i in 0..self.density.len() {
            let index = if self.density[i] > 0.0 {
                emission_index(f64::from(self.temperature[i])) as f32
            } else {
                0.0
            };
            params.push([
                self.density[i],
                self.attenuation[i],
                index,
                self.temperature[i],
            ]);
            let a = self.albedo[i];
            albedo.push([a[0], a[1], a[2], 0.0]);
        }
        (params, albedo)
    }

    /// The index of the sample containing `p` (frame coordinates), clamped
    /// to the grid on the far faces.
    fn sample_at(&self, p: Vec3) -> usize {
        let n = self.n();
        let step = self.edge / n as f64;
        let idx = |c: f64, o: f64| (((c - o) / step).max(0.0) as usize).min(n - 1);
        idx(p.x, self.origin.x) + n * (idx(p.y, self.origin.y) + n * idx(p.z, self.origin.z))
    }

    /// Marches the ray from `eye` along `dir` (frame coordinates and axes,
    /// `dir` need not be unit), clipped to the box around the gas and plasma
    /// samples and to `max_distance` meters from `eye` along the ray, in
    /// `steps` steps: the CPU mirror of `shaders/volume.wgsl` (see the module
    /// docs). Emission uses the 1 K temperature buckets of the emission
    /// table.
    pub fn march(&self, eye: Vec3, dir: Vec3, max_distance: f64, steps: u32) -> March {
        let clear = March {
            radiance: [0.0; 3],
            transmittance: 1.0,
        };
        let Some(dir) = dir.normalized() else {
            return clear;
        };
        let Some((t0, t1)) = ray_box(eye, dir, self.bounds_min, self.bounds_max) else {
            return clear;
        };
        let (t0, t1) = (t0.max(0.0), t1.min(max_distance));
        if t1 <= t0 || steps == 0 {
            return clear;
        }
        let ds = (t1 - t0) / f64::from(steps);
        let mut radiance = [0.0; 3];
        let mut t = 1.0;
        for i in 0..steps {
            let p = eye + dir.scale(t0 + (f64::from(i) + 0.5) * ds);
            let s = self.sample_at(p);
            let k = f64::from(self.density[s]) * f64::from(self.attenuation[s]);
            if k <= 0.0 {
                continue;
            }
            let bucket = f64::from(temperature_bucket(f64::from(self.temperature[s])));
            let band = band_radiance(Kelvin::new(bucket));
            for b in 0..3 {
                let e = band[b] * (1.0 - f64::from(self.albedo[s][b]));
                radiance[b] += t * e * k * ds;
            }
            t *= (-k * ds).exp();
        }
        March {
            radiance,
            transmittance: t,
        }
    }
}

/// The entry and exit distances of the ray from `origin` along unit `dir`
/// through the box `[lo, hi]`, or `None` when it misses. The entry may be
/// negative when `origin` is inside.
pub fn ray_box(origin: Vec3, dir: Vec3, lo: [f64; 3], hi: [f64; 3]) -> Option<(f64, f64)> {
    let o = [origin.x, origin.y, origin.z];
    let d = [dir.x, dir.y, dir.z];
    let mut t0 = f64::NEG_INFINITY;
    let mut t1 = f64::INFINITY;
    for a in 0..3 {
        if d[a] == 0.0 {
            if o[a] < lo[a] || o[a] > hi[a] {
                return None;
            }
            continue;
        }
        let (u, v) = ((lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]);
        t0 = t0.max(u.min(v));
        t1 = t1.min(u.max(v));
    }
    (t1 > t0).then_some((t0, t1))
}

/// Radiance per band leaving a path of `path` meters through uniform
/// matter of emitted band radiance `emitted`: `E * (1 - T)` with `T` from
/// [`gx_core::extinction::transmittance`]. The analytic limit of the march.
pub fn uniform_radiance(
    emitted: [f64; 3],
    density: Density,
    attenuation: Attenuation,
    path: Meters,
) -> [f64; 3] {
    let t = transmittance(density, attenuation, path);
    emitted.map(|e| e * (1.0 - t))
}

/// The unit cube as 12 triangles, counterclockwise seen from outside. The
/// volume pass scales it to a cell's gas box and draws its back faces, so
/// every covered pixel gets exactly one fragment, camera inside or out.
pub const CUBE_TRIANGLES: [[f32; 3]; 36] = {
    // Corner c sits at (c & 1, (c >> 1) & 1, (c >> 2) & 1).
    const C: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    // Each face as a quad, counterclockwise from outside.
    const FACES: [[usize; 4]; 6] = [
        [1, 3, 7, 5], // +x
        [0, 4, 6, 2], // -x
        [2, 6, 7, 3], // +y
        [0, 1, 5, 4], // -y
        [4, 5, 7, 6], // +z
        [0, 2, 3, 1], // -z
    ];
    let mut out = [[0.0f32; 3]; 36];
    let mut f = 0;
    while f < 6 {
        let q = FACES[f];
        let tri = [q[0], q[1], q[2], q[0], q[2], q[3]];
        let mut k = 0;
        while k < 6 {
            out[f * 6 + k] = C[tri[k]];
            k += 1;
        }
        f += 1;
    }
    out
};

/// Returns `true` when `section` would be drawn as a volume only: its
/// densest sample is gas or plasma and no sample forms a surface.
pub fn volume_only(section: &Section) -> bool {
    section.samples().is_some_and(|s| {
        let mut densest = State::Vacuum;
        let mut max = 0.0;
        let mut surface = false;
        for x in s.iter() {
            surface |= forms_surface(x.state);
            if x.density.value() > max {
                max = x.density.value();
                densest = x.state;
            }
        }
        forms_volume(densest) && !surface
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::matter::{Sample, Samples};
    use gx_core::radiance::emitted_band_radiance;
    use gx_core::units::Ratio;

    fn plasma(density: f64, attenuation: f64) -> Sample {
        Sample {
            density: Density::new(density),
            state: State::Plasma,
            temperature: Kelvin::new(6000.0),
            albedo: [Ratio::new(0.25); 3],
            roughness: Ratio::new(0.0),
            attenuation: Attenuation::new(attenuation),
        }
    }

    fn section(res: u8, f: impl FnMut(u32, u32, u32) -> Sample) -> Section {
        Section::new(
            CellKey::new(3, 0, 0, 0, 0).unwrap(),
            Vec3::new(-4.0, -4.0, -4.0),
            Meters::new(8.0),
            res,
            Samples::from_fn(res, f),
        )
        .unwrap()
    }

    #[test]
    fn uniform_march_tends_to_the_analytic_result() {
        // k = 0.125 / m over 8 m: optical depth 1.
        let s = section(4, |_, _, _| plasma(0.5, 0.25));
        let grid = VolumeGrid::from_section(&s).unwrap();
        assert_eq!(grid.bounds_min, [-4.0; 3]);
        assert_eq!(grid.bounds_max, [4.0; 3]);
        let eye = Vec3::new(0.0, 0.0, 20.0);
        let m = grid.march(eye, Vec3::new(0.0, 0.0, -1.0), f64::INFINITY, VOLUME_STEPS);
        let albedo = [Ratio::new(f64::from(0.25f32)); 3];
        let e = emitted_band_radiance(Kelvin::new(6000.0), albedo);
        let want = uniform_radiance(
            e,
            Density::new(0.5),
            Attenuation::new(0.25),
            Meters::new(8.0),
        );
        let t = transmittance(Density::new(0.5), Attenuation::new(0.25), Meters::new(8.0));
        assert!(
            (m.transmittance - t).abs() < 1e-12,
            "{} vs {t}",
            m.transmittance
        );
        for (b, (got, want)) in m.radiance.iter().zip(want).enumerate() {
            // The midpoint sum overshoots by about k ds / 2 = 0.8 percent.
            let rel = (got - want) / want;
            assert!(rel > 0.0 && rel < 0.01, "band {b}: {rel}");
        }
        // Clipped to 2 m inside the cell (16 m from the eye): a shorter path.
        let short = grid.march(eye, Vec3::new(0.0, 0.0, -1.0), 18.0, VOLUME_STEPS);
        assert!((short.transmittance - (-0.25f64).exp()).abs() < 1e-12);
        // A ray that misses the box sees nothing.
        let miss = grid.march(eye, Vec3::new(1.0, 0.0, 0.0), f64::INFINITY, VOLUME_STEPS);
        assert_eq!(miss.radiance, [0.0; 3]);
        assert_eq!(miss.transmittance, 1.0);
    }

    #[test]
    fn only_gas_and_plasma_enter_the_grid() {
        let s = section(4, |x, _, _| match x {
            0 => Sample {
                state: State::Solid,
                ..plasma(3000.0, 0.0)
            },
            1 => Sample {
                state: State::Gas,
                ..plasma(1.0, 1.0)
            },
            _ => Sample::VACUUM,
        });
        let g = VolumeGrid::from_section(&s).unwrap();
        assert_eq!(g.density[0], 0.0);
        assert_eq!(g.density[1], 1.0);
        // The box is the x = 1 slab of samples.
        assert_eq!(g.bounds_min, [-2.0, -4.0, -4.0]);
        assert_eq!(g.bounds_max, [0.0, 4.0, 4.0]);
        assert!(!volume_only(&s));
        let (params, albedo) = g.texels(|t| t as u32);
        assert_eq!(params[1], [1.0, 1.0, 6000.0, 6000.0]);
        assert_eq!(params[0], [0.0; 4]);
        assert_eq!(albedo[1], [0.25, 0.25, 0.25, 0.0]);

        let solid_only = section(2, |_, _, _| Sample {
            state: State::Solid,
            ..plasma(1.0, 0.0)
        });
        assert!(VolumeGrid::from_section(&solid_only).is_none());
        assert!(volume_only(&section(2, |_, _, _| plasma(1.0, 1.0))));
    }

    #[test]
    fn ray_box_entry_and_exit() {
        let lo = [0.0; 3];
        let hi = [1.0; 3];
        let (a, b) = ray_box(Vec3::new(0.5, 0.5, -1.0), Vec3::new(0.0, 0.0, 1.0), lo, hi).unwrap();
        assert_eq!((a, b), (1.0, 2.0));
        let (a, b) = ray_box(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0), lo, hi).unwrap();
        assert_eq!((a, b), (-0.5, 0.5));
        assert!(ray_box(Vec3::new(2.0, 0.5, -1.0), Vec3::new(0.0, 0.0, 1.0), lo, hi).is_none());
    }

    #[test]
    fn cube_triangles_face_outward() {
        for tri in CUBE_TRIANGLES.chunks(3) {
            let p = |i: usize| {
                Vec3::new(
                    f64::from(tri[i][0]),
                    f64::from(tri[i][1]),
                    f64::from(tri[i][2]),
                )
            };
            let (a, b, c) = (p(0), p(1), p(2));
            let normal = (b - a).cross(c - a);
            let centroid = (a + b + c).scale(1.0 / 3.0) - Vec3::new(0.5, 0.5, 0.5);
            assert!(normal.dot(centroid) > 0.0, "{tri:?}");
        }
    }
}
