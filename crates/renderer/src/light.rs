//! Lights from hot matter, and the emission lookup for hot surfaces.
//!
//! Implements the renderer's step "derives emission from temperature" of
//! `space-model.md` section 2 with the emission derived quantity of
//! `matter-format.md` section 3.3 (blackbody radiance from temperature,
//! scaled by `1 - albedo` per band). Nothing here knows what is hot: any
//! sample at or above [`MIN_EMITTER_TEMPERATURE`] radiates.
//!
//! # From cells to lights
//!
//! 1. Every drawn `Ready` cell is reduced to one emitter by
//!    [`gx_core::emission::summarize`] with `min_temperature` 1000 K
//!    ([`cell_emitter`]).
//! 2. The emitters of one frame are merged ([`merge_frame_emitters`]): the
//!    band powers add, and the position is the centroid weighted by total
//!    power.
//! 3. The eight strongest merged emitters in the scene become point lights
//!    ([`strongest_lights`]) with radiant intensity `band_power / (4 pi)` per
//!    band, in watts per steradian.
//!
//! A surface point at distance `d` with unit normal `n` and unit direction
//! `l` toward a light of intensity `I` receives the irradiance
//! `E = I / d^2 * max(0, n . l)` per band ([`irradiance`]). The distance is
//! computed in `f64` camera-relative space on the CPU, and the light
//! position reaches the shader as an `f32` offset from the camera. There are
//! no shadows in v1.
//!
//! # Emission of a surface
//!
//! A hot surface glows whether or not any light reaches it: its own
//! `emitted_band_radiance(temperature, albedo)` is added to the reflected
//! light. `gx-core` computes radiance in `f64` on the CPU, so the renderer
//! evaluates [`gx_core::radiance::band_radiance`] once per 1 K temperature
//! bucket that appears in a drawn mesh ([`EmissionTable`]), uploads the
//! values as a small lookup texture, and the shader multiplies by
//! `1 - albedo`.

use crate::camera::Camera;
use gx_core::emission::{summarize, Emitter};
use gx_core::frames::FrameSystem;
use gx_core::matter::Section;
use gx_core::radiance::band_radiance;
use gx_core::units::{Kelvin, Vec3};
use std::collections::BTreeMap;

/// Samples at or above this temperature, kelvin, become light sources.
pub const MIN_EMITTER_TEMPERATURE: f64 = 1000.0;

/// Most point lights in a scene.
pub const MAX_LIGHTS: usize = 8;

/// The emitter of one cell's composited section, or `None` when nothing in
/// it is hot enough.
pub fn cell_emitter(section: &Section) -> Option<Emitter> {
    summarize(section, Kelvin::new(MIN_EMITTER_TEMPERATURE))
}

/// The merged emitters of one frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrameEmitter {
    /// The frame the emitting matter is in.
    pub frame_id: u64,
    /// Power-weighted centroid, meters, frame coordinates.
    pub position: Vec3,
    /// Power per band, watts, long wavelength first.
    pub band_power: [f64; 3],
}

impl FrameEmitter {
    /// Power over all three bands, watts.
    pub fn total_power(&self) -> f64 {
        self.band_power[0] + self.band_power[1] + self.band_power[2]
    }
}

/// Merges cell emitters per frame: band powers add, the position is the
/// centroid weighted by total power. Inputs are summed in the order given;
/// the output is in ascending frame id. A frame whose emitters carry no
/// power at all gives no output.
pub fn merge_frame_emitters<'a>(
    cells: impl IntoIterator<Item = (u64, &'a Emitter)>,
) -> Vec<FrameEmitter> {
    struct Acc {
        band_power: [f64; 3],
        weight: f64,
        weighted: Vec3,
    }
    let mut frames: BTreeMap<u64, Acc> = BTreeMap::new();
    for (frame_id, e) in cells {
        let acc = frames.entry(frame_id).or_insert(Acc {
            band_power: [0.0; 3],
            weight: 0.0,
            weighted: Vec3::zero(),
        });
        let w = e.band_power[0] + e.band_power[1] + e.band_power[2];
        for b in 0..3 {
            acc.band_power[b] += e.band_power[b];
        }
        acc.weight += w;
        acc.weighted = acc.weighted + e.position.scale(w);
    }
    frames
        .into_iter()
        .filter(|(_, a)| a.weight > 0.0)
        .map(|(frame_id, a)| FrameEmitter {
            frame_id,
            position: a.weighted.scale(1.0 / a.weight),
            band_power: a.band_power,
        })
        .collect()
}

/// A point light ready for the GPU.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PointLight {
    /// The frame the light's matter is in.
    pub frame_id: u64,
    /// Light position relative to the camera, root axes, meters: the
    /// `f64` offset cast to `f32`.
    pub position: [f32; 3],
    /// Radiant intensity per band, watts per steradian.
    pub intensity: [f32; 3],
}

/// Radiant intensity per band of an isotropic emitter of `band_power`
/// watts: `band_power / (4 pi)` watts per steradian.
pub fn radiant_intensity(band_power: [f64; 3]) -> [f64; 3] {
    band_power.map(|p| p / (4.0 * core::f64::consts::PI))
}

/// The [`MAX_LIGHTS`] strongest emitters as point lights relative to the
/// camera, strongest first; equal powers go to the lower frame id.
pub fn strongest_lights(
    emitters: &[FrameEmitter],
    system: &FrameSystem,
    camera: &Camera,
) -> Vec<PointLight> {
    let mut order: Vec<&FrameEmitter> = emitters.iter().collect();
    order.sort_by(|a, b| {
        b.total_power()
            .total_cmp(&a.total_power())
            .then(a.frame_id.cmp(&b.frame_id))
    });
    order
        .into_iter()
        .take(MAX_LIGHTS)
        .map(|e| {
            let rel = camera.relative(system, e.position, e.frame_id);
            PointLight {
                frame_id: e.frame_id,
                position: [rel.x as f32, rel.y as f32, rel.z as f32],
                intensity: radiant_intensity(e.band_power).map(|v| v as f32),
            }
        })
        .collect()
}

/// Irradiance per band, watts per square meter, at a surface point with
/// unit `normal` from a light of `intensity` watts per steradian per band:
/// `I / d^2 * max(0, n . l)`. Points and light share any one origin; the
/// renderer uses the camera. The shader evaluates the same formula in
/// `f32`.
pub fn irradiance(intensity: [f64; 3], light: Vec3, point: Vec3, normal: Vec3) -> [f64; 3] {
    let to_light = light - point;
    let d2 = to_light.length_squared();
    if d2 <= 0.0 {
        return [0.0; 3];
    }
    let cos = (normal.dot(to_light) / d2.sqrt()).max(0.0);
    intensity.map(|i| i / d2 * cos)
}

/// The 1 K bucket of a temperature: the nearest whole kelvin.
pub fn temperature_bucket(temperature: f64) -> u32 {
    let t = temperature.max(0.0) + 0.5;
    if t >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        t as u32
    }
}

/// Width of the emission lookup texture, texels. Entry `i` sits at
/// `(i % width, i / width)`.
pub const EMISSION_TABLE_WIDTH: u32 = 1024;

/// Blackbody band radiance per 1 K temperature bucket, in the order the
/// buckets were first seen, for the GPU lookup texture.
///
/// Append only: an index handed out stays valid for the life of the table,
/// so uploaded meshes never need rewriting.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmissionTable {
    index: BTreeMap<u32, u32>,
    radiance: Vec<[f32; 4]>,
}

impl EmissionTable {
    /// An empty table.
    pub fn new() -> EmissionTable {
        EmissionTable::default()
    }

    /// The table index for a temperature, adding its bucket if new.
    pub fn index_for(&mut self, temperature: f64) -> u32 {
        let bucket = temperature_bucket(temperature);
        if let Some(&i) = self.index.get(&bucket) {
            return i;
        }
        let i = self.radiance.len() as u32;
        let b = band_radiance(Kelvin::new(f64::from(bucket)));
        self.radiance
            .push([b[0] as f32, b[1] as f32, b[2] as f32, 0.0]);
        self.index.insert(bucket, i);
        i
    }

    /// Number of buckets.
    pub fn len(&self) -> usize {
        self.radiance.len()
    }

    /// Returns `true` when no bucket has been added.
    pub fn is_empty(&self) -> bool {
        self.radiance.is_empty()
    }

    /// Band radiance per entry, W m^-2 sr^-1, long wavelength first, the
    /// fourth channel 0.
    pub fn texels(&self) -> &[[f32; 4]] {
        &self.radiance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;
    use gx_core::key::CellKey;
    use gx_core::matter::{Sample, Samples, State};
    use gx_core::radiance::emitted_band_radiance;
    use gx_core::units::{Attenuation, Density, Meters, Ratio};

    fn hot(t: f64) -> Sample {
        Sample {
            density: Density::new(1.0e-3),
            state: State::Plasma,
            temperature: Kelvin::new(t),
            albedo: [Ratio::new(0.1), Ratio::new(0.2), Ratio::new(0.3)],
            roughness: Ratio::new(1.0),
            attenuation: Attenuation::new(0.0),
        }
    }

    fn section(frame: u64) -> Section {
        Section::new(
            CellKey::new(frame, 0, 0, 0, 0).unwrap(),
            Vec3::new(-4.0, -4.0, -4.0),
            Meters::new(8.0),
            4,
            Samples::from_fn(4, |x, y, z| match (x, y, z) {
                (1, 1, 1) => hot(6000.0),
                (2, 1, 1) => hot(4000.0),
                (3, 3, 3) => hot(500.0),
                _ => Sample::VACUUM,
            }),
        )
        .unwrap()
    }

    #[test]
    fn hot_samples_give_one_emitter_with_their_summed_power() {
        let s = section(3);
        let e = cell_emitter(&s).expect("hot samples emit");
        // Sum over the qualifying samples as gx-core defines it: emitted
        // band radiance times pi times six faces of edge e / n.
        let face = 6.0 * 2.0 * 2.0;
        let mut want = [0.0; 3];
        for t in [6000.0, 4000.0] {
            // The section stores albedo as f32, as on the wire.
            let albedo = [0.1f32, 0.2, 0.3].map(|a| Ratio::new(f64::from(a)));
            let l = emitted_band_radiance(Kelvin::new(t), albedo);
            for b in 0..3 {
                want[b] += l[b] * PI * face;
            }
        }
        for (got, want) in e.band_power.iter().zip(want) {
            assert!((got - want).abs() <= 1e-9 * want, "{got} vs {want}");
        }
        // Between the two hot centers, nearer the hotter one.
        assert!(e.position.x > -1.0 && e.position.x < 0.0);

        let merged = merge_frame_emitters([(3, &e)]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].band_power, e.band_power);
    }

    #[test]
    fn merging_sums_power_and_weights_position() {
        let a = Emitter {
            position: Vec3::new(0.0, 0.0, 0.0),
            band_power: [1.0, 1.0, 1.0],
            radius: Meters::new(0.0),
        };
        let b = Emitter {
            position: Vec3::new(4.0, 0.0, 0.0),
            band_power: [3.0, 3.0, 3.0],
            radius: Meters::new(0.0),
        };
        let m = merge_frame_emitters([(7, &a), (9, &a), (7, &b)]);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].frame_id, 7);
        assert_eq!(m[0].band_power, [4.0; 3]);
        assert_eq!(m[0].position, Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(m[1].frame_id, 9);
    }

    #[test]
    fn irradiance_falls_with_square_and_cosine() {
        let i = [4.0, 8.0, 12.0];
        let n = Vec3::new(0.0, 0.0, 1.0);
        let e = irradiance(i, Vec3::new(0.0, 0.0, 2.0), Vec3::zero(), n);
        assert_eq!(e, [1.0, 2.0, 3.0]);
        let back = irradiance(i, Vec3::new(0.0, 0.0, -2.0), Vec3::zero(), n);
        assert_eq!(back, [0.0; 3]);
        let slant = irradiance([1.0; 3], Vec3::new(1.0, 0.0, 1.0), Vec3::zero(), n);
        assert!((slant[0] - 0.5 / 2.0f64.sqrt()).abs() < 1e-15);
        assert_eq!(radiant_intensity([4.0 * PI; 3]), [1.0; 3]);
    }

    #[test]
    fn emission_table_buckets_by_kelvin() {
        let mut t = EmissionTable::new();
        assert_eq!(t.index_for(5999.6), 0);
        assert_eq!(t.index_for(6000.4), 0);
        assert_eq!(t.index_for(300.0), 1);
        assert_eq!(t.index_for(6000.0), 0);
        assert_eq!(t.len(), 2);
        let b = band_radiance(Kelvin::new(6000.0));
        assert_eq!(t.texels()[0][0], b[0] as f32);
        assert_eq!(temperature_bucket(-5.0), 0);
        assert_eq!(temperature_bucket(f64::INFINITY), u32::MAX);
    }
}
