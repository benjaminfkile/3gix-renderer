//! The far field: frames too small on screen for cells, drawn as one point
//! sprite each.
//!
//! The renderer "renders whatever matter is near the camera and lets the
//! camera go anywhere" (`space-model.md` sections 1 and 2). Matter far away
//! still has to show up: a hot frame must be visible from anywhere in the
//! system, and a cold one should be a faint dot where the lights make it
//! bright. Cells cannot do that once a frame covers a pixel or two, so this
//! module replaces them with a point sprite of the right brightness, worked
//! out from the frame's depth-0 cell with the emission and albedo of
//! `matter-format.md` section 3.3.
//!
//! # Which frames
//!
//! Only frames with mass above 0 take part. A frame's size on screen is the
//! projected size of its `root_extent / 8` region ([`region_projected_px`]):
//!
//! | Projected size | Drawn as |
//! |---|---|
//! | under 2 px | a sprite only; its cells are not selected |
//! | 2 to 8 px | the sprite fading out while the cells fade in |
//! | 8 px and up | cells only |
//!
//! In the transition the sprite is weighted by `(8 - px) / 6` and the
//! frame's cells by the rest ([`sprite_weight`]).
//!
//! # Brightness
//!
//! The depth-0 cell of every frame that may need a sprite is fetched once
//! for this purpose and kept by the cell cache (it is pinned and never
//! evicted). From it ([`FarFieldCell`]):
//!
//! - a hot frame (its depth-0 cell has an emitter, see
//!   [`crate::light::cell_emitter`]) delivers the irradiance of a point source
//!   of intensity `band_power / (4 pi)` at the camera: `I / D^2`
//!   ([`hot_irradiance`]);
//! - a cold frame reflects the active lights as a Lambertian disc of radius
//!   `root_extent / 8` facing each light, with the mass-weighted mean albedo
//!   of its depth-0 cell: `a E_l R^2 max(0, cos theta) / D^2` per light, with
//!   `E_l = I_l / d_l^2` the irradiance from the light and `theta` the angle
//!   at the frame between the light and the camera ([`reflected_irradiance`]).
//!   With no light it delivers nothing and no sprite is drawn.
//!
//! The sprite is a disc whose diameter grows by one pixel per decade of
//! irradiance from 2 px at [`SPRITE_MIN_IRRADIANCE`] to 6 px
//! ([`sprite_size_px`]). Its radiance spreads the irradiance over the disc:
//! `L = E f^2 / (pi r^2)` with `f` the focal length in pixels and `r` the
//! disc radius in pixels, so the disc's pixels together deliver `E` to the
//! camera ([`sprite_radiance`]). Sprites are drawn after volumes with
//! additive blending.

use crate::camera::{Camera, FIELD_OF_VIEW_Y};
use crate::light::{radiant_intensity, PointLight};
use gx_core::emission::Emitter;
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::matter::Section;
use gx_core::registry::Frame;
use gx_core::units::Vec3;

/// Below this projected size of the `root_extent / 8` region, pixels, a
/// frame is a sprite only.
pub const SPRITE_ONLY_BELOW_PX: f64 = 2.0;

/// From this projected size, pixels, a frame is drawn by its cells only.
pub const CELLS_ONLY_FROM_PX: f64 = 8.0;

/// Smallest sprite diameter, pixels.
pub const SPRITE_MIN_PX: f64 = 2.0;

/// Largest sprite diameter, pixels.
pub const SPRITE_MAX_PX: f64 = 6.0;

/// Irradiance at the camera, W m^-2 (luminance weighted over the bands),
/// that gets the smallest sprite. Each decade above adds a pixel.
pub const SPRITE_MIN_IRRADIANCE: f64 = 1.0e-8;

/// Rec. 709 weights on the three bands (red is band 0), the weights of the
/// exposure pass.
pub const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// Pixels per unit of `size / distance` for a view `view_height_px` tall.
pub fn focal_px(view_height_px: f64) -> f64 {
    view_height_px / (2.0 * (0.5 * FIELD_OF_VIEW_Y).tan())
}

/// The projected size, pixels, of `frame`'s `root_extent / 8` region seen
/// from `camera` in a view `view_height_px` tall: `region / D * f`, with `D`
/// the distance to the frame origin. Infinite when the camera is inside the
/// region.
pub fn region_projected_px(
    system: &FrameSystem,
    camera: &Camera,
    frame: &Frame,
    view_height_px: f64,
) -> f64 {
    let region = frame.root_extent.value() / 8.0;
    let d = camera
        .relative(system, Vec3::zero(), frame.frame_id)
        .length();
    if d <= region {
        return f64::INFINITY;
    }
    region / d * focal_px(view_height_px)
}

/// The weight of a frame's sprite for its projected size: 1 below
/// [`SPRITE_ONLY_BELOW_PX`], 0 from [`CELLS_ONLY_FROM_PX`], linear between.
/// The frame's cells are drawn with weight `1 - w`.
pub fn sprite_weight(projected_px: f64) -> f64 {
    ((CELLS_ONLY_FROM_PX - projected_px) / (CELLS_ONLY_FROM_PX - SPRITE_ONLY_BELOW_PX))
        .clamp(0.0, 1.0)
}

/// How one frame is drawn this frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FarFrame {
    /// The frame.
    pub frame_id: u64,
    /// Projected size of its `root_extent / 8` region, pixels.
    pub projected_px: f64,
    /// Weight of its sprite, 0 to 1 ([`sprite_weight`]).
    pub sprite_weight: f64,
}

impl FarFrame {
    /// Returns `true` when the frame is a sprite only and its cells are not
    /// selected.
    pub fn sprite_only(&self) -> bool {
        self.projected_px < SPRITE_ONLY_BELOW_PX
    }

    /// The weight its cells are drawn with: `1 - sprite_weight`.
    pub fn cell_weight(&self) -> f64 {
        1.0 - self.sprite_weight
    }
}

/// Every frame with mass above 0 whose region projects to less than
/// [`CELLS_ONLY_FROM_PX`], in ascending frame id order.
pub fn far_frames(system: &FrameSystem, camera: &Camera, view_height_px: f64) -> Vec<FarFrame> {
    system
        .tree()
        .frames()
        .iter()
        .filter(|f| f.mass.value() > 0.0)
        .filter_map(|f| {
            let px = region_projected_px(system, camera, f, view_height_px);
            (px < CELLS_ONLY_FROM_PX).then(|| FarFrame {
                frame_id: f.frame_id,
                projected_px: px,
                sprite_weight: sprite_weight(px),
            })
        })
        .collect()
}

/// The depth-0 cell of a frame, the cell that stands for it in the far
/// field.
pub fn depth_zero_key(frame_id: u64) -> CellKey {
    CellKey::new(frame_id, 0, 0, 0, 0).expect("depth 0 cell 0 is always valid")
}

/// What the far field needs from a frame's depth-0 cell.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FarFieldCell {
    /// Its hot matter as one emitter, if any.
    pub emitter: Option<Emitter>,
    /// Mass-weighted mean albedo of its samples, three bands.
    pub mean_albedo: [f64; 3],
}

/// Mass-weighted mean albedo of a section's samples, three bands; 0 for an
/// empty section. Sums run in sample order.
pub fn mean_albedo(section: &Section) -> [f64; 3] {
    let Some(samples) = section.samples() else {
        return [0.0; 3];
    };
    let mut sum = [0.0; 3];
    let mut weight = 0.0;
    for s in samples.iter() {
        let w = s.density.value();
        if w <= 0.0 {
            continue;
        }
        for (acc, a) in sum.iter_mut().zip(s.albedo) {
            *acc += w * a.value();
        }
        weight += w;
    }
    if weight > 0.0 {
        sum.map(|v| v / weight)
    } else {
        [0.0; 3]
    }
}

/// Irradiance per band, W m^-2, at a camera `distance` meters from a hot
/// frame's emitter: `band_power / (4 pi) / D^2`.
pub fn hot_irradiance(emitter: &Emitter, distance: f64) -> [f64; 3] {
    let d2 = (distance * distance).max(f64::MIN_POSITIVE);
    radiant_intensity(emitter.band_power).map(|i| i / d2)
}

/// Irradiance per band, W m^-2, at the camera from a cold frame at
/// `frame_rel` (camera-relative, meters) reflecting `lights` as a Lambertian
/// disc of `radius` meters and `albedo` facing each light (see the module
/// docs).
pub fn reflected_irradiance(
    albedo: [f64; 3],
    radius: f64,
    frame_rel: Vec3,
    lights: &[PointLight],
) -> [f64; 3] {
    let d2 = frame_rel.length_squared();
    let Some(to_camera) = (-frame_rel).normalized() else {
        return [0.0; 3];
    };
    let mut out = [0.0; 3];
    for l in lights {
        let p = Vec3::new(
            f64::from(l.position[0]),
            f64::from(l.position[1]),
            f64::from(l.position[2]),
        );
        let to_light = p - frame_rel;
        let dl2 = to_light.length_squared();
        let Some(n) = to_light.normalized() else {
            continue;
        };
        let cos = n.dot(to_camera).max(0.0);
        for b in 0..3 {
            let e_l = f64::from(l.intensity[b]) / dl2;
            out[b] += albedo[b] * e_l * radius * radius * cos / d2;
        }
    }
    out
}

/// Luminance-weighted irradiance, W m^-2.
pub fn luminance(irradiance: [f64; 3]) -> f64 {
    irradiance[0] * LUMA[0] + irradiance[1] * LUMA[1] + irradiance[2] * LUMA[2]
}

/// Sprite diameter, pixels, for an irradiance at the camera: one pixel per
/// decade above [`SPRITE_MIN_IRRADIANCE`], from [`SPRITE_MIN_PX`] to
/// [`SPRITE_MAX_PX`].
pub fn sprite_size_px(irradiance: [f64; 3]) -> f64 {
    let y = luminance(irradiance);
    if y.is_nan() || y <= 0.0 {
        return SPRITE_MIN_PX;
    }
    (SPRITE_MIN_PX + (y / SPRITE_MIN_IRRADIANCE).log10()).clamp(SPRITE_MIN_PX, SPRITE_MAX_PX)
}

/// Radiance per band, W m^-2 sr^-1, of a sprite disc `size_px` across that
/// delivers `irradiance` to the camera in a view with focal length
/// `focal_px`: each pixel subtends `1 / f^2` steradians, so
/// `L = E f^2 / (pi (size / 2)^2)`.
pub fn sprite_radiance(irradiance: [f64; 3], size_px: f64, focal_px: f64) -> [f64; 3] {
    let r = 0.5 * size_px;
    let area = core::f64::consts::PI * r * r;
    irradiance.map(|e| e * focal_px * focal_px / area)
}

/// One point sprite, ready for the GPU.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sprite {
    /// The frame it stands for.
    pub frame_id: u64,
    /// Frame origin relative to the camera, root axes, meters.
    pub position: [f32; 3],
    /// Diameter, pixels.
    pub size_px: f32,
    /// Radiance per band, W m^-2 sr^-1, already weighted.
    pub radiance: [f32; 3],
}

/// The sprite of a far frame, or `None` when it delivers no light. `cell`
/// is what its depth-0 cell holds; `lights` are the active lights, camera
/// relative.
pub fn frame_sprite(
    system: &FrameSystem,
    camera: &Camera,
    far: &FarFrame,
    cell: &FarFieldCell,
    lights: &[PointLight],
    view_height_px: f64,
) -> Option<Sprite> {
    if far.sprite_weight <= 0.0 {
        return None;
    }
    let frame = system.tree().get(far.frame_id)?;
    let rel = camera.relative(system, Vec3::zero(), far.frame_id);
    let irradiance = match &cell.emitter {
        Some(e) => {
            let at = camera.relative(system, e.position, far.frame_id);
            hot_irradiance(e, at.length())
        }
        None => reflected_irradiance(
            cell.mean_albedo,
            frame.root_extent.value() / 8.0,
            rel,
            lights,
        ),
    };
    let y = luminance(irradiance);
    if y.is_nan() || y <= 0.0 {
        return None;
    }
    let size = sprite_size_px(irradiance);
    let radiance = sprite_radiance(irradiance, size, focal_px(view_height_px));
    Some(Sprite {
        frame_id: far.frame_id,
        position: [rel.x as f32, rel.y as f32, rel.z as f32],
        size_px: size as f32,
        radiance: radiance.map(|l| (l * far.sprite_weight) as f32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::units::Meters;

    #[test]
    fn weights_across_the_transition() {
        assert_eq!(sprite_weight(1.0), 1.0);
        assert_eq!(sprite_weight(2.0), 1.0);
        assert_eq!(sprite_weight(5.0), 0.5);
        assert_eq!(sprite_weight(8.0), 0.0);
        assert_eq!(sprite_weight(f64::INFINITY), 0.0);
        let f = FarFrame {
            frame_id: 1,
            projected_px: 1.5,
            sprite_weight: sprite_weight(1.5),
        };
        assert!(f.sprite_only());
        assert_eq!(f.cell_weight(), 0.0);
    }

    #[test]
    fn sizes_by_decade_and_radiance_conserves_irradiance() {
        assert_eq!(sprite_size_px([0.0; 3]), 2.0);
        assert_eq!(sprite_size_px([1.0e-8; 3]), 2.0);
        assert!((sprite_size_px([1.0e-6; 3]) - 4.0).abs() < 1e-12);
        assert_eq!(sprite_size_px([1.0; 3]), 6.0);
        // The disc's pixels together give back E: L * pi r^2 / f^2.
        let e = [2.0, 3.0, 4.0];
        let l = sprite_radiance(e, 4.0, 100.0);
        for b in 0..3 {
            let back = l[b] * core::f64::consts::PI * 4.0 / 1.0e4;
            assert!((back - e[b]).abs() < 1e-12);
        }
    }

    #[test]
    fn hot_and_reflected_irradiance() {
        let e = Emitter {
            position: Vec3::zero(),
            band_power: [4.0 * core::f64::consts::PI * 100.0; 3],
            radius: Meters::new(0.0),
        };
        assert_eq!(hot_irradiance(&e, 10.0), [1.0; 3]);

        // A light behind the camera, the frame ahead: full phase.
        let light = PointLight {
            frame_id: 9,
            position: [0.0, 0.0, 10.0],
            intensity: [100.0; 3],
        };
        let frame = Vec3::new(0.0, 0.0, -10.0);
        let full = reflected_irradiance([0.5; 3], 2.0, frame, &[light]);
        // E_l = 100 / 400; a E_l R^2 / D^2 = 0.5 * 0.25 * 4 / 100.
        assert!((full[0] - 0.005).abs() < 1e-15, "{full:?}");
        // The light beyond the frame: the camera sees the unlit side.
        let behind = PointLight {
            position: [0.0, 0.0, -30.0],
            ..light
        };
        assert_eq!(
            reflected_irradiance([0.5; 3], 2.0, frame, &[behind]),
            [0.0; 3]
        );
        assert_eq!(reflected_irradiance([0.5; 3], 2.0, frame, &[]), [0.0; 3]);
    }
}
