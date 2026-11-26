//! Whether a frame needs drawing: the renderer draws only when what would
//! be drawn differs visibly from what was drawn last.
//!
//! The window loop ticks (input, simulation, streaming, selection) on its
//! own schedule and builds the [`Scene`] it would draw; [`Redraw::decide`]
//! compares it with the last scene drawn and says whether to draw. A scene
//! needs drawing when:
//!
//! - what is drawn changed: another mesh, tile, volume, sprite, light,
//!   marker, or line set, a different mesh mode or field of view, or the
//!   exposure settings (bias, fixed stops);
//! - anything drawn moved on screen by [`MOVE_PX`] or more. Everything is
//!   compared in camera axes, so a camera carried along with a rotating
//!   frame sees no change from the rotation itself. An item's offset `t`
//!   and rotation `R` move its points by at most `|dt| + |dR| rho` (`rho`
//!   its extent, `|dR|` the Frobenius norm), seen from at least
//!   `max(near, |t| - rho)` away;
//! - a light seen from a drawn item turned by [`LIGHT_TURN`] radians or
//!   more, or any intensity, radiance, or weight changed by
//!   [`RELATIVE_CHANGE`] or more of itself;
//! - the overlay text changed and [`OVERLAY_SECONDS`] passed since the last
//!   draw (it shows clocks and counters that tick all the time);
//! - the automatic exposure is still adapting: after a draw for a change,
//!   frames keep coming at most every [`SETTLE_INTERVAL`] for
//!   [`SETTLE_SECONDS`] (five adaptation time constants).
//!
//! Window events (resize, the system asking for a redraw, any key or
//! button) force a draw.

use crate::render::gpu::{Exposure, ADAPTATION_SECONDS};
use crate::render::scene::Scene;
use std::sync::Arc;

/// Smallest on-screen movement, pixels, that is drawn.
pub const MOVE_PX: f64 = 0.25;

/// Smallest turn of a light's direction from a drawn item, radians, that
/// is drawn: the irradiance on a surface changes by about this fraction of
/// itself, about half of one 8-bit step.
pub const LIGHT_TURN: f64 = 1.0 / 512.0;

/// Smallest relative change of an intensity, radiance, or weight that is
/// drawn.
pub const RELATIVE_CHANGE: f64 = 1.0 / 512.0;

/// Longest the overlay text may lag, seconds, when nothing else changes.
pub const OVERLAY_SECONDS: f64 = 1.0;

/// How long frames keep coming after a draw for a change, seconds, while
/// the automatic exposure adapts.
pub const SETTLE_SECONDS: f64 = 5.0 * ADAPTATION_SECONDS;

/// Shortest time between frames that are drawn only for the exposure to
/// settle, seconds.
pub const SETTLE_INTERVAL: f64 = 0.05;

/// What the last draw drew, and when.
#[derive(Default)]
pub struct Redraw {
    last: Option<(Scene, String, Exposure)>,
    last_draw: f64,
    settle_until: f64,
}

impl Redraw {
    /// No draw yet: the first scene is always drawn.
    pub fn new() -> Redraw {
        Redraw::default()
    }

    /// Whether to draw `scene` with `overlay` and `exposure` at `now`
    /// (seconds), seen with `focal` pixels per unit of size over distance.
    /// `force` draws regardless (window events).
    pub fn decide(
        &self,
        scene: &Scene,
        overlay: &str,
        exposure: &Exposure,
        now: f64,
        focal: f64,
        force: bool,
    ) -> Need {
        let Some((last, last_overlay, last_exposure)) = &self.last else {
            return Need::Change;
        };
        if force || !same_exposure(last_exposure, exposure) || scene_changed(last, scene, focal) {
            return Need::Change;
        }
        let since = now - self.last_draw;
        if exposure.fixed_stops.is_none() && now < self.settle_until && since >= SETTLE_INTERVAL {
            return Need::Settle;
        }
        if overlay != last_overlay && since >= OVERLAY_SECONDS {
            return Need::Overlay;
        }
        Need::None
    }

    /// Records a draw of `scene` at `now` for `need`.
    pub fn drawn(
        &mut self,
        scene: Scene,
        overlay: String,
        exposure: Exposure,
        now: f64,
        need: Need,
    ) {
        if need == Need::Change {
            self.settle_until = now + SETTLE_SECONDS;
        }
        self.last = Some((scene, overlay, exposure));
        self.last_draw = now;
    }

    /// The last scene drawn, to draw again when the window asks.
    pub fn last(&self) -> Option<&(Scene, String, Exposure)> {
        self.last.as_ref()
    }

    /// Whether frames are still coming for the exposure to settle.
    pub fn settling(&self, now: f64) -> bool {
        now < self.settle_until
    }
}

/// Why a frame is drawn, or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Need {
    /// Nothing visible changed.
    None,
    /// What is drawn changed (or the first frame, or a window event).
    Change,
    /// The automatic exposure is still adapting.
    Settle,
    /// Only the overlay text changed.
    Overlay,
}

impl Need {
    /// Whether to draw.
    pub fn draw(self) -> bool {
        self != Need::None
    }
}

fn same_exposure(a: &Exposure, b: &Exposure) -> bool {
    a.bias_stops == b.bias_stops && a.fixed_stops == b.fixed_stops
}

type V3 = [f64; 3];

fn v(a: [f32; 3]) -> V3 {
    a.map(f64::from)
}

fn len(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `m * a` for the rotation from root into camera axes (column major 4x4).
fn to_camera(m: &[[f32; 4]; 4], a: V3) -> V3 {
    core::array::from_fn(|r| (0..3).map(|c| f64::from(m[c][r]) * a[c]).sum())
}

/// The rotation of an item from frame into camera axes: `V * R`, `R` given
/// by its columns.
fn rotation_in_camera(m: &[[f32; 4]; 4], cols: &[[f32; 3]; 3]) -> [V3; 3] {
    core::array::from_fn(|c| to_camera(m, v(cols[c])))
}

fn frobenius(a: &[V3; 3], b: &[V3; 3]) -> f64 {
    let mut s = 0.0;
    for c in 0..3 {
        for r in 0..3 {
            let d = a[c][r] - b[c][r];
            s += d * d;
        }
    }
    s.sqrt()
}

fn relative_change(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs());
    scale > 0.0 && (a - b).abs() > RELATIVE_CHANGE * scale
}

fn bands_changed(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).any(|i| relative_change(f64::from(a[i]), f64::from(b[i])))
}

/// One drawn item in camera axes: offset, rotation, extent.
struct Pose {
    t: V3,
    r: [V3; 3],
    rho: f64,
}

/// Relative rounding of the `f32` offsets and rotations a scene holds, with
/// room for the arithmetic around them: changes within it are rounding,
/// not movement (a camera carried by a rotating frame sees its anchor's
/// offset recomputed every tick, a few millimeters different at 50 km).
const F32_NOISE: f64 = 8.0 * f32::EPSILON as f64;

/// Whether an item moved by [`MOVE_PX`] or more on screen.
fn moved(a: &Pose, b: &Pose, near: f64, focal: f64) -> bool {
    let noise = F32_NOISE * (len(a.t) + len(b.t) + 2.0 * b.rho);
    let shift = len(sub(a.t, b.t)) + frobenius(&a.r, &b.r) * b.rho - noise;
    if shift <= 0.0 {
        return false;
    }
    let d = (len(b.t) - b.rho).max(near).max(f64::MIN_POSITIVE);
    shift / d * focal >= MOVE_PX
}

/// Whether a light seen from an item at `t` (camera axes) turned by
/// [`LIGHT_TURN`] or more: `la`, `lb` are the light positions, camera
/// relative, root axes, before and after.
fn light_turned(
    ma: &[[f32; 4]; 4],
    mb: &[[f32; 4]; 4],
    ta: V3,
    tb: V3,
    la: [f32; 3],
    lb: [f32; 3],
) -> bool {
    let da = sub(to_camera(ma, v(la)), ta);
    let db = sub(to_camera(mb, v(lb)), tb);
    let (na, nb) = (len(da), len(db));
    if na == 0.0 || nb == 0.0 {
        return na != nb;
    }
    let cos = (da[0] * db[0] + da[1] * db[1] + da[2] * db[2]) / (na * nb);
    cos.clamp(-1.0, 1.0).acos() >= LIGHT_TURN
}

/// Whether `next` differs visibly from `prev` (module docs).
pub fn scene_changed(prev: &Scene, next: &Scene, focal: f64) -> bool {
    change_reason(prev, next, focal).is_some()
}

/// What differs visibly between `prev` and `next`, if anything: a short
/// name of the first difference found (for logs).
pub fn change_reason(prev: &Scene, next: &Scene, focal: f64) -> Option<&'static str> {
    if prev.mesh_mode != next.mesh_mode
        || prev.fov_y != next.fov_y
        || prev.surfaces.len() != next.surfaces.len()
        || prev.tiles.len() != next.tiles.len()
        || prev.volumes.len() != next.volumes.len()
        || prev.sprites.len() != next.sprites.len()
        || prev.lights.len() != next.lights.len()
        || prev.markers.len() != next.markers.len()
        || prev.lines.len() != next.lines.len()
    {
        return Some("set");
    }
    let (ma, mb) = (&prev.view_rotation, &next.view_rotation);
    let near = f64::from(next.near.min(prev.near)).max(1e-3);
    let lights = next.lights.len();
    for (a, b) in prev.lights.iter().zip(&next.lights) {
        if a.frame_id != b.frame_id || bands_changed(a.intensity, b.intensity) {
            return Some("light");
        }
    }
    let lit = |ta: V3, tb: V3, la: &[[f32; 3]], lb: &[[f32; 3]]| {
        (0..lights).any(|i| light_turned(ma, mb, ta, tb, la[i], lb[i]))
    };
    for (a, b) in prev.surfaces.iter().zip(&next.surfaces) {
        if !Arc::ptr_eq(&a.mesh, &b.mesh)
            || relative_change(f64::from(a.weight), f64::from(b.weight))
        {
            return Some("surface");
        }
        let m = &b.mesh;
        let rho = len(sub(m.bounds_max, m.bounds_min))
            + len(sub(m.bounds_min, [m.origin.x, m.origin.y, m.origin.z]));
        let pa = Pose {
            t: to_camera(ma, v(a.offset)),
            r: rotation_in_camera(ma, &a.rotation),
            rho,
        };
        let pb = Pose {
            t: to_camera(mb, v(b.offset)),
            r: rotation_in_camera(mb, &b.rotation),
            rho,
        };
        if moved(&pa, &pb, near, focal) || lit(pa.t, pb.t, &a.lights, &b.lights) {
            return Some("surface moved");
        }
    }
    for (a, b) in prev.tiles.iter().zip(&next.tiles) {
        let (ga, gb) = (&a.tile.geometry, &b.tile.geometry);
        if !Arc::ptr_eq(ga, gb)
            || a.tile.anchor != b.tile.anchor
            || a.tile.morph != b.tile.morph
            || relative_change(f64::from(a.weight), f64::from(b.weight))
        {
            return Some("tile");
        }
        let (lo, hi) = gb.bounds;
        let p = b.tile.anchor.position;
        let rho = (0..8)
            .map(|c| {
                let corner = [
                    if c & 1 == 0 { lo[0] } else { hi[0] },
                    if c & 2 == 0 { lo[1] } else { hi[1] },
                    if c & 4 == 0 { lo[2] } else { hi[2] },
                ];
                len(sub(corner, p))
            })
            .fold(0.0, f64::max);
        let pa = Pose {
            t: to_camera(ma, v(a.offset)),
            r: rotation_in_camera(ma, &a.rotation),
            rho,
        };
        let pb = Pose {
            t: to_camera(mb, v(b.offset)),
            r: rotation_in_camera(mb, &b.rotation),
            rho,
        };
        if moved(&pa, &pb, near, focal) || lit(pa.t, pb.t, &a.lights, &b.lights) {
            return Some("tile moved");
        }
    }
    for (a, b) in prev.volumes.iter().zip(&next.volumes) {
        if !Arc::ptr_eq(&a.grid, &b.grid)
            || relative_change(f64::from(a.weight), f64::from(b.weight))
        {
            return Some("volume");
        }
        let rho = b.grid.edge * 3.0f64.sqrt();
        let pa = Pose {
            t: to_camera(ma, v(a.offset)),
            r: rotation_in_camera(ma, &a.rotation),
            rho,
        };
        let pb = Pose {
            t: to_camera(mb, v(b.offset)),
            r: rotation_in_camera(mb, &b.rotation),
            rho,
        };
        if moved(&pa, &pb, near, focal) {
            return Some("volume moved");
        }
    }
    let point = |a: [f32; 3], b: [f32; 3]| {
        let still = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pa = Pose {
            t: to_camera(ma, v(a)),
            r: still,
            rho: 0.0,
        };
        let pb = Pose {
            t: to_camera(mb, v(b)),
            r: still,
            rho: 0.0,
        };
        moved(&pa, &pb, near, focal)
    };
    for (a, b) in prev.sprites.iter().zip(&next.sprites) {
        if a.frame_id != b.frame_id
            || f64::from((a.size_px - b.size_px).abs()) >= MOVE_PX
            || bands_changed(a.radiance, b.radiance)
            || point(a.position, b.position)
        {
            return Some("sprite");
        }
    }
    for (a, b) in prev.markers.iter().zip(&next.markers) {
        if a.frame_id != b.frame_id
            || a.size_px != b.size_px
            || a.color != b.color
            || point(a.position, b.position)
        {
            return Some("marker");
        }
    }
    for (a, b) in prev.lines.iter().zip(&next.lines) {
        if a.color != b.color || point(a.position, b.position) {
            return Some("line");
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::farfield::Sprite;
    use crate::render::scene::{Marker, MeshMode};

    /// A rotation about camera-space y by `a` radians, column major 4x4.
    fn yaw(a: f64) -> [[f32; 4]; 4] {
        let (s, c) = (a.sin() as f32, a.cos() as f32);
        [
            [c, 0.0, -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn scene(view: [[f32; 4]; 4], at: [f32; 3]) -> Scene {
        Scene {
            mesh_mode: MeshMode::Surfaces,
            markers: vec![Marker {
                frame_id: 67,
                position: at,
                size_px: 6.0,
                color: [1.0; 4],
            }],
            lines: Vec::new(),
            surfaces: Vec::new(),
            tiles: Vec::new(),
            volumes: Vec::new(),
            sprites: vec![Sprite {
                frame_id: 1,
                position: [0.0, 0.0, -1.0e11],
                size_px: 4.0,
                radiance: [1.0, 1.0, 1.0],
            }],
            lights: Vec::new(),
            view_rotation: view,
            near: 1.0,
            fov_y: 1.0,
        }
    }

    /// `m^T * a` in root axes for a camera-space point `a`.
    fn to_root(m: &[[f32; 4]; 4], a: [f64; 3]) -> [f32; 3] {
        core::array::from_fn(|c| (0..3).map(|r| f64::from(m[c][r]) * a[r]).sum::<f64>() as f32)
    }

    #[test]
    fn the_same_scene_is_not_drawn_again() {
        let s = scene(yaw(0.0), [0.0, 0.0, -1000.0]);
        assert!(!scene_changed(&s, &s.clone(), 1000.0));
    }

    #[test]
    fn movement_below_a_quarter_pixel_is_not_drawn() {
        let a = scene(yaw(0.0), [0.0, 0.0, -1000.0]);
        // 0.1 m across at 1000 m with a 1000 px focal length: 0.1 px.
        let b = scene(yaw(0.0), [0.1, 0.0, -1000.0]);
        assert!(!scene_changed(&a, &b, 1000.0));
        // 1 m: 1 px.
        let c = scene(yaw(0.0), [1.0, 0.0, -1000.0]);
        assert!(scene_changed(&a, &c, 1000.0));
        // Turning the camera moves the far sprite.
        let d = scene(yaw(1.0e-3), [0.0, 0.0, -1000.0]);
        assert!(scene_changed(&a, &d, 1000.0));
    }

    #[test]
    fn a_camera_carried_by_a_turning_frame_sees_no_change() {
        // Everything (the marker and the sprite) and the view turn
        // together, as when the camera rides a rotating frame: in camera
        // axes nothing moved.
        let m0 = yaw(0.0);
        let m1 = yaw(0.3);
        let mut a = scene(m0, to_root(&m0, [0.0, 0.0, -1000.0]));
        let mut b = scene(m1, to_root(&m1, [0.0, 0.0, -1000.0]));
        a.sprites[0].position = to_root(&m0, [0.0, 0.0, -1.0e11]);
        b.sprites[0].position = to_root(&m1, [0.0, 0.0, -1.0e11]);
        assert!(!scene_changed(&a, &b, 1000.0));
    }

    #[test]
    fn radiance_and_set_changes_are_drawn() {
        let a = scene(yaw(0.0), [0.0, 0.0, -1000.0]);
        let mut b = a.clone();
        b.sprites[0].radiance = [1.01, 1.0, 1.0];
        assert!(scene_changed(&a, &b, 1000.0));
        let mut c = a.clone();
        c.markers.clear();
        assert!(scene_changed(&a, &c, 1000.0));
    }

    #[test]
    fn overlay_waits_and_exposure_settles() {
        let s = scene(yaw(0.0), [0.0, 0.0, -1000.0]);
        let e = Exposure::default();
        let mut r = Redraw::new();
        assert_eq!(r.decide(&s, "a", &e, 0.0, 1000.0, false), Need::Change);
        r.drawn(s.clone(), "a".into(), e, 0.0, Need::Change);
        // Settling: frames at most every SETTLE_INTERVAL.
        assert_eq!(r.decide(&s, "a", &e, 0.01, 1000.0, false), Need::None);
        assert_eq!(r.decide(&s, "a", &e, 0.06, 1000.0, false), Need::Settle);
        r.drawn(s.clone(), "a".into(), e, 3.0, Need::Settle);
        // Settled: nothing, until the overlay changed for a second.
        assert_eq!(r.decide(&s, "a", &e, 3.5, 1000.0, false), Need::None);
        assert_eq!(r.decide(&s, "b", &e, 3.5, 1000.0, false), Need::None);
        assert_eq!(r.decide(&s, "b", &e, 4.1, 1000.0, false), Need::Overlay);
        assert_eq!(r.decide(&s, "a", &e, 3.5, 1000.0, true), Need::Change);
    }
}
