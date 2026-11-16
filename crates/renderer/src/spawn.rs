//! Spawning the camera onto a frame: putting it just above the outermost
//! matter of that frame and carrying it with the frame.
//!
//! `G` opens a chooser of the frames whose depth-0 cell has arrived with
//! matter in it ([`spawn_frames`]); a number key picks one. The camera is put
//! on the line from that frame's origin toward where the camera was (or
//! along the `--start-on` direction), [`SPAWN_HEIGHT`] above the outermost
//! matter the loaded cells show on that line ([`surface_along`]), looking
//! level. While finer cells arrive the point is found again with them, for
//! at most [`REFINE_SECONDS`] and only until the camera is flown.
//!
//! A spawned camera is attached to its frame: it moves and turns with it
//! ([`crate::camera::advance`] with `attached`), so the ground stays under
//! it, and it flies relative to the frame. Nothing else acts on it: no
//! gravity, no collision. `Home` and the number-key views detach it.

use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::registry::FrameTree;
use gx_core::units::{Meters, Vec3};

use crate::camera::{look_rotation, Camera};
use crate::farfield::depth_zero_key;
use crate::stream::{CellCache, CellState};

/// Height of a spawned camera above the outermost matter, meters.
pub const SPAWN_HEIGHT: f64 = 2.0;

/// How long a spawn keeps finding its point again as finer cells arrive,
/// seconds.
pub const REFINE_SECONDS: f64 = 20.0;

/// The frames that can be spawned onto: those whose depth-0 cell is loaded
/// with matter, ascending by id.
pub fn spawn_frames(cache: &CellCache, tree: &FrameTree) -> Vec<u64> {
    tree.frames()
        .iter()
        .filter(|f| !f.is_root())
        .filter(|f| {
            matches!(
                cache.state(&depth_zero_key(f.frame_id)),
                Some(CellState::Ready(c)) if c.has_matter
            )
        })
        .map(|f| f.frame_id)
        .collect()
}

/// The overlay line of the open chooser.
pub fn chooser_line(frames: &[u64]) -> String {
    if frames.is_empty() {
        return "spawn: no frame with matter loaded yet  (G closes)".to_string();
    }
    let keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'];
    let choices: Vec<String> = frames
        .iter()
        .zip(keys)
        .map(|(f, k)| format!("{k} frame {f}"))
        .collect();
    format!("spawn: {}  (G closes)", choices.join("  "))
}

/// The distance from `frame_id`'s origin along the unit `direction` (frame
/// axes) of the outermost matter the loaded cells show, meters: in the
/// deepest loaded cells the line crosses that hold matter on it, the
/// farthest point whose nearest sample is at or above the frame's surface
/// isovalue ([`CellCache::frame_isovalue`]; any matter without one). `None` when no loaded cell
/// shows matter on the line.
pub fn surface_along(
    cache: &CellCache,
    tree: &FrameTree,
    frame_id: u64,
    direction: Vec3,
) -> Option<f64> {
    let extent = Meters::new(tree.get(frame_id)?.root_extent.value());
    let frame_iso = cache.frame_isovalue(frame_id);
    let mut best: Option<(u8, f64)> = None;
    for (key, entry) in cache.entries() {
        if key.frame_id != frame_id {
            continue;
        }
        let CellState::Ready(cell) = &entry.state else {
            continue;
        };
        if best.is_some_and(|(d, _)| d > key.depth) {
            continue;
        }
        let Some(samples) = cell.section.samples() else {
            continue;
        };
        let Some((t0, t1)) = ray_box(key, extent, direction) else {
            continue;
        };
        let iso = frame_iso
            .or(cell.surface_isovalue)
            .unwrap_or(f64::MIN_POSITIVE);
        let g = key.geometry(extent);
        let n = usize::from(cell.section.resolution());
        let e = g.edge.value() / n as f64;
        let index = |p: Vec3| -> usize {
            let c = |v: f64, o: f64| (((v - o) / e).floor() as i64).clamp(0, n as i64 - 1) as usize;
            c(p.x, g.origin.x) + n * (c(p.y, g.origin.y) + n * c(p.z, g.origin.z))
        };
        // From the far end inward, a quarter sample at a time.
        let step = e / 4.0;
        let mut t = t1;
        while t >= t0 {
            if samples.density(index(direction.scale(t))).value() >= iso {
                let better = match best {
                    None => true,
                    Some((d, r)) => key.depth > d || t > r,
                };
                if better {
                    best = Some((key.depth, t));
                }
                break;
            }
            t -= step;
        }
    }
    best.map(|(_, r)| r)
}

/// Where the ray `t * direction`, `t >= 0`, crosses the cell's box: the
/// entering and leaving `t`, or `None` if it misses.
fn ray_box(key: &CellKey, extent: Meters, direction: Vec3) -> Option<(f64, f64)> {
    let g = key.geometry(extent);
    let e = g.edge.value();
    let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
    for (o, d) in [
        (g.origin.x, direction.x),
        (g.origin.y, direction.y),
        (g.origin.z, direction.z),
    ] {
        if d == 0.0 {
            if 0.0 < o || 0.0 > o + e {
                return None;
            }
            continue;
        }
        let (a, b) = (o / d, (o + e) / d);
        lo = lo.max(a.min(b));
        hi = hi.min(a.max(b));
    }
    (lo <= hi).then_some((lo, hi))
}

/// A spawn still finding its point as finer cells arrive.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Spawn {
    /// The frame spawned onto.
    pub frame_id: u64,
    /// Unit direction from the frame origin, frame axes.
    pub direction: Vec3,
    /// Session time after which the point is no longer refined, seconds.
    pub until: f64,
}

/// Spawns `camera` onto `frame_id` along `direction` (frame axes, or the
/// line toward the camera when `None`), at session time `now`. Returns the
/// spawn to keep refining, or `None` (changing nothing) when the frame
/// shows no matter on that line yet.
pub fn spawn(
    system: &FrameSystem,
    cache: &CellCache,
    frame_id: u64,
    direction: Option<Vec3>,
    camera: &mut Camera,
    now: f64,
) -> Option<Spawn> {
    let direction = match direction {
        Some(d) => d,
        None => crate::stream::camera_in_frame(system, camera, frame_id).normalized()?,
    };
    let r = surface_along(cache, system.tree(), frame_id, direction)?;
    place(camera, frame_id, direction, r);
    Some(Spawn {
        frame_id,
        direction,
        until: now + REFINE_SECONDS,
    })
}

/// Finds the point of `spawn` again with the cells loaded now and moves the
/// camera there. Returns `false` once the spawn is over (its time is up).
pub fn refine(
    system: &FrameSystem,
    cache: &CellCache,
    spawn: &Spawn,
    camera: &mut Camera,
    now: f64,
) -> bool {
    if now > spawn.until {
        return false;
    }
    if let Some(r) = surface_along(cache, system.tree(), spawn.frame_id, spawn.direction) {
        let want = spawn.direction.scale(r + SPAWN_HEIGHT);
        if (camera.position - want).length() > 0.01 || camera.frame_id != spawn.frame_id {
            place(camera, spawn.frame_id, spawn.direction, r);
        }
    }
    true
}

/// What the frame loop keeps between frames for spawning: the spawn being
/// refined, and a `--start-on` spawn waiting for its frame's matter.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Spawner {
    /// The spawn still finding its point.
    pub active: Option<Spawn>,
    /// `--start-on` (without a distance) until it has spawned.
    pub pending: Option<crate::config::StartOn>,
}

impl Spawner {
    /// One frame, before the camera moves: spawns on a chooser pick (an
    /// index into [`spawn_frames`], [`crate::controls::Controls::take_spawn`])
    /// or a pending `--start-on`,
    /// refines the active spawn, and ends it once the camera is flown
    /// (`input` moves it) or detached (`controls.attached` cleared by
    /// `Home` or a view key). Sets `controls.attached` when it spawns.
    pub fn frame(
        &mut self,
        system: &FrameSystem,
        cache: &CellCache,
        camera: &mut Camera,
        controls: &mut crate::controls::Controls,
        input: &crate::camera::FlightInput,
        now: f64,
    ) {
        let picked = controls.take_spawn();
        let mut attached = controls.attached;
        if !attached {
            self.active = None;
        }
        if let Some(i) = picked {
            self.pending = None;
            if let Some(&f) = spawn_frames(cache, system.tree()).get(i) {
                if let Some(s) = spawn(system, cache, f, None, camera, now) {
                    self.active = Some(s);
                    attached = true;
                }
            }
        }
        if let Some(p) = self.pending {
            if spawn_frames(cache, system.tree()).contains(&p.frame_id) {
                if let Some(s) = spawn(system, cache, p.frame_id, Some(p.direction), camera, now) {
                    self.active = Some(s);
                    self.pending = None;
                    attached = true;
                }
            }
        }
        let flown =
            input.forward || input.back || input.left || input.right || input.up || input.down;
        if flown {
            self.active = None;
        }
        if let Some(s) = self.active {
            if !refine(system, cache, &s, camera, now) {
                self.active = None;
            }
        }
        controls.attached = attached;
    }

    /// The overlay line of an attached camera.
    pub fn status_line(camera: &Camera) -> String {
        format!(
            "camera on frame {}  (moves with it; Home or a view key frees it)",
            camera.frame_id
        )
    }
}

/// Puts the camera in `frame_id`, [`SPAWN_HEIGHT`] beyond `r` along
/// `direction`, looking level toward the frame's `+z` (or `+x` at a pole).
fn place(camera: &mut Camera, frame_id: u64, direction: Vec3, r: f64) {
    let up = direction;
    let z = Vec3::new(0.0, 0.0, 1.0);
    let forward = (z - up.scale(z.dot(up)))
        .normalized()
        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    camera.frame_id = frame_id;
    camera.position = direction.scale(r + SPAWN_HEIGHT);
    camera.orientation = look_rotation(forward, up);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_cross_boxes() {
        let k = CellKey::new(1, 1, 1, 1, 1).unwrap();
        let extent = Meters::new(8.0);
        // Depth 1, cell (1,1,1): from the origin to (4,4,4).
        let d = Vec3::new(1.0, 1.0, 1.0).normalized().unwrap();
        let (a, b) = ray_box(&k, extent, d).unwrap();
        assert!(a.abs() < 1e-12 && (b - 48.0f64.sqrt()).abs() < 1e-9);
        // Depth 2, cell (3,3,3): (2,2,2) to (4,4,4), behind a ray along -x.
        let far = CellKey::new(1, 2, 3, 3, 3).unwrap();
        assert!(ray_box(&far, extent, Vec3::new(-1.0, 0.0, 0.0)).is_none());
        let (a, b) = ray_box(&far, extent, d).unwrap();
        assert!((a - 12.0f64.sqrt()).abs() < 1e-9 && (b - 48.0f64.sqrt()).abs() < 1e-9);
    }
}
