//! Spawning the camera onto a frame: putting it just above the outermost
//! matter of that frame and carrying it with the frame.
//!
//! `G` opens a chooser of the frames whose depth-0 cell has arrived with
//! matter in it ([`spawn_frames`]); a number key picks one. The camera is put
//! on the line from that frame's origin toward where the camera was (or
//! along the `--start-on` direction), [`SPAWN_HEIGHT`] above the outermost
//! surface drawn on that line ([`crate::world::World::surface_along`]),
//! looking level. While finer meshes replace coarser ones the point is
//! found again on what is drawn, for at most [`REFINE_SECONDS`] and only
//! until the camera is flown.
//!
//! A spawned camera is attached to its frame: it moves and turns with it
//! ([`crate::camera::advance`] with `attached`), so the matter it spawned
//! over stays under it, and it flies relative to the frame. Nothing else acts on it: no
//! gravity, no collision. `Home` and the number-key views detach it.

use gx_core::frames::FrameSystem;
use gx_core::registry::FrameTree;
use gx_core::units::Vec3;

use crate::camera::{pitched_look, Camera};
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

/// Where the ray `t * direction`, `t >= 0`, from the origin crosses the box
/// from `min` to `max`: the entering and leaving `t`, or `None` if it
/// misses.
pub fn ray_box(min: [f64; 3], max: [f64; 3], direction: Vec3) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
    for (k, d) in [direction.x, direction.y, direction.z]
        .into_iter()
        .enumerate()
    {
        let (a0, a1) = (min[k], max[k]);
        if d == 0.0 {
            if 0.0 < a0 || 0.0 > a1 {
                return None;
            }
            continue;
        }
        let (a, b) = (a0 / d, a1 / d);
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
    /// Height above the surface, meters.
    pub height: f64,
    /// Degrees below level the camera looks; `None` for level.
    pub pitch_deg: Option<f64>,
    /// Session time after which the point is no longer refined, seconds.
    pub until: f64,
}

/// Spawns `camera` onto `frame_id` along `direction` (frame axes, or the
/// line toward the camera when `None`), `height` meters above the surface
/// and looking `pitch_deg` below level (level when `None`), at session
/// time `now`. Returns the spawn to keep refining, or `None` (changing
/// nothing) when the frame shows no matter on that line yet.
#[allow(clippy::too_many_arguments)]
pub fn spawn(
    system: &FrameSystem,
    world: &crate::world::World,
    frame_id: u64,
    direction: Option<Vec3>,
    height: f64,
    pitch_deg: Option<f64>,
    camera: &mut Camera,
    now: f64,
) -> Option<Spawn> {
    let direction = match direction {
        Some(d) => d,
        None => crate::stream::camera_in_frame(system, camera, frame_id).normalized()?,
    };
    let r = world.surface_along(frame_id, direction)?;
    let s = Spawn {
        frame_id,
        direction,
        height,
        pitch_deg,
        until: now + REFINE_SECONDS,
    };
    place(camera, &s, r);
    Some(s)
}

/// Finds the point of `spawn` again with the cells loaded now and moves the
/// camera there. Returns `false` once the spawn is over (its time is up).
pub fn refine(world: &crate::world::World, spawn: &Spawn, camera: &mut Camera, now: f64) -> bool {
    if now > spawn.until {
        return false;
    }
    if let Some(r) = world.surface_along(spawn.frame_id, spawn.direction) {
        let want = spawn.direction.scale(r + spawn.height);
        if (camera.position - want).length() > 0.01 || camera.frame_id != spawn.frame_id {
            place(camera, spawn, r);
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
    /// Only where the controls exist (native, or the browser build).
    #[cfg(any(not(target_arch = "wasm32"), feature = "web"))]
    pub fn frame(
        &mut self,
        system: &FrameSystem,
        world: &crate::world::World,
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
            if let Some(&f) = spawn_frames(world.cache(), system.tree()).get(i) {
                if let Some(s) = spawn(system, world, f, None, SPAWN_HEIGHT, None, camera, now) {
                    self.active = Some(s);
                    attached = true;
                }
            }
        }
        if let Some(p) = self.pending {
            if spawn_frames(world.cache(), system.tree()).contains(&p.frame_id) {
                let height = p.height.unwrap_or(SPAWN_HEIGHT);
                if let Some(s) = spawn(
                    system,
                    world,
                    p.frame_id,
                    Some(p.direction),
                    height,
                    p.pitch_deg,
                    camera,
                    now,
                ) {
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
            if !refine(world, &s, camera, now) {
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

/// Puts the camera in the spawn's frame, its height beyond `r` along its
/// direction, looking its pitch below level toward the frame's `+z` axis,
/// or `+x` when the camera is on the z axis.
fn place(camera: &mut Camera, spawn: &Spawn, r: f64) {
    camera.frame_id = spawn.frame_id;
    camera.position = spawn.direction.scale(r + spawn.height);
    camera.orientation = pitched_look(spawn.direction, spawn.pitch_deg.unwrap_or(0.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_cross_boxes() {
        // The box from the origin to (4,4,4).
        let d = Vec3::new(1.0, 1.0, 1.0).normalized().unwrap();
        let (a, b) = ray_box([0.0; 3], [4.0; 3], d).unwrap();
        assert!(a.abs() < 1e-12 && (b - 48.0f64.sqrt()).abs() < 1e-9);
        // The box from (2,2,2) to (4,4,4), behind a ray along -x.
        assert!(ray_box([2.0; 3], [4.0; 3], Vec3::new(-1.0, 0.0, 0.0)).is_none());
        let (a, b) = ray_box([2.0; 3], [4.0; 3], d).unwrap();
        assert!((a - 12.0f64.sqrt()).abs() < 1e-9 && (b - 48.0f64.sqrt()).abs() < 1e-9);
    }
}
