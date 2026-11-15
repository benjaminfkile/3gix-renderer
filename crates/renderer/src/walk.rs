//! Fly and walk: the observer as a free camera or as a body the laws act
//! on, and spawning onto any frame that has matter.
//!
//! Implements `space-model.md` section 3, rule 3 ("either a free observer
//! that nothing acts on (fly) or a body that the laws act on (walk)"), and
//! section 6.3 on the renderer side, with the core's observer body
//! ([`gx_core::observer::step_observer`]) and the probe over the loaded
//! matter ([`crate::probe`]).
//!
//! # Fly
//!
//! The free camera of [`crate::camera`]. Nothing acts on it, so it keeps
//! whatever velocity it had: zero for a camera placed by a view key, or the
//! velocity it left walk mode with ([`Observer::drift`]), along which it
//! coasts over simulation time.
//!
//! # Walk
//!
//! A [`Walker`]: a core [`ObserverState`] relative to one frame, stepped
//! every rendered frame by the simulation time that passed, before the
//! frames are integrated by the same time, as the core asks. The camera
//! sits at [`EYE_HEIGHT_FRACTION`] of the observer's height above its feet
//! along the up direction ([`up_direction`]), and the view is kept in the
//! frame's axes, carried along as the up direction changes, so it turns
//! with the matter the observer stands on and mouse look yaws about up and
//! pitches about the horizontal.
//!
//! The walking keys set the core's `intent` along the contact surface: a
//! push along the walking direction while the speed along it, relative to
//! the ground, is below [`WALK_SPEED`], in proportion to what is missing
//! (reached in about [`WALK_RESPONSE_SECONDS`]). Running doubles the speed
//! it drives toward, and with it the intent from rest. Friction stops every other
//! motion over the ground, including the walk once the keys are released.
//! A jump adds [`JUMP_SPEED`] along up. Out of contact the keys do
//! nothing.
//!
//! The observer is stepped by at most [`MAX_OBSERVER_STEP`] of simulation
//! time per rendered frame. Whatever simulation time passes beyond that
//! (a fast playback rate, a jump in time, time running backwards) carries
//! the observer with its frame: the same place and the same velocity
//! relative to the matter, in the frame's axes ([`Anchor`]).
//!
//! While the cells around the feet at the frame's deepest depth are not
//! loaded ([`crate::probe::waiting_for_matter`]) the observer is held in
//! place with its frame, not stepped, and the overlay says so. Those cells
//! are the streaming priority ([`Observer::priority_cells`]).
//!
//! # Spawning
//!
//! [`spawn_frames`] lists the frames with matter: any depth-0 or depth-1
//! cell loaded with a sample above zero density, in ascending frame id
//! ([`survey_keys`] are the cells that tells). [`Walker::spawn`] puts the
//! observer on the line from the frame origin to the camera,
//! [`SPAWN_CLEARANCE`] above the outermost matter on that line, moving with
//! the matter there. While held for finer cells, the spawn point is found
//! again on the same line with what has arrived, so the observer starts
//! above the finest matter available. Then the laws take over: it comes
//! to rest on solid matter, and in fluid, gas, or plasma it sinks until
//! buoyancy holds it.
//!
//! # Toggling
//!
//! [`Observer::toggle`] keeps the eye where it is and turns the camera's
//! velocity (its coasting velocity plus the held flight keys) into the
//! observer's velocity relative to its frame, and back.

use crate::camera::{axis_angle, look_rotation, Camera, FlightInput, LOOK_RADIANS_PER_PIXEL};
use crate::probe::{feet_cells, waiting_for_matter, CacheProbe};
use crate::sim::{ClockAction, Simulation};
use crate::stream::{camera_in_frame, CellCache, CellState};
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use gx_core::observer::{
    step_observer, up_direction, Contact, MatterProbe, ObserverParams, ObserverState,
};
use gx_core::registry::FrameTree;
use gx_core::rotation::advance_orientation;
use gx_core::units::{Seconds, Vec3};

/// Eye height as a fraction of the observer's height.
pub const EYE_HEIGHT_FRACTION: f64 = 0.94;

/// Speed a jump adds along up, meters per second.
pub const JUMP_SPEED: f64 = 3.0;

/// Walking speed the walking keys push toward, relative to the ground,
/// meters per second.
pub const WALK_SPEED: f64 = 1.4;

/// Factor on the walking push and speed while running (Shift).
pub const RUN_FACTOR: f64 = 2.0;

/// Time the walking push takes to reach the walking speed from rest,
/// seconds: the push is `mass * (target - v) / WALK_RESPONSE_SECONDS`
/// along the walking direction, `v` the speed along it and `target`
/// [`WALK_SPEED`], doubled when running.
pub const WALK_RESPONSE_SECONDS: f64 = 0.25;

/// Height of a spawn above the outermost matter, meters.
pub const SPAWN_CLEARANCE: f64 = 2.0;

/// Most simulation time the observer is stepped per rendered frame,
/// seconds.
pub const MAX_OBSERVER_STEP: f64 = 0.25;

/// Closest the view may pitch to straight up or down, radians.
pub const PITCH_MARGIN: f64 = 0.01;

/// Most times a spawn point is found again as finer cells arrive.
pub const SPAWN_REFINEMENTS: u32 = 8;

/// Spawn points closer than this to the last one are taken as the same,
/// meters.
const SPAWN_SETTLED: f64 = 1.0e-3;

/// The walking keys and mouse movement of one rendered frame.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct WalkInput {
    /// Walk along the view direction (W).
    pub forward: bool,
    /// Walk against the view direction (S).
    pub back: bool,
    /// Walk left (A).
    pub left: bool,
    /// Walk right (D).
    pub right: bool,
    /// Run (Shift held).
    pub run: bool,
    /// Jump (Space pressed since the last frame).
    pub jump: bool,
    /// Right mouse drag since the last frame, pixels, `+x` right, `+y` down.
    pub look: (f64, f64),
}

impl WalkInput {
    /// The walking reading of the flight keys: W, A, S, D, Shift, and the
    /// mouse, with `jump` from Space.
    pub fn from_flight(f: &FlightInput, jump: bool) -> WalkInput {
        WalkInput {
            forward: f.forward,
            back: f.back,
            left: f.left,
            right: f.right,
            run: f.boost,
            jump,
            look: f.look,
        }
    }
}

/// The observer at one moment, in its frame's axes, to put back after the
/// frames move ([`Walker::anchor`], [`Walker::reanchor`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Anchor {
    position: Vec3,
    relative_velocity: Vec3,
    contact: Option<Contact>,
}

/// A spawn that is still looking for the finest matter on its line.
#[derive(Copy, Clone, Debug, PartialEq)]
struct SpawnLine {
    /// Unit direction from the frame origin, frame axes.
    direction: Vec3,
    /// Times the point was found again.
    refinements: u32,
}

/// The observer as a body in walk mode (module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct Walker {
    /// The frame the observer's state is relative to.
    pub frame_id: u64,
    /// The core state: the feet relative to the frame origin, root axes.
    pub state: ObserverState,
    /// The constants of the body.
    pub params: ObserverParams,
    /// `true` while matter under the observer is not loaded and it is held
    /// in place.
    pub held: bool,
    /// The view direction, a unit vector in the frame's axes.
    view: Vec3,
    /// The last known up direction, a unit vector in the frame's axes.
    up: Vec3,
    spawn: Option<SpawnLine>,
}

/// Points `view` away from `up` by at least [`PITCH_MARGIN`], keeping its
/// heading; with no heading (along up), the heading of `fallback`, else
/// any horizontal direction.
fn level_enough(view: Vec3, up: Vec3, fallback: Vec3) -> Vec3 {
    if view.dot(up).abs() < PITCH_MARGIN.cos() {
        if let Some(v) = view.normalized() {
            return v;
        }
    }
    let axes = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
    [view, fallback, axes[0], axes[1]]
        .into_iter()
        .map(|c| c - up.scale(c.dot(up)))
        .find(|flat| flat.length() > 1.0e-6)
        .and_then(Vec3::normalized)
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0))
}

/// The rotation taking unit vector `from` to unit vector `to` by the
/// smallest angle, applied to `v`.
fn transport(v: Vec3, from: Vec3, to: Vec3) -> Vec3 {
    let axis = from.cross(to);
    let sin = axis.length();
    let cos = from.dot(to);
    match axis.normalized() {
        Some(a) if sin > 0.0 => axis_angle(a, sin.atan2(cos)).rotate(v),
        _ => v,
    }
}

impl Walker {
    /// A walker whose eye is where `camera` is, looking where it looks,
    /// relative to the camera's frame, moving at `velocity` (root axes,
    /// meters per second, relative to root coordinates).
    pub fn from_camera(
        system: &FrameSystem,
        camera: &Camera,
        velocity: Vec3,
        params: ObserverParams,
    ) -> Walker {
        let frame_id = camera.frame_id;
        let q = system.root_orientation(frame_id);
        let eye = q.rotate(camera.position);
        let camera_up = camera.orientation.rotate(Vec3::new(0.0, 1.0, 0.0));
        let up_root = up_direction(eye, frame_id, system)
            .or_else(|| eye.normalized())
            .unwrap_or_else(|| q.rotate(camera_up));
        let feet = eye - up_root.scale(EYE_HEIGHT_FRACTION * params.height.value());
        let up = q.conjugate().rotate(up_root);
        let view = camera.orientation.rotate(Vec3::new(0.0, 0.0, -1.0));
        Walker {
            frame_id,
            state: ObserverState::new(feet, velocity - system.root_velocity(frame_id)),
            params,
            held: false,
            view: level_enough(view, up, camera_up),
            up,
            spawn: None,
        }
    }

    /// A walker spawned onto `frame_id` (module docs, Spawning) for a camera
    /// at `camera`, keeping the camera's heading. `None` if the frame is not
    /// in the system or the line from its origin to the camera meets no
    /// loaded matter.
    pub fn spawn(
        system: &FrameSystem,
        cache: &CellCache,
        frame_id: u64,
        camera: &Camera,
        params: ObserverParams,
    ) -> Option<Walker> {
        let probe = CacheProbe::new(cache, system, frame_id)?;
        let direction = camera_in_frame(system, camera, frame_id)
            .normalized()
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        let r = probe.outermost_along(direction)?;
        let to_frame = system.root_orientation(frame_id).conjugate();
        let axes = to_frame * camera.root_orientation(system);
        let view = axes.rotate(Vec3::new(0.0, 0.0, -1.0));
        let camera_up = axes.rotate(Vec3::new(0.0, 1.0, 0.0));
        let mut walker = Walker {
            frame_id,
            state: ObserverState::new(Vec3::zero(), Vec3::zero()),
            params,
            held: false,
            view: level_enough(view, direction, camera_up),
            up: direction,
            spawn: Some(SpawnLine {
                direction,
                refinements: 0,
            }),
        };
        walker.place_on_line(system, &probe, direction, r);
        walker.update_up(system);
        Some(walker)
    }

    /// Puts the feet [`SPAWN_CLEARANCE`] beyond `r` along `direction`
    /// (frame axes) moving with the matter there.
    fn place_on_line(&mut self, system: &FrameSystem, probe: &CacheProbe, direction: Vec3, r: f64) {
        let q = system.root_orientation(self.frame_id);
        let at = direction.scale(r + SPAWN_CLEARANCE);
        let matter = probe
            .sample(direction.scale(r))
            .map_or(Vec3::zero(), |s| s.velocity);
        let w = system
            .state(self.frame_id)
            .map_or(Vec3::zero(), |s| s.angular_velocity);
        // The matter's velocity at the outermost sample, carried to the
        // spawn point with the frame's rotation.
        let velocity = matter + w.cross(at - direction.scale(r));
        self.state = ObserverState::new(q.rotate(at), q.rotate(velocity));
    }

    /// The feet in the frame's coordinates.
    pub fn feet_in_frame(&self, system: &FrameSystem) -> Vec3 {
        system
            .root_orientation(self.frame_id)
            .conjugate()
            .rotate(self.state.position)
    }

    /// The up direction at the feet, root axes: against local gravity, or
    /// the last known one where gravity is zero.
    pub fn up_root(&self, system: &FrameSystem) -> Vec3 {
        let q = system.root_orientation(self.frame_id);
        up_direction(self.state.position, self.frame_id, system)
            .unwrap_or_else(|| q.rotate(self.up))
    }

    /// The eye relative to the frame origin, root axes.
    pub fn eye(&self, system: &FrameSystem) -> Vec3 {
        let h = EYE_HEIGHT_FRACTION * self.params.height.value();
        self.state.position + self.up_root(system).scale(h)
    }

    /// The observer's velocity relative to root coordinates, root axes:
    /// what a camera leaving walk mode coasts at.
    pub fn root_velocity(&self, system: &FrameSystem) -> Vec3 {
        self.state.velocity + system.root_velocity(self.frame_id)
    }

    /// The speed relative to matter at rest in the frame, meters per
    /// second.
    pub fn speed_over_matter(&self, system: &FrameSystem) -> f64 {
        let q = system.root_orientation(self.frame_id);
        let w = system
            .state(self.frame_id)
            .map_or(Vec3::zero(), |s| s.angular_velocity);
        let p = self.feet_in_frame(system);
        (q.conjugate().rotate(self.state.velocity) - w.cross(p)).length()
    }

    /// The camera at the eye, looking along the view with up as its up
    /// axis, with the flight speeds of `template`.
    pub fn camera(&self, system: &FrameSystem, template: &Camera) -> Camera {
        let q = system.root_orientation(self.frame_id);
        let up = q.conjugate().rotate(self.up_root(system));
        Camera {
            frame_id: self.frame_id,
            position: q.conjugate().rotate(self.eye(system)),
            orientation: look_rotation(self.view, up),
            ..*template
        }
    }

    /// Follows the up direction at the feet, turning the view with it.
    fn update_up(&mut self, system: &FrameSystem) {
        let q = system.root_orientation(self.frame_id);
        if let Some(up) = up_direction(self.state.position, self.frame_id, system) {
            let up = q.conjugate().rotate(up);
            let view = transport(self.view, self.up, up);
            self.view = level_enough(view, up, self.view);
            self.up = up;
        }
    }

    /// Mouse look: yaw about up, pitch about the horizontal, never closer
    /// than [`PITCH_MARGIN`] to straight up or down.
    fn look(&mut self, (dx, dy): (f64, f64)) {
        if dx != 0.0 {
            self.view = axis_angle(self.up, -dx * LOOK_RADIANS_PER_PIXEL).rotate(self.view);
        }
        if dy != 0.0 {
            if let Some(right) = self.view.cross(self.up).normalized() {
                let pitched = axis_angle(right, -dy * LOOK_RADIANS_PER_PIXEL).rotate(self.view);
                if pitched.dot(self.up).abs() < PITCH_MARGIN.cos() {
                    self.view = pitched;
                }
            }
        }
        self.view = level_enough(self.view, self.up, self.view);
    }

    /// The walking push for this frame, newtons, root axes (module docs);
    /// zero out of contact or with no walking key held.
    pub fn intent(&self, system: &FrameSystem, input: &WalkInput) -> Vec3 {
        let Some(c) = self.state.contact else {
            return Vec3::zero();
        };
        let n = c.normal;
        let tangent = |v: Vec3| v - n.scale(v.dot(n));
        let q = system.root_orientation(self.frame_id);
        let Some(forward) = tangent(q.rotate(self.view)).normalized() else {
            return Vec3::zero();
        };
        let right = forward.cross(n);
        let axis = |plus: bool, minus: bool| f64::from(u8::from(plus)) - f64::from(u8::from(minus));
        let Some(dir) = (forward.scale(axis(input.forward, input.back))
            + right.scale(axis(input.right, input.left)))
        .normalized() else {
            return Vec3::zero();
        };
        let factor = if input.run { RUN_FACTOR } else { 1.0 };
        // Along the walking direction only: friction spares motion along
        // the push and stops the rest, so a push with a sideways part would
        // swing the velocity from side to side.
        let along = (self.state.velocity - c.ground_velocity).dot(dir);
        let push = (WALK_SPEED * factor - along).max(0.0);
        dir.scale(push * self.params.mass.value() / WALK_RESPONSE_SECONDS)
    }

    /// Finds the spawn point again with the cells loaded now, while the
    /// spawn is still settling. Returns `true` if the observer moved.
    fn refine_spawn(&mut self, system: &FrameSystem, cache: &CellCache) -> bool {
        let Some(line) = self.spawn else {
            return false;
        };
        let Some(probe) = CacheProbe::new(cache, system, self.frame_id) else {
            self.spawn = None;
            return false;
        };
        let Some(r) = probe.outermost_along(line.direction) else {
            self.spawn = None;
            return false;
        };
        let before = self.feet_in_frame(system);
        let at = line.direction.scale(r + SPAWN_CLEARANCE);
        if (at - before).length() <= SPAWN_SETTLED || line.refinements >= SPAWN_REFINEMENTS {
            self.spawn = None;
            return false;
        }
        self.place_on_line(system, &probe, line.direction, r);
        self.spawn = Some(SpawnLine {
            refinements: line.refinements + 1,
            ..line
        });
        true
    }

    /// One rendered frame of walking with `system` at the current time:
    /// look, then, unless held for matter, the walking push, a jump, and
    /// [`step_observer`] for `dt` seconds of simulation time (at most
    /// [`MAX_OBSERVER_STEP`]). The caller integrates the frames afterwards
    /// and puts the observer back with [`Walker::anchor`] and
    /// [`Walker::reanchor`]. Returns the time stepped, 0 while held.
    pub fn step(
        &mut self,
        system: &FrameSystem,
        cache: &CellCache,
        input: &WalkInput,
        dt: f64,
    ) -> f64 {
        self.update_up(system);
        self.look(input.look);
        let mut feet = self.feet_in_frame(system);
        self.held = waiting_for_matter(cache, system, self.frame_id, feet);
        if !self.held && self.refine_spawn(system, cache) {
            feet = self.feet_in_frame(system);
            self.held = waiting_for_matter(cache, system, self.frame_id, feet);
        }
        if !self.held {
            // Released: from here on only the laws move the observer.
            self.spawn = None;
        }
        let dt = dt.clamp(0.0, MAX_OBSERVER_STEP);
        if self.held || dt == 0.0 {
            return 0.0;
        }
        let Some(probe) = CacheProbe::new(cache, system, self.frame_id) else {
            return 0.0;
        };
        if input.jump && self.state.contact.is_some() {
            let up = self.up_root(system);
            self.state.velocity = self.state.velocity + up.scale(JUMP_SPEED);
            self.state.contact = None;
        }
        let intent = self.intent(system, input);
        step_observer(
            &mut self.state,
            self.frame_id,
            system,
            &probe,
            intent,
            Seconds::new(dt),
            &self.params,
        );
        dt
    }

    /// The observer in its frame's axes after `stepped` seconds of
    /// [`Walker::step`] from `system`'s time, the frame turned the way the
    /// core turned it.
    pub fn anchor(&self, system: &FrameSystem, stepped: f64) -> Anchor {
        let w = system
            .state(self.frame_id)
            .map_or(Vec3::zero(), |s| s.angular_velocity);
        let q = advance_orientation(system.root_orientation(self.frame_id), w, stepped);
        let to_frame = q.conjugate();
        let position = to_frame.rotate(self.state.position);
        let relative_velocity = to_frame.rotate(self.state.velocity) - w.cross(position);
        let contact = self.state.contact.map(|c| Contact {
            normal: to_frame.rotate(c.normal),
            ground_velocity: to_frame.rotate(c.ground_velocity),
            ..c
        });
        Anchor {
            position,
            relative_velocity,
            contact,
        }
    }

    /// Puts the observer back at `anchor` with `system` at its new time:
    /// the same place and velocity relative to the matter, in the frame's
    /// axes.
    pub fn reanchor(&mut self, system: &FrameSystem, anchor: Anchor) {
        let w = system
            .state(self.frame_id)
            .map_or(Vec3::zero(), |s| s.angular_velocity);
        let q = system.root_orientation(self.frame_id);
        self.state.position = q.rotate(anchor.position);
        self.state.velocity = q.rotate(w.cross(anchor.position) + anchor.relative_velocity);
        self.state.contact = anchor.contact.map(|c| Contact {
            normal: q.rotate(c.normal),
            ground_velocity: q.rotate(c.ground_velocity),
            ..c
        });
    }
}

/// The cells that tell which frames have matter: depth 0 and depth 1 of
/// every frame, in key order.
pub fn survey_keys(tree: &FrameTree) -> Vec<CellKey> {
    let mut out = Vec::new();
    for f in tree.frames() {
        if let Ok(k) = CellKey::new(f.frame_id, 0, 0, 0, 0) {
            out.push(k);
            out.extend(k.children().into_iter().flatten());
        }
    }
    out
}

/// The frames that can be spawned onto: those with a depth-0 or depth-1
/// cell loaded with a sample above zero density, in ascending frame id.
/// Number key `1` selects the first, `0` the tenth.
pub fn spawn_frames(cache: &CellCache, tree: &FrameTree) -> Vec<u64> {
    let mut out: Vec<u64> = survey_keys(tree)
        .iter()
        .filter(|k| matches!(cache.state(k), Some(CellState::Ready(c)) if c.has_matter))
        .map(|k| k.frame_id)
        .collect();
    out.dedup();
    out
}

/// The overlay lines of the spawn chooser: each frame with its number key.
pub fn spawn_menu_lines(frames: &[u64]) -> Vec<String> {
    if frames.is_empty() {
        return vec!["spawn: no frame with matter loaded yet (G closes)".to_string()];
    }
    let keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'];
    let choices: Vec<String> = frames
        .iter()
        .zip(keys)
        .map(|(f, k)| format!("{k} frame {f}"))
        .collect();
    vec![format!("spawn: {}  (G closes)", choices.join("  "))]
}

/// The overlay lines of the mode ([`Observer::status_lines`]) and, while
/// the spawn chooser is open, of the chooser ([`spawn_menu_lines`]).
pub fn mode_lines(
    observer: &Observer,
    system: &FrameSystem,
    cache: &CellCache,
    spawning: bool,
) -> Vec<String> {
    let mut lines = observer.status_lines(system);
    if spawning {
        lines.extend(spawn_menu_lines(&spawn_frames(cache, system.tree())));
    }
    lines
}

/// Resolves a spawn chooser index (`0` for key `1`) against
/// [`spawn_frames`] and spawns there. Returns `true` if the observer
/// spawned.
pub fn spawn_by_index(
    observer: &mut Observer,
    system: &FrameSystem,
    cache: &CellCache,
    index: usize,
    camera: &mut Camera,
) -> bool {
    match spawn_frames(cache, system.tree()).get(index) {
        Some(&frame_id) => observer.spawn(system, cache, frame_id, camera),
        None => false,
    }
}

/// Fly or walk, and the velocity fly mode coasts at (module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Observer {
    /// The body in walk mode; `None` in fly mode.
    pub walker: Option<Walker>,
    /// The velocity the camera coasts at in fly mode, root axes, meters per
    /// second: what it kept from walk mode, zero after a view key.
    pub drift: Vec3,
    /// The constants of the body in walk mode.
    pub params: ObserverParams,
}

impl Observer {
    /// Fly mode at rest with the default body.
    pub fn new() -> Observer {
        Observer::default()
    }

    /// `true` in walk mode.
    pub fn walking(&self) -> bool {
        self.walker.is_some()
    }

    /// `Tab`: from fly to walk with the eye where the camera is and the
    /// camera's velocity (its drift plus `flight_velocity`, the held flight
    /// keys, root axes) as the observer's, or from walk to fly with the
    /// observer's velocity as the drift.
    pub fn toggle(&mut self, system: &FrameSystem, camera: &mut Camera, flight_velocity: Vec3) {
        match self.walker.take() {
            Some(w) => {
                *camera = w.camera(system, camera);
                self.drift = w.root_velocity(system);
            }
            None => {
                let v = self.drift + flight_velocity;
                let w = Walker::from_camera(system, camera, v, self.params);
                *camera = w.camera(system, camera);
                self.walker = Some(w);
                self.drift = Vec3::zero();
            }
        }
    }

    /// Spawns onto `frame_id` in walk mode ([`Walker::spawn`]). Returns
    /// `false`, changing nothing, if there is no matter to spawn on.
    pub fn spawn(
        &mut self,
        system: &FrameSystem,
        cache: &CellCache,
        frame_id: u64,
        camera: &mut Camera,
    ) -> bool {
        match Walker::spawn(system, cache, frame_id, camera, self.params) {
            Some(w) => {
                *camera = w.camera(system, camera);
                self.walker = Some(w);
                self.drift = Vec3::zero();
                true
            }
            None => false,
        }
    }

    /// Back to fly mode at rest, for a view key that places the camera.
    pub fn stop(&mut self) {
        self.walker = None;
        self.drift = Vec3::zero();
    }

    /// The streaming priority: the cells around the walking observer's feet
    /// ([`feet_cells`]); none in fly mode.
    pub fn priority_cells(&self, system: &FrameSystem) -> Vec<CellKey> {
        match &self.walker {
            Some(w) => feet_cells(system, w.frame_id, w.feet_in_frame(system)),
            None => Vec::new(),
        }
    }

    /// One rendered frame: ticks `sim` by `wall_dt` with the hub's `now`
    /// (seconds of TDB since J2000) and moves the observer and `camera`
    /// through it. Fly: the camera keeps its place in root coordinates,
    /// flies by `input`, coasts at the drift, and re-parents. Walk: the
    /// observer is stepped first ([`Walker::step`], with the matter in
    /// `cache`), the frames are integrated, the observer is put back with
    /// its frame, and the camera is placed at its eye.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        sim: &mut Simulation,
        wall_dt: f64,
        hub_now: Seconds,
        cache: &CellCache,
        input: &FlightInput,
        jump: bool,
        camera: &mut Camera,
    ) {
        let start = sim.time();
        match &mut self.walker {
            Some(w) => {
                let mut clock = *sim.clock();
                clock.tick(Seconds::new(wall_dt), hub_now);
                let dt = (clock.time() - start).value();
                let walk = WalkInput::from_flight(input, jump);
                let stepped = w.step(sim.system(), cache, &walk, dt);
                let anchor = w.anchor(sim.system(), stepped);
                sim.update(Seconds::new(wall_dt), hub_now);
                w.reanchor(sim.system(), anchor);
                *camera = w.camera(sim.system(), camera);
            }
            None => {
                let held = camera.hold(sim.system());
                sim.update(Seconds::new(wall_dt), hub_now);
                camera.keep(sim.system(), held);
                camera.fly(input, wall_dt);
                let dt = (sim.time() - start).value();
                camera.coast(sim.system(), self.drift, dt);
                camera.update_parent(sim.system());
            }
        }
    }

    /// Applies a clock action and integrates to the new time, keeping the
    /// observer where it was: in fly mode at the same place in root
    /// coordinates (plus the drift over the time jumped), in walk mode at
    /// the same place on its frame.
    pub fn apply_clock(&mut self, sim: &mut Simulation, camera: &mut Camera, action: ClockAction) {
        let start = sim.time();
        match &mut self.walker {
            Some(w) => {
                let anchor = w.anchor(sim.system(), 0.0);
                sim.apply(action);
                w.reanchor(sim.system(), anchor);
                *camera = w.camera(sim.system(), camera);
            }
            None => {
                let held = camera.hold(sim.system());
                sim.apply(action);
                camera.keep(sim.system(), held);
                let dt = (sim.time() - start).value();
                camera.coast(sim.system(), self.drift, dt);
            }
        }
    }

    /// The overlay lines for the mode.
    pub fn status_lines(&self, system: &FrameSystem) -> Vec<String> {
        match &self.walker {
            None if self.drift == Vec3::zero() => vec!["mode fly".to_string()],
            None => vec![format!("mode fly  drift {:.3e} m/s", self.drift.length())],
            Some(w) if w.held => vec![
                format!("mode walk frame {}", w.frame_id),
                "waiting for matter under the observer".to_string(),
            ],
            Some(w) => vec![format!(
                "mode walk frame {}  {}  {:.2} m/s over matter",
                w.frame_id,
                if w.state.contact.is_some() {
                    "contact"
                } else {
                    "no contact"
                },
                w.speed_over_matter(system),
            )],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_turns_a_view_with_up() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let z = Vec3::new(0.0, 0.0, 1.0);
        let v = transport(x, z, x);
        assert!((v - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-12, "{v:?}");
        assert_eq!(transport(x, z, z), x);
    }

    #[test]
    fn the_view_keeps_away_from_up() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(level_enough(z, z, y), y);
        let v = level_enough(z, z, z);
        assert!(v.dot(z).abs() < 1e-12);
        let tilted = Vec3::new(1.0, 0.0, 0.5).normalized().unwrap();
        assert!((level_enough(tilted, z, y) - tilted).length() < 1e-15);
    }

    #[test]
    fn the_spawn_menu_numbers_frames_in_order() {
        assert_eq!(
            spawn_menu_lines(&[3, 7]),
            vec!["spawn: 1 frame 3  2 frame 7  (G closes)".to_string()]
        );
        assert!(spawn_menu_lines(&[])[0].contains("no frame"));
    }
}
