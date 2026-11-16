//! What the observer sees: every frame where it was when the light now
//! reaching the observer left it, and every light where it was when the
//! light now reaching the lit frame left it.
//!
//! Implements `space-model.md` section 6.4 with the core light-time solution
//! ([`gx_core::light`]):
//!
//! - Each frame is drawn at its retarded state for the observer
//!   ([`gx_core::light::retarded_state`]): position and orientation at the
//!   time `t_r` with `t_r = t - |x_obs - x_frame(t_r)| / c`, solved every
//!   draw against the simulation's fine history
//!   ([`crate::sim::Simulation::light_history`]).
//! - Each lit frame is lit by each emitting frame from where the emitter was
//!   one light-travel time before the lit frame's own retarded time
//!   ([`gx_core::light::retarded_emitter`]): [`Seen::emitter_shift`].
//!
//! The light of a frame travels from its matter nearest the camera when the
//! draw loop knows that distance ([`Seen::retarded_near`], from the drawn
//! meshes' boxes, [`crate::world::World::matter_distances`]), and from its
//! origin otherwise. From the center of a body the size of Earth the light
//! takes 0.02 s, in which the body moves hundreds of meters; a camera at
//! its surface would see the ground drawn that far from where it is.
//!
//! The observer is the camera's position in root coordinates at the
//! simulation time. Only its position enters, so the solution holds at any
//! flight speed.
//!
//! # The drawn system
//!
//! [`Seen::system`] is a [`FrameSystem`] whose frame states are the retarded
//! ones, and [`Seen::camera`] is the camera re-expressed in it at the same
//! place and view direction in space. Everything that draws (cell
//! selection, meshes, volumes, markers, lines, far field sprites, lights)
//! reads these two, so it all uses the same retarded positions. The drawn
//! system is built as a one-registry frame tree with the retarded states as
//! its records. Its root sits at the origin of its coordinates: drawn
//! coordinates are root coordinates shifted by the root frame's retarded
//! position, which leaves every vector between two points unchanged.

use crate::camera::Camera;
use gx_core::frames::{FrameState, FrameSystem};
use gx_core::history::History;
use gx_core::light::{retarded_emitter, retarded_state, LightTime};
use gx_core::registry::{FrameTree, Registry};
use gx_core::units::{Seconds, Vec3};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// One frame as seen: its retarded root state and the light-time solution.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SeenFrame {
    /// The frame.
    pub frame_id: u64,
    /// Position and velocity of the frame origin in root coordinates,
    /// orientation of its axes relative to root axes, and angular velocity
    /// about its own axes, at the retarded time.
    pub state: FrameState,
    /// The retarded time.
    pub light: LightTime,
}

/// The frames and the camera as the observer sees them at one moment
/// (module docs).
#[derive(Debug)]
pub struct Seen<'a> {
    system: FrameSystem,
    camera: Camera,
    time: Seconds,
    /// Root coordinates minus drawn coordinates.
    shift: Vec3,
    /// In ascending frame id.
    frames: Vec<SeenFrame>,
    history: Option<&'a History>,
    emitter_shifts: RefCell<BTreeMap<(u64, u64), Vec3>>,
}

impl Seen<'static> {
    /// What `camera` would see of `system` if light were instantaneous:
    /// every frame at the system's time. For drawing a bare frame system
    /// with no history (tests, tools).
    pub fn instant(system: &FrameSystem, camera: &Camera) -> Seen<'static> {
        let t = system.time();
        let frames = system
            .tree()
            .frames()
            .iter()
            .map(|f| SeenFrame {
                frame_id: f.frame_id,
                state: root_state(system, f.frame_id),
                light: LightTime {
                    time: t,
                    clamped: false,
                    iterations: 0,
                },
            })
            .collect();
        Seen {
            system: system.clone(),
            camera: *camera,
            time: t,
            shift: Vec3::zero(),
            frames,
            history: None,
            emitter_shifts: RefCell::default(),
        }
    }
}

impl<'a> Seen<'a> {
    /// What `camera` sees of `system` (every frame at the simulation time)
    /// with light time: every frame at its retarded state for the camera's
    /// position, solved against `history`, which must cover every retarded
    /// time ([`crate::sim::Simulation::light_history`]).
    pub fn retarded(history: &'a History, system: &FrameSystem, camera: &Camera) -> Seen<'a> {
        Seen::retarded_near(history, system, camera, &BTreeMap::new())
    }

    /// [`Seen::retarded`], with the light of each frame in `near` measured
    /// over that distance instead of the distance to its origin: `near`
    /// maps a frame id to the distance, meters, from the camera to the
    /// frame's matter nearest it. A camera close to a
    /// large body sees its near surface with the light time of that
    /// surface, not of the body's center (6400 km away on Earth, 0.02 s, in
    /// which the body moves hundreds of meters). An error `e` in a distance
    /// moves the drawn body by about `e v / c`.
    pub fn retarded_near(
        history: &'a History,
        system: &FrameSystem,
        camera: &Camera,
        near: &BTreeMap<u64, f64>,
    ) -> Seen<'a> {
        let t = system.time();
        let observer = camera.root_position(system);
        let view = camera.root_orientation(system);
        let tree = system.tree();
        let frames: Vec<SeenFrame> = tree
            .frames()
            .iter()
            .map(|f| {
                // The light-time equation measures from `from` to the frame
                // origin: put `from` `d` from the origin, on the line to the
                // camera, so the light travels `d`.
                let from = match near.get(&f.frame_id) {
                    Some(&d) => {
                        let origin = system.root_position(f.frame_id);
                        let to_camera = observer - origin;
                        let full = to_camera.length();
                        if full > d && full > 0.0 {
                            origin + to_camera.scale(d / full)
                        } else {
                            observer
                        }
                    }
                    None => observer,
                };
                let r = retarded_state(from, t, history, f.frame_id);
                SeenFrame {
                    frame_id: f.frame_id,
                    state: r.state,
                    light: r.light,
                }
            })
            .collect();
        let state_of = |id: u64| {
            let i = frames
                .binary_search_by_key(&id, |s| s.frame_id)
                .expect("every frame is seen");
            frames[i].state
        };
        let root = state_of(tree.root().frame_id);
        let records = tree
            .frames()
            .iter()
            .map(|f| {
                let s = state_of(f.frame_id);
                let mut record = *f;
                record.angular_velocity = s.angular_velocity;
                match tree.parent(f.frame_id) {
                    Some(p) => {
                        let p = state_of(p.frame_id);
                        record.position = s.position - p.position;
                        record.velocity = s.velocity - p.velocity;
                        record.orientation = (p.orientation.conjugate() * s.orientation)
                            .normalized()
                            .unwrap_or(f.orientation);
                    }
                    None => {
                        record.position = Vec3::zero();
                        record.velocity = Vec3::zero();
                        record.orientation = s.orientation.normalized().unwrap_or(f.orientation);
                    }
                }
                record
            })
            .collect();
        let registry = Registry::new(t, records)
            .expect("retarded states of a valid tree are a valid registry");
        let drawn = FrameSystem::from_tree(
            FrameTree::from_registries(&[registry]).expect("the same tree as the registry's"),
        );
        let shift = root.position;
        let to_frame = drawn.root_orientation(camera.frame_id).conjugate();
        let seen_camera = Camera {
            position: drawn.to_frame(observer - shift, camera.frame_id),
            orientation: (to_frame * view).normalized().unwrap_or(camera.orientation),
            ..*camera
        };
        Seen {
            system: drawn,
            camera: seen_camera,
            time: t,
            shift,
            frames,
            history: Some(history),
            emitter_shifts: RefCell::default(),
        }
    }

    /// Every frame at its retarded state: what is drawn.
    pub fn system(&self) -> &FrameSystem {
        &self.system
    }

    /// The camera in [`Seen::system`], at the same place and view direction
    /// in space as the camera it was made from.
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// The simulation time the frames are seen at, seconds of TDB since
    /// J2000.
    pub fn time(&self) -> Seconds {
        self.time
    }

    /// Every frame as seen, in ascending frame id.
    pub fn frames(&self) -> &[SeenFrame] {
        &self.frames
    }

    /// One frame as seen, or `None` if the id is not in the tree.
    pub fn frame(&self, frame_id: u64) -> Option<&SeenFrame> {
        self.frames
            .binary_search_by_key(&frame_id, |s| s.frame_id)
            .ok()
            .map(|i| &self.frames[i])
    }

    /// The light-travel time from the frame's origin to the observer,
    /// seconds: the simulation time minus the retarded time.
    pub fn light_time(&self, frame_id: u64) -> Option<Seconds> {
        self.frame(frame_id).map(|f| self.time - f.light.time)
    }

    /// The drawn position of a frame origin in root coordinates.
    pub fn root_position(&self, frame_id: u64) -> Vec3 {
        self.system.root_position(frame_id) + self.shift
    }

    /// How far the light of `emitter` moves for the matter of frame `lit`,
    /// root axes, meters: where the emitter's origin was when the light
    /// reaching the lit frame (at the lit frame's own retarded time) left
    /// it, minus where the emitter is drawn. Zero without light time, for a
    /// frame lit by its own matter, or for an unknown id.
    pub fn emitter_shift(&self, lit: u64, emitter: u64) -> Vec3 {
        let Some(history) = self.history else {
            return Vec3::zero();
        };
        if lit == emitter {
            return Vec3::zero();
        }
        let (Some(l), Some(e)) = (self.frame(lit), self.frame(emitter)) else {
            return Vec3::zero();
        };
        if let Some(s) = self.emitter_shifts.borrow().get(&(lit, emitter)) {
            return *s;
        }
        let from = retarded_emitter(l.state.position, l.light.time, history, emitter);
        let s = from.position - e.state.position;
        self.emitter_shifts.borrow_mut().insert((lit, emitter), s);
        s
    }

    /// The frame whose origin is nearest the crosshair (the view direction)
    /// in angle, among those in front of the camera, with its light time.
    /// Ties go to the lower id. `None` when no frame is in front.
    pub fn nearest_to_crosshair(&self) -> Option<(u64, Seconds)> {
        let forward = self
            .camera
            .root_orientation(&self.system)
            .rotate(Vec3::new(0.0, 0.0, -1.0));
        let mut best: Option<(u64, f64)> = None;
        for f in &self.frames {
            let rel = self.camera.relative(&self.system, Vec3::zero(), f.frame_id);
            let Some(dir) = rel.normalized() else {
                continue;
            };
            let c = dir.dot(forward);
            if c > 0.0 && best.is_none_or(|(_, b)| c > b) {
                best = Some((f.frame_id, c));
            }
        }
        best.map(|(id, _)| (id, self.time - self.frame(id).expect("seen").light.time))
    }
}

/// A frame's state in root coordinates.
fn root_state(system: &FrameSystem, frame_id: u64) -> FrameState {
    FrameState {
        position: system.root_position(frame_id),
        velocity: system.root_velocity(frame_id),
        orientation: system.root_orientation(frame_id),
        angular_velocity: system
            .state(frame_id)
            .expect("a frame of the tree")
            .angular_velocity,
    }
}
