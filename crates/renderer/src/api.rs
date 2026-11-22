//! The renderer API, native binding: `docs/api.md` sections 1, 2, 4, and 8
//! (the `ready`, `camera`, and `frameChanged` events).
//!
//! One implementation for every build (`docs/api.md` section 1 item 1):
//! nothing here is native or web specific. A loop owns the simulation, the
//! camera, and the spawner, and drives an [`Api`] each tick:
//!
//! 1. the [`Consumer`] (the application) calls setters, which queue
//!    commands, and getters, which read the state of the last completed
//!    tick,
//! 2. [`Api::begin_tick`] applies the queued commands in call order,
//! 3. the loop flies the camera with [`Api::flight`] (the held flight
//!    input combined with the keys when the controls are
//!    [`Controls::Builtin`]), simulates, and draws,
//! 4. [`Api::end_tick`] takes the snapshot the getters read and queues the
//!    events, drained by [`Api::events`].

use crate::camera::{Camera, FlightInput, FIELD_OF_VIEW_Y, SPEED_LIMITS};
use crate::config::StartOn;
use crate::sim::Simulation;
use crate::spawn::Spawner;
use gx_core::registry::FrameTree;
use gx_core::units::{Quat, Vec3};
use std::collections::{BTreeSet, VecDeque};

/// How far from one the length of a pose's quaternion may be.
pub const UNIT_TOLERANCE: f64 = 1e-6;

/// Limits of the vertical field of view, degrees.
pub const FOV_LIMITS_DEG: (f64, f64) = (1.0, 179.0);

/// An API error: the message is the same in both bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError(pub String);

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ApiError {}

/// The result of an API item.
pub type ApiResult<T> = Result<T, ApiError>;

fn error<T>(message: impl Into<String>) -> ApiResult<T> {
    Err(ApiError(message.into()))
}

/// The camera's pose (`docs/api.md` section 2): the fields of [`Camera`]
/// an application sets and reads.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Pose {
    /// The frame the camera is parented to.
    pub frame_id: u64,
    /// Position relative to the frame origin, in the frame axes, meters.
    pub position: [f64; 3],
    /// Camera axes relative to the frame axes, a unit quaternion
    /// `[x, y, z, w]`.
    pub orientation: [f64; 4],
    /// The set flight speed, meters per second.
    pub speed: f64,
}

impl Pose {
    /// The pose of `camera`.
    pub fn of(camera: &Camera) -> Pose {
        let (p, q) = (camera.position, camera.orientation);
        Pose {
            frame_id: camera.frame_id,
            position: [p.x, p.y, p.z],
            orientation: [q.x, q.y, q.z, q.w],
            speed: camera.speed,
        }
    }
}

/// Who flies the camera from the keyboard and mouse (`setControls`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Controls {
    /// The key and mouse bindings of `docs/controls.md` (the default).
    #[default]
    Builtin,
    /// Keyboard and mouse do nothing to the camera, choosers, clock, and
    /// exposure; window events (resize, focus, close) still apply.
    None,
}

/// An event (`docs/api.md` section 8).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Event {
    /// The loop runs.
    Ready,
    /// The pose changed during the tick (at most one per tick).
    Camera(Pose),
    /// The camera's frame changed.
    FrameChanged {
        /// The frame before.
        from: u64,
        /// The frame after.
        to: u64,
    },
}

/// A queued setter, applied at the start of the next tick.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Command {
    SetCamera(Pose),
    PlaceOver(StartOn),
    SetSpeed(f64),
}

/// The application's side of the loop: runs once per tick on the render
/// thread, before simulation and drawing.
pub trait Consumer {
    /// One tick: call setters and getters, drain [`Api::events`].
    fn tick(&mut self, api: &mut Api);
}

/// A consumer that does nothing: the desktop window with only the built-in
/// controls.
impl Consumer for () {
    fn tick(&mut self, _api: &mut Api) {}
}

/// The renderer API: the command queue, the snapshot getters read, and the
/// event queue.
#[derive(Clone, Debug)]
pub struct Api {
    frames: BTreeSet<u64>,
    commands: Vec<Command>,
    events: VecDeque<Event>,
    pose: Pose,
    ready: bool,
    controls: Controls,
    flight: Option<FlightInput>,
    fov_deg: f64,
}

impl Api {
    /// The API of a loop over the frames of `tree`, its camera starting at
    /// `camera`.
    pub fn new(tree: &FrameTree, camera: &Camera) -> Api {
        Api {
            frames: tree.frames().iter().map(|f| f.frame_id).collect(),
            commands: Vec::new(),
            events: VecDeque::new(),
            pose: Pose::of(camera),
            ready: false,
            controls: Controls::Builtin,
            flight: None,
            fov_deg: FIELD_OF_VIEW_Y.to_degrees(),
        }
    }

    /// `getCamera`: the pose as of the last completed tick.
    pub fn get_camera(&self) -> Pose {
        self.pose
    }

    /// `setCamera`: puts the camera exactly at `pose` next tick, detached
    /// from any spawn. Fails for a frame not in the registry, a non-finite
    /// value, or a quaternion whose length is off one by more than
    /// [`UNIT_TOLERANCE`].
    pub fn set_camera(&mut self, pose: Pose) -> ApiResult<()> {
        if !self.frames.contains(&pose.frame_id) {
            return error(format!("frame {} is not in the registry", pose.frame_id));
        }
        let values = pose.position.iter().chain(&pose.orientation);
        if values.chain([&pose.speed]).any(|v| !v.is_finite()) {
            return error("the pose has a value that is not finite");
        }
        let [x, y, z, w] = pose.orientation;
        let length = (x * x + y * y + z * z + w * w).sqrt();
        if (length - 1.0).abs() > UNIT_TOLERANCE {
            return error(format!(
                "the orientation is not a unit quaternion (length {length})"
            ));
        }
        let (lo, hi) = SPEED_LIMITS;
        if !(lo..=hi).contains(&pose.speed) {
            return error(format!("speed {} is outside {lo} to {hi} m/s", pose.speed));
        }
        self.commands.push(Command::SetCamera(pose));
        Ok(())
    }

    /// `placeOver`: the placement of `--start-on
    /// <frame>:<x>,<y>,<z>@+<height>^<pitch>`: looks at the frame until its
    /// drawn surface along `direction` (frame axes) has arrived, then
    /// spawns `height` meters above it looking `pitch_deg` below level
    /// ([`crate::spawn::Spawner`]).
    pub fn place_over(
        &mut self,
        frame_id: u64,
        direction: [f64; 3],
        height: f64,
        pitch_deg: f64,
    ) -> ApiResult<()> {
        if !self.frames.contains(&frame_id) {
            return error(format!("frame {frame_id} is not in the registry"));
        }
        let Some(direction) = Vec3::new(direction[0], direction[1], direction[2]).normalized()
        else {
            return error("the direction is zero or not finite");
        };
        if !(height.is_finite() && height > 0.0) {
            return error("the height must be above zero");
        }
        if !(-90.0..=90.0).contains(&pitch_deg) {
            return error("the pitch must be between -90 and 90 degrees");
        }
        self.commands.push(Command::PlaceOver(StartOn {
            frame_id,
            direction,
            distance: None,
            height: Some(height),
            pitch_deg: Some(pitch_deg),
        }));
        Ok(())
    }

    /// `setControls`: applies at once to the input the window passes on.
    pub fn set_controls(&mut self, controls: Controls) {
        self.controls = controls;
    }

    /// The controls set.
    pub fn controls(&self) -> Controls {
        self.controls
    }

    /// `setFlightInput`: the free camera's input, the same
    /// [`FlightInput`] the keys produce, held every tick until replaced;
    /// `None` clears it. Works in both control modes.
    pub fn set_flight_input(&mut self, input: Option<FlightInput>) {
        self.flight = input;
    }

    /// `setSpeed`: the set flight speed, clamped to [`SPEED_LIMITS`].
    /// Fails when not finite.
    pub fn set_speed(&mut self, mps: f64) -> ApiResult<()> {
        if !mps.is_finite() {
            return error("the speed is not finite");
        }
        let (lo, hi) = SPEED_LIMITS;
        self.commands.push(Command::SetSpeed(mps.clamp(lo, hi)));
        Ok(())
    }

    /// `setFov`: the vertical field of view, degrees, within
    /// [`FOV_LIMITS_DEG`].
    pub fn set_fov(&mut self, deg: f64) -> ApiResult<()> {
        let (lo, hi) = FOV_LIMITS_DEG;
        if !(lo..=hi).contains(&deg) {
            return error(format!(
                "field of view {deg} is outside {lo} to {hi} degrees"
            ));
        }
        self.fov_deg = deg;
        Ok(())
    }

    /// `getFov`: the vertical field of view, degrees.
    pub fn get_fov(&self) -> f64 {
        self.fov_deg
    }

    /// Drains the events queued since the last call, oldest first.
    pub fn events(&mut self) -> Vec<Event> {
        self.events.drain(..).collect()
    }

    /// The flight input of this tick: `keys` (the built-in controls' input,
    /// ignored with [`Controls::None`]) with the held input added: a key
    /// or the held input holding a direction moves the camera along it,
    /// look and wheel add, and boost holds when either does.
    pub fn flight(&self, keys: FlightInput) -> FlightInput {
        let keys = match self.controls {
            Controls::Builtin => keys,
            Controls::None => FlightInput::default(),
        };
        let Some(held) = self.flight else {
            return keys;
        };
        FlightInput {
            forward: keys.forward || held.forward,
            back: keys.back || held.back,
            left: keys.left || held.left,
            right: keys.right || held.right,
            up: keys.up || held.up,
            down: keys.down || held.down,
            look: (keys.look.0 + held.look.0, keys.look.1 + held.look.1),
            wheel: keys.wheel + held.wheel,
            boost: keys.boost || held.boost,
        }
    }

    /// Start of a tick: applies the queued commands in call order to the
    /// loop's camera and spawner. `attached` is the loop's flag of a
    /// camera attached to the frame it spawned onto; a placed camera is
    /// free.
    pub fn begin_tick(
        &mut self,
        sim: &Simulation,
        camera: &mut Camera,
        spawner: &mut Spawner,
        attached: &mut bool,
    ) {
        for command in std::mem::take(&mut self.commands) {
            match command {
                Command::SetCamera(p) => {
                    let [x, y, z, w] = p.orientation;
                    *camera = Camera {
                        frame_id: p.frame_id,
                        position: Vec3::new(p.position[0], p.position[1], p.position[2]),
                        orientation: Quat::new(x, y, z, w)
                            .normalized()
                            .unwrap_or(Quat::identity()),
                        speed: p.speed,
                        cruise: p.speed,
                    };
                    *spawner = Spawner::default();
                    *attached = false;
                }
                Command::PlaceOver(s) => {
                    if let Some(c) = crate::options::launch_camera_at(sim.system(), s) {
                        *camera = Camera {
                            speed: camera.speed,
                            cruise: camera.cruise,
                            ..c
                        };
                        *spawner = Spawner {
                            active: None,
                            pending: Some(s),
                        };
                        *attached = false;
                    }
                }
                Command::SetSpeed(v) => camera.speed = v,
            }
        }
    }

    /// End of a tick: takes the snapshot the getters read, and queues
    /// `ready` on the first tick, then `frameChanged` and `camera` when
    /// the camera's frame or pose changed during the tick.
    pub fn end_tick(&mut self, camera: &Camera) {
        let pose = Pose::of(camera);
        if !self.ready {
            self.ready = true;
            self.events.push_back(Event::Ready);
        } else if pose != self.pose {
            if pose.frame_id != self.pose.frame_id {
                self.events.push_back(Event::FrameChanged {
                    from: self.pose.frame_id,
                    to: pose.frame_id,
                });
            }
            self.events.push_back(Event::Camera(pose));
        }
        self.pose = pose;
    }
}
