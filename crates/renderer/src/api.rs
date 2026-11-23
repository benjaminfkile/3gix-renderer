//! The renderer API, native binding: `docs/api.md` sections 1 to 5, 7
//! (all but `setComputeBudget`), and 8 (the `ready`, `error`, `camera`, and
//! `frameChanged` events).
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

use crate::camera::{Camera, FIELD_OF_VIEW_Y, SPEED_LIMITS};
use crate::config::StartOn;
use crate::controls::{Controls as Keys, EXPOSURE_BIAS_LIMITS, EXPOSURE_FIXED_LIMITS};
use crate::protocol::BuildDescription;
pub use crate::render::scene::MeshMode;
use crate::sim::{SimClock, Simulation};
use crate::spawn::Spawner;
pub use crate::tiles::SurfaceShape as Surface;
use crate::world::World;
use gx_core::frames::FrameSystem;
use gx_core::registry::FrameTree;
use gx_core::timescale::{tdb_seconds_to_utc_ms, utc_ms_to_tdb_seconds};
use gx_core::units::{Quat, Seconds, Vec3};
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
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The loop runs.
    Ready,
    /// A call failed when it was applied, after it was queued.
    Error(String),
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

/// The free camera's input from the application (`setFlightInput`,
/// `docs/api.md` section 4).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct FlightInput {
    /// Movement along the camera axes `[x right, y up, z back]`, each -1
    /// to 1 of the flight speed.
    pub r#move: [f64; 3],
    /// Turn this tick, radians: `[yaw, pitch, roll]` (yaw positive left,
    /// pitch positive up, roll positive counterclockwise seen from behind).
    pub look: [f64; 3],
    /// Fly at [`crate::camera::BOOST_FACTOR`] times the set speed.
    pub boost: bool,
}

impl FlightInput {
    /// What [`Camera::fly`] needs: the movement as analog axes, the look
    /// as a turn in radians.
    pub fn to_camera(self) -> crate::camera::FlightInput {
        let [x, y, z] = self.r#move;
        let [yaw, pitch, roll] = self.look;
        crate::camera::FlightInput {
            axes: (x, y, z),
            turn: (yaw, pitch, roll),
            boost: self.boost,
            ..crate::camera::FlightInput::default()
        }
    }
}

/// The time (`getTime`, `docs/api.md` section 5), UTC milliseconds.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Time {
    /// The instant drawn.
    pub utc_ms: i64,
    /// The universe's origin.
    pub origin_utc_ms: i64,
    /// Real now, as the hub reports it.
    pub now_utc_ms: i64,
    /// The playback rate, simulated seconds per second: negative runs
    /// back, 0 while the clock is paused.
    pub time_scale: f64,
}

impl Time {
    /// The time of `clock`.
    pub fn of(clock: &SimClock) -> Time {
        let rate = if clock.paused() {
            0.0
        } else if clock.rewinding() {
            -clock.rate()
        } else {
            clock.rate()
        };
        Time {
            utc_ms: tdb_seconds_to_utc_ms(clock.time().value()),
            origin_utc_ms: tdb_seconds_to_utc_ms(clock.origin().value()),
            now_utc_ms: tdb_seconds_to_utc_ms(clock.now().value()),
            time_scale: rate,
        }
    }
}

/// One frame of the registry (`frames()`, `docs/api.md` section 5).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrameInfo {
    /// The frame id.
    pub id: u64,
    /// The parent frame, `None` for a root.
    pub parent_id: Option<u64>,
    /// Mass, kilograms.
    pub mass: f64,
    /// Angular velocity about the frame axes, radians per second.
    pub angular_velocity: [f64; 3],
    /// The frame's level-0 surface tiles have arrived and one is drawn.
    pub has_surface: bool,
    /// The surface its level-0 tiles declare, once one has arrived.
    pub surface: Option<Surface>,
}

/// What [`Api::transform`] carries between frames.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A point: offset by the frames' origins and turned by their axes.
    Point,
    /// A direction: turned by the frames' axes only.
    Direction,
}

/// One layer of the build (`layers()`, `docs/api.md` section 7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    /// The layer id.
    pub id: String,
    /// A surface overlay layer.
    pub overlay: bool,
    /// The layer owns its frames exclusively.
    pub exclusive: bool,
    /// Shown as currently applied.
    pub shown: bool,
}

/// The exposure (`setExposure`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Exposure {
    /// Automatic, with a bias in stops (the exposure keys).
    Auto {
        /// Stops added to the automatic exposure.
        bias_stops: f64,
    },
    /// Fixed, as `--exposure-stops`.
    Fixed {
        /// The exposure, stops.
        stops: f64,
    },
}

/// What the loop gives [`Api::begin_tick`] to apply the queued commands to.
pub struct Target<'a> {
    /// The simulation (`setTime`, `setTimeScale`).
    pub sim: &'a mut Simulation,
    /// The camera.
    pub camera: &'a mut Camera,
    /// The spawner (`placeOver`).
    pub spawner: &'a mut Spawner,
    /// The built-in controls' state: attachment, exposure, mesh mode, and
    /// the layer chooser's toggles.
    pub keys: &'a mut Keys,
}

/// A queued setter, applied at the start of the next tick.
#[derive(Clone, Debug, PartialEq)]
enum Command {
    SetCamera(Pose),
    PlaceOver(StartOn),
    SetSpeed(f64),
    SetFov(f64),
    SetTime(i64),
    SetTimeScale(f64),
    SetShown(String, bool),
    SetExposure(Exposure),
    SetMeshMode(MeshMode),
}

/// Replaces the clock, keeping a free camera at the same place and view
/// direction in root coordinates across the jump in time, as the clock
/// keys do ([`crate::camera::apply_clock`]).
fn set_clock(sim: &mut Simulation, camera: &mut Camera, attached: bool, clock: SimClock) {
    if attached {
        sim.set_clock(clock);
        return;
    }
    let held = camera.hold(sim.system());
    sim.set_clock(clock);
    camera.keep(sim.system(), held);
    camera.update_parent(sim.system());
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
    time: Option<Time>,
    system: Option<FrameSystem>,
    frame_infos: Vec<FrameInfo>,
    layers: Vec<Layer>,
    paused: bool,
    destroyed: bool,
    hud: bool,
    max_fps: Option<u32>,
    size: Option<(u32, u32, f64)>,
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
            time: None,
            system: None,
            frame_infos: Vec::new(),
            layers: Vec::new(),
            paused: false,
            destroyed: false,
            hud: true,
            max_fps: None,
            size: None,
        }
    }

    /// The API with the layers of `build` (their flags; whether each is
    /// shown is read from the loop's world every tick).
    pub fn with_build(mut self, build: &BuildDescription) -> Api {
        self.layers = build
            .layers
            .iter()
            .map(|l| Layer {
                id: l.layer.clone(),
                overlay: l.overlay,
                exclusive: l.exclusive,
                shown: true,
            })
            .collect();
        self
    }

    fn live(&self) -> ApiResult<()> {
        if self.destroyed {
            return error("the renderer is destroyed");
        }
        Ok(())
    }

    /// `destroy`: the loop stops at the end of this tick
    /// ([`Api::destroyed`]); any later call is an error.
    pub fn destroy(&mut self) -> ApiResult<()> {
        self.live()?;
        self.destroyed = true;
        self.commands.clear();
        Ok(())
    }

    /// Whether [`Api::destroy`] was called: the loop stops and frees what
    /// it holds.
    pub fn destroyed(&self) -> bool {
        self.destroyed
    }

    /// `pause`: from the next tick, no drawing, streaming, or computing,
    /// and the simulation time stays where it is.
    pub fn pause(&mut self) -> ApiResult<()> {
        self.live()?;
        self.paused = true;
        Ok(())
    }

    /// `resume`: the loop runs again from the time it was paused at.
    pub fn resume(&mut self) -> ApiResult<()> {
        self.live()?;
        self.paused = false;
        Ok(())
    }

    /// Whether the loop is paused: it skips simulation, streaming, and
    /// drawing ([`Api::begin_tick`] still applies commands).
    pub fn paused(&self) -> bool {
        self.paused
    }

    /// `resize`: the surface size, physical pixels, and the pixel ratio,
    /// for hosts that size the surface themselves. Fails for a zero size
    /// or a ratio that is not above zero.
    pub fn resize(&mut self, width: u32, height: u32, pixel_ratio: f64) -> ApiResult<()> {
        self.live()?;
        if width == 0 || height == 0 {
            return error("the size must be above zero");
        }
        if !(pixel_ratio.is_finite() && pixel_ratio > 0.0) {
            return error("the pixel ratio must be above zero");
        }
        self.size = Some((width, height, pixel_ratio));
        Ok(())
    }

    /// The size asked for by [`Api::resize`] since the last call; clears
    /// it. The loop resizes its surface to it.
    pub fn take_size(&mut self) -> Option<(u32, u32, f64)> {
        self.size.take()
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
        self.live()?;
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
    /// ([`crate::spawn::Spawner`]). Queues an [`Event::Error`] when the
    /// placement cannot be made as it is applied.
    pub fn place_over(
        &mut self,
        frame_id: u64,
        direction: [f64; 3],
        height: f64,
        pitch_deg: f64,
    ) -> ApiResult<()> {
        self.live()?;
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

    /// `setFlightInput`: the free camera's input, held every tick until
    /// replaced; `None` clears it. Works in both control modes.
    pub fn set_flight_input(&mut self, input: Option<FlightInput>) {
        self.flight = input;
    }

    /// `setSpeed`: the set flight speed, clamped to [`SPEED_LIMITS`].
    /// Fails when not finite.
    pub fn set_speed(&mut self, mps: f64) -> ApiResult<()> {
        self.live()?;
        if !mps.is_finite() {
            return error("the speed is not finite");
        }
        let (lo, hi) = SPEED_LIMITS;
        self.commands.push(Command::SetSpeed(mps.clamp(lo, hi)));
        Ok(())
    }

    /// `setFov`: the vertical field of view, degrees, within
    /// [`FOV_LIMITS_DEG`], applied next tick to [`Camera::fov_y`], the one
    /// value projection, selection, and streaming read. [`Api::get_fov`]
    /// returns it at once.
    pub fn set_fov(&mut self, deg: f64) -> ApiResult<()> {
        self.live()?;
        let (lo, hi) = FOV_LIMITS_DEG;
        if !(lo..=hi).contains(&deg) {
            return error(format!(
                "field of view {deg} is outside {lo} to {hi} degrees"
            ));
        }
        self.fov_deg = deg;
        self.commands.push(Command::SetFov(deg));
        Ok(())
    }

    /// `getFov`: the vertical field of view, degrees.
    pub fn get_fov(&self) -> f64 {
        self.fov_deg
    }

    /// `getTime`: the time as of the last completed tick; `None` before
    /// the first.
    pub fn get_time(&self) -> Option<Time> {
        self.time
    }

    /// `setTime`: jumps to `utc_ms` next tick, clamped between the origin
    /// and now as the start options are ([`SimClock::starting_at`]), at the
    /// current playback rate.
    pub fn set_time(&mut self, utc_ms: i64) -> ApiResult<()> {
        self.live()?;
        self.commands.push(Command::SetTime(utc_ms));
        Ok(())
    }

    /// `setTimeScale`: the playback rate next tick; negative runs back.
    /// Its size is clamped to [`crate::sim::RATE_LIMITS`], and at now a
    /// forward rate stays at now ([`SimClock::with_rate`]). Fails for 0 or
    /// a value that is not finite.
    pub fn set_time_scale(&mut self, x: f64) -> ApiResult<()> {
        self.live()?;
        if !x.is_finite() || x == 0.0 {
            return error("the time scale must be finite and not 0");
        }
        self.commands.push(Command::SetTimeScale(x));
        Ok(())
    }

    /// `frames()`: every frame of the registry as of the last completed
    /// tick, in id order.
    pub fn frames(&self) -> Vec<FrameInfo> {
        self.frame_infos.clone()
    }

    /// `transform`: `point` (or a direction) in `from` frame axes into `to`
    /// frame axes at the instant drawn in the last completed tick. Fails
    /// for a frame not in the registry or before the first tick.
    pub fn transform(
        &self,
        point: [f64; 3],
        from: u64,
        to: u64,
        kind: Kind,
    ) -> ApiResult<[f64; 3]> {
        self.live()?;
        for id in [from, to] {
            if !self.frames.contains(&id) {
                return error(format!("frame {id} is not in the registry"));
            }
        }
        let Some(system) = &self.system else {
            return error("no tick has completed yet");
        };
        let p = Vec3::new(point[0], point[1], point[2]);
        let in_root_axes = match kind {
            Kind::Point => system.relative(p, from, Vec3::zero(), to),
            Kind::Direction => system.root_orientation(from).rotate(p),
        };
        let v = system.root_orientation(to).conjugate().rotate(in_root_axes);
        Ok([v.x, v.y, v.z])
    }

    /// `layers()`: the build's layers, shown as applied in the last
    /// completed tick.
    pub fn layers(&self) -> Vec<Layer> {
        self.layers.clone()
    }

    /// `setShown`: shows or hides the layer `layer_id` next tick, through
    /// the layer chooser's path (`rendering.md` section 7 item 7). A layer
    /// the build does not have queues an [`Event::Error`] when applied.
    pub fn set_shown(&mut self, layer_id: &str, shown: bool) -> ApiResult<()> {
        self.live()?;
        self.commands
            .push(Command::SetShown(layer_id.to_string(), shown));
        Ok(())
    }

    /// `setExposure`: as the exposure keys (`Auto`, the bias clamped to
    /// [`EXPOSURE_BIAS_LIMITS`]) and `--exposure-stops` (`Fixed`, clamped
    /// to [`EXPOSURE_FIXED_LIMITS`]). Fails when not finite.
    pub fn set_exposure(&mut self, exposure: Exposure) -> ApiResult<()> {
        self.live()?;
        let exposure = match exposure {
            Exposure::Auto { bias_stops: s } if s.is_finite() => {
                let (lo, hi) = EXPOSURE_BIAS_LIMITS;
                Exposure::Auto {
                    bias_stops: s.clamp(lo, hi),
                }
            }
            Exposure::Fixed { stops: s } if s.is_finite() => {
                let (lo, hi) = EXPOSURE_FIXED_LIMITS;
                Exposure::Fixed {
                    stops: s.clamp(lo, hi),
                }
            }
            _ => return error("the exposure is not finite"),
        };
        self.commands.push(Command::SetExposure(exposure));
        Ok(())
    }

    /// `setHud`: the built-in text overlay on (the default) or off.
    pub fn set_hud(&mut self, on: bool) -> ApiResult<()> {
        self.live()?;
        self.hud = on;
        Ok(())
    }

    /// Whether the built-in text overlay is drawn.
    pub fn hud(&self) -> bool {
        self.hud
    }

    /// The overlay text the loop draws: `text` with the overlay on, none
    /// with it off.
    pub fn overlay_text(&self, text: String) -> Option<String> {
        self.hud.then_some(text)
    }

    /// `setMeshMode`: as the `mesh-mode` option.
    pub fn set_mesh_mode(&mut self, mode: MeshMode) -> ApiResult<()> {
        self.live()?;
        self.commands.push(Command::SetMeshMode(mode));
        Ok(())
    }

    /// `setMaxFps`: as the `max-fps` option, `None` (or 0) for no cap.
    pub fn set_max_fps(&mut self, fps: Option<u32>) -> ApiResult<()> {
        self.live()?;
        self.max_fps = fps.filter(|&n| n > 0);
        Ok(())
    }

    /// The frame cap set by [`Api::set_max_fps`], if any.
    pub fn max_fps(&self) -> Option<u32> {
        self.max_fps
    }

    /// Drains the events queued since the last call, oldest first.
    pub fn events(&mut self) -> Vec<Event> {
        self.events.drain(..).collect()
    }

    /// The flight input of this tick: `keys` (the built-in controls' input,
    /// ignored with [`Controls::None`]) with the held input added: the
    /// keys' direction and the held movement add, the held look turns,
    /// and boost holds when either does.
    pub fn flight(&self, keys: crate::camera::FlightInput) -> crate::camera::FlightInput {
        let keys = match self.controls {
            Controls::Builtin => keys,
            Controls::None => crate::camera::FlightInput::default(),
        };
        let Some(held) = self.flight else {
            return keys;
        };
        let held = held.to_camera();
        crate::camera::FlightInput {
            axes: held.axes,
            turn: held.turn,
            boost: keys.boost || held.boost,
            ..keys
        }
    }

    /// Start of a tick: applies the queued commands in call order to the
    /// loop's state. A placed camera is free; a command that cannot be
    /// applied queues an [`Event::Error`].
    pub fn begin_tick(&mut self, target: Target<'_>) {
        let Target {
            sim,
            camera,
            spawner,
            keys,
        } = target;
        let mut shown: Vec<bool> = self.layers.iter().map(|l| l.shown).collect();
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
                        fov_y: camera.fov_y,
                    };
                    *spawner = Spawner::default();
                    keys.attached = false;
                }
                Command::PlaceOver(s) => match crate::options::launch_camera_at(sim.system(), s) {
                    Some(c) => {
                        *camera = Camera {
                            speed: camera.speed,
                            cruise: camera.cruise,
                            fov_y: camera.fov_y,
                            ..c
                        };
                        *spawner = Spawner {
                            active: None,
                            pending: Some(s),
                        };
                        keys.attached = false;
                    }
                    None => self.events.push_back(Event::Error(format!(
                        "placeOver: frame {} cannot be placed over",
                        s.frame_id
                    ))),
                },
                Command::SetSpeed(v) => camera.speed = v,
                Command::SetFov(deg) => camera.fov_y = deg.to_radians(),
                Command::SetTime(utc_ms) => {
                    let c = *sim.clock();
                    let time = Seconds::new(utc_ms_to_tdb_seconds(utc_ms));
                    let rate = Time::of(&c).time_scale;
                    let rate = if rate == 0.0 { 1.0 } else { rate };
                    let mut next = SimClock::starting_at(c.origin(), c.now(), time, rate);
                    if c.paused() && !next.is_at_now() {
                        next.apply(crate::sim::ClockAction::TogglePause);
                    }
                    set_clock(sim, camera, keys.attached, next);
                }
                Command::SetTimeScale(x) => {
                    let next = sim.clock().with_rate(x);
                    set_clock(sim, camera, keys.attached, next);
                }
                Command::SetShown(id, on) => match self.layers.iter().position(|l| l.id == id) {
                    Some(i) if shown[i] != on => {
                        shown[i] = on;
                        keys.toggle_layer(i);
                    }
                    Some(_) => {}
                    None => self.events.push_back(Event::Error(format!(
                        "setShown: the build has no layer {id:?}"
                    ))),
                },
                Command::SetExposure(Exposure::Auto { bias_stops }) => {
                    keys.exposure_fixed = None;
                    keys.exposure_bias = bias_stops;
                }
                Command::SetExposure(Exposure::Fixed { stops }) => {
                    keys.exposure_fixed = Some(stops);
                }
                Command::SetMeshMode(mode) => keys.mesh_mode = mode,
            }
        }
    }

    /// End of a tick: takes the snapshot the getters read (pose, time,
    /// frames at the instant drawn, layers), and queues `ready` on the
    /// first tick, then `frameChanged` and `camera` when the camera's
    /// frame or pose changed during the tick.
    pub fn end_tick(&mut self, sim: &Simulation, camera: &Camera, world: &World) {
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
        self.time = Some(Time::of(sim.clock()));
        let system = sim.system();
        let surfaces = world.surfaces();
        self.frame_infos = system
            .tree()
            .frames()
            .iter()
            .map(|f| {
                let w = f.angular_velocity;
                FrameInfo {
                    id: f.frame_id,
                    parent_id: (!f.is_root()).then_some(f.parent_frame_id),
                    mass: f.mass.value(),
                    angular_velocity: [w.x, w.y, w.z],
                    has_surface: surfaces.has_surface(f.frame_id),
                    surface: surfaces.shape(f.frame_id),
                }
            })
            .collect();
        self.system = Some(system.clone());
        let hidden = world.hidden_layers();
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.shown = !hidden.contains(&i);
        }
    }
}
