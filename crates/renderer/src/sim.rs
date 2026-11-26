//! Simulation time, the integration of every frame to it, and the history
//! the light-time solution reads.
//!
//! Implements `space-model.md` section 7 (time) on the renderer side, with
//! the laws of sections 6.1 and 6.2 through the core integrator and the
//! history section 6.4 solves the light-time equation against.
//!
//! # Time
//!
//! Simulation time is universe time (`space-model.md` section 7 item 5):
//! seconds since the origin, which is 0 and the epoch of every registry
//! (`matter-format.md` section 5). The hub reports the origin and its now
//! as wall-clock milliseconds; "now" is the hub's now, kept by
//! [`crate::protocol::ClockSync`]. The [`SimClock`] holds the origin's
//! wall-clock milliseconds and converts both ways with
//! [`gx_core::clock::universe_seconds`] and [`gx_core::clock::wall_ms`]
//! ([`SimClock::universe_seconds`], [`SimClock::wall_ms`]). It holds the
//! simulation time, always between the origin and now:
//!
//! - It starts at now and, while at now, follows now at one second per
//!   second.
//! - It can leave now for the past: step back one hour, rewind, pause, or
//!   play at half speed. In the past it plays forward or backward at any
//!   rate between [`RATE_LIMITS`].
//! - Playing forward into now locks it to now again at rate 1; it can never
//!   go past now. Going back stops at the origin.
//!
//! # Laws
//!
//! Every frame is integrated with [`gx_core::integrate::Scheme::Relativistic`]
//! (first post-Newtonian gravity, oblateness, and rotation under torques)
//! from the registry at its epoch. A coarse [`History`] keeps a checkpoint
//! every [`CHECKPOINT_SECONDS`] of simulated time, so a jump anywhere
//! between the origin and now integrates at most one checkpoint interval
//! from the nearest checkpoint before it.
//!
//! The integration step is [`max_step`]: at most [`MAX_STEP_SECONDS`], and
//! at most [`STEPS_PER_DYNAMICAL_TIME`] steps per the shortest dynamical
//! time between two frames, so a tight orbit gets as many steps as a wide
//! one.
//!
//! A second, fine history (checkpoints every step or more) covers
//! the simulation time and the light-travel time before it: every retarded
//! time the frames are drawn and lit at ([`crate::seen`]) lies in it, so
//! each pass of the light-time solution costs at most one integration step.
//! It starts [`RECENT_SLACK_SECONDS`] before the earliest retarded time
//! needed, so playing backwards rebuilds it only once per that span, and it
//! is rebuilt from the coarse history whenever the time leaves it. The
//! frame states at the simulation time come from it too, so drawing,
//! lighting, and the camera all read one integration.
//!
//! All of it is a pure function of the registry and the times asked for.

use crate::camera::Camera;
use crate::seen::Seen;
use gx_core::frames::FrameSystem;
use gx_core::gravity::C;
use gx_core::history::History;
use gx_core::integrate::{advance, Scheme};
use gx_core::units::{Seconds, Vec3};

/// The integration scheme: post-Newtonian gravity, oblateness, and
/// rotation with torques (`space-model.md` sections 6.1 and 6.2).
pub const SCHEME: Scheme = Scheme::Relativistic;

/// The longest integration step, seconds ([`max_step`]).
pub const MAX_STEP_SECONDS: f64 = 600.0;

/// Integration steps at least per dynamical time `sqrt(d^3 / (G (m1 + m2)))`
/// of any two frames ([`max_step`]): about 1600 per circular orbit.
pub const STEPS_PER_DYNAMICAL_TIME: f64 = 256.0;

/// Simulated seconds between two checkpoints of the coarse history:
/// 86 400 s.
pub const CHECKPOINT_SECONDS: f64 = 86_400.0;

/// The most checkpoints the coarse history holds before it thins itself
/// (doubling its interval): 1024 intervals of 86 400 s.
pub const CHECKPOINT_CAPACITY: usize = 1024;

/// The most checkpoints the fine history holds.
pub const RECENT_CAPACITY: usize = 1024;

/// How far the fine history reaches before the earliest retarded time it
/// is built for, seconds.
pub const RECENT_SLACK_SECONDS: f64 = 21_600.0;

/// How far before the origin the histories reach, seconds: the light-travel
/// time across the larger of the root frame's cube (corner to corner) and
/// twice the farthest frame origin from the root origin at `system`'s time,
/// plus [`LIGHT_TIME_MARGIN`]. The origin starts the recorded time a client
/// can go back to, not the universe: light reaching an observer just after
/// the origin left its frames before it. The registry is integrated
/// backwards this far (the laws allow it).
pub fn before_origin(system: &FrameSystem) -> f64 {
    let tree = system.tree();
    let farthest = tree
        .frames()
        .iter()
        .map(|f| system.root_position(f.frame_id).length())
        .fold(0.0, f64::max);
    let across = (tree.root().root_extent.value() * 3.0f64.sqrt()).max(2.0 * farthest);
    across / C * (1.0 + LIGHT_TIME_MARGIN)
}

/// The integration step for `system`, seconds: [`MAX_STEP_SECONDS`], or
/// less where two frames with mass are close enough that their dynamical
/// time `sqrt(d^3 / (G (m1 + m2)))` (distance `d` at the system's time)
/// over [`STEPS_PER_DYNAMICAL_TIME`] is shorter (a massless frame counts:
/// it moves in the other's field). The root frame is the coordinate origin
/// and is left out. Nothing about the frames but their masses and
/// positions is used.
pub fn max_step(system: &FrameSystem) -> f64 {
    // The root is the coordinate origin, not a body: it does not move.
    let frames: Vec<_> = system
        .tree()
        .frames()
        .iter()
        .filter(|f| !f.is_root())
        .collect();
    let mut step = MAX_STEP_SECONDS;
    for (i, a) in frames.iter().enumerate() {
        for b in &frames[i + 1..] {
            let gm = gx_core::gravity::G * (a.mass.value() + b.mass.value());
            let d = (system.root_position(a.frame_id) - system.root_position(b.frame_id)).length();
            if gm > 0.0 && d > 0.0 {
                let t = (d * d * d / gm).sqrt();
                step = step.min(t / STEPS_PER_DYNAMICAL_TIME);
            }
        }
    }
    step
}

/// Relative margin on the light-travel time the fine history covers:
/// frames move at a small fraction of the speed of light, so their
/// distances at the retarded times differ from the current ones by less
/// than this.
pub const LIGHT_TIME_MARGIN: f64 = 0.01;

/// The step of `,` and `.`, seconds: one hour.
pub const STEP_SECONDS: f64 = 3_600.0;

/// The smallest and largest playback rate, simulated seconds per
/// wall-clock second, that `[` and `]` reach: 2^-10 to 2^20. At the top
/// rate one rendered frame at 60 frames per second asks for about 30
/// integration steps.
pub const RATE_LIMITS: (f64, f64) = (1.0 / 1024.0, 1_048_576.0);

/// A change to the clock requested by the user.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClockAction {
    /// Pause or resume. Pausing at now leaves now.
    TogglePause,
    /// Halve the playback rate. At now this leaves now at half speed, so
    /// the simulation time falls behind now.
    HalveRate,
    /// Double the playback rate. Does nothing at now, where the rate is 1.
    DoubleRate,
    /// Step back [`STEP_SECONDS`], at most to the origin.
    StepBack,
    /// Step forward [`STEP_SECONDS`], at most to now.
    StepForward,
    /// Toggle playing backwards at the current rate. At now this leaves
    /// now going backwards at rate 1.
    ToggleRewind,
    /// Jump to now and follow it.
    JumpToNow,
}

/// The simulation clock: a time between the origin and now, a playback
/// rate and direction, and whether it follows now (module docs).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimClock {
    time: Seconds,
    origin: Seconds,
    origin_ms: i64,
    now: Seconds,
    rate: f64,
    rewinding: bool,
    paused: bool,
    at_now: bool,
}

impl SimClock {
    /// A clock at now, following it. Times are universe seconds; a `now`
    /// before the origin is taken as the origin. The origin's wall-clock
    /// instant is 0 milliseconds until [`SimClock::with_origin_ms`] sets
    /// it.
    pub fn at_now(origin: Seconds, now: Seconds) -> SimClock {
        let now = if now < origin { origin } else { now };
        SimClock {
            time: now,
            origin,
            origin_ms: 0,
            now,
            rate: 1.0,
            rewinding: false,
            paused: false,
            at_now: true,
        }
    }

    /// A clock at `time`, clamped between the origin and now, playing
    /// forward at `rate` (its magnitude, clamped to [`RATE_LIMITS`]; a
    /// negative rate plays backwards). A time at or after now follows now.
    pub fn starting_at(origin: Seconds, now: Seconds, time: Seconds, rate: f64) -> SimClock {
        let mut c = SimClock::at_now(origin, now);
        if time < c.now {
            let (lo, hi) = RATE_LIMITS;
            c.at_now = false;
            c.time = time;
            c.rate = if rate.is_finite() && rate != 0.0 {
                rate.abs().clamp(lo, hi)
            } else {
                1.0
            };
            c.rewinding = rate < 0.0;
            c.settle();
        }
        c
    }

    /// This clock with the origin at the wall-clock instant `origin_ms`,
    /// milliseconds, as the hub reports it. Universe times are unchanged.
    pub fn with_origin_ms(self, origin_ms: i64) -> SimClock {
        SimClock { origin_ms, ..self }
    }

    /// The origin's wall-clock instant, milliseconds.
    pub fn origin_ms(&self) -> i64 {
        self.origin_ms
    }

    /// The universe time of the wall-clock instant `wall_ms`, milliseconds
    /// ([`gx_core::clock::universe_seconds`] from this clock's origin).
    pub fn universe_seconds(&self, wall_ms: i64) -> Seconds {
        gx_core::clock::universe_seconds(wall_ms, self.origin_ms)
    }

    /// The wall-clock instant, milliseconds, of universe time `t`
    /// ([`gx_core::clock::wall_ms`] from this clock's origin).
    pub fn wall_ms(&self, t: Seconds) -> i64 {
        gx_core::clock::wall_ms(t, self.origin_ms)
    }

    /// The simulation time, universe seconds.
    pub fn time(&self) -> Seconds {
        self.time
    }

    /// The origin, universe seconds: the registry epoch, 0.
    pub fn origin(&self) -> Seconds {
        self.origin
    }

    /// The latest now the clock was given, universe seconds.
    pub fn now(&self) -> Seconds {
        self.now
    }

    /// The simulation time minus now, seconds: 0 at now, negative in the
    /// past.
    pub fn offset_from_now(&self) -> Seconds {
        self.time - self.now
    }

    /// Simulated seconds per wall-clock second, always above 0; the
    /// direction is [`SimClock::rewinding`].
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// `true` while playing backwards.
    pub fn rewinding(&self) -> bool {
        self.rewinding
    }

    /// `true` while paused.
    pub fn paused(&self) -> bool {
        self.paused
    }

    /// `true` while the clock follows now.
    pub fn is_at_now(&self) -> bool {
        self.at_now
    }

    /// Takes the hub's `now` and advances the time by `wall_dt` times the
    /// rate, backwards while rewinding, unless paused or at now. The result
    /// is clamped between the origin and now; reaching now follows it.
    pub fn tick(&mut self, wall_dt: Seconds, now: Seconds) {
        self.now = if now < self.origin { self.origin } else { now };
        if !self.at_now && !self.paused {
            let rate = if self.rewinding {
                -self.rate
            } else {
                self.rate
            };
            self.time = self.time + wall_dt * rate;
        }
        self.settle();
    }

    /// Applies a user action (see [`ClockAction`]).
    pub fn apply(&mut self, action: ClockAction) {
        let (lo, hi) = RATE_LIMITS;
        match action {
            ClockAction::TogglePause => {
                if self.at_now {
                    self.at_now = false;
                    self.paused = true;
                } else {
                    self.paused = !self.paused;
                }
            }
            ClockAction::HalveRate => {
                if self.at_now {
                    self.at_now = false;
                    self.rate = 0.5;
                } else {
                    self.rate = (self.rate * 0.5).max(lo);
                }
            }
            ClockAction::DoubleRate => {
                if !self.at_now {
                    self.rate = (self.rate * 2.0).min(hi);
                }
            }
            ClockAction::StepBack => {
                self.at_now = false;
                self.time = self.time - Seconds::new(STEP_SECONDS);
            }
            ClockAction::StepForward => {
                if !self.at_now {
                    self.time = self.time + Seconds::new(STEP_SECONDS);
                }
            }
            ClockAction::ToggleRewind => {
                if self.at_now {
                    self.at_now = false;
                    self.rewinding = true;
                } else {
                    self.rewinding = !self.rewinding;
                }
            }
            ClockAction::JumpToNow => self.follow_now(),
        }
        self.settle();
    }

    /// This clock at playback rate `rate` (`setTimeScale`): its size
    /// clamped to [`RATE_LIMITS`], backwards when negative, playing. At
    /// now a forward rate of 1 or more stays at now, since the time never
    /// passes now; any other rate leaves now.
    pub fn with_rate(self, rate: f64) -> SimClock {
        let mut c = self;
        if !rate.is_finite() || rate == 0.0 || (c.at_now && rate >= 1.0) {
            return c;
        }
        let (lo, hi) = RATE_LIMITS;
        c.at_now = false;
        c.paused = false;
        c.rate = rate.abs().clamp(lo, hi);
        c.rewinding = rate < 0.0;
        c.settle();
        c
    }

    /// Clamps the time between the origin and now: past now (or at now)
    /// it follows now.
    fn settle(&mut self) {
        if self.at_now || self.time > self.now {
            self.follow_now();
        }
        if self.time < self.origin {
            self.time = self.origin;
        }
    }

    fn follow_now(&mut self) {
        self.at_now = true;
        self.time = self.now;
        self.rate = 1.0;
        self.rewinding = false;
        self.paused = false;
    }
}

/// Every frame integrated to a clock, with the coarse and fine histories
/// (module docs).
#[derive(Clone, Debug)]
pub struct Simulation {
    /// The clock the system follows. Change it through
    /// [`Simulation::update`] and [`Simulation::apply`], which integrate.
    clock: SimClock,
    /// Checkpoints every [`CHECKPOINT_SECONDS`] from the origin.
    history: History,
    /// Checkpoints every [`MAX_STEP_SECONDS`] or more around the time.
    recent: History,
    /// Every frame at the clock's time.
    system: FrameSystem,
    /// The integration step, seconds ([`max_step`]).
    step: Seconds,
}

impl Simulation {
    /// The simulation of `system`, a frame system at the registry epoch
    /// (the origin), on `clock`.
    pub fn new(system: FrameSystem, clock: SimClock) -> Simulation {
        let step = Seconds::new(max_step(&system));
        tracing::info!(
            step_seconds = step.value(),
            before_origin_seconds = before_origin(&system),
            "integration"
        );
        let mut early = system.clone();
        let start = early.time() - Seconds::new(before_origin(&system));
        advance(&mut early, start, step, SCHEME);
        let history = History::new(
            early,
            Seconds::new(CHECKPOINT_SECONDS),
            step,
            SCHEME,
            CHECKPOINT_CAPACITY,
        );
        let recent = History::new(system.clone(), step, step, SCHEME, RECENT_CAPACITY);
        let mut sim = Simulation {
            clock,
            history,
            recent,
            system,
            step,
        };
        sim.sync();
        sim
    }

    /// The simulation of `system` (at its epoch, the origin) entered at
    /// `now`, universe seconds, following now.
    pub fn at_now(system: FrameSystem, now: Seconds) -> Simulation {
        let clock = SimClock::at_now(system.tree().epoch(), now);
        Simulation::new(system, clock)
    }

    /// The clock.
    pub fn clock(&self) -> &SimClock {
        &self.clock
    }

    /// Every frame at the simulation time.
    pub fn system(&self) -> &FrameSystem {
        &self.system
    }

    /// The simulation time, universe seconds.
    pub fn time(&self) -> Seconds {
        self.system.time()
    }

    /// The origin, universe seconds: 0.
    pub fn origin(&self) -> Seconds {
        self.clock.origin()
    }

    /// The coarse history.
    pub fn history(&self) -> &History {
        &self.history
    }

    /// Ticks the clock by `wall_dt` with the hub's `now` (universe
    /// seconds) and integrates every frame to the new time.
    pub fn update(&mut self, wall_dt: Seconds, now: Seconds) {
        self.clock.tick(wall_dt, now);
        self.sync();
    }

    /// Applies a user action to the clock and integrates every frame to the
    /// new time.
    pub fn apply(&mut self, action: ClockAction) {
        self.clock.apply(action);
        self.sync();
    }

    /// Replaces the clock and integrates every frame to its time.
    pub fn set_clock(&mut self, clock: SimClock) {
        self.clock = clock;
        self.sync();
    }

    /// The fine history, extended so it covers every retarded time of
    /// every frame seen from `observer_root` (root coordinates) at the
    /// simulation time: what [`crate::seen::Seen::retarded`] solves
    /// against.
    pub fn light_history(&mut self, observer_root: Vec3) -> &History {
        let t = self.time();
        let farthest = self
            .system
            .tree()
            .frames()
            .iter()
            .map(|f| (self.system.root_position(f.frame_id) - observer_root).length())
            .fold(0.0, f64::max);
        let light = farthest / C * (1.0 + LIGHT_TIME_MARGIN) + 1.0;
        self.cover(t - Seconds::new(light), t);
        &self.recent
    }

    /// Makes the fine history cover every retarded time of every frame
    /// seen from `camera` ([`Simulation::light_history`]). Call before
    /// [`Simulation::seen`] whenever the time or the camera moved.
    pub fn prepare_light(&mut self, camera: &Camera) {
        let observer = camera.root_position(&self.system);
        self.light_history(observer);
    }

    /// What `camera` sees at the simulation time: every frame at its
    /// retarded state ([`Seen::retarded`]), solved against the fine
    /// history. Call [`Simulation::prepare_light`] for the camera first, or
    /// retarded times before the fine history's start are clamped to it.
    pub fn seen(&self, camera: &Camera) -> Seen<'_> {
        Seen::retarded(&self.recent, &self.system, camera)
    }

    /// [`Simulation::seen`] with each frame in `near` seen over that
    /// distance to its matter ([`Seen::retarded_near`]).
    pub fn seen_near(
        &self,
        camera: &Camera,
        near: &std::collections::BTreeMap<u64, f64>,
    ) -> Seen<'_> {
        Seen::retarded_near(&self.recent, &self.system, camera, near)
    }

    /// Integrates every frame to the clock's time.
    fn sync(&mut self) {
        let t = self.clock.time();
        if self.system.time() == t {
            return;
        }
        self.cover(t, t);
        self.system = self.recent.peek(t);
    }

    /// Makes the fine history cover `lo` (or the origin, if later) to `t`,
    /// rebuilding it from the coarse history when it does not, or when its
    /// checkpoints are much farther apart than the span needs.
    fn cover(&mut self, lo: Seconds, t: Seconds) {
        let first = self.history.epoch();
        let lo = if lo < first { first } else { lo };
        let step = self.step;
        let wanted = |span: Seconds| {
            Seconds::new(
                step.value()
                    .max(span.value() / (RECENT_CAPACITY / 4) as f64),
            )
        };
        let epoch = self.recent.epoch();
        let interval = self.recent.interval().value();
        // It must start early enough, its checkpoints must not be much
        // sparser than the span needs, and reaching `t` must not fill more
        // than half of it (which would start thinning it).
        let fits = epoch <= lo
            && interval <= 2.0 * wanted(t - lo + Seconds::new(RECENT_SLACK_SECONDS)).value()
            && (t - epoch).value() <= (RECENT_CAPACITY / 2) as f64 * interval;
        if !fits {
            let start = lo - Seconds::new(RECENT_SLACK_SECONDS);
            let start = if start < first { first } else { start };
            let interval = wanted(t - start + Seconds::new(RECENT_SLACK_SECONDS));
            self.recent = History::new(
                self.history.state_at(start),
                interval,
                self.step,
                SCHEME,
                RECENT_CAPACITY,
            );
        }
        self.recent.extend_to(t);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use gx_core::integrate;
    use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
    use gx_core::units::{Kilograms, Meters, Quat};

    /// The registry epoch: 0, the origin.
    pub(crate) const EPOCH: f64 = 0.0;

    /// A heavy root frame and a spinning child frame in orbit around it.
    pub(crate) fn system() -> FrameSystem {
        let root = Frame {
            frame_id: 1,
            parent_frame_id: ROOT_PARENT,
            root_extent: Meters::new(1.0e12),
            max_depth: 4,
            mass: Kilograms::new(2.0e30),
            position: Vec3::zero(),
            velocity: Vec3::zero(),
            orientation: Quat::identity(),
            angular_velocity: Vec3::zero(),
            j2: 0.0,
            reference_radius: Meters::new(0.0),
            principal_moments: Vec3::zero(),
        };
        let child = Frame {
            frame_id: 2,
            parent_frame_id: 1,
            root_extent: Meters::new(1.0e7),
            mass: Kilograms::new(6.0e24),
            position: Vec3::new(1.5e11, 0.0, 0.0),
            velocity: Vec3::new(0.0, 2.98e4, 0.0),
            angular_velocity: Vec3::new(0.0, 0.0, 7.3e-5),
            j2: 1.08e-3,
            reference_radius: Meters::new(6.4e6),
            principal_moments: Vec3::new(8.0e37, 8.0e37, 8.03e37),
            ..root
        };
        let reg = Registry::new(Seconds::new(EPOCH), vec![root, child]).unwrap();
        FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
    }

    fn s(v: f64) -> Seconds {
        Seconds::new(v)
    }

    #[test]
    fn enters_at_now_and_follows_it() {
        let mut c = SimClock::at_now(s(EPOCH), s(EPOCH + 5000.0));
        assert!(c.is_at_now());
        assert_eq!(c.time(), s(EPOCH + 5000.0));
        c.tick(s(0.5), s(EPOCH + 5000.5));
        assert_eq!(c.time(), s(EPOCH + 5000.5));
        assert_eq!(c.offset_from_now(), s(0.0));
        // At now the rate is 1 and doubling does nothing.
        c.apply(ClockAction::DoubleRate);
        assert_eq!(c.rate(), 1.0);
        assert!(c.is_at_now());
        // A now before the origin is the origin.
        assert_eq!(SimClock::at_now(s(EPOCH), s(EPOCH - 1.0)).time(), s(EPOCH));
    }

    #[test]
    fn playing_past_now_never_passes_now() {
        let now = EPOCH + 1.0e6;
        let mut c = SimClock::starting_at(s(EPOCH), s(now), s(EPOCH), 1.0);
        assert!(!c.is_at_now());
        for _ in 0..40 {
            c.apply(ClockAction::DoubleRate);
        }
        assert_eq!(c.rate(), RATE_LIMITS.1);
        // One wall second at this rate asks for far more than is left.
        c.tick(s(1.0), s(now));
        assert_eq!(c.time(), s(now));
        assert!(c.is_at_now());
        assert_eq!(c.rate(), 1.0);
        for k in 0..100 {
            c.tick(s(0.25), s(now + 0.25 * f64::from(k)));
            assert!(c.time() <= c.now());
        }
        // Stepping forward and asking for a later start both stop at now.
        let mut c = SimClock::starting_at(s(EPOCH), s(now), s(now - 10.0), 1.0);
        c.apply(ClockAction::StepForward);
        assert_eq!(c.time(), s(now));
        assert!(c.is_at_now());
        let c = SimClock::starting_at(s(EPOCH), s(now), s(now + 1.0e9), 1.0);
        assert_eq!(c.time(), s(now));
        assert!(c.is_at_now());
        // A simulation asked to play past now stays at now too.
        let mut sim = Simulation::new(
            system(),
            SimClock::starting_at(s(EPOCH), s(EPOCH + 7200.0), s(EPOCH), 1.0e6),
        );
        sim.update(s(1.0), s(EPOCH + 7200.0));
        assert_eq!(sim.time(), s(EPOCH + 7200.0));
        assert!(sim.clock().is_at_now());
    }

    #[test]
    fn steps_rewind_and_rates() {
        let now = EPOCH + 1.0e6;
        let mut c = SimClock::at_now(s(EPOCH), s(now));
        c.apply(ClockAction::StepBack);
        assert!(!c.is_at_now());
        assert_eq!(c.offset_from_now(), s(-STEP_SECONDS));
        c.apply(ClockAction::StepBack);
        c.apply(ClockAction::StepForward);
        assert_eq!(c.offset_from_now(), s(-STEP_SECONDS));
        // The time does not move with now in the past at rate 1.
        c.apply(ClockAction::TogglePause);
        c.tick(s(10.0), s(now + 10.0));
        assert_eq!(c.time(), s(now - STEP_SECONDS));
        c.apply(ClockAction::TogglePause);
        c.apply(ClockAction::ToggleRewind);
        c.apply(ClockAction::DoubleRate);
        c.tick(s(1.0), s(now + 11.0));
        assert_eq!(c.time(), s(now - STEP_SECONDS - 2.0));
        assert!(c.rewinding());
        // Rewinding stops at the origin.
        for _ in 0..30 {
            c.apply(ClockAction::DoubleRate);
        }
        c.tick(s(100.0), s(now + 111.0));
        assert_eq!(c.time(), s(EPOCH));
        for _ in 0..1000 {
            c.apply(ClockAction::StepBack);
        }
        assert_eq!(c.time(), s(EPOCH));
        for _ in 0..40 {
            c.apply(ClockAction::HalveRate);
        }
        assert_eq!(c.rate(), RATE_LIMITS.0);
        c.apply(ClockAction::JumpToNow);
        assert!(c.is_at_now());
        assert_eq!(c.time(), s(now + 111.0));
        assert!(!c.rewinding());
        // Halving at now falls behind now; rewinding at now goes back.
        c.apply(ClockAction::HalveRate);
        assert!(!c.is_at_now());
        c.tick(s(2.0), s(now + 113.0));
        assert_eq!(c.time(), s(now + 112.0));
        c.apply(ClockAction::JumpToNow);
        c.apply(ClockAction::ToggleRewind);
        c.tick(s(3.0), s(now + 116.0));
        assert_eq!(c.time(), s(now + 110.0));
        c.apply(ClockAction::JumpToNow);
        c.apply(ClockAction::TogglePause);
        assert!(c.paused() && !c.is_at_now());
    }

    #[test]
    fn step_follows_the_shortest_dynamical_time() {
        // Wide pair: the cap.
        assert_eq!(max_step(&system()), MAX_STEP_SECONDS);
        // A one-hour orbit: far more steps than the cap would give.
        let gm = gx_core::gravity::G * 6.0e24;
        let d = (gm * (3600.0 / (2.0 * std::f64::consts::PI)).powi(2)).cbrt();
        let mut tree = system().tree().frames().to_vec();
        let mut close = tree[1];
        close.frame_id = 3;
        close.parent_frame_id = 2;
        close.mass = Kilograms::new(1.0e3);
        close.position = Vec3::new(d, 0.0, 0.0);
        close.root_extent = Meters::new(1.0e6);
        tree.push(close);
        let reg = Registry::new(Seconds::new(EPOCH), tree).unwrap();
        let sys = FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap());
        let step = max_step(&sys);
        let t = (d * d * d / (gm + gx_core::gravity::G * 1.0e3)).sqrt();
        assert!(
            (step - t / STEPS_PER_DYNAMICAL_TIME).abs() < 1e-9 * step,
            "{step}"
        );
        assert!(3600.0 / step > 1000.0);
    }

    #[test]
    fn jumps_match_direct_integration() {
        let now = EPOCH + 30.0 * CHECKPOINT_SECONDS + 1234.5;
        let mut sim = Simulation::at_now(system(), s(now));
        let mut direct = system();
        integrate::advance(&mut direct, s(now), s(MAX_STEP_SECONDS), SCHEME);
        let close = |a: &FrameSystem, b: &FrameSystem| {
            for f in [1, 2] {
                let (pa, pb) = (a.root_position(f), b.root_position(f));
                let scale = pb.length().max(1.0);
                assert!((pa - pb).length() <= 1e-9 * scale, "{pa:?} vs {pb:?}");
            }
        };
        close(sim.system(), &direct);
        // Back to the origin: the registry state, reached through the
        // checkpoints that start before it.
        sim.set_clock(SimClock::starting_at(s(EPOCH), s(now), s(EPOCH), 1.0));
        assert_eq!(sim.time(), s(EPOCH));
        close(sim.system(), &system());
        sim.apply(ClockAction::JumpToNow);
        assert_eq!(sim.time(), s(now));
        close(sim.system(), &direct);
        // Halfway, from the checkpoints.
        let mid = EPOCH + 12.5 * CHECKPOINT_SECONDS;
        sim.set_clock(SimClock::starting_at(s(EPOCH), s(now), s(mid), 1.0));
        let mut half = system();
        integrate::advance(&mut half, s(mid), s(MAX_STEP_SECONDS), SCHEME);
        close(sim.system(), &half);
    }

    #[test]
    fn light_history_reaches_back_one_light_time() {
        let now = EPOCH + 3.0 * CHECKPOINT_SECONDS;
        let mut sim = Simulation::at_now(system(), s(now));
        // From 40 light-minutes away (inside the root frame) the fine
        // history reaches back further.
        let far = Vec3::new(0.0, 0.0, C * 2_400.0);
        let h = sim.light_history(far);
        assert!(h.epoch() <= s(now - 2_400.0 * (1.0 + LIGHT_TIME_MARGIN)));
        assert!(h.latest() <= s(now) && h.latest() + h.interval() > s(now));
        // Just after the origin the light left before it: the fine history
        // reaches back before the origin, as far as the light needs.
        sim.set_clock(SimClock::starting_at(
            s(EPOCH),
            s(now),
            s(EPOCH + 60.0),
            1.0,
        ));
        let h = sim.light_history(far);
        assert!(h.epoch() <= s(EPOCH + 60.0 - 2_400.0 * (1.0 + LIGHT_TIME_MARGIN)));
        assert!(h.epoch() >= s(EPOCH - before_origin(&system())));
        // The clock itself never goes before the origin.
        sim.apply(ClockAction::StepBack);
        assert_eq!(sim.time(), s(EPOCH));
    }
}
