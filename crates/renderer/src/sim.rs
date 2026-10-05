//! The simulation clock and the integration of every frame to the
//! simulation time.
//!
//! Implements `space-model.md` section 6 (time) on the renderer side: the
//! renderer keeps a simulation time that advances from the build epoch at a
//! user-controlled rate, and integrates all frames to that time with
//! [`gx_core::integrate::advance`] and the fourth order Yoshida scheme.
//!
//! The [`SimClock`] holds the offset of the simulation time from the epoch.
//! Each rendered frame it advances by the wall-clock delta times the time
//! scale, and [`Simulation::update`] integrates the [`FrameSystem`] to the
//! clock's target time in steps of at most [`MAX_STEP_SECONDS`].
//!
//! When the time scale asks for more than [`MAX_SUBSTEPS_PER_FRAME`] steps in
//! one rendered frame, the advance is clamped to exactly that many steps and
//! the simulation reports that it lags (the overlay shows "sim lag"). The
//! integrator is never skipped to catch up: the frame states always come from
//! integration, and the system chases the clock over the following frames.

use gx_core::frames::FrameSystem;
use gx_core::integrate::{self, Scheme};
use gx_core::units::Seconds;

/// Seconds in one day, the step of the day keys while paused.
pub const SECONDS_PER_DAY: f64 = 86_400.0;

/// The longest integration step, seconds.
pub const MAX_STEP_SECONDS: f64 = 600.0;

/// The most integration steps taken in one rendered frame.
pub const MAX_SUBSTEPS_PER_FRAME: u64 = 20_000;

/// The smallest and largest time scale magnitude the halve and double keys
/// reach.
pub const SCALE_LIMITS: (f64, f64) = (1.0 / 1024.0, 1.0e12);

/// A change to the clock requested by the user.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClockAction {
    /// Pause or resume.
    TogglePause,
    /// Halve the time scale.
    HalveScale,
    /// Double the time scale.
    DoubleScale,
    /// Step one day backwards; only while paused.
    StepBack,
    /// Step one day forwards; only while paused.
    StepForward,
    /// Return to the offset at launch.
    Reset,
}

/// The simulation clock: an offset from the epoch, a rate, and a pause flag.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimClock {
    /// Simulation time minus the registry epoch, seconds.
    pub epoch_offset: Seconds,
    /// Simulation seconds per wall-clock second. May be negative to run
    /// backwards.
    pub scale: f64,
    /// When `true`, wall-clock time does not advance the clock.
    pub paused: bool,
    /// The offset at launch, restored by [`ClockAction::Reset`].
    launch_offset: Seconds,
}

impl SimClock {
    /// A running clock at `launch_offset` from the epoch with the given
    /// scale.
    pub fn new(launch_offset: Seconds, scale: f64) -> SimClock {
        SimClock {
            epoch_offset: launch_offset,
            scale,
            paused: false,
            launch_offset,
        }
    }

    /// The offset at launch.
    pub fn launch_offset(&self) -> Seconds {
        self.launch_offset
    }

    /// Advances the clock by `wall_dt` times the scale, unless paused.
    pub fn tick(&mut self, wall_dt: Seconds) {
        if !self.paused {
            self.epoch_offset = self.epoch_offset + wall_dt * self.scale;
        }
    }

    /// Applies a user action. The day steps do nothing while running.
    pub fn apply(&mut self, action: ClockAction) {
        let (lo, hi) = SCALE_LIMITS;
        let clamp = |s: f64| s.signum() * s.abs().clamp(lo, hi);
        match action {
            ClockAction::TogglePause => self.paused = !self.paused,
            ClockAction::HalveScale => self.scale = clamp(self.scale * 0.5),
            ClockAction::DoubleScale => self.scale = clamp(self.scale * 2.0),
            ClockAction::StepBack if self.paused => {
                self.epoch_offset = self.epoch_offset - Seconds::new(SECONDS_PER_DAY)
            }
            ClockAction::StepForward if self.paused => {
                self.epoch_offset = self.epoch_offset + Seconds::new(SECONDS_PER_DAY)
            }
            ClockAction::StepBack | ClockAction::StepForward => {}
            ClockAction::Reset => self.epoch_offset = self.launch_offset,
        }
    }
}

/// The number of steps [`integrate::advance`] takes over `interval` with
/// steps of at most `max_step`: `ceil(|interval| / max_step)`.
pub fn substeps_for(interval: Seconds, max_step: Seconds) -> u64 {
    let n = (interval.value().abs() / max_step.value()).ceil();
    if n >= u64::MAX as f64 {
        u64::MAX
    } else {
        n as u64
    }
}

/// What one [`Simulation::update`] did.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StepReport {
    /// Integration steps taken.
    pub substeps: u64,
    /// `true` when the advance was clamped and the system is behind the
    /// clock.
    pub lagging: bool,
}

/// A frame system integrated to a clock.
#[derive(Clone, Debug)]
pub struct Simulation {
    /// Every frame's state at the current simulation time.
    pub system: FrameSystem,
    /// The clock the system follows.
    pub clock: SimClock,
    lagging: bool,
}

impl Simulation {
    /// Wraps a system, typically at its epoch, and a clock.
    pub fn new(system: FrameSystem, clock: SimClock) -> Simulation {
        Simulation {
            system,
            clock,
            lagging: false,
        }
    }

    /// The time the clock asks for: epoch plus offset, seconds since J2000.
    pub fn target_time(&self) -> Seconds {
        self.system.tree().epoch() + self.clock.epoch_offset
    }

    /// `true` when the last update was clamped.
    pub fn lagging(&self) -> bool {
        self.lagging
    }

    /// Ticks the clock by `wall_dt` and integrates the system toward the
    /// target time, clamped to [`MAX_SUBSTEPS_PER_FRAME`] steps.
    pub fn update(&mut self, wall_dt: Seconds) -> StepReport {
        self.clock.tick(wall_dt);
        self.integrate_toward_target()
    }

    /// Integrates toward the target time without ticking the clock, clamped
    /// to [`MAX_SUBSTEPS_PER_FRAME`] steps.
    pub fn integrate_toward_target(&mut self) -> StepReport {
        let max_step = Seconds::new(MAX_STEP_SECONDS);
        let target = self.target_time();
        let interval = target - self.system.time();
        let needed = substeps_for(interval, max_step);
        let (to, substeps, lagging) = if needed > MAX_SUBSTEPS_PER_FRAME {
            let span = MAX_SUBSTEPS_PER_FRAME as f64 * MAX_STEP_SECONDS;
            let to = self.system.time() + Seconds::new(span.copysign(interval.value()));
            (to, MAX_SUBSTEPS_PER_FRAME, true)
        } else {
            (target, needed, false)
        };
        integrate::advance(&mut self.system, to, max_step, Scheme::Yoshida4);
        self.lagging = lagging;
        StepReport { substeps, lagging }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
    use gx_core::units::{Kilograms, Meters, Quat, Vec3};

    fn system() -> FrameSystem {
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
        };
        let child = Frame {
            frame_id: 2,
            parent_frame_id: 1,
            root_extent: Meters::new(1.0e7),
            mass: Kilograms::new(6.0e24),
            position: Vec3::new(1.5e11, 0.0, 0.0),
            velocity: Vec3::new(0.0, 2.98e4, 0.0),
            ..root
        };
        let reg = Registry::new(Seconds::new(1000.0), vec![root, child]).unwrap();
        FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
    }

    #[test]
    fn clock_ticks_with_scale_and_pause() {
        let mut c = SimClock::new(Seconds::new(10.0), 4.0);
        c.tick(Seconds::new(0.5));
        assert_eq!(c.epoch_offset, Seconds::new(12.0));
        c.apply(ClockAction::TogglePause);
        c.tick(Seconds::new(100.0));
        assert_eq!(c.epoch_offset, Seconds::new(12.0));
        c.apply(ClockAction::TogglePause);
        c.tick(Seconds::new(1.0));
        assert_eq!(c.epoch_offset, Seconds::new(16.0));
    }

    #[test]
    fn scale_keys_halve_and_double_within_limits() {
        let mut c = SimClock::new(Seconds::new(0.0), 1.0);
        c.apply(ClockAction::DoubleScale);
        c.apply(ClockAction::DoubleScale);
        assert_eq!(c.scale, 4.0);
        c.apply(ClockAction::HalveScale);
        assert_eq!(c.scale, 2.0);
        for _ in 0..40 {
            c.apply(ClockAction::HalveScale);
        }
        assert_eq!(c.scale, SCALE_LIMITS.0);
        for _ in 0..100 {
            c.apply(ClockAction::DoubleScale);
        }
        assert_eq!(c.scale, SCALE_LIMITS.1);
        let mut back = SimClock::new(Seconds::new(0.0), -1.0);
        back.apply(ClockAction::DoubleScale);
        assert_eq!(back.scale, -2.0);
    }

    #[test]
    fn day_steps_only_while_paused_and_reset() {
        let mut c = SimClock::new(Seconds::new(5.0), 1.0);
        c.apply(ClockAction::StepForward);
        assert_eq!(c.epoch_offset, Seconds::new(5.0), "ignored while running");
        c.apply(ClockAction::TogglePause);
        c.apply(ClockAction::StepForward);
        c.apply(ClockAction::StepForward);
        c.apply(ClockAction::StepBack);
        assert_eq!(c.epoch_offset, Seconds::new(5.0 + SECONDS_PER_DAY));
        c.apply(ClockAction::Reset);
        assert_eq!(c.epoch_offset, Seconds::new(5.0));
        assert_eq!(c.launch_offset(), Seconds::new(5.0));
    }

    #[test]
    fn substep_count() {
        let m = Seconds::new(MAX_STEP_SECONDS);
        assert_eq!(substeps_for(Seconds::new(0.0), m), 0);
        assert_eq!(substeps_for(Seconds::new(600.0), m), 1);
        assert_eq!(substeps_for(Seconds::new(600.5), m), 2);
        assert_eq!(substeps_for(Seconds::new(-1200.0), m), 2);
    }

    #[test]
    fn update_reaches_target_when_within_budget() {
        let mut sim = Simulation::new(system(), SimClock::new(Seconds::new(0.0), 3600.0));
        let r = sim.update(Seconds::new(1.0));
        assert_eq!(
            r,
            StepReport {
                substeps: 6,
                lagging: false
            }
        );
        assert_eq!(sim.system.time(), Seconds::new(1000.0 + 3600.0));
        assert!(!sim.lagging());
    }

    #[test]
    fn update_clamps_and_lags_without_skipping() {
        // One wall second at this scale asks for 30000 steps of 600 s.
        let scale = 30_000.0 * MAX_STEP_SECONDS;
        let mut sim = Simulation::new(system(), SimClock::new(Seconds::new(0.0), scale));
        let r = sim.update(Seconds::new(1.0));
        assert_eq!(r.substeps, MAX_SUBSTEPS_PER_FRAME);
        assert!(r.lagging && sim.lagging());
        let reached = 1000.0 + MAX_SUBSTEPS_PER_FRAME as f64 * MAX_STEP_SECONDS;
        assert_eq!(sim.system.time(), Seconds::new(reached));
        // The clock keeps its target; the system catches up next frame.
        sim.clock.apply(ClockAction::TogglePause);
        let r = sim.update(Seconds::new(1.0));
        assert_eq!(r.substeps, 10_000);
        assert!(!r.lagging);
        assert_eq!(sim.system.time(), sim.target_time());
    }

    #[test]
    fn clamped_steps_match_unclamped_integration() {
        // Never skipping means the clamped path lands on the same states as
        // integrating the same span in one call with the same step count.
        let scale = 25_000.0 * MAX_STEP_SECONDS;
        let mut sim = Simulation::new(system(), SimClock::new(Seconds::new(0.0), scale));
        sim.update(Seconds::new(1.0));
        let mut direct = system();
        let to = Seconds::new(1000.0 + MAX_SUBSTEPS_PER_FRAME as f64 * MAX_STEP_SECONDS);
        integrate::advance(
            &mut direct,
            to,
            Seconds::new(MAX_STEP_SECONDS),
            Scheme::Yoshida4,
        );
        assert_eq!(sim.system, direct);
    }

    #[test]
    fn backwards_clamp() {
        let scale = -40_000.0 * MAX_STEP_SECONDS;
        let mut sim = Simulation::new(system(), SimClock::new(Seconds::new(0.0), scale));
        let r = sim.update(Seconds::new(1.0));
        assert!(r.lagging);
        let reached = 1000.0 - MAX_SUBSTEPS_PER_FRAME as f64 * MAX_STEP_SECONDS;
        assert_eq!(sim.system.time(), Seconds::new(reached));
    }
}
