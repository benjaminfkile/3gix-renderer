//! Keyboard and mouse controls, shared by the desktop window and the
//! browser canvas.
//!
//! Turns `winit` window events into flight input for the [`Camera`],
//! actions on the [`SimClock`](crate::sim::SimClock), view jumps, the
//! followed frame (see [`Camera::hold`]), and the
//! exposure: the bias of the automatic exposure, or the fixed exposure when
//! one was given (`--exposure-stops`). The bindings are listed in `docs/controls.md`. Keys are
//! matched by physical position, so they sit in the same place on every
//! keyboard layout.

use crate::camera::{Camera, FlightInput};
use crate::render::gpu::{Exposure, EXPOSURE_STEP_STOPS};
use crate::sim::{ClockAction, Simulation};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Pixels of trackpad scroll that count as one wheel notch.
pub const PIXELS_PER_NOTCH: f64 = 40.0;

/// Limits of the exposure bias, stops.
pub const EXPOSURE_BIAS_LIMITS: (f64, f64) = (-16.0, 16.0);

/// Limits of a fixed exposure, stops relative to
/// [`crate::render::gpu::REFERENCE_LUMINANCE`].
pub const EXPOSURE_FIXED_LIMITS: (f64, f64) = (-64.0, 64.0);

/// Held keys, accumulated mouse movement, and the exposure.
#[derive(Clone, Debug, Default)]
pub struct Controls {
    flight: FlightInput,
    looking: bool,
    cursor: Option<(f64, f64)>,
    /// User exposure bias of the automatic exposure, stops.
    pub exposure_bias: f64,
    /// The fixed exposure, stops (see [`crate::render::gpu::fixed_exposure`]);
    /// `None` for the automatic exposure.
    pub exposure_fixed: Option<f64>,
    /// The frame the camera moves along with, or `None` for a camera that
    /// floats free in space.
    pub follow: Option<u64>,
}

impl Controls {
    /// Controls with the automatic exposure, or a fixed exposure of
    /// `fixed_stops` (clamped to [`EXPOSURE_FIXED_LIMITS`]).
    pub fn new(fixed_stops: Option<f64>) -> Controls {
        let (lo, hi) = EXPOSURE_FIXED_LIMITS;
        Controls {
            exposure_fixed: fixed_stops.map(|s| s.clamp(lo, hi)),
            ..Controls::default()
        }
    }

    /// How to expose this frame, `adapt_seconds` after the previous one
    /// (see [`Exposure::adapt_seconds`]).
    pub fn exposure(&self, adapt_seconds: Option<f64>) -> Exposure {
        Exposure {
            bias_stops: self.exposure_bias,
            fixed_stops: self.exposure_fixed,
            adapt_seconds,
        }
    }

    /// This frame's flight input; clears the accumulated mouse movement.
    pub fn take_flight(&mut self) -> FlightInput {
        let out = self.flight;
        self.flight.look = (0.0, 0.0);
        self.flight.wheel = 0.0;
        out
    }

    /// Applies one window event. Returns `true` when the user asked to
    /// quit (`Escape`). Events that are not input are ignored.
    pub fn handle(
        &mut self,
        event: &WindowEvent,
        sim: &mut Simulation,
        camera: &mut Camera,
    ) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    return self.key(code, pressed, event.repeat, sim, camera);
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.looking = *state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x, position.y);
                if let (true, Some(last)) = (self.looking, self.cursor) {
                    self.flight.look.0 += p.0 - last.0;
                    self.flight.look.1 += p.1 - last.1;
                }
                self.cursor = Some(p);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.flight.wheel += match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(*y),
                    MouseScrollDelta::PixelDelta(p) => p.y / PIXELS_PER_NOTCH,
                };
            }
            _ => {}
        }
        false
    }

    /// Applies one key press or release. Returns `true` for `Escape`.
    pub fn key(
        &mut self,
        code: KeyCode,
        pressed: bool,
        repeat: bool,
        sim: &mut Simulation,
        camera: &mut Camera,
    ) -> bool {
        let f = &mut self.flight;
        match code {
            KeyCode::KeyW => f.forward = pressed,
            KeyCode::KeyS => f.back = pressed,
            KeyCode::KeyA => f.left = pressed,
            KeyCode::KeyD => f.right = pressed,
            KeyCode::KeyE => f.up = pressed,
            KeyCode::KeyQ => f.down = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => f.boost = pressed,
            _ if !pressed || repeat => {}
            KeyCode::Space => sim.clock.apply(ClockAction::TogglePause),
            KeyCode::BracketLeft => sim.clock.apply(ClockAction::HalveScale),
            KeyCode::BracketRight => sim.clock.apply(ClockAction::DoubleScale),
            KeyCode::Comma => sim.clock.apply(ClockAction::StepBack),
            KeyCode::Period => sim.clock.apply(ClockAction::StepForward),
            KeyCode::KeyR => sim.clock.apply(ClockAction::Reset),
            KeyCode::Home => {
                *camera = Camera::home(&sim.system);
                self.follow = None;
            }
            KeyCode::KeyF => {
                self.follow = match self.follow {
                    Some(_) => None,
                    None => Some(camera.frame_id),
                };
            }
            KeyCode::Equal | KeyCode::NumpadAdd => self.bias_exposure(EXPOSURE_STEP_STOPS),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.bias_exposure(-EXPOSURE_STEP_STOPS),
            KeyCode::Escape => return true,
            other => {
                if let Some(index) = digit_index(other) {
                    if let Some(cam) = Camera::view_of_index(&sim.system, index) {
                        *camera = Camera {
                            speed: camera.speed,
                            cruise: camera.cruise,
                            ..cam
                        };
                        // A jump never follows: the camera floats free
                        // where it lands until `F` is pressed.
                        self.follow = None;
                    }
                }
            }
        }
        false
    }

    /// `+` and `-`: shift the fixed exposure if there is one, the bias of
    /// the automatic exposure otherwise.
    fn bias_exposure(&mut self, stops: f64) {
        match &mut self.exposure_fixed {
            Some(fixed) => {
                let (lo, hi) = EXPOSURE_FIXED_LIMITS;
                *fixed = (*fixed + stops).clamp(lo, hi);
            }
            None => {
                let (lo, hi) = EXPOSURE_BIAS_LIMITS;
                self.exposure_bias = (self.exposure_bias + stops).clamp(lo, hi);
            }
        }
    }
}

/// Number keys: `1` to `9` select indices 0 to 8, `0` selects index 9.
pub fn digit_index(code: KeyCode) -> Option<usize> {
    const DIGITS: [KeyCode; 10] = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
        KeyCode::Digit0,
    ];
    DIGITS.iter().position(|&d| d == code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_keys_jump_without_following_and_f_toggles_follow() {
        use crate::sim::SimClock;
        use gx_core::frames::FrameSystem;
        use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
        use gx_core::units::{Kilograms, Meters, Quat, Seconds, Vec3};
        let frame = |id, parent, x| Frame {
            frame_id: id,
            parent_frame_id: parent,
            root_extent: Meters::new(1.0e3),
            max_depth: 4,
            mass: Kilograms::new(0.0),
            position: Vec3::new(x, 0.0, 0.0),
            velocity: Vec3::zero(),
            orientation: Quat::identity(),
            angular_velocity: Vec3::zero(),
        };
        let reg = Registry::new(
            Seconds::new(0.0),
            vec![frame(1, ROOT_PARENT, 0.0), frame(2, 1, 1.0e4)],
        )
        .unwrap();
        let system = FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap());
        let mut sim = Simulation::new(system, SimClock::new(Seconds::new(0.0), 1.0));
        let mut camera = Camera::home(&sim.system);
        let mut c = Controls::default();
        c.key(KeyCode::Digit1, true, false, &mut sim, &mut camera);
        assert_eq!(camera.frame_id, 2, "jumped to the first frame");
        assert_eq!(c.follow, None, "a jump floats free");
        c.key(KeyCode::KeyF, true, false, &mut sim, &mut camera);
        assert_eq!(c.follow, Some(2));
        c.key(KeyCode::Digit1, true, false, &mut sim, &mut camera);
        assert_eq!(c.follow, None, "a jump stops following");
    }

    #[test]
    fn keys_move_the_fixed_exposure_when_there_is_one() {
        let mut auto = Controls::new(None);
        auto.bias_exposure(EXPOSURE_STEP_STOPS);
        assert_eq!(auto.exposure_bias, 0.5);
        assert_eq!(auto.exposure(None).fixed_stops, None);

        let mut fixed = Controls::new(Some(12.81));
        fixed.bias_exposure(-EXPOSURE_STEP_STOPS);
        assert_eq!(fixed.exposure_bias, 0.0);
        let e = fixed.exposure(Some(0.016));
        assert_eq!(e.fixed_stops, Some(12.31));
        assert_eq!(e.adapt_seconds, Some(0.016));

        assert_eq!(Controls::new(Some(100.0)).exposure_fixed, Some(64.0));
        let mut low = Controls::new(Some(-64.0));
        low.bias_exposure(-EXPOSURE_STEP_STOPS);
        assert_eq!(low.exposure_fixed, Some(-64.0));
    }
}
