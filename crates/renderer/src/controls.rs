//! Keyboard and mouse controls, shared by the desktop window and the
//! browser canvas.
//!
//! Turns `winit` window events into flight input for the [`Camera`],
//! actions on the [`SimClock`](crate::sim::SimClock), view jumps, and the
//! exposure bias. The bindings are listed in `docs/controls.md`. Keys are
//! matched by physical position, so they sit in the same place on every
//! keyboard layout.

use crate::camera::{Camera, FlightInput};
use crate::render::gpu::EXPOSURE_STEP_STOPS;
use crate::sim::{ClockAction, Simulation};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Pixels of trackpad scroll that count as one wheel notch.
pub const PIXELS_PER_NOTCH: f64 = 40.0;

/// Limits of the exposure bias, stops.
pub const EXPOSURE_BIAS_LIMITS: (f64, f64) = (-16.0, 16.0);

/// Held keys, accumulated mouse movement, and the exposure bias.
#[derive(Clone, Debug, Default)]
pub struct Controls {
    flight: FlightInput,
    looking: bool,
    cursor: Option<(f64, f64)>,
    /// User exposure bias, stops.
    pub exposure_bias: f64,
}

impl Controls {
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
            _ if !pressed || repeat => {}
            KeyCode::Space => sim.clock.apply(ClockAction::TogglePause),
            KeyCode::BracketLeft => sim.clock.apply(ClockAction::HalveScale),
            KeyCode::BracketRight => sim.clock.apply(ClockAction::DoubleScale),
            KeyCode::Comma => sim.clock.apply(ClockAction::StepBack),
            KeyCode::Period => sim.clock.apply(ClockAction::StepForward),
            KeyCode::KeyR => sim.clock.apply(ClockAction::Reset),
            KeyCode::Home => *camera = Camera::home(&sim.system),
            KeyCode::Equal | KeyCode::NumpadAdd => self.bias_exposure(EXPOSURE_STEP_STOPS),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.bias_exposure(-EXPOSURE_STEP_STOPS),
            KeyCode::Escape => return true,
            other => {
                if let Some(index) = digit_index(other) {
                    if let Some(cam) = Camera::view_of_index(&sim.system, index) {
                        *camera = Camera {
                            speed: camera.speed,
                            ..cam
                        };
                    }
                }
            }
        }
        false
    }

    fn bias_exposure(&mut self, stops: f64) {
        let (lo, hi) = EXPOSURE_BIAS_LIMITS;
        self.exposure_bias = (self.exposure_bias + stops).clamp(lo, hi);
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
