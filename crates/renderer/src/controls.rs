//! Keyboard and mouse controls, shared by the desktop window and the
//! browser canvas.
//!
//! Turns `winit` window events into flight or walking input for the
//! [`Camera`] and the [`Observer`] (fly and walk, `Tab`), the spawn chooser
//! (`G`), actions on the simulation clock ([`crate::sim::ClockAction`]),
//! view jumps, and the
//! exposure: the bias of the automatic exposure, or the fixed exposure when
//! one was given (`--exposure-stops`). The bindings are listed in `docs/controls.md`. Keys are
//! matched by physical position, so they sit in the same place on every
//! keyboard layout.

use crate::camera::{Camera, FlightInput, SPEED_LIMITS};
use crate::render::gpu::{Exposure, EXPOSURE_STEP_STOPS};
use crate::render::scene::MeshMode;
use crate::sim::{ClockAction, Simulation};
use crate::walk::Observer;
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
    /// `true` while frame markers and frame-to-parent lines are hidden
    /// (`M`): they are drawing aids, not matter.
    pub hide_guides: bool,
    /// `true` while the spawn chooser is open (`G`): the number keys pick a
    /// frame to spawn onto ([`crate::walk::spawn_frames`]).
    pub spawning: bool,
    /// The spawn chooser index picked since the last frame.
    spawn: Option<usize>,
    /// Space pressed in walk mode since the last frame.
    jump: bool,
    /// The speed being typed after `V`, meters per second, as typed; `None`
    /// when no speed is being typed.
    pub speed_entry: Option<String>,
    /// How surface meshes are drawn (`X`).
    pub mesh_mode: MeshMode,
}

/// Longest typed speed, characters.
pub const SPEED_ENTRY_MAX_CHARS: usize = 16;

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

    /// Whether Space was pressed in walk mode since the last frame; clears
    /// it.
    pub fn take_jump(&mut self) -> bool {
        std::mem::take(&mut self.jump)
    }

    /// The spawn chooser index (`0` for key `1`, up to `9` for key `0`)
    /// picked since the last frame; clears it. The frame loop resolves it
    /// against [`crate::walk::spawn_frames`].
    pub fn take_spawn(&mut self) -> Option<usize> {
        self.spawn.take()
    }

    /// Applies one window event. Returns `true` when the user asked to
    /// quit (`Escape`). Events that are not input are ignored.
    pub fn handle(
        &mut self,
        event: &WindowEvent,
        sim: &mut Simulation,
        camera: &mut Camera,
        observer: &mut Observer,
    ) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    return self.key(code, pressed, event.repeat, sim, camera, observer);
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
        observer: &mut Observer,
    ) -> bool {
        if self.speed_entry.is_some() && pressed {
            self.type_speed(code, camera, observer);
            return false;
        }
        if code == KeyCode::Space && pressed && !repeat && observer.walking() {
            self.jump = true;
            return false;
        }
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
            KeyCode::Space => observer.apply_clock(sim, camera, ClockAction::TogglePause),
            KeyCode::BracketLeft => observer.apply_clock(sim, camera, ClockAction::HalveRate),
            KeyCode::BracketRight => observer.apply_clock(sim, camera, ClockAction::DoubleRate),
            KeyCode::Comma => observer.apply_clock(sim, camera, ClockAction::StepBack),
            KeyCode::Period => observer.apply_clock(sim, camera, ClockAction::StepForward),
            KeyCode::Backspace => observer.apply_clock(sim, camera, ClockAction::ToggleRewind),
            KeyCode::KeyN => observer.apply_clock(sim, camera, ClockAction::JumpToNow),
            KeyCode::Tab => {
                let v = camera.flight_velocity(sim.system(), &self.flight);
                observer.toggle(sim.system(), camera, v);
            }
            KeyCode::KeyG => self.spawning = !self.spawning,
            KeyCode::Home => {
                observer.stop();
                *camera = Camera::home(sim.system());
            }
            KeyCode::KeyM => self.hide_guides = !self.hide_guides,
            KeyCode::KeyV => self.speed_entry = Some(String::new()),
            KeyCode::KeyX => self.mesh_mode = self.mesh_mode.next(),
            KeyCode::Equal | KeyCode::NumpadAdd => self.bias_exposure(EXPOSURE_STEP_STOPS),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.bias_exposure(-EXPOSURE_STEP_STOPS),
            KeyCode::Escape => return true,
            other => {
                if let Some(index) = digit_index(other) {
                    if self.spawning {
                        self.spawning = false;
                        self.spawn = Some(index);
                    } else if let Some(cam) = Camera::view_of_index(sim.system(), index) {
                        observer.stop();
                        *camera = Camera {
                            speed: camera.speed,
                            cruise: camera.cruise,
                            ..cam
                        };
                    }
                }
            }
        }
        false
    }

    /// One key while a speed is being typed: digits and one decimal point
    /// add to it, `Backspace` removes the last character, `Enter` sets it
    /// (the walking speed in walk mode, the flight speed in fly mode, each
    /// clamped to its limits) and `Escape` drops it. Other keys do nothing
    /// until the entry is closed.
    fn type_speed(&mut self, code: KeyCode, camera: &mut Camera, observer: &mut Observer) {
        let Some(text) = self.speed_entry.as_mut() else {
            return;
        };
        let digit = digit_index(code)
            .map(|i| char::from(b'0' + ((i + 1) % 10) as u8))
            .or_else(|| numpad_digit(code));
        match code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                if let Ok(v) = text.parse::<f64>() {
                    if observer.walking() {
                        observer.set_walk_speed(v);
                    } else if !v.is_nan() {
                        let (lo, hi) = SPEED_LIMITS;
                        camera.speed = v.clamp(lo, hi);
                    }
                }
                self.speed_entry = None;
            }
            KeyCode::Escape => self.speed_entry = None,
            KeyCode::Backspace => {
                text.pop();
            }
            KeyCode::Period | KeyCode::NumpadDecimal if !text.contains('.') => {
                if text.len() < SPEED_ENTRY_MAX_CHARS {
                    text.push('.');
                }
            }
            _ => {
                if let Some(c) = digit {
                    if text.len() < SPEED_ENTRY_MAX_CHARS {
                        text.push(c);
                    }
                }
            }
        }
    }

    /// The controls' own overlay lines: the mesh debug mode, and the speed
    /// entry while a speed is being typed.
    pub fn status_lines(&self, observer: &Observer) -> Vec<String> {
        let mut lines: Vec<String> = self
            .mesh_mode
            .label()
            .map(String::from)
            .into_iter()
            .collect();
        if let Some(t) = &self.speed_entry {
            let what = if observer.walking() { "walk" } else { "flight" };
            lines.push(format!(
                "type a {what} speed: {t}_ m/s  (Enter sets, Esc cancels)"
            ));
        }
        lines
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

/// Numpad digits as characters.
fn numpad_digit(code: KeyCode) -> Option<char> {
    const PAD: [KeyCode; 10] = [
        KeyCode::Numpad0,
        KeyCode::Numpad1,
        KeyCode::Numpad2,
        KeyCode::Numpad3,
        KeyCode::Numpad4,
        KeyCode::Numpad5,
        KeyCode::Numpad6,
        KeyCode::Numpad7,
        KeyCode::Numpad8,
        KeyCode::Numpad9,
    ];
    PAD.iter()
        .position(|&d| d == code)
        .map(|i| char::from(b'0' + i as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_typed_speed_sets_the_walk_or_flight_speed() {
        let mut c = Controls::new(None);
        let mut camera = Camera {
            frame_id: 0,
            position: gx_core::units::Vec3::zero(),
            orientation: gx_core::units::Quat::identity(),
            speed: 1.0,
            cruise: 1.0,
        };
        let mut observer = Observer::new();
        let typing = |c: &mut Controls, keys: &[KeyCode], cam: &mut Camera, o: &mut Observer| {
            c.speed_entry = Some(String::new());
            for &k in keys {
                c.type_speed(k, cam, o);
            }
        };
        // Fly mode: the flight speed.
        typing(
            &mut c,
            &[
                KeyCode::Digit2,
                KeyCode::Digit5,
                KeyCode::Period,
                KeyCode::Numpad5,
                KeyCode::Enter,
            ],
            &mut camera,
            &mut observer,
        );
        assert_eq!(camera.speed, 25.5);
        assert_eq!(c.speed_entry, None);
        // Backspace, a second point ignored, Escape drops it.
        typing(
            &mut c,
            &[
                KeyCode::Digit9,
                KeyCode::Backspace,
                KeyCode::Digit7,
                KeyCode::Period,
                KeyCode::Period,
            ],
            &mut camera,
            &mut observer,
        );
        assert_eq!(c.speed_entry.as_deref(), Some("7."));
        c.type_speed(KeyCode::Escape, &mut camera, &mut observer);
        assert_eq!(c.speed_entry, None);
        assert_eq!(camera.speed, 25.5);
        // The walking speed clamps to Mach 4.
        observer.set_walk_speed(1.0e6);
        assert_eq!(observer.walk_speed, crate::walk::WALK_SPEED_LIMITS.1);
        observer.set_walk_speed(0.0);
        assert_eq!(observer.walk_speed, crate::walk::WALK_SPEED_LIMITS.0);
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
