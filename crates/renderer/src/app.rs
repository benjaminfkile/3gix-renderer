//! The desktop window loop and the headless run.
//!
//! Each rendered frame follows the same order in both modes:
//!
//! 1. advance the [`SimClock`] by the wall-clock delta and integrate every
//!    frame toward the simulation time ([`crate::sim`], `space-model.md`
//!    section 6),
//! 2. apply free flight to the [`Camera`] and re-parent it to its nearest
//!    frame ([`crate::camera`], `space-model.md` section 5),
//! 3. build the camera-relative scene and draw it with the overlay
//!    ([`crate::render`]).
//!
//! Key and mouse bindings are listed in `docs/controls.md`.

use crate::camera::{Camera, FlightInput};
use crate::config::Config;
use crate::render::gpu::Renderer;
use crate::render::headless::{Headless, Image};
use crate::render::overlay::OverlayInfo;
use crate::render::scene::build_scene;
use crate::sim::{ClockAction, SimClock, Simulation};
use anyhow::{anyhow, Context, Result};
use gx_core::frames::FrameSystem;
use gx_core::units::Seconds;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// How many clamped integration rounds a headless run takes to reach the
/// launch offset before it renders anyway, showing "sim lag".
pub const HEADLESS_CATCH_UP_ROUNDS: usize = 64;

/// Longest wall-clock delta one frame applies, seconds, so a stalled window
/// does not turn into one giant step of flight or time.
pub const MAX_FRAME_SECONDS: f64 = 0.25;

/// Pixels of trackpad scroll that count as one wheel notch.
pub const PIXELS_PER_NOTCH: f64 = 40.0;

/// A simulation at the configured launch offset and time scale, with the
/// system still at its epoch.
pub fn launch_simulation(system: FrameSystem, config: &Config) -> Simulation {
    let clock = SimClock::new(Seconds::new(config.start_offset_seconds), config.time_scale);
    Simulation::new(system, clock)
}

/// The overlay values for the current state.
pub fn overlay_info(sim: &Simulation, camera: &Camera, fps: Option<f64>) -> OverlayInfo {
    OverlayInfo {
        sim_time: sim.system.time(),
        time_scale: sim.clock.scale,
        paused: sim.clock.paused,
        camera_frame: camera.frame_id,
        camera_distance: camera.position.length(),
        frame_count: sim.system.tree().frames().len(),
        fps,
        sim_lag: sim.lagging(),
    }
}

/// Renders one headless frame of `sim` from the `Home` view: integrates
/// toward the launch offset (at most [`HEADLESS_CATCH_UP_ROUNDS`] clamped
/// rounds), places the camera, and reads the image back.
pub fn render_headless_frame(sim: &mut Simulation, width: u32, height: u32) -> Result<Image> {
    for _ in 0..HEADLESS_CATCH_UP_ROUNDS {
        if !sim.integrate_toward_target().lagging {
            break;
        }
    }
    let mut camera = Camera::home(&sim.system);
    camera.update_parent(&sim.system);
    let scene = build_scene(&sim.system, &camera, f64::from(width) / f64::from(height));
    let mut headless = Headless::new(width, height)?;
    tracing::info!(adapter = headless.adapter_name(), "headless rendering");
    let overlay = overlay_info(sim, &camera, None).text();
    headless.render(&scene, Some(&overlay))
}

/// The `--headless` run: one frame offscreen, written to the screenshot path
/// if one was given.
pub fn run_headless(config: &Config, system: FrameSystem) -> Result<()> {
    let mut sim = launch_simulation(system, config);
    let image = render_headless_frame(&mut sim, config.width, config.height)?;
    match &config.screenshot {
        Some(path) => {
            image.write_png(path)?;
            tracing::info!(path = %path.display(), "screenshot written");
        }
        None => tracing::info!("headless frame rendered"),
    }
    Ok(())
}

/// The desktop run: a window, the frame loop, and the controls.
pub fn run_windowed(config: &Config, system: FrameSystem) -> Result<()> {
    let event_loop = EventLoop::new().context("opening the window system")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let sim = launch_simulation(system, config);
    let camera = Camera::home(&sim.system);
    let mut app = App {
        sim,
        camera,
        input: Input::default(),
        gfx: None,
        size: (config.width, config.height),
        last_frame: None,
        fps: Fps::default(),
        error: None,
    };
    event_loop
        .run_app(&mut app)
        .context("running the window loop")?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Held keys and accumulated mouse movement between frames.
#[derive(Default)]
struct Input {
    flight: FlightInput,
    looking: bool,
    cursor: Option<(f64, f64)>,
}

impl Input {
    /// This frame's flight input; clears the accumulated mouse movement.
    fn take(&mut self) -> FlightInput {
        let out = self.flight;
        self.flight.look = (0.0, 0.0);
        self.flight.wheel = 0.0;
        out
    }
}

/// Frames per second over half-second windows.
#[derive(Default)]
struct Fps {
    window_start: Option<Instant>,
    frames: u32,
    value: Option<f64>,
}

impl Fps {
    fn frame(&mut self, now: Instant) -> Option<f64> {
        let start = *self.window_start.get_or_insert(now);
        self.frames += 1;
        let span = now - start;
        if span >= Duration::from_millis(500) {
            self.value = Some(f64::from(self.frames) / span.as_secs_f64());
            self.window_start = Some(now);
            self.frames = 0;
        }
        self.value
    }
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
}

impl Gfx {
    fn new(event_loop: &ActiveEventLoop, size: (u32, u32)) -> Result<Gfx> {
        let attrs = Window::default_attributes()
            .with_title("gx-renderer")
            .with_inner_size(PhysicalSize::new(size.0, size.1));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .context("creating the window")?,
        );
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(
                Box::new(event_loop.owned_display_handle()),
            ));
        let surface = instance
            .create_surface(window.clone())
            .context("creating the window surface")?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no graphics adapter for the window: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("window"),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            ..Default::default()
        }))
        .context("opening the graphics device")?;
        let inner = window.inner_size();
        let (w, h) = (inner.width.max(1), inner.height.max(1));
        let mut surface_config = surface
            .get_default_config(&adapter, w, h)
            .ok_or_else(|| anyhow!("the window surface is not supported by the adapter"))?;
        let caps = surface.get_capabilities(&adapter);
        if let Some(srgb) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            surface_config.format = srgb;
        }
        surface.configure(&device, &surface_config);
        let renderer = Renderer::new(&device, &queue, surface_config.format, w, h);
        Ok(Gfx {
            window,
            surface,
            device,
            queue,
            surface_config,
            renderer,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer.resize(width, height);
    }
}

struct App {
    sim: Simulation,
    camera: Camera,
    input: Input,
    gfx: Option<Gfx>,
    size: (u32, u32),
    last_frame: Option<Instant>,
    fps: Fps,
    error: Option<anyhow::Error>,
}

impl App {
    fn key(&mut self, code: KeyCode, pressed: bool, repeat: bool, event_loop: &ActiveEventLoop) {
        let f = &mut self.input.flight;
        match code {
            KeyCode::KeyW => f.forward = pressed,
            KeyCode::KeyS => f.back = pressed,
            KeyCode::KeyA => f.left = pressed,
            KeyCode::KeyD => f.right = pressed,
            KeyCode::KeyE => f.up = pressed,
            KeyCode::KeyQ => f.down = pressed,
            _ if !pressed || repeat => {}
            KeyCode::Space => self.sim.clock.apply(ClockAction::TogglePause),
            KeyCode::BracketLeft => self.sim.clock.apply(ClockAction::HalveScale),
            KeyCode::BracketRight => self.sim.clock.apply(ClockAction::DoubleScale),
            KeyCode::Comma => self.sim.clock.apply(ClockAction::StepBack),
            KeyCode::Period => self.sim.clock.apply(ClockAction::StepForward),
            KeyCode::KeyR => self.sim.clock.apply(ClockAction::Reset),
            KeyCode::Home => self.camera = Camera::home(&self.sim.system),
            KeyCode::Escape => event_loop.exit(),
            other => {
                if let Some(index) = digit_index(other) {
                    if let Some(cam) = Camera::view_of_index(&self.sim.system, index) {
                        self.camera = Camera {
                            speed: self.camera.speed,
                            ..cam
                        };
                    }
                }
            }
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = self
            .last_frame
            .map_or(0.0, |t| (now - t).as_secs_f64().min(MAX_FRAME_SECONDS));
        self.last_frame = Some(now);
        let fps = self.fps.frame(now);

        self.sim.update(Seconds::new(dt));
        let input = self.input.take();
        self.camera.fly(&self.sim.system, &input, dt);
        self.camera.update_parent(&self.sim.system);

        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };
        let (w, h) = gfx.renderer.size();
        let scene = build_scene(&self.sim.system, &self.camera, f64::from(w) / f64::from(h));
        let overlay = overlay_info(&self.sim, &self.camera, fps).text();
        let texture = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.device, &gfx.surface_config);
                return;
            }
            _ => return,
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        gfx.renderer.render(&view, &scene, Some(&overlay));
        gfx.window.pre_present_notify();
        gfx.queue.present(texture);
    }
}

/// Number keys: `1` to `9` select indices 0 to 8, `0` selects index 9.
fn digit_index(code: KeyCode) -> Option<usize> {
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        match Gfx::new(event_loop, self.size) {
            Ok(gfx) => {
                gfx.window.request_redraw();
                self.gfx = Some(gfx);
            }
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    self.key(code, pressed, event.repeat, event_loop);
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.input.looking = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x, position.y);
                if let (true, Some(last)) = (self.input.looking, self.input.cursor) {
                    self.input.flight.look.0 += p.0 - last.0;
                    self.input.flight.look.1 += p.1 - last.1;
                }
                self.input.cursor = Some(p);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.flight.wheel += match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(p) => p.y / PIXELS_PER_NOTCH,
                };
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gfx) = &self.gfx {
            gfx.window.request_redraw();
        }
    }
}
