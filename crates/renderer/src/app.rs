//! The desktop window loop and the headless run.
//!
//! Each rendered frame follows the same order in both modes:
//!
//! 1. advance the [`SimClock`] by the wall-clock delta and integrate every
//!    frame toward the simulation time ([`crate::sim`], `space-model.md`
//!    section 6),
//! 2. apply free flight to the [`Camera`] and re-parent it to its nearest
//!    frame ([`crate::camera`], `space-model.md` section 5),
//! 3. stream matter: select cells (every 100 ms), start requests, apply
//!    fetch results and readiness pushes, collect extracted meshes, and
//!    evict ([`crate::world`], [`crate::stream`]),
//! 4. build the camera-relative scene with the drawn cells and their lights
//!    and draw it with the overlay ([`crate::render`]).
//!
//! Key and mouse bindings are listed in `docs/controls.md`.

use crate::camera::Camera;
use crate::config::{Config, View};
use crate::controls::Controls;
pub use crate::controls::{EXPOSURE_BIAS_LIMITS, PIXELS_PER_NOTCH};
use crate::extract::default_threads;
use crate::hub::HubClient;
use crate::render::gpu::{wanted_features, Exposure, Renderer};
use crate::render::headless::{Headless, Image};
use crate::render::overlay::{MatterStats, OverlayInfo};
pub use crate::render::scene::matter_scene;
use crate::render::scene::Scene;
use crate::sim::{SimClock, Simulation};
use crate::stream::{FetchEvent, Fetcher};
use crate::world::World;
use anyhow::{anyhow, Context, Result};
use gx_core::frames::FrameSystem;
use gx_core::units::Seconds;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// How many clamped integration rounds a headless run takes to reach the
/// launch offset before it renders anyway, showing "sim lag".
pub const HEADLESS_CATCH_UP_ROUNDS: usize = 64;

/// Longest wall-clock delta one frame applies, seconds, so a stalled window
/// does not turn into one giant step of flight or time.
pub const MAX_FRAME_SECONDS: f64 = 0.25;

/// Longest a headless run waits for the selected cells, seconds, before it
/// renders whatever has arrived.
pub const HEADLESS_STREAM_SECONDS: f64 = 60.0;

/// Starts requests for the cells the world wants and applies every fetch
/// event that has arrived.
pub fn stream_step(
    world: &mut World,
    fetcher: &mut Fetcher,
    system: &FrameSystem,
    events: Vec<FetchEvent>,
    now: f64,
) {
    for event in events {
        match event {
            FetchEvent::Completed { key, outcome, .. } => world.complete(key, outcome, now),
            FetchEvent::ChunkReady(key) => world.chunk_ready(&key),
        }
    }
    for key in world.take_requests(now) {
        if let Some(frame) = system.tree().get(key.frame_id) {
            fetcher.request(key, frame.root_extent);
        }
    }
    world.pump();
}

/// The matter values for the overlay.
pub fn matter_stats(world: &World, fetcher: Option<&Fetcher>, scene: &Scene) -> MatterStats {
    let counts = world.counts();
    MatterStats {
        cells_selected: counts.cache.selected,
        cells_ready: counts.cache.ready,
        cells_pending: counts.cache.pending,
        meshes_drawn: scene.surfaces.len(),
        triangles_drawn: scene.surfaces.iter().map(|d| d.mesh.triangle_count()).sum(),
        volumes_drawn: scene.volumes.len(),
        sprites_drawn: scene.sprites.len(),
        lights_active: scene.lights.len(),
        ..MatterStats::from_tally(fetcher.map(Fetcher::tally).unwrap_or_default())
    }
}

/// A simulation at the configured launch offset and time scale, with the
/// system still at its epoch.
pub fn launch_simulation(system: FrameSystem, config: &Config) -> Simulation {
    let clock = SimClock::new(Seconds::new(config.start_offset_seconds), config.time_scale);
    Simulation::new(system, clock)
}

/// The overlay values for the current state.
pub fn overlay_info(
    sim: &Simulation,
    camera: &Camera,
    fps: Option<f64>,
    matter: MatterStats,
) -> OverlayInfo {
    OverlayInfo {
        sim_time: sim.system.time(),
        time_scale: sim.clock.scale,
        paused: sim.clock.paused,
        camera_frame: camera.frame_id,
        camera_distance: camera.position.length(),
        frame_count: sim.system.tree().frames().len(),
        fps,
        sim_lag: sim.lagging(),
        matter,
    }
}

/// What a headless run renders and how long it waits for matter.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HeadlessRun {
    /// Image width, pixels.
    pub width: u32,
    /// Image height, pixels.
    pub height: u32,
    /// The camera view.
    pub view: View,
    /// Factor on the view's distance from its frame origin.
    pub view_distance_scale: f64,
    /// Wait at most this long, seconds, for every selected and pinned cell
    /// to be ready ([`World::ready`]). `None` waits at most
    /// [`HEADLESS_STREAM_SECONDS`] for them to settle ([`World::settled`]).
    pub wait_ready_seconds: Option<f64>,
    /// Draw the overlay text, the frame markers, and the lines between
    /// them into the image. Without them only matter is drawn.
    pub overlay: bool,
}

impl HeadlessRun {
    /// The `Home` view with the overlay, waiting as long as
    /// [`HEADLESS_STREAM_SECONDS`].
    pub fn home(width: u32, height: u32) -> HeadlessRun {
        HeadlessRun {
            width,
            height,
            view: View::Home,
            view_distance_scale: 1.0,
            wait_ready_seconds: None,
            overlay: true,
        }
    }

    /// The run a configuration asks for.
    pub fn from_config(config: &Config) -> HeadlessRun {
        HeadlessRun {
            width: config.width,
            height: config.height,
            view: config.view,
            view_distance_scale: config.view_distance_scale,
            wait_ready_seconds: config.wait_ready_seconds,
            overlay: config.overlay,
        }
    }
}

/// What a headless run saw: the overlay values and how the wait for
/// matter ended. Written by `--stats-json`.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadlessStats {
    /// The view rendered.
    pub view: View,
    /// The overlay values at the rendered frame.
    pub overlay: OverlayInfo,
    /// Pinned far field cells, besides the selection.
    pub cells_pinned: usize,
    /// Every selected and pinned cell was ready when the frame was drawn.
    pub ready: bool,
    /// Wall-clock seconds spent streaming before drawing.
    pub stream_seconds: f64,
}

impl HeadlessStats {
    /// The statistics as a JSON object, keys in a fixed order.
    pub fn to_json(&self) -> serde_json::Value {
        let o = &self.overlay;
        let m = &o.matter;
        serde_json::json!({
            "view": self.view.to_string(),
            "sim_time_seconds_since_j2000": o.sim_time.value(),
            "camera_frame": o.camera_frame,
            "camera_distance_m": o.camera_distance,
            "frames": o.frame_count,
            "sim_lag": o.sim_lag,
            "cells_selected": m.cells_selected,
            "cells_pinned": self.cells_pinned,
            "cells_ready": m.cells_ready,
            "cells_pending": m.cells_pending,
            "requests": m.requests,
            "cells_fetched": m.cells_fetched,
            "bytes_fetched": m.bytes_fetched,
            "first_round_trip_seconds": m.first_round_trip,
            "last_round_trip_seconds": m.last_round_trip,
            "meshes_drawn": m.meshes_drawn,
            "triangles_drawn": m.triangles_drawn,
            "volumes_drawn": m.volumes_drawn,
            "sprites_drawn": m.sprites_drawn,
            "lights_active": m.lights_active,
            "ready": self.ready,
            "stream_seconds": self.stream_seconds,
        })
    }
}

/// One headless frame and its statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadlessFrame {
    /// The image read back.
    pub image: Image,
    /// What the run saw.
    pub stats: HeadlessStats,
}

/// The camera of a view, at the view's distance scale. Fails when no frame
/// has the number key the view names.
pub fn view_camera(system: &FrameSystem, view: View, distance_scale: f64) -> Result<Camera> {
    let camera = match view {
        View::Home => Camera::home(system),
        View::Key(index) => Camera::view_of_index(system, index)
            .ok_or_else(|| anyhow!("no frame is bound to the number key of view {view}"))?,
    };
    Ok(camera.with_distance_scale(distance_scale))
}

/// Renders one headless frame of `sim` from the `Home` view (see
/// [`render_headless_view`] and [`HeadlessRun::home`]).
pub fn render_headless_frame(
    sim: &mut Simulation,
    fetcher: Option<&mut Fetcher>,
    width: u32,
    height: u32,
) -> Result<Image> {
    Ok(render_headless_view(sim, fetcher, &HeadlessRun::home(width, height))?.image)
}

/// Renders one headless frame of `sim`: integrates toward the launch
/// offset (at most [`HEADLESS_CATCH_UP_ROUNDS`] clamped rounds), places the
/// camera for the run's view, streams the selected cells through
/// `fetcher` until they are ready (or the wait ends), and reads the image
/// back.
pub fn render_headless_view(
    sim: &mut Simulation,
    mut fetcher: Option<&mut Fetcher>,
    run: &HeadlessRun,
) -> Result<HeadlessFrame> {
    for _ in 0..HEADLESS_CATCH_UP_ROUNDS {
        if !sim.integrate_toward_target().lagging {
            break;
        }
    }
    let (width, height) = (run.width, run.height);
    let mut camera = view_camera(&sim.system, run.view, run.view_distance_scale)?;
    camera.update_parent(&sim.system);
    let mut world = World::new(default_threads());
    let start = Instant::now();
    world.select(&sim.system, &camera, (width, height), 0.0, true);
    let limit = run.wait_ready_seconds.unwrap_or(HEADLESS_STREAM_SECONDS);
    let done = |world: &World| match run.wait_ready_seconds {
        Some(_) => world.ready(),
        None => world.settled(),
    };
    if let Some(f) = fetcher.as_deref_mut() {
        loop {
            let now = start.elapsed().as_secs_f64();
            let events = f.wait(Duration::from_millis(50));
            stream_step(&mut world, f, &sim.system, events, now);
            if done(&world) {
                break;
            }
            if now > limit {
                tracing::warn!(seconds = limit, "not every cell arrived in time");
                break;
            }
        }
    }
    world.finish_extraction();
    let stream_seconds = start.elapsed().as_secs_f64();
    let mut scene = matter_scene(
        &mut world,
        &sim.system,
        &camera,
        (width, height),
        stream_seconds,
    );
    if !run.overlay {
        // The markers and lines are visual aids like the text: only matter
        // remains.
        scene.markers.clear();
        scene.lines.clear();
    }
    let mut headless = Headless::new(width, height)?;
    tracing::info!(adapter = headless.adapter_name(), "headless rendering");
    let info = overlay_info(
        sim,
        &camera,
        None,
        matter_stats(&world, fetcher.as_deref(), &scene),
    );
    let text = info.text();
    let image = headless.render(&scene, run.overlay.then_some(text.as_str()))?;
    Ok(HeadlessFrame {
        image,
        stats: HeadlessStats {
            view: run.view,
            overlay: info,
            cells_pinned: world.counts().cache.pinned,
            ready: world.ready(),
            stream_seconds,
        },
    })
}

/// The `--headless` run: one frame offscreen, written to the screenshot path
/// if one was given, with its statistics written to the `--stats-json`
/// path. Cells are fetched through `client` on `runtime`. With
/// `--wait-ready-seconds`, fails after writing both when the selection was
/// not ready in time.
pub fn run_headless(
    config: &Config,
    system: FrameSystem,
    client: Arc<HubClient>,
    runtime: tokio::runtime::Handle,
) -> Result<()> {
    let mut sim = launch_simulation(system, config);
    let mut fetcher = Fetcher::new(runtime, client);
    let run = HeadlessRun::from_config(config);
    let frame = render_headless_view(&mut sim, Some(&mut fetcher), &run)?;
    match &config.screenshot {
        Some(path) => {
            frame.image.write_png(path)?;
            tracing::info!(path = %path.display(), "screenshot written");
        }
        None => tracing::info!("headless frame rendered"),
    }
    if let Some(path) = &config.stats_json {
        let json = serde_json::to_string_pretty(&frame.stats.to_json())?;
        std::fs::write(path, json + "\n").with_context(|| format!("writing {}", path.display()))?;
        tracing::info!(path = %path.display(), "statistics written");
    }
    if let Some(limit) = config.wait_ready_seconds {
        if !frame.stats.ready {
            let m = &frame.stats.overlay.matter;
            return Err(anyhow!(
                "the selection was not ready after {limit} s: {} of {} selected cells ready, {} pending",
                m.cells_ready,
                m.cells_selected,
                m.cells_pending
            ));
        }
    }
    Ok(())
}

/// The desktop run: a window, the frame loop, and the controls. Cells are
/// fetched through `client` on `runtime`.
pub fn run_windowed(
    config: &Config,
    system: FrameSystem,
    client: Arc<HubClient>,
    runtime: tokio::runtime::Handle,
) -> Result<()> {
    let event_loop = EventLoop::new().context("opening the window system")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let sim = launch_simulation(system, config);
    let camera = Camera::home(&sim.system);
    let mut app = App {
        sim,
        camera,
        world: World::new(default_threads()),
        fetcher: Fetcher::new(runtime, client),
        started: Instant::now(),
        controls: Controls::default(),
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
            required_features: wanted_features(&adapter),
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
    world: World,
    fetcher: Fetcher,
    started: Instant,
    controls: Controls,
    gfx: Option<Gfx>,
    size: (u32, u32),
    last_frame: Option<Instant>,
    fps: Fps,
    error: Option<anyhow::Error>,
}

impl App {
    fn frame(&mut self) {
        let now = Instant::now();
        let dt = self
            .last_frame
            .map_or(0.0, |t| (now - t).as_secs_f64().min(MAX_FRAME_SECONDS));
        self.last_frame = Some(now);
        let fps = self.fps.frame(now);

        self.sim.update(Seconds::new(dt));
        let input = self.controls.take_flight();
        self.camera.fly(&self.sim.system, &input, dt);
        self.camera.update_parent(&self.sim.system);

        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };
        let size = gfx.renderer.size();
        let t = self.started.elapsed().as_secs_f64();
        let system = &self.sim.system;
        self.world.select(system, &self.camera, size, t, false);
        let events = self.fetcher.drain();
        stream_step(&mut self.world, &mut self.fetcher, system, events, t);
        self.world.evict(t);
        let scene = matter_scene(&mut self.world, system, &self.camera, size, t);
        let stats = matter_stats(&self.world, Some(&self.fetcher), &scene);
        let overlay = overlay_info(&self.sim, &self.camera, fps, stats).text();
        let exposure = Exposure {
            bias_stops: self.controls.exposure_bias,
            adapt_seconds: Some(dt),
        };
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
        gfx.renderer.render(&view, &scene, Some(&overlay), exposure);
        gfx.window.pre_present_notify();
        gfx.queue.present(texture);
    }
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
            WindowEvent::RedrawRequested => self.frame(),
            other => {
                if self
                    .controls
                    .handle(&other, &mut self.sim, &mut self.camera)
                {
                    event_loop.exit();
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gfx) = &self.gfx {
            gfx.window.request_redraw();
        }
    }
}
