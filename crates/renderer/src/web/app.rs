//! The browser frame loop on a canvas.
//!
//! The same steps as the desktop window loop of [`crate::app`]: advance the
//! simulation clock (clamped to the hub's now) and integrate, keep and
//! fly the free camera ([`crate::camera::advance`]), solve the light time
//! of every frame
//! ([`crate::seen`]), stream
//! matter through [`WebFetcher`], build the camera-relative scene, and draw
//! it with [`Renderer`]. Differences are only in the platform: `winit`
//! drives a canvas element instead of a window, wgpu runs on the browser's
//! WebGPU, the device is created asynchronously, time comes from
//! `performance.now()`, and extraction runs inline because the page has no
//! threads.

use super::binding::Shared;
use super::fetch::WebFetcher;
use super::hub::now_seconds;
use crate::camera::Camera;
use crate::controls::Controls;
use crate::extract::default_threads;
use crate::render::gpu::{describe_adapter, present_mode, wanted_features, Renderer};
use crate::render::overlay::{MatterStats, OverlayInfo};
use crate::render::scene::{scene_with, Scene};
use crate::seen::Seen;
use crate::sim::Simulation;
use crate::stream::FetchEvent;
use crate::world::{DrawList, World, COMPOSITE_SLICE_SECONDS};
use anyhow::{anyhow, Context, Result};
use gx_core::units::Seconds;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, OwnedDisplayHandle};
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::{Window, WindowId};

/// Longest wall-clock delta one frame applies, seconds.
pub const MAX_FRAME_SECONDS: f64 = 0.25;

/// The graphics state, created asynchronously once the canvas window
/// exists.
struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// The sRGB view format the renderer draws in.
    view_format: wgpu::TextureFormat,
    renderer: Renderer,
}

impl Gfx {
    async fn new(window: Arc<Window>, display: OwnedDisplayHandle) -> Result<Gfx> {
        let mut desc = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display));
        desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(desc);
        let surface = instance
            .create_surface(window.clone())
            .context("creating the canvas surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("no WebGPU adapter (does this browser support WebGPU?): {e}"))?;
        web_sys::console::info_1(
            &format!("gx-renderer: rendering with {}", describe_adapter(&adapter)).into(),
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("canvas"),
                required_features: wanted_features(&adapter),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .context("opening the WebGPU device")?;
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));
        let mut config = surface
            .get_default_config(&adapter, w, h)
            .ok_or_else(|| anyhow!("the canvas surface is not supported by the adapter"))?;
        // Canvases offer only linear formats; draw through an sRGB view so
        // the output is encoded the same way as on the desktop.
        let view_format = config.format.add_srgb_suffix();
        if view_format != config.format {
            config.view_formats.push(view_format);
        }
        config.present_mode = present_mode(&surface.get_capabilities(&adapter));
        surface.configure(&device, &config);
        let renderer = Renderer::new(&device, &queue, view_format, w, h);
        Ok(Gfx {
            surface,
            device,
            queue,
            config,
            view_format,
            renderer,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.renderer.resize(width, height);
    }
}

/// One step of the frame thread's matter work, as natively
/// ([`crate::app::matter_step`]): composites what has arrived until the
/// step's slice is spent, runs [`World::step`], hands the requests to the
/// fetcher, and records the step's duration. Returns the draw list.
fn matter_step(
    world: &mut World,
    fetcher: &mut WebFetcher,
    sim: &Simulation,
    seen: &Seen,
    size: (u32, u32),
    now: f64,
) -> DrawList {
    let started = crate::clock::now_seconds();
    fetcher.set_time_utc_ms(gx_core::timescale::tdb_seconds_to_utc_ms(
        sim.time().value(),
    ));
    let mut cells = Vec::new();
    for e in fetcher.drain_until(started + COMPOSITE_SLICE_SECONDS) {
        match e {
            FetchEvent::Completed { key, outcome, .. } => cells.push((key, outcome)),
            FetchEvent::Surface {
                key,
                outcome,
                partial,
                ..
            } => world.complete_surface(key, outcome, partial),
            FetchEvent::Registry(_) => {}
        }
    }
    let (requests, draw) = world.step(seen, size, now, cells, started);
    if let Some(cells) = requests {
        let tree = seen.system().tree();
        fetcher.set_wanted_cells(
            cells
                .into_iter()
                .filter_map(|(key, px)| tree.get(key.frame_id).map(|f| (key, f.root_extent, px)))
                .collect(),
        );
    }
    if let Some(tiles) = world.take_wanted_surfaces() {
        fetcher.set_wanted_surfaces(tiles);
    }
    world.record_step((crate::clock::now_seconds() - started) * 1e3);
    draw
}

/// The API's `stats` event values (`docs/api.md` section 8) of a drawn
/// scene, as natively (`crate::app::api_stats`).
fn api_stats(world: &World, fetcher: &WebFetcher, scene: &Scene, fps: f64) -> crate::api::Stats {
    let m = MatterStats::from_stats(fetcher.stats());
    crate::api::Stats {
        fps,
        tiles_drawn: scene.tiles.len(),
        triangles: scene
            .surfaces
            .iter()
            .map(|d| d.mesh.triangle_count())
            .sum::<usize>()
            + scene.tiles.len() * gx_core::surface::TILE_QUADS * gx_core::surface::TILE_QUADS * 2,
        cells_drawn: scene.surfaces.len(),
        streaming: world.counts().cache.pending,
        computing: m.module_runs,
        uploading: m.uploads_pending,
    }
}

/// The picture the API's queries run against (`docs/api.md` section 6),
/// as natively (`crate::app::api_picture`).
fn api_picture(
    world: &World,
    draw: &DrawList,
    seen: &Seen,
    size: (u32, u32),
) -> crate::api::Picture {
    crate::api::Picture::new(seen.system(), seen.camera(), draw, size)
        .with_coarse(world.surfaces().coarse_drawn())
}

/// The browser application: the simulation, the camera, the matter
/// pipeline, and the canvas.
pub struct WebApp {
    sim: Simulation,
    camera: Camera,
    spawner: crate::spawn::Spawner,
    world: World,
    fetcher: WebFetcher,
    controls: Controls,
    canvas: Option<web_sys::HtmlCanvasElement>,
    window: Option<Arc<Window>>,
    gfx: Rc<RefCell<Option<Gfx>>>,
    started: f64,
    last_frame: Option<f64>,
    fps_window: (f64, u32, Option<f64>),
    /// When the last frame was drawn.
    last_draw: Option<f64>,
    /// Draw the next frame's scene whatever it is (a window event came).
    force_draw: bool,
    /// What was drawn last (`crate::redraw`).
    redraw: crate::redraw::Redraw,
    /// The API and the listeners, shared with the `Renderer` class.
    shared: Rc<Shared>,
    /// The cursor, physical pixels.
    cursor: (f64, f64),
    /// The canvas asked to close or the built-in controls asked to quit:
    /// this renderer stops at the next [`WebApp::after_events`].
    closing: bool,
    /// When the memory report was last logged ([`MEMORY_LOG_SECONDS`]).
    #[cfg(feature = "memory-report")]
    memory_logged: f64,
}

/// Seconds between two `gx-memory` console lines (the memory report,
/// `docs/architecture.md`).
#[cfg(feature = "memory-report")]
pub const MEMORY_LOG_SECONDS: f64 = 5.0;

impl WebApp {
    /// An application drawing the simulation of `launch` from its camera
    /// on `canvas`, fetching through `fetcher`, with the exposure and mesh
    /// mode of `options` ([`crate::options::launch`], as natively), driving
    /// the API in `shared` each frame.
    pub fn new(
        launch: crate::options::Launch,
        options: &crate::options::ViewOptions,
        mut fetcher: WebFetcher,
        canvas: web_sys::HtmlCanvasElement,
        shared: Rc<Shared>,
    ) -> WebApp {
        WebApp {
            sim: launch.sim,
            camera: launch.camera,
            spawner: crate::spawn::Spawner {
                active: None,
                pending: launch.pending_spawn,
            },
            world: {
                let mut w = World::new(default_threads());
                w.start_layers(fetcher.hub().build(), options.hide_layers.as_deref());
                fetcher.set_hidden_layers(w.hidden_layers().clone());
                w
            },
            fetcher,
            controls: {
                let mut c = Controls::new(options.exposure_stops);
                c.mesh_mode = options.mesh_mode;
                c
            },
            canvas: Some(canvas),
            window: None,
            gfx: Rc::default(),
            started: now_seconds(),
            last_frame: None,
            #[cfg(feature = "memory-report")]
            memory_logged: f64::NEG_INFINITY,
            fps_window: (0.0, 0, None),
            last_draw: None,
            force_draw: true,
            redraw: crate::redraw::Redraw::new(),
            shared,
            cursor: (0.0, 0.0),
            closing: false,
        }
    }

    /// Frames drawn per second over half-second windows; called every
    /// display frame, `drew` when this one was drawn, so it falls when the
    /// loop stops drawing.
    fn fps(&mut self, now: f64, drew: bool) -> Option<f64> {
        let (start, frames, value) = &mut self.fps_window;
        if *frames == 0 && value.is_none() && *start == 0.0 {
            *start = now;
        }
        if drew {
            *frames += 1;
        }
        if now - *start >= 0.5 {
            *value = Some(f64::from(*frames) / (now - *start));
            *start = now;
            *frames = 0;
        }
        *value
    }

    /// One frame, then the API's events to the listeners.
    fn frame(&mut self) {
        self.tick();
        self.shared.deliver();
    }

    fn tick(&mut self) {
        let now = now_seconds();
        let dt = self
            .last_frame
            .map_or(0.0, |t| (now - t).clamp(0.0, MAX_FRAME_SECONDS));
        self.last_frame = Some(now);
        let fps = self.fps(now, false);
        #[cfg(feature = "memory-report")]
        if now - self.memory_logged >= MEMORY_LOG_SECONDS {
            self.memory_logged = now;
            if let Ok(json) = serde_json::to_string(&self.world.memory_report()) {
                web_sys::console::log_2(&"gx-memory".into(), &json.into());
            }
        }

        let hub_now = gx_core::timescale::utc_ms_to_tdb_seconds(self.fetcher.now_utc_ms());
        let shared = self.shared.clone();
        let mut api = shared.api.borrow_mut();
        api.begin_tick(crate::api::Target {
            sim: &mut self.sim,
            camera: &mut self.camera,
            spawner: &mut self.spawner,
            keys: &mut self.controls,
        });
        if api.get_compute_budget() != self.fetcher.compute_budget() {
            self.fetcher.set_compute_budget(api.get_compute_budget());
        }
        if let (Some((w, h, _)), Some(gfx)) = (api.take_size(), self.gfx.borrow_mut().as_mut()) {
            gfx.resize(w, h);
            self.force_draw = true;
        }
        if api.paused() || api.destroyed() {
            // Paused: no simulation, streaming, computing, or drawing; the
            // clock holds its time (`docs/api.md` section 3).
            api.end_tick(&self.sim, &self.camera, &self.world);
            return;
        }
        let input = api.flight(self.controls.take_flight());
        self.spawner.frame(
            self.sim.system(),
            &self.world,
            &mut self.camera,
            &mut self.controls,
            &input,
            now - self.started,
        );
        crate::camera::advance(
            &mut self.sim,
            &mut self.camera,
            dt,
            Seconds::new(hub_now),
            &input,
            self.controls.attached,
        );
        api.end_tick(&self.sim, &self.camera, &self.world);

        let mut slot = self.gfx.borrow_mut();
        let Some(gfx) = slot.as_mut() else {
            return;
        };
        let size = gfx.renderer.size();
        let t = now - self.started;
        self.sim.prepare_light(&self.camera);
        let near = self.world.matter_distances(self.sim.system(), &self.camera);
        let seen = self.sim.seen_near(&self.camera, &near);
        if self
            .world
            .toggle_layers(&self.controls.take_layer_toggles())
        {
            (self.fetcher).set_hidden_layers(self.world.hidden_layers().clone());
        }
        let draw = matter_step(
            &mut self.world,
            &mut self.fetcher,
            &self.sim,
            &seen,
            size,
            t,
        );
        let mut scene = scene_with(&draw, &seen, size);
        scene.mesh_mode = self.controls.mesh_mode;
        if self.controls.hide_guides {
            scene.markers.clear();
            scene.lines.clear();
        }
        api.set_picture(api_picture(&self.world, &draw, &seen, size));
        api.observe(
            self.spawner.pending.is_none() && self.world.ready(),
            api_stats(&self.world, &self.fetcher, &scene, fps.unwrap_or(0.0)),
            t,
        );
        let overlay = api
            .overlay_text(self.overlay(&seen, &scene, fps))
            .unwrap_or_default();
        // The frame cap of `setMaxFps`: a frame due sooner after the last
        // one drawn is skipped.
        if let (Some(cap), Some(last)) = (api.max_fps(), self.last_draw) {
            if now - last < 1.0 / f64::from(cap) && !self.force_draw {
                return;
            }
        }
        drop(api);
        // Draw only what differs visibly from the last frame drawn
        // (`crate::redraw`); the browser still calls every display frame
        // for input, streaming, and the simulation.
        let focal = crate::farfield::focal_px(f64::from(size.1.max(1)), self.camera.fov_y);
        let need = self.redraw.decide(
            &scene,
            &overlay,
            &self.controls.exposure(None),
            t,
            focal,
            self.force_draw,
        );
        if !need.draw() {
            return;
        }
        self.force_draw = false;
        let adapt = self
            .last_draw
            .map_or(0.0, |d| (now - d).clamp(0.0, MAX_FRAME_SECONDS));
        self.last_draw = Some(now);
        self.fps_window.1 += 1;
        let exposure = self.controls.exposure(Some(adapt));
        let texture = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.device, &gfx.config);
                return;
            }
            _ => return,
        };
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(gfx.view_format),
            ..Default::default()
        });
        gfx.renderer.render(&view, &scene, Some(&overlay), exposure);
        gfx.queue.present(texture);
        self.redraw.drawn(scene, overlay, exposure, t, need);
    }

    fn overlay(&self, seen: &Seen, scene: &Scene, fps: Option<f64>) -> String {
        let counts = self.world.counts();
        let clock = self.sim.clock();
        let mut text = OverlayInfo {
            sim_time: self.sim.time(),
            offset_from_now: clock.offset_from_now(),
            at_now: clock.is_at_now(),
            rate: clock.rate(),
            rewinding: clock.rewinding(),
            paused: clock.paused(),
            crosshair: seen.nearest_to_crosshair().map(|(f, t)| (f, t.value())),
            camera_frame: self.camera.frame_id,
            camera_distance: self.camera.position.length(),
            camera_speed: self.camera.cruise,
            frame_count: self.sim.system().tree().frames().len(),
            fps,
            exposure_fixed: self.controls.exposure_fixed,
            matter: MatterStats {
                cells_selected: counts.cache.selected,
                cells_ready: counts.cache.ready,
                cells_pending: counts.cache.pending,
                meshes_drawn: scene.surfaces.len(),
                triangles_drawn: scene.surfaces.iter().map(|d| d.mesh.triangle_count()).sum(),
                volumes_drawn: scene.volumes.len(),
                sprites_drawn: scene.sprites.len(),
                lights_active: scene.lights.len(),
                frame_step_p95_ms: self
                    .world
                    .step_times()
                    .recent_percentile(crate::clock::OVERLAY_STEPS, 95.0),
                ..MatterStats::from_stats(self.fetcher.stats())
            },
        }
        .text();
        if self.controls.attached {
            text.push('\n');
            text.push_str(&crate::spawn::Spawner::status_line(&self.camera));
        }
        if self.controls.spawning {
            text.push('\n');
            text.push_str(&crate::spawn::chooser_line(&crate::spawn::spawn_frames(
                self.world.cache(),
                self.sim.system().tree(),
            )));
        }
        if self.controls.choosing_layers {
            text.push('\n');
            text.push_str(&self.world.layer_view().chooser_line());
        }
        if let Some(line) = self.world.layer_view().hidden_line() {
            text.push('\n');
            text.push_str(&line);
        }
        for line in self.controls.status_lines() {
            text.push('\n');
            text.push_str(&line);
        }
        text
    }
}

/// What the page's one event loop ([`super::host`]) calls on each
/// renderer: one window per renderer, events routed by window.
impl WebApp {
    /// Creates this renderer's canvas window and, asynchronously, its
    /// device. Called once, when the host first sees the renderer.
    pub fn open(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("gx-renderer")
            .with_canvas(self.canvas.take())
            .with_prevent_default(true)
            .with_focusable(true);
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                let message = format!("creating the canvas window: {e}");
                web_sys::console::error_1(&format!("gx-renderer: {message}").into());
                self.shared.fail(&message);
                return;
            }
        };
        self.shared.scale.set(window.scale_factor());
        let slot = self.gfx.clone();
        let shared = self.shared.clone();
        let display = event_loop.owned_display_handle();
        let for_gfx = window.clone();
        wasm_bindgen_futures::spawn_local(async move {
            match Gfx::new(for_gfx.clone(), display).await {
                Ok(gfx) => {
                    *slot.borrow_mut() = Some(gfx);
                    for_gfx.request_redraw();
                }
                Err(e) => {
                    web_sys::console::error_1(&format!("gx-renderer: {e:#}").into());
                    shared.fail(&format!("{e:#}"));
                }
            }
        });
        self.window = Some(window);
    }

    /// The window of this renderer, once [`WebApp::open`] made it.
    pub fn window_id(&self) -> Option<WindowId> {
        self.window.as_ref().map(|w| w.id())
    }

    /// One event of this renderer's window.
    pub fn window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.closing = true,
            WindowEvent::Resized(size) => {
                if let Some(gfx) = self.gfx.borrow_mut().as_mut() {
                    gfx.resize(size.width, size.height);
                }
                self.force_draw = true;
            }
            WindowEvent::RedrawRequested => self.frame(),
            other => {
                match &other {
                    WindowEvent::CursorMoved { position, .. } => {
                        self.cursor = (position.x, position.y);
                    }
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        self.shared.scale.set(*scale_factor);
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        // `hit` in both control modes (`docs/api.md`
                        // section 8), physical pixels as natively.
                        let (x, y) = self.cursor;
                        let b = match button {
                            winit::event::MouseButton::Left => 0,
                            winit::event::MouseButton::Right => 1,
                            winit::event::MouseButton::Middle => 2,
                            winit::event::MouseButton::Back => 3,
                            winit::event::MouseButton::Forward => 4,
                            winit::event::MouseButton::Other(n) => 5 + u32::from(*n),
                        };
                        let mut api = self.shared.api.borrow_mut();
                        match state {
                            winit::event::ElementState::Pressed => api.mouse_down(x, y, b),
                            winit::event::ElementState::Released => api.mouse_up(x, y, b),
                        }
                    }
                    _ => {}
                }
                if matches!(
                    other,
                    WindowEvent::KeyboardInput { .. }
                        | WindowEvent::MouseInput { .. }
                        | WindowEvent::MouseWheel { .. }
                        | WindowEvent::Focused(_)
                ) {
                    self.force_draw = true;
                }
                // With controls `None` the keyboard and mouse leave the
                // camera, choosers, clock, and exposure alone.
                let builtin = self.shared.api.borrow().controls() == crate::api::Controls::Builtin;
                if builtin
                    && self
                        .controls
                        .handle(&other, &mut self.sim, &mut self.camera)
                {
                    self.closing = true;
                }
            }
        }
    }

    /// After each batch of events: asks for the next frame, or returns
    /// false when this renderer is done (`destroy`, or a close), having
    /// dropped its surface and device; the host then drops the rest of it,
    /// the fetcher and its module worker included. The page's loop keeps
    /// running for later renderers.
    pub fn after_events(&mut self) -> bool {
        if self.closing && !self.shared.api.borrow().destroyed() {
            let _ = self.shared.api.borrow_mut().destroy();
        }
        if self.shared.api.borrow().destroyed() {
            *self.gfx.borrow_mut() = None;
            self.window = None;
            return false;
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
        true
    }
}
