//! The browser frame loop on a canvas.
//!
//! The same steps as the desktop window loop of [`crate::app`]: advance the
//! simulation clock and integrate, fly and re-parent the camera, stream
//! matter through [`WebFetcher`], build the camera-relative scene, and draw
//! it with [`Renderer`]. Differences are only in the platform: `winit`
//! drives a canvas element instead of a window, wgpu runs on the browser's
//! WebGPU, the device is created asynchronously, time comes from
//! `performance.now()`, and extraction runs inline because the page has no
//! threads.

use super::hub::{now_seconds, WebFetcher};
use crate::camera::Camera;
use crate::controls::Controls;
use crate::extract::default_threads;
use crate::render::gpu::{wanted_features, Renderer};
use crate::render::overlay::{MatterStats, OverlayInfo};
use crate::render::scene::{matter_scene, Scene};
use crate::sim::Simulation;
use crate::stream::FetchEvent;
use crate::world::World;
use anyhow::{anyhow, Context, Result};
use gx_core::units::Seconds;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use winit::application::ApplicationHandler;
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
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("no WebGPU adapter (does this browser support WebGPU?): {e}"))?;
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

/// Starts requests for the cells the world wants and applies every fetch
/// event that has arrived.
fn stream_step(world: &mut World, fetcher: &mut WebFetcher, sim: &Simulation, now: f64) {
    for event in fetcher.drain() {
        match event {
            FetchEvent::Completed { key, outcome, .. } => world.complete(key, outcome, now),
            FetchEvent::ChunkReady(key) => world.chunk_ready(&key),
        }
    }
    for key in world.take_requests(now) {
        if let Some(frame) = sim.system.tree().get(key.frame_id) {
            fetcher.request(key, frame.root_extent);
        }
    }
    world.pump();
}

/// The browser application: the simulation, the camera, the matter
/// pipeline, and the canvas.
pub struct WebApp {
    sim: Simulation,
    camera: Camera,
    world: World,
    fetcher: WebFetcher,
    controls: Controls,
    canvas: Option<web_sys::HtmlCanvasElement>,
    window: Option<Arc<Window>>,
    gfx: Rc<RefCell<Option<Gfx>>>,
    started: f64,
    last_frame: Option<f64>,
    fps_window: (f64, u32, Option<f64>),
}

impl WebApp {
    /// An application drawing `sim` on `canvas`, fetching through
    /// `fetcher`.
    pub fn new(sim: Simulation, fetcher: WebFetcher, canvas: web_sys::HtmlCanvasElement) -> WebApp {
        let camera = Camera::home(&sim.system);
        WebApp {
            sim,
            camera,
            world: World::new(default_threads()),
            fetcher,
            controls: Controls::default(),
            canvas: Some(canvas),
            window: None,
            gfx: Rc::default(),
            started: now_seconds(),
            last_frame: None,
            fps_window: (0.0, 0, None),
        }
    }

    fn fps(&mut self, now: f64) -> Option<f64> {
        let (start, frames, value) = &mut self.fps_window;
        if *frames == 0 && value.is_none() && *start == 0.0 {
            *start = now;
        }
        *frames += 1;
        if now - *start >= 0.5 {
            *value = Some(f64::from(*frames) / (now - *start));
            *start = now;
            *frames = 0;
        }
        *value
    }

    fn frame(&mut self) {
        let now = now_seconds();
        let dt = self
            .last_frame
            .map_or(0.0, |t| (now - t).clamp(0.0, MAX_FRAME_SECONDS));
        self.last_frame = Some(now);
        let fps = self.fps(now);

        self.sim.update(Seconds::new(dt));
        let input = self.controls.take_flight();
        self.camera.fly(&self.sim.system, &input, dt);
        self.camera.update_parent(&self.sim.system);

        let mut slot = self.gfx.borrow_mut();
        let Some(gfx) = slot.as_mut() else {
            return;
        };
        let size = gfx.renderer.size();
        let t = now - self.started;
        self.world
            .select(&self.sim.system, &self.camera, size, t, false);
        stream_step(&mut self.world, &mut self.fetcher, &self.sim, t);
        self.world.evict(t);
        let scene = matter_scene(&mut self.world, &self.sim.system, &self.camera, size, t);
        let overlay = self.overlay(&scene, fps);
        let exposure = self.controls.exposure(Some(dt));
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
    }

    fn overlay(&self, scene: &Scene, fps: Option<f64>) -> String {
        let counts = self.world.counts();
        OverlayInfo {
            sim_time: self.sim.system.time(),
            time_scale: self.sim.clock.scale,
            paused: self.sim.clock.paused,
            camera_frame: self.camera.frame_id,
            camera_distance: self.camera.position.length(),
            frame_count: self.sim.system.tree().frames().len(),
            fps,
            sim_lag: self.sim.lagging(),
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
                ..MatterStats::from_tally(self.fetcher.tally())
            },
        }
        .text()
    }
}

impl ApplicationHandler for WebApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
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
                web_sys::console::error_1(
                    &format!("gx-renderer: creating the canvas window: {e}").into(),
                );
                return;
            }
        };
        let slot = self.gfx.clone();
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
                }
            }
        });
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gfx) = self.gfx.borrow_mut().as_mut() {
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
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}
