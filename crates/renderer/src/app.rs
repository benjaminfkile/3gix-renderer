//! The desktop window loop and the headless run.
//!
//! Each rendered frame follows the same order in both modes:
//!
//! 1. advance the [`SimClock`] by the wall-clock delta, clamped to the
//!    hub's now, and integrate every frame to the simulation time
//!    ([`crate::sim`], `space-model.md` sections 6 and 7),
//! 2. keep the free camera at its place in space through that step, fly
//!    it, and re-parent it to its nearest frame ([`crate::camera::advance`],
//!    `space-model.md` section 5),
//! 3. solve the light time of every frame for the camera's position
//!    ([`Seen`], `space-model.md` section 6.4); everything after this
//!    reads the retarded states,
//! 4. one step of matter ([`matter_step`], `rendering.md` section 4):
//!    apply what the fetcher brought in (streamed from the edge or the
//!    local cache, or computed here by the worker, decoded off the frame
//!    thread), select cells when something changed, start requests, gather
//!    extraction inputs within the step's time slice, evict, and build the
//!    draw list ([`crate::world`], [`crate::stream`], [`crate::directions`]);
//!    the step's duration is recorded for the statistics and the overlay,
//! 5. build the camera-relative scene with the drawn cells and their lights
//!    and draw it with the overlay ([`crate::render`]).
//!
//! The window, the headless run, and the budget test run the same
//! [`matter_step`].
//!
//! Key and mouse bindings are listed in `docs/controls.md`.

use crate::camera::Camera;
use crate::clock::now_seconds;
use crate::config::{Config, Start, View};
use crate::controls::Controls;
pub use crate::controls::{EXPOSURE_BIAS_LIMITS, PIXELS_PER_NOTCH};
use crate::extract::default_threads;
use crate::farfield::focal_px;
use crate::render::gpu::{
    describe_adapter, pick_adapter, present_mode, wanted_features, Exposure, Renderer,
};
use crate::render::headless::{Headless, Image};
use crate::render::overlay::{MatterStats, OverlayInfo};
pub use crate::render::scene::matter_scene;
use crate::render::scene::scene_with;
use crate::render::scene::Scene;
use crate::seen::Seen;
use crate::sim::{SimClock, Simulation};
use crate::stream::{FetchEvent, Fetcher};
use crate::world::{DrawList, World};
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

/// Longest wall-clock delta one frame applies, seconds, so a stalled window
/// does not turn into one giant step of flight or time.
pub const MAX_FRAME_SECONDS: f64 = 0.25;

/// Longest a headless run waits for the selected cells, seconds, before it
/// renders whatever has arrived.
pub const HEADLESS_STREAM_SECONDS: f64 = 60.0;

/// Longest a headless run waits for the fetcher between two steps.
pub const HEADLESS_STEP_WAIT: Duration = Duration::from_millis(4);

/// One step of the frame thread's matter work (`rendering.md` section 4):
/// takes what the fetcher has brought in, runs [`World::step`] for the view
/// `seen` of `view_px` pixels at session time `now` (seconds), hands the
/// requests to the fetcher, and records how long all of it took on this
/// thread ([`World::record_step`]). Returns the draw list.
pub fn matter_step(
    world: &mut World,
    fetcher: &mut Fetcher,
    seen: &Seen,
    view_px: (u32, u32),
    now: f64,
) -> DrawList {
    let started = now_seconds();
    let mut cells = Vec::new();
    for e in fetcher.drain() {
        match e {
            FetchEvent::Completed { key, outcome, .. } => cells.push((key, outcome)),
            FetchEvent::Surface { key, outcome, .. } => world.complete_surface(key, outcome),
            // Loaded once at startup, before the integrator.
            FetchEvent::Registry(_) => {}
        }
    }
    let (requests, draw) = world.step(seen, view_px, now, cells, started);
    let system = seen.system();
    for key in requests {
        if let Some(frame) = system.tree().get(key.frame_id) {
            fetcher.request(key, frame.root_extent);
        }
    }
    for key in world.take_surface_requests() {
        fetcher.request_surface(key);
    }
    fetcher.flush();
    world.record_step((now_seconds() - started) * 1e3);
    draw
}

/// The matter values for the overlay.
pub fn matter_stats(world: &World, fetcher: Option<&Fetcher>, scene: &Scene) -> MatterStats {
    let counts = world.counts();
    MatterStats {
        cells_selected: counts.cache.selected,
        cells_ready: counts.cache.ready,
        cells_pending: counts.cache.pending,
        meshes_drawn: scene.surfaces.len() + scene.tiles.len(),
        triangles_drawn: scene
            .surfaces
            .iter()
            .map(|d| d.mesh.triangle_count())
            .sum::<usize>()
            + scene.tiles.len() * gx_core::surface::TILE_QUADS * gx_core::surface::TILE_QUADS * 2,
        volumes_drawn: scene.volumes.len(),
        sprites_drawn: scene.sprites.len(),
        lights_active: scene.lights.len(),
        frame_step_p95_ms: world
            .step_times()
            .recent_percentile(crate::clock::OVERLAY_STEPS, 95.0),
        ..MatterStats::from_stats(fetcher.map(Fetcher::stats).unwrap_or_default())
    }
}

/// Tells the fetcher which moment is drawn, so directions are asked for it
/// (`client-compute.md` section 2.4): the simulation time converted from
/// TDB to UTC milliseconds.
pub fn set_fetch_time(fetcher: &mut Fetcher, sim: &Simulation) {
    let t = gx_core::timescale::tdb_seconds_to_utc_ms(sim.time().value());
    fetcher.set_time_utc_ms(t);
}

/// A UTC time in milliseconds as seconds of TDB since J2000.
pub fn tdb_of_utc_ms(utc_ms: i64) -> Seconds {
    Seconds::new(gx_core::timescale::utc_ms_to_tdb_seconds(utc_ms))
}

/// Seconds the registry epoch may differ from the hub's origin before
/// [`check_origin`] warns.
pub const ORIGIN_TOLERANCE_SECONDS: f64 = 2.0;

/// Logs a warning when the registry epoch, which the renderer takes as the
/// origin, is not the hub's origin converted to TDB (`matter-format.md`
/// section 5.2 requires them to be equal). Returns whether they agree, or
/// `true` while the hub has not reported an origin.
pub fn check_origin(system: &FrameSystem, origin_utc_ms: Option<i64>) -> bool {
    let Some(origin) = origin_utc_ms else {
        return true;
    };
    let epoch = system.tree().epoch().value();
    let hub = tdb_of_utc_ms(origin).value();
    let agree = (epoch - hub).abs() <= ORIGIN_TOLERANCE_SECONDS;
    if !agree {
        tracing::warn!(
            registry_epoch_tdb_s = epoch,
            hub_origin_tdb_s = hub,
            "the registry epoch is not the hub's origin; time runs from the registry epoch"
        );
    }
    agree
}

/// The configured start (`--start`, `--start-offset-seconds`,
/// `--time-scale`) of a simulation of `system` (at its epoch, the origin)
/// whose now is `now`, seconds of TDB since J2000: at now by default.
pub fn launch_simulation(system: FrameSystem, config: &Config, now: Seconds) -> Simulation {
    let origin = system.tree().epoch();
    let from = match config.start {
        Start::Now => now,
        Start::Origin => origin,
    };
    let clock = SimClock::starting_at(
        origin,
        now,
        from + Seconds::new(config.start_offset_seconds),
        config.time_scale,
    );
    Simulation::new(system, clock)
}

/// The overlay values for the current state, with the fixed exposure in
/// stops or `None` for the automatic exposure.
pub fn overlay_info(
    sim: &Simulation,
    seen: &Seen,
    camera: &Camera,
    fps: Option<f64>,
    exposure_fixed: Option<f64>,
    matter: MatterStats,
) -> OverlayInfo {
    let clock = sim.clock();
    OverlayInfo {
        sim_time: sim.time(),
        offset_from_now: clock.offset_from_now(),
        at_now: clock.is_at_now(),
        rate: clock.rate(),
        rewinding: clock.rewinding(),
        paused: clock.paused(),
        crosshair: seen.nearest_to_crosshair().map(|(f, t)| (f, t.value())),
        camera_frame: camera.frame_id,
        camera_distance: camera.position.length(),
        camera_speed: camera.cruise,
        frame_count: sim.system().tree().frames().len(),
        fps,
        exposure_fixed,
        matter,
    }
}

/// What a headless run renders and how long it waits for matter.
#[derive(Clone, Debug, PartialEq)]
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
    /// A fixed exposure, stops ([`crate::render::gpu::fixed_exposure`]);
    /// `None` exposes for the frame's own log-average luminance.
    pub exposure_stops: Option<f64>,
    /// How surface meshes are drawn.
    pub mesh_mode: crate::render::scene::MeshMode,
    /// A part of the graphics adapter's name ([`crate::render::gpu::pick_adapter`]).
    pub adapter: Option<String>,
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
            exposure_stops: None,
            mesh_mode: crate::render::scene::MeshMode::Surfaces,
            adapter: None,
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
            exposure_stops: config.exposure_stops,
            mesh_mode: config.mesh_mode,
            adapter: config.adapter.clone(),
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
    /// Wall-clock seconds until every selected cell was drawn at its
    /// selected depth ([`World::ready`]), or `None` if that never happened
    /// within the wait.
    pub settle_seconds: Option<f64>,
    /// Open edges of the drawn meshes of the camera's frame
    /// ([`World::open_edges`]): 0 when what is drawn is watertight
    /// (`rendering.md` 2.5).
    pub open_edges: usize,
    /// Frame-thread matter work per step over the run, milliseconds: the
    /// 50th and 95th percentiles and the maximum ([`matter_step`]).
    pub frame_step_ms: (Option<f64>, Option<f64>, Option<f64>),
    /// Steps measured.
    pub frame_steps: u64,
    /// Extraction time per cell over the run, milliseconds: the 50th and
    /// 95th percentiles ([`World::extraction_times`]).
    pub extraction_ms: (Option<f64>, Option<f64>),
    /// Extractions measured.
    pub extractions: u64,
    /// Selections made during the run ([`World::selection_rounds`]).
    pub selection_rounds: u64,
    /// Cells requested before their parent had arrived
    /// ([`crate::stream::CellCache::requested_ahead`]).
    pub cells_requested_ahead: u64,
    /// The far field sprites drawn, in ascending frame id order.
    pub sprites: Vec<SpriteStats>,
}

/// What one far field sprite delivered, for `--stats-json`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpriteStats {
    /// The frame it stands for.
    pub frame_id: u64,
    /// Where its center projects, pixels from the top left corner, or
    /// `None` behind the camera.
    pub position_px: Option<(f32, f32)>,
    /// Diameter, pixels.
    pub size_px: f32,
    /// Luminance-weighted irradiance at the camera, W m^-2: the Rec. 709
    /// luma of the band irradiances, recovered from the sprite's radiance
    /// as `L pi (s / 2)^2 / f^2` (`docs/shading.md`, far field).
    pub irradiance: f64,
}

/// The [`SpriteStats`] of every sprite of `scene` in a view of
/// `width` x `height` pixels.
pub fn sprite_stats(scene: &Scene, width: u32, height: u32) -> Vec<SpriteStats> {
    let f = focal_px(f64::from(height.max(1)));
    scene
        .sprites
        .iter()
        .map(|sp| {
            let luma = 0.2126 * f64::from(sp.radiance[0])
                + 0.7152 * f64::from(sp.radiance[1])
                + 0.0722 * f64::from(sp.radiance[2]);
            let r = 0.5 * f64::from(sp.size_px);
            SpriteStats {
                frame_id: sp.frame_id,
                position_px: scene.project(sp.position, width, height),
                size_px: sp.size_px,
                irradiance: luma * std::f64::consts::PI * r * r / (f * f),
            }
        })
        .collect()
}

impl HeadlessStats {
    /// The statistics as a JSON object, keys in a fixed order.
    pub fn to_json(&self) -> serde_json::Value {
        let o = &self.overlay;
        let m = &o.matter;
        let mut out = serde_json::json!({
            "view": self.view.to_string(),
            "sim_time_seconds_since_j2000": o.sim_time.value(),
            "camera_frame": o.camera_frame,
            "camera_distance_m": o.camera_distance,
            "frames": o.frame_count,
            "cells_selected": m.cells_selected,
            "cells_pinned": self.cells_pinned,
            "cells_ready": m.cells_ready,
            "cells_pending": m.cells_pending,
            "requests": m.requests,
            "cells_fetched": m.cells_fetched,
            "bytes_fetched": m.bytes_fetched,
            "sections_streamed": m.sections_streamed,
            "sections_cached": m.sections_cached,
            "sections_computed": m.sections_computed,
            "module_runs": m.module_runs,
            "modules_loaded": m.modules_loaded,
            "uploads_pending": m.uploads_pending,
            "uploads_done": m.uploads_done,
            "uploads_failed": m.uploads_failed,
            "directions_requests": m.directions_requests,
            "first_round_trip_seconds": m.first_round_trip,
            "last_round_trip_seconds": m.last_round_trip,
            "meshes_drawn": m.meshes_drawn,
            "triangles_drawn": m.triangles_drawn,
            "volumes_drawn": m.volumes_drawn,
            "sprites_drawn": m.sprites_drawn,
            "lights_active": m.lights_active,
            "exposure_fixed_stops": o.exposure_fixed,
            "ready": self.ready,
            "stream_seconds": self.stream_seconds,
            "sprites": self.sprites.iter().map(|s| serde_json::json!({
                "frame_id": s.frame_id,
                "position_px": s.position_px.map(|(x, y)| [x, y]),
                "size_px": s.size_px,
                "irradiance_w_m2": s.irradiance,
            })).collect::<Vec<_>>(),
        });
        let time = serde_json::json!({
            "sim_time_utc_ms": gx_core::timescale::tdb_seconds_to_utc_ms(o.sim_time.value()),
            "offset_from_now_seconds": o.offset_from_now.value(),
            "at_now": o.at_now,
            "crosshair_frame": o.crosshair.map(|(f, _)| f),
            "crosshair_light_time_seconds": o.crosshair.map(|(_, t)| t),
        });
        let budget = serde_json::json!({
            "settle_seconds": self.settle_seconds,
            "open_edges": self.open_edges,
            "frame_step_ms_p50": self.frame_step_ms.0,
            "frame_step_ms_p95": self.frame_step_ms.1,
            "frame_step_ms_max": self.frame_step_ms.2,
            "frame_steps": self.frame_steps,
            "extraction_ms_p50": self.extraction_ms.0,
            "extraction_ms_p95": self.extraction_ms.1,
            "extractions": self.extractions,
            "selection_rounds": self.selection_rounds,
            "cells_requested_ahead": self.cells_requested_ahead,
        });
        for more in [time, budget] {
            if let (Some(all), serde_json::Value::Object(more)) = (out.as_object_mut(), more) {
                all.extend(more);
            }
        }
        out
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
        View::Over(s) => {
            // A spawn (no distance) looks at the frame until it lands; its
            // pitch applies once it has spawned.
            Camera::over_pitched(
                system,
                s.frame_id,
                s.direction,
                s.distance,
                s.distance.and(s.pitch_deg),
            )
            .ok_or_else(|| anyhow!("frame {} is not in the registry", s.frame_id))?
        }
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

/// Renders one headless frame of `sim` at its simulation time: places the
/// camera for the run's view, solves the light time of every frame for it
/// ([`Seen`]), streams the selected cells through `fetcher` until they are
/// ready (or the wait ends), and reads the image back.
pub fn render_headless_view(
    sim: &mut Simulation,
    mut fetcher: Option<&mut Fetcher>,
    run: &HeadlessRun,
) -> Result<HeadlessFrame> {
    let (width, height) = (run.width, run.height);
    let mut camera = view_camera(sim.system(), run.view, run.view_distance_scale)?;
    camera.update_parent(sim.system());
    sim.prepare_light(&camera);
    // A `--start-on` without a distance spawns the camera onto the frame's
    // matter, as in the window ([`crate::spawn`]).
    let mut spawner = crate::spawn::Spawner {
        active: None,
        pending: match run.view {
            View::Over(s) if s.distance.is_none() => Some(s),
            _ => None,
        },
    };
    let mut controls = Controls::new(None);
    let mut world = World::new(default_threads());
    let start = Instant::now();
    let limit = run.wait_ready_seconds.unwrap_or(HEADLESS_STREAM_SECONDS);
    let done = |world: &World| match run.wait_ready_seconds {
        Some(_) => world.ready(),
        None => world.settled(),
    };
    let mut settle_seconds = None;
    match fetcher.as_deref_mut() {
        Some(f) => {
            set_fetch_time(f, sim);
            // The same step as the window's, as fast as results come in:
            // waiting for the fetcher between steps, never longer than
            // `HEADLESS_STEP_WAIT`, so extraction is picked up as it ends.
            loop {
                let now = start.elapsed().as_secs_f64();
                {
                    let near = world.matter_distances(sim.system(), &camera);
                    let seen = sim.seen_near(&camera, &near);
                    matter_step(&mut world, f, &seen, (width, height), now);
                }
                let placed = (camera.frame_id, camera.position);
                spawner.frame(
                    sim.system(),
                    &world,
                    &mut camera,
                    &mut controls,
                    &crate::camera::FlightInput::default(),
                    now,
                );
                let moved = placed != (camera.frame_id, camera.position);
                if moved {
                    sim.prepare_light(&camera);
                }
                if !moved && spawner.pending.is_none() && done(&world) {
                    if world.ready() {
                        settle_seconds = Some(start.elapsed().as_secs_f64());
                    }
                    break;
                }
                if now > limit {
                    tracing::warn!(seconds = limit, "not every cell arrived in time");
                    break;
                }
                f.wait_for_events(HEADLESS_STEP_WAIT);
            }
        }
        None => {
            let seen = sim.seen(&camera);
            world.select(seen.system(), seen.camera(), (width, height), 0.0, true);
        }
    }
    let near = world.matter_distances(sim.system(), &camera);
    let sim = &*sim;
    let seen = sim.seen_near(&camera, &near);
    world.finish_extraction();
    let stream_seconds = start.elapsed().as_secs_f64();
    let mut scene = matter_scene(&mut world, &seen, (width, height), stream_seconds);
    scene.mesh_mode = run.mesh_mode;
    if !run.overlay {
        // The markers and lines are visual aids like the text: only matter
        // remains.
        scene.markers.clear();
        scene.lines.clear();
    }
    let mut headless = Headless::new(width, height, run.adapter.as_deref())?;
    tracing::info!(adapter = headless.adapter_name(), "headless rendering");
    let info = overlay_info(
        sim,
        &seen,
        &camera,
        None,
        run.exposure_stops,
        matter_stats(&world, fetcher.as_deref(), &scene),
    );
    let text = info.text();
    let exposure = Exposure {
        fixed_stops: run.exposure_stops,
        ..Exposure::default()
    };
    let image = headless.render_with(&scene, run.overlay.then_some(text.as_str()), exposure)?;
    Ok(HeadlessFrame {
        image,
        stats: HeadlessStats {
            view: run.view,
            overlay: info,
            cells_pinned: world.counts().cache.pinned,
            ready: world.ready(),
            stream_seconds,
            settle_seconds,
            open_edges: sim.system().tree().get(camera.frame_id).map_or(0, |f| {
                let extent = f.root_extent.value();
                world.open_edges(f.frame_id, extent, extent * 1e-12)
            }),
            frame_step_ms: (
                world.step_times().percentile(50.0),
                world.step_times().percentile(95.0),
                world.step_times().max(),
            ),
            frame_steps: world.step_times().count(),
            extraction_ms: (
                world.extraction_times().percentile(50.0),
                world.extraction_times().percentile(95.0),
            ),
            extractions: world.extraction_times().count(),
            selection_rounds: world.selection_rounds(),
            cells_requested_ahead: world.cache().requested_ahead(),
            sprites: sprite_stats(&scene, width, height),
        },
    })
}

/// The `--headless` run: one frame offscreen, written to the screenshot path
/// if one was given, with its statistics written to the `--stats-json`
/// path. Cells come through `fetcher`. With `--wait-ready-seconds`, fails
/// after writing both when the selection was not ready in time.
pub fn run_headless(config: &Config, system: FrameSystem, mut fetcher: Fetcher) -> Result<()> {
    check_origin(&system, fetcher.origin_utc_ms());
    let now = tdb_of_utc_ms(fetcher.now_utc_ms());
    let mut sim = launch_simulation(system, config, now);
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

/// The desktop run: a window, the frame loop, and the controls. Cells come
/// through `fetcher`.
pub fn run_windowed(config: &Config, system: FrameSystem, fetcher: Fetcher) -> Result<()> {
    let event_loop = EventLoop::new().context("opening the window system")?;
    // Wait, not Poll: each frame asks for the next redraw, so the loop
    // still runs continuously, but paced by the surface's present mode (one
    // frame per display refresh) and the optional cap instead of spinning.
    event_loop.set_control_flow(ControlFlow::Wait);
    check_origin(&system, fetcher.origin_utc_ms());
    let now = tdb_of_utc_ms(fetcher.now_utc_ms());
    let sim = launch_simulation(system, config, now);
    let camera = config
        .start_on
        .and_then(|s| {
            Camera::over_pitched(
                sim.system(),
                s.frame_id,
                s.direction,
                s.distance,
                s.distance.and(s.pitch_deg),
            )
        })
        .unwrap_or_else(|| Camera::home(sim.system()));
    let mut app = App {
        sim,
        camera,
        spawner: crate::spawn::Spawner {
            active: None,
            pending: config.start_on.filter(|s| s.distance.is_none()),
        },
        world: World::new(default_threads()),
        fetcher,
        started: Instant::now(),
        controls: {
            let mut c = Controls::new(config.exposure_stops);
            c.mesh_mode = config.mesh_mode;
            c
        },
        adapter: config.adapter.clone(),
        frame_interval: (config.max_fps > 0)
            .then(|| Duration::from_secs_f64(1.0 / f64::from(config.max_fps))),
        next_frame: None,
        gfx: None,
        size: (config.width, config.height),
        last_tick: None,
        next_tick: Instant::now(),
        last_draw: None,
        force_draw: true,
        redraw: crate::redraw::Redraw::new(),
        pending: None,
        occluded: false,
        occluded_try: None,
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

/// Frames drawn in the last second. The loop draws only when something
/// visible changes, so this is low when nothing does.
#[derive(Default)]
struct Fps {
    draws: std::collections::VecDeque<Instant>,
}

impl Fps {
    fn frame(&mut self, now: Instant) {
        self.draws.push_back(now);
        self.trim(now);
    }

    fn trim(&mut self, now: Instant) {
        while self
            .draws
            .front()
            .is_some_and(|t| now - *t > Duration::from_secs(1))
        {
            self.draws.pop_front();
        }
    }

    /// Draws in the last second.
    fn value(&mut self, now: Instant) -> Option<f64> {
        self.trim(now);
        Some(self.draws.len() as f64)
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
    fn new(event_loop: &ActiveEventLoop, size: (u32, u32), adapter: Option<&str>) -> Result<Gfx> {
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
        let adapter = pollster::block_on(pick_adapter(&instance, Some(&surface), adapter))
            .context("choosing the graphics adapter for the window")?;
        tracing::info!(adapter = describe_adapter(&adapter), "rendering with");
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
        surface_config.present_mode = present_mode(&caps);
        tracing::info!(present_mode = ?surface_config.present_mode, "window surface");
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
    spawner: crate::spawn::Spawner,
    world: World,
    fetcher: Fetcher,
    started: Instant,
    controls: Controls,
    /// A part of the graphics adapter's name to render with, if chosen.
    adapter: Option<String>,
    /// Minimum time between frames when `--max-fps` is given.
    frame_interval: Option<Duration>,
    /// With a cap: the earliest the next frame may start.
    next_frame: Option<Instant>,
    gfx: Option<Gfx>,
    size: (u32, u32),
    /// When the last tick ran (input, simulation, streaming).
    last_tick: Option<Instant>,
    /// When the next tick is due.
    next_tick: Instant,
    /// When the last frame was drawn.
    last_draw: Option<Instant>,
    /// Draw the next tick's scene whatever it is (a window event came).
    force_draw: bool,
    /// What was drawn last and whether the next scene needs drawing
    /// (`crate::redraw`).
    redraw: crate::redraw::Redraw,
    /// The scene a tick decided to draw, until the window draws it.
    pending: Option<(Scene, String, crate::redraw::Need)>,
    /// The window is hidden (minimized, covered, the display asleep): no
    /// frame can be shown, so ticks go at the idle rate and a draw is
    /// tried at most once a second until it shows again.
    occluded: bool,
    /// When a draw was last asked for while occluded.
    occluded_try: Option<Instant>,
    fps: Fps,
    error: Option<anyhow::Error>,
}

/// Time between ticks while anything moves, streams, or settles: about one
/// display frame.
const ACTIVE_TICK: Duration = Duration::from_millis(16);

/// Time between ticks while nothing changes: time still passes (the clock
/// may move what is seen), cells and tiles may arrive, so the loop looks
/// ten times a second; it draws only if something visible changed.
const IDLE_TICK: Duration = Duration::from_millis(100);

impl App {
    /// One tick: input, the simulation, streaming, and the scene; draws
    /// only when the scene differs visibly from the last one drawn
    /// (`crate::redraw`). Schedules the next tick.
    fn tick(&mut self) {
        let now = Instant::now();
        let dt = self
            .last_tick
            .map_or(0.0, |t| (now - t).as_secs_f64().min(MAX_FRAME_SECONDS));
        self.last_tick = Some(now);

        let hub_now = tdb_of_utc_ms(self.fetcher.now_utc_ms());
        let input = self.controls.take_flight();
        let flying = input.forward
            || input.back
            || input.left
            || input.right
            || input.up
            || input.down
            || input.look != (0.0, 0.0)
            || input.wheel != 0.0;
        self.spawner.frame(
            self.sim.system(),
            &self.world,
            &mut self.camera,
            &mut self.controls,
            &input,
            self.started.elapsed().as_secs_f64(),
        );
        crate::camera::advance(
            &mut self.sim,
            &mut self.camera,
            dt,
            hub_now,
            &input,
            self.controls.attached,
        );

        let Some(gfx) = self.gfx.as_mut() else {
            self.next_tick = now + ACTIVE_TICK;
            return;
        };
        let size = gfx.renderer.size();
        let t = self.started.elapsed().as_secs_f64();
        self.sim.prepare_light(&self.camera);
        let near = self.world.matter_distances(self.sim.system(), &self.camera);
        let seen = self.sim.seen_near(&self.camera, &near);
        set_fetch_time(&mut self.fetcher, &self.sim);
        let draw = matter_step(&mut self.world, &mut self.fetcher, &seen, size, t);
        let mut scene = scene_with(&draw, &seen, size);
        scene.mesh_mode = self.controls.mesh_mode;
        if self.controls.hide_guides {
            scene.markers.clear();
            scene.lines.clear();
        }
        let stats = matter_stats(&self.world, Some(&self.fetcher), &scene);
        let mut overlay = OverlayInfo {
            ..overlay_info(
                &self.sim,
                &seen,
                &self.camera,
                self.fps.value(now),
                self.controls.exposure_fixed,
                stats,
            )
        }
        .text();
        if self.controls.attached {
            overlay.push('\n');
            overlay.push_str(&crate::spawn::Spawner::status_line(&self.camera));
        }
        if self.controls.spawning {
            overlay.push('\n');
            overlay.push_str(&crate::spawn::chooser_line(&crate::spawn::spawn_frames(
                self.world.cache(),
                self.sim.system().tree(),
            )));
        }
        for line in self.controls.status_lines() {
            overlay.push('\n');
            overlay.push_str(&line);
        }
        let exposure = self.controls.exposure(None);
        let focal = crate::farfield::focal_px(f64::from(size.1.max(1)));
        let need = self
            .redraw
            .decide(&scene, &overlay, &exposure, t, focal, self.force_draw);
        if tracing::enabled!(tracing::Level::DEBUG) {
            if let Some((last, _, _)) = self.redraw.last() {
                tracing::debug!(
                    reason = crate::redraw::change_reason(last, &scene, focal),
                    "change"
                );
            }
        }
        self.force_draw = false;
        if need.draw() {
            self.pending = Some((scene, overlay, need));
            let retry = self
                .occluded_try
                .is_none_or(|t| now - t >= Duration::from_secs(1));
            if !self.occluded || retry {
                if self.occluded {
                    self.occluded_try = Some(now);
                }
                gfx.window.request_redraw();
            }
        }
        let busy = flying
            || (self.pending.is_some() && !self.occluded)
            || self.world.busy()
            || (self.redraw.settling(t) && !self.occluded);
        tracing::debug!(
            ?need,
            flying,
            unsettled = self.world.unsettled(),
            settling = self.redraw.settling(t),
            occluded = self.occluded,
            "tick"
        );
        self.next_tick = now + if busy { ACTIVE_TICK } else { IDLE_TICK };
        // With a frame cap, the next draw waits for its slot.
        if let Some(next) = self.next_frame {
            if self.pending.is_some() && next > self.next_tick {
                self.next_tick = next;
            }
        }
    }

    /// Draws the pending scene, or the last one again when the window asks
    /// without a tick having asked.
    fn draw(&mut self) {
        let now = Instant::now();
        if let Some(next) = self.next_frame {
            if now < next && self.pending.is_some() {
                // Capped: the tick at the slot asks again.
                return;
            }
        }
        let (scene, overlay, need) = match self.pending.take() {
            Some(p) => p,
            None => match self.redraw.last() {
                Some((s, o, _)) => (s.clone(), o.clone(), crate::redraw::Need::Change),
                None => return,
            },
        };
        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };
        // The exposure adapts over the wall-clock time since the last draw.
        let adapt = self
            .last_draw
            .map_or(0.0, |t| (now - t).as_secs_f64().min(MAX_FRAME_SECONDS));
        self.last_draw = Some(now);
        self.next_frame = self.frame_interval.map(|i| now + i);
        self.fps.frame(now);
        let exposure = self.controls.exposure(Some(adapt));
        let texture = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gfx.surface.configure(&gfx.device, &gfx.surface_config);
                self.force_draw = true;
                return;
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                // Keep the scene pending; ticks slow down until the window
                // shows again (`WindowEvent::Occluded(false)` or a retry).
                self.occluded = true;
                return;
            }
            _ => return,
        };
        self.occluded = false;
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        gfx.renderer.render(&view, &scene, Some(&overlay), exposure);
        gfx.window.pre_present_notify();
        gfx.queue.present(texture);
        let t = self.started.elapsed().as_secs_f64();
        self.redraw.drawn(scene, overlay, exposure, t, need);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        match Gfx::new(event_loop, self.size, self.adapter.as_deref()) {
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
                self.force_draw = true;
                self.next_tick = Instant::now();
            }
            WindowEvent::RedrawRequested => self.draw(),
            other => {
                if let WindowEvent::Occluded(o) = other {
                    self.occluded = o;
                }
                let input = matches!(
                    other,
                    WindowEvent::KeyboardInput { .. }
                        | WindowEvent::MouseInput { .. }
                        | WindowEvent::MouseWheel { .. }
                        | WindowEvent::CursorMoved { .. }
                        | WindowEvent::Focused(_)
                        | WindowEvent::ScaleFactorChanged { .. }
                        | WindowEvent::Occluded(false)
                );
                if self
                    .controls
                    .handle(&other, &mut self.sim, &mut self.camera)
                {
                    event_loop.exit();
                }
                if input {
                    // Tick now: the input may change what is seen. Keys and
                    // buttons may change settings the scene does not show
                    // (clock, exposure): draw for them.
                    if !matches!(other, WindowEvent::CursorMoved { .. }) {
                        self.force_draw = true;
                    }
                    self.next_tick = Instant::now();
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_none() {
            return;
        }
        // Tick when due; draw only what a tick decided to draw
        // (`crate::redraw`). Between ticks the loop sleeps.
        if Instant::now() >= self.next_tick {
            self.tick();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_tick));
    }
}
