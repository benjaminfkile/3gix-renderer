//! `create` (`docs/api.md` section 3), native binding: connects to the
//! hub, loads the registry, and runs the loop with a [`Consumer`] in a
//! window or headless. `gx-renderer` (`src/main.rs`) is a caller of it.

use super::{Api, ApiError, ApiResult, Consumer, Target};
use crate::app::{
    api_picture, api_stats, check_origin, matter_step, set_fetch_time, tdb_of_utc_ms, view_settled,
};
use crate::config::Config;
use crate::controls::Controls as Keys;
use crate::extract::default_threads;
use crate::hub::HubClient;
use crate::options::ViewOptions;
use crate::render::scene::scene_with;
use crate::spawn::Spawner;
use crate::stream::{Fetcher, FetcherOptions};
use crate::world::World;
use anyhow::Context;
use gx_core::frames::FrameSystem;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Longest wait for every layer's registry section.
pub const REGISTRY_DEADLINE: Duration = Duration::from_secs(300);

/// Time between headless ticks at most: the loop waits for the fetcher up
/// to this long between ticks.
pub const HEADLESS_TICK: Duration = Duration::from_millis(16);

/// Where `create` runs the loop.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CreateTarget {
    /// A desktop window of the connection's size.
    Window,
    /// No window: the loop ticks, streams, and answers queries for a view
    /// of this many physical pixels, without drawing.
    Headless {
        /// Width, pixels.
        width: u32,
        /// Height, pixels.
        height: u32,
    },
    /// The `--headless` run: one frame offscreen, with its screenshot and
    /// statistics files ([`crate::app::run_headless`]); no consumer runs.
    Snapshot,
}

/// The connection of `create`: the resolved configuration, of which the
/// hub (`hub_url`, `api_key`, `space_id`, `build_id`), the cache, the
/// workers, the adapter, the size, and the frame cap are used. Its view
/// options are replaced by `create`'s `options`.
pub type Connection = Config;

/// The fetcher, the registry, and the runtime they run on.
struct Connected {
    system: FrameSystem,
    fetcher: Fetcher,
    _runtime: tokio::runtime::Runtime,
}

fn connect(config: &Config) -> anyhow::Result<Connected> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let client = Arc::new(runtime.block_on(HubClient::connect(config))?);
    let cache = match &config.cache_dir {
        Some(dir) => {
            let cache = gx_worker::DiskCache::open(dir, config.cache_max_bytes)
                .with_context(|| format!("opening the cache at {}", dir.display()))?;
            tracing::info!(
                dir = %dir.display(),
                entries = cache.len(),
                bytes = cache.total_bytes(),
                "local cache"
            );
            Some(Arc::new(cache))
        }
        None => {
            tracing::info!("no local cache");
            None
        }
    };
    let mut options = FetcherOptions {
        cache,
        ..FetcherOptions::default()
    };
    if let Some(n) = config.workers {
        options.worker_threads = n;
    }
    let mut fetcher = Fetcher::new(runtime.handle().clone(), client, options)?;
    // The start budget applies from the first key, the registry included.
    if let Some(n) = config.options.compute_budget {
        fetcher.set_compute_budget(n);
    }
    let system = fetcher.load_registry(REGISTRY_DEADLINE)?;
    Ok(Connected {
        system,
        fetcher,
        _runtime: runtime,
    })
}

/// `create`: connects to the hub of `connection`, loads the registry, and
/// runs the loop on `target` with `consumer` each tick, under `options`
/// (the view and time options parsed by `crate::options`). The consumer
/// sees [`super::Event::Ready`] on the first tick. Blocks on this thread
/// while the loop runs (a native window loop owns its thread) and returns
/// when it stops: by [`Api::destroy`], the window closing, or an error.
pub fn create(
    target: CreateTarget,
    connection: &Connection,
    options: ViewOptions,
    consumer: Box<dyn Consumer>,
) -> ApiResult<()> {
    let config = Config {
        options,
        ..connection.clone()
    };
    let run = || -> anyhow::Result<()> {
        let c = connect(&config)?;
        match target {
            CreateTarget::Window => {
                crate::app::run_windowed(&config, c.system, c.fetcher, consumer)
            }
            CreateTarget::Snapshot => crate::app::run_headless(&config, c.system, c.fetcher),
            CreateTarget::Headless { width, height } => {
                run_headless_loop(&config, c.system, c.fetcher, (width, height), consumer);
                Ok(())
            }
        }
    };
    run().map_err(|e| ApiError(format!("{e:#}")))
}

/// The headless loop: the window's tick order without a window or
/// drawing, until the consumer calls [`Api::destroy`].
fn run_headless_loop(
    config: &Config,
    system: FrameSystem,
    mut fetcher: Fetcher,
    size: (u32, u32),
    mut consumer: Box<dyn Consumer>,
) {
    check_origin(&system, fetcher.origin_utc_ms());
    let now = tdb_of_utc_ms(fetcher.now_utc_ms());
    let launch = crate::options::launch(system, &config.options, now);
    let (mut sim, mut camera) = (launch.sim, launch.camera);
    let mut api = Api::new(sim.system().tree(), &camera)
        .with_build(fetcher.client().build())
        .with_compute_budget(fetcher.max_compute_budget(), config.options.compute_budget);
    let mut spawner = Spawner {
        active: None,
        pending: launch.pending_spawn,
    };
    let mut keys = Keys::new(config.options.exposure_stops);
    keys.mesh_mode = config.options.mesh_mode;
    let mut world = World::new(default_threads());
    world.start_layers(
        fetcher.client().build(),
        config.options.hide_layers.as_deref(),
    );
    fetcher.set_hidden_layers(world.hidden_layers().clone());
    let started = Instant::now();
    let mut last: Option<Instant> = None;
    loop {
        let tick = Instant::now();
        let dt = last.map_or(0.0, |t| (tick - t).as_secs_f64());
        last = Some(tick);
        let hub_now = tdb_of_utc_ms(fetcher.now_utc_ms());
        consumer.tick(&mut api);
        if api.destroyed() {
            break;
        }
        api.begin_tick(Target {
            sim: &mut sim,
            camera: &mut camera,
            spawner: &mut spawner,
            keys: &mut keys,
        });
        if api.paused() {
            api.end_tick(&sim, &camera, &world);
            std::thread::sleep(HEADLESS_TICK);
            continue;
        }
        let input = api.flight(keys.take_flight());
        let t = started.elapsed().as_secs_f64();
        spawner.frame(sim.system(), &world, &mut camera, &mut keys, &input, t);
        crate::camera::advance(&mut sim, &mut camera, dt, hub_now, &input, keys.attached);
        api.end_tick(&sim, &camera, &world);
        sim.prepare_light(&camera);
        let near = world.matter_distances(sim.system(), &camera);
        let seen = sim.seen_near(&camera, &near);
        set_fetch_time(&mut fetcher, &sim);
        if world.toggle_layers(&keys.take_layer_toggles()) {
            fetcher.set_hidden_layers(world.hidden_layers().clone());
        }
        let draw = matter_step(&mut world, &mut fetcher, &seen, size, t);
        let scene = scene_with(&draw, &seen, size);
        api.set_picture(api_picture(&world, &draw, &seen, size));
        api.observe(
            view_settled(&world, &spawner),
            api_stats(&world, &fetcher, &scene, 0.0),
            t,
        );
        fetcher.wait_for_events(HEADLESS_TICK);
    }
}
