//! `gx-renderer`: the desktop and headless entry point.
//!
//! Loads the configuration ([`renderer::config`]), connects to the hub
//! ([`renderer::hub`]), opens the local cache and starts the worker
//! ([`gx_worker`]), loads the frame registry through directions like any
//! other key ([`renderer::stream::Fetcher::load_registry`]), then opens a
//! window or renders one headless frame ([`renderer::app`]), streaming and
//! computing matter on the same runtime. Exits with a clear error and a
//! non-zero status when the configuration is incomplete or the registry is
//! invalid or never arrives.
//!
//! Native only: the browser build starts from `web/index.html` instead, so
//! on `wasm32` this binary is an empty stub that lets the whole package
//! build for that target.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use anyhow::Context;
    use anyhow::Result;
    use renderer::config::{load_env, parse_flags, Config, USAGE};
    use renderer::hub::HubClient;
    use renderer::stream::{Fetcher, FetcherOptions};
    use std::process::ExitCode;
    use std::sync::Arc;
    use std::time::Duration;

    /// Longest wait for every layer's registry section.
    const REGISTRY_DEADLINE: Duration = Duration::from_secs(300);

    pub fn main() -> ExitCode {
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info,wgpu_core=error,wgpu_hal=error,naga=warn".into()),
            )
            .init();
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("gx-renderer: {e:#}");
                ExitCode::FAILURE
            }
        }
    }

    fn run() -> Result<()> {
        let flags = parse_flags(std::env::args().skip(1))?;
        if flags.help {
            print!("{USAGE}");
            return Ok(());
        }
        let config = Config::resolve(&load_env()?, &flags)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let client = Arc::new(runtime.block_on(HubClient::connect(&config))?);
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
        let system = fetcher.load_registry(REGISTRY_DEADLINE)?;
        if config.headless {
            renderer::app::run_headless(&config, system, fetcher)
        } else {
            renderer::app::run_windowed(&config, system, fetcher)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    native::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
