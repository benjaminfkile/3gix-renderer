//! `gx-renderer`: the desktop and headless entry point.
//!
//! Loads the configuration ([`renderer::config`]), connects to the hub and
//! loads the frame registry ([`renderer::hub`]), then opens a window or
//! renders one headless frame ([`renderer::app`]), streaming matter cells
//! from the hub on the same runtime. Exits with a clear error
//! and a non-zero status when the configuration is incomplete or the
//! registry is invalid or never becomes ready.

use anyhow::Result;
use renderer::config::{load_env, parse_flags, Config, USAGE};
use renderer::hub::{load_registry, HubClient, ReadyPolicy};
use std::process::ExitCode;
use std::sync::Arc;

fn main() -> ExitCode {
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
    let (client, system) = runtime.block_on(async {
        let client = HubClient::connect(&config).await?;
        let system = load_registry(&client, ReadyPolicy::default()).await?;
        anyhow::Ok((Arc::new(client), system))
    })?;
    let handle = runtime.handle().clone();
    if config.headless {
        renderer::app::run_headless(&config, system, client, handle)
    } else {
        renderer::app::run_windowed(&config, system, client, handle)
    }
}
