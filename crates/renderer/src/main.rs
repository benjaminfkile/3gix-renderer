//! `gx-renderer`: the desktop and headless entry point.
//!
//! Loads the configuration ([`renderer::config`]) and calls the API's
//! `create` ([`renderer::api::create`], `docs/api.md` section 3), which
//! connects to the hub ([`renderer::hub`]), opens the local cache and
//! starts the worker ([`gx_worker`]), loads the frame registry through
//! directions like any other key
//! ([`renderer::stream::Fetcher::load_registry`]), then opens a window or
//! renders one headless frame ([`renderer::app`]), streaming and
//! computing matter on the same runtime. Exits with a clear error and a
//! non-zero status when the configuration is incomplete or the registry is
//! invalid or never arrives.
//!
//! Native only: the browser build starts from `web/index.html` instead, so
//! on `wasm32` this binary is an empty stub that lets the whole package
//! build for that target.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use anyhow::Result;
    use renderer::api::{create, CreateTarget};
    use renderer::config::{load_env, parse_flags, Config, USAGE};
    use std::process::ExitCode;

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
        let target = if config.headless {
            CreateTarget::Snapshot
        } else {
            CreateTarget::Window
        };
        create(target, &config, config.options.clone(), Box::new(()))?;
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    native::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
