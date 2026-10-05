//! `gx-mock-hub`: serves a registry the way the hub does, for local runs and
//! screenshots without a real hub.
//!
//! Serves the registry container built from a registry file (by default the
//! frame tree from `tests/data/tree.bin`) with one `202` before the `200`,
//! the builds listing, and the readiness WebSocket. See
//! [`renderer::mock_hub`] for the exact behavior.
//!
//! ```text
//! gx-mock-hub [--port <u16>] [--registry <file.bin>]
//! ```
//!
//! It prints the `GX_*` variables to point `gx-renderer` at it.

use anyhow::{bail, Context, Result};
use renderer::mock_hub::{MockHub, MockHubOptions, MOCK_API_KEY, MOCK_SPACE_ID};
use std::net::{Ipv4Addr, SocketAddr};

/// The registry served when no file is given.
const DEFAULT_REGISTRY: &[u8] = include_bytes!("../../tests/data/tree.bin");

#[tokio::main]
async fn main() -> Result<()> {
    let mut port = 0u16;
    let mut registry = DEFAULT_REGISTRY.to_vec();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => port = args.next().context("--port needs a value")?.parse()?,
            "--registry" => {
                let path = args.next().context("--registry needs a value")?;
                registry = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
            }
            other => bail!("unknown argument {other:?}; usage: gx-mock-hub [--port <u16>] [--registry <file.bin>]"),
        }
    }
    gx_core::registry::validate(&registry)
        .map_err(|e| anyhow::anyhow!("registry is invalid: code {}: {}", e.code, e.reason))?;
    let chunk = gx_core::container::encode_chunk(&[&registry], &["layer-0"]);
    let hub = MockHub::bind(
        MockHubOptions::new(chunk),
        SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
    )
    .await?;
    println!("GX_HUB_URL={}", hub.url());
    println!("GX_API_KEY={MOCK_API_KEY}");
    println!("GX_SPACE_ID={MOCK_SPACE_ID}");
    tokio::signal::ctrl_c().await?;
    Ok(())
}
