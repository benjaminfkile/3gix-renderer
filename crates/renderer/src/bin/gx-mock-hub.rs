//! `gx-mock-hub`: a whole space without a hub or a compiler, for local runs
//! and screenshots.
//!
//! Serves the v2 client routes ([`renderer::mock_hub`]) for a build of two
//! layers: one declaring the frames of a registry file (by default the
//! frame tree from `tests/data/tree.bin`) through a registry-only module,
//! and one running the core library's example module
//! (`testdata/example-module/module.wasm`) on every frame up to a depth, so
//! the first client computes and uploads every section it sees and the
//! next one streams them.
//!
//! ```text
//! gx-mock-hub [--port <u16>] [--registry <file.bin>] [--depth <0-31>]
//!             [--density <kg/m^3>] [--temperature <K>]
//! ```
//!
//! It prints the `GX_*` variables to point `gx-renderer` at it.
//!
//! Native only: the browser build starts from `web/index.html` instead, so
//! on `wasm32` this binary is an empty stub that lets the whole package
//! build for that target.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use anyhow::{bail, Context, Result};
    use renderer::mock_hub::{
        example_matter_layer, registry_layer, MockHub, MockHubOptions, MOCK_API_KEY, MOCK_SPACE_ID,
    };
    use std::net::{Ipv4Addr, SocketAddr};

    /// The registry served when no file is given.
    const DEFAULT_REGISTRY: &[u8] = include_bytes!("../../tests/data/tree.bin");

    /// The core library's example module.
    const EXAMPLE_MODULE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/example-module/module.wasm"
    ));

    const USAGE: &str = "usage: gx-mock-hub [--port <u16>] [--registry <file.bin>] [--depth <0-31>] [--density <kg/m^3>] [--temperature <K>]";

    #[tokio::main]
    pub async fn main() -> Result<()> {
        let mut port = 0u16;
        let mut registry = DEFAULT_REGISTRY.to_vec();
        let mut depth = 2u8;
        let mut density = 1.0e-3;
        let mut temperature = 5000.0;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value =
                |name: &str| args.next().with_context(|| format!("{name} needs a value"));
            match arg.as_str() {
                "--port" => port = value("--port")?.parse()?,
                "--registry" => {
                    let path = value("--registry")?;
                    registry = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
                }
                "--depth" => depth = value("--depth")?.parse()?,
                "--density" => density = value("--density")?.parse()?,
                "--temperature" => temperature = value("--temperature")?.parse()?,
                other => bail!("unknown argument {other:?}; {USAGE}"),
            }
        }
        if depth > 31 {
            bail!("--depth must be 0 to 31");
        }
        gx_core::registry::validate(&registry)
            .map_err(|e| anyhow::anyhow!("registry is invalid: code {}: {}", e.code, e.reason))?;
        let frames: Vec<(u64, f64)> = gx_core::registry::decode(&registry)
            .map_err(|e| anyhow::anyhow!("registry is invalid: {e}"))?
            .frames()
            .iter()
            .map(|f| (f.frame_id, f.root_extent.value()))
            .collect();
        let (matter, tiles) = example_matter_layer(
            "matter",
            EXAMPLE_MODULE,
            &frames,
            depth,
            density,
            temperature,
        );
        let options = MockHubOptions::new(vec![registry_layer("frames", &registry), matter], tiles);
        let hub = MockHub::bind(options, SocketAddr::from((Ipv4Addr::LOCALHOST, port))).await?;
        println!("GX_HUB_URL={}", hub.url());
        println!("GX_API_KEY={MOCK_API_KEY}");
        println!("GX_SPACE_ID={MOCK_SPACE_ID}");
        tokio::signal::ctrl_c().await?;
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
