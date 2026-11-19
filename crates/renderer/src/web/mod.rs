//! The browser build: the renderer on a canvas with WebGPU.
//!
//! Compiled only for `wasm32-unknown-unknown` with the `web` feature, and
//! packaged by `web/build.sh` with `wasm-pack`. `web/index.html` loads the
//! package and calls [`start`] with the settings from its panel. The
//! renderer is the same library as on the desktop (`space-model.md`
//! section 10: "Native desktop and WebAssembly from one codebase"); only the
//! platform glue lives here:
//!
//! - [`hub`]: the hub client over `fetch`.
//! - [`cache`]: the local cache in the Cache API.
//! - [`worker`]: modules run in a dedicated Web Worker.
//! - [`fetch`]: the client loop (directions, streams, compute, and the
//!   upload lane) on the page's event loop.
//! - [`app`]: the canvas frame loop.
//!
//! There is no browser in CI. The build is checked there; running it is
//! verified by hand on a desktop browser (`docs/web.md`).

pub mod app;
pub mod cache;
pub mod fetch;
pub mod hub;
pub mod worker;

use crate::config::ApiKey;
use crate::options::ViewOptions;
use crate::protocol::Routes;
use fetch::WebFetcher;
use hub::WebHub;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::web::EventLoopExtWebSys;

/// Installs the panic hook, so a panic shows in the browser console with
/// its message.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

fn js_error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Checks the view and time options without starting: `Ok` when every
/// pair parses, the parser's message otherwise. The page calls it to show
/// a bad value in the panel before asking for anything.
#[wasm_bindgen]
pub fn check_options(options: Vec<String>) -> Result<(), JsValue> {
    parse_options(&options).map(|_| ()).map_err(js_error)
}

/// Parses flattened camelCase name and value pairs into the options.
fn parse_options(pairs: &[String]) -> Result<ViewOptions, crate::config::ConfigError> {
    if !pairs.len().is_multiple_of(2) {
        return Err(crate::config::ConfigError(
            "the options are not name and value pairs".into(),
        ));
    }
    ViewOptions::from_camel_pairs(
        pairs
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[k, v]| (k.as_str(), v.as_str())),
    )
}

/// Connects to the hub, loads the frame registry of the build, and starts
/// drawing on the canvas with id `canvas_id`.
///
/// `build_id` may be empty for the active build of the space. `options`
/// holds the view and time options as camelCase name and value pairs,
/// flattened (`["startOn", "3:0,0,1", "timeScale", "60"]`; later pairs
/// override earlier ones), parsed and applied by [`crate::options`] the same
/// way as the native flags. Without any the client enters at the hub's now
/// (`space-model.md` section 7) at the `Home` view. The returned promise
/// resolves once the frame loop runs, and rejects with a message if the
/// settings are incomplete, an option is invalid (the parser's message,
/// before anything is fetched), the hub refuses, or the registry is invalid or
/// never becomes ready.
#[wasm_bindgen]
pub async fn start(
    canvas_id: String,
    hub_url: String,
    api_key: String,
    space_id: String,
    build_id: String,
    options: Vec<String>,
) -> Result<(), JsValue> {
    if hub_url.trim().is_empty() || api_key.is_empty() || space_id.trim().is_empty() {
        return Err(js_error("hub URL, API key, and space id are required"));
    }
    let options = parse_options(&options).map_err(js_error)?;
    let canvas = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(&canvas_id))
        .ok_or_else(|| js_error(format!("no element with id {canvas_id}")))?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| js_error(format!("{canvas_id} is not a canvas")))?;
    let build = build_id.trim();
    let hub = WebHub::open(
        Routes::new(&hub_url, None, &space_id),
        ApiKey::new(api_key),
        (!build.is_empty()).then_some(build),
    )
    .await
    .map_err(|e| js_error(format!("{e:#}")))?;
    let cache = cache::WebCache::open(cache::DEFAULT_MAX_BYTES).await;
    if cache.is_none() {
        web_sys::console::warn_1(&"gx-renderer: no Cache API here; nothing is cached".into());
    }
    let mut fetcher = WebFetcher::new(Rc::new(hub), cache);
    let system = fetcher
        .load_registry()
        .await
        .map_err(|e| js_error(format!("{e:#}")))?;
    let now = gx_core::timescale::utc_ms_to_tdb_seconds(fetcher.now_utc_ms());
    let launch = crate::options::launch(system, &options, gx_core::units::Seconds::new(now));
    let event_loop = EventLoop::new().map_err(js_error)?;
    // Wait, not Poll: on the web Poll is a busy loop that starves the page.
    // Each frame requests the next redraw, which the browser delivers once
    // per display refresh.
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.spawn_app(app::WebApp::new(launch, &options, fetcher, canvas));
    Ok(())
}
