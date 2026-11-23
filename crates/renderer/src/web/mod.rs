//! The browser build: the renderer on a canvas with WebGPU.
//!
//! Compiled only for `wasm32-unknown-unknown` with the `web` feature, and
//! packaged by `web/build.sh` with `wasm-pack`. `web/index.html` loads the
//! package and creates a [`binding::Renderer`] with the settings from its
//! panel. The
//! renderer is the same library as on the desktop (`space-model.md`
//! section 10: "Native desktop and WebAssembly from one codebase"); only the
//! platform glue lives here:
//!
//! - [`hub`]: the hub client over `fetch`.
//! - [`cache`]: the local cache in the Cache API.
//! - [`worker`]: modules run in a dedicated Web Worker.
//! - [`fetch`]: the client loop (directions, streams, compute, and the
//!   upload lane) on the page's event loop.
//! - [`app`]: the canvas frame loop, driving the [`crate::api::Api`].
//! - [`host`]: the page's one event loop, which runs every renderer the
//!   page creates, one after another or side by side.
//! - [`binding`]: the `Renderer` class, the web binding of the API
//!   (`docs/api.md` sections 3 to 8).
//! - [`value`]: values across the binding (`docs/api.md` section 2).
//!
//! There is no browser in CI. The build is checked there; running it is
//! verified by hand on a desktop browser (`docs/web.md`).

pub mod app;
pub mod binding;
pub mod cache;
pub mod fetch;
pub mod host;
pub mod hub;
pub mod value;
pub mod worker;

use crate::config::ApiKey;
use crate::options::ViewOptions;
use crate::protocol::Routes;
use fetch::WebFetcher;
use hub::WebHub;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

/// Installs the panic hook, so a panic shows in the browser console with
/// its message.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Checks the view and time options without starting: `Ok` when every
/// pair parses, the parser's message otherwise. The page calls it to show
/// a bad value in the panel before asking for anything.
#[wasm_bindgen]
pub fn check_options(options: Vec<String>) -> Result<(), JsValue> {
    parse_options(&options).map(|_| ()).map_err(value::js_error)
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

/// Reads a string field of `connection`, empty when absent.
fn field(connection: &JsValue, key: &str) -> Result<String, JsValue> {
    let v = js_sys::Reflect::get(connection, &JsValue::from_str(key))?;
    if v.is_undefined() || v.is_null() {
        return Ok(String::new());
    }
    v.as_string()
        .ok_or_else(|| value::js_error(format!("connection.{key} is not a string")))
}

/// The view and time options of `create`: an object of camelCase names
/// (values converted to strings) or flattened name and value pairs.
fn option_pairs(options: &JsValue) -> Result<Vec<String>, JsValue> {
    if options.is_undefined() || options.is_null() {
        return Ok(Vec::new());
    }
    let text = |v: JsValue| {
        v.as_string()
            .unwrap_or_else(|| String::from(js_sys::JSON::stringify(&v).unwrap_or_default()))
    };
    if js_sys::Array::is_array(options) {
        return Ok(js_sys::Array::from(options).iter().map(text).collect());
    }
    if !options.is_object() {
        return Err(value::js_error("the options are not an object"));
    }
    let mut pairs = Vec::new();
    for entry in js_sys::Object::entries(&js_sys::Object::from(options.clone())).iter() {
        let entry = js_sys::Array::from(&entry);
        pairs.push(text(entry.get(0)));
        pairs.push(text(entry.get(1)));
    }
    Ok(pairs)
}

/// The canvas of `create`'s target: a canvas element or its id.
fn canvas_of(target: &JsValue) -> Result<web_sys::HtmlCanvasElement, JsValue> {
    if let Some(id) = target.as_string() {
        return web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id(&id))
            .ok_or_else(|| value::js_error(format!("no element with id {id}")))?
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| value::js_error(format!("{id} is not a canvas")));
    }
    target
        .clone()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| value::js_error("the target is not a canvas element or its id"))
}

/// `create` up to the loop (`docs/api.md` section 3): connects to the hub
/// of `connection`, loads the frame registry of the build, and spawns the
/// frame loop on the canvas `target` with the API the
/// [`binding::Renderer`] shares. Fails if the settings are incomplete, an
/// option is invalid (the parser's message, before anything is fetched),
/// the hub refuses, or the registry is invalid or never becomes ready.
/// Without options the client enters at the hub's now
/// (`space-model.md` section 7) at the `Home` view.
async fn start(
    target: JsValue,
    connection: JsValue,
    options: JsValue,
) -> Result<Rc<binding::Shared>, JsValue> {
    let hub_url = field(&connection, "hubUrl")?;
    let api_key = field(&connection, "apiKey")?;
    let space_id = field(&connection, "spaceId")?;
    let build_id = field(&connection, "buildId")?;
    if hub_url.trim().is_empty() || api_key.is_empty() || space_id.trim().is_empty() {
        return Err(value::js_error(
            "hub URL, API key, and space id are required",
        ));
    }
    let options = parse_options(&option_pairs(&options)?).map_err(value::js_error)?;
    let canvas = canvas_of(&target)?;
    let build = build_id.trim();
    let hub = WebHub::open(
        Routes::new(&hub_url, None, &space_id),
        ApiKey::new(api_key),
        (!build.is_empty()).then_some(build),
    )
    .await
    .map_err(|e| value::js_error(format!("{e:#}")))?;
    let cache = cache::WebCache::open(cache::DEFAULT_MAX_BYTES).await;
    if cache.is_none() {
        web_sys::console::warn_1(&"gx-renderer: no Cache API here; nothing is cached".into());
    }
    let mut fetcher = WebFetcher::new(Rc::new(hub), cache);
    // The start budget applies from the first key, the registry included.
    if let Some(n) = options.compute_budget {
        fetcher.set_compute_budget(n);
    }
    let system = fetcher
        .load_registry()
        .await
        .map_err(|e| value::js_error(format!("{e:#}")))?;
    let now = gx_core::timescale::utc_ms_to_tdb_seconds(fetcher.now_utc_ms());
    let launch = crate::options::launch(system, &options, gx_core::units::Seconds::new(now));
    let api = crate::api::Api::new(launch.sim.system().tree(), &launch.camera)
        .with_build(fetcher.hub().build())
        .with_compute_budget(fetch::WEB_WORKERS, options.compute_budget);
    let shared = Rc::new(binding::Shared::new(api));
    // The page's one event loop takes it (`host`): a page may create a
    // renderer after destroying another.
    host::run(app::WebApp::new(
        launch,
        &options,
        fetcher,
        canvas,
        shared.clone(),
    ))
    .map_err(value::js_error)?;
    Ok(shared)
}
