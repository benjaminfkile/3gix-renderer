//! The browser build: the renderer on a canvas with WebGPU.
//!
//! Compiled only for `wasm32-unknown-unknown` with the `web` feature, and
//! packaged by `web/build.sh` with `wasm-pack`. `web/index.html` loads the
//! package and calls [`start`] with the settings from its panel. The
//! renderer is the same library as on the desktop (`space-model.md`
//! section 10: "Native desktop and WebAssembly from one codebase"); only the
//! platform glue lives here:
//!
//! - [`hub`]: the hub client over `fetch` and a browser `WebSocket`.
//! - [`app`]: the canvas frame loop.
//!
//! There is no browser in CI. The build is checked there; running it is
//! verified by hand on a desktop browser (`docs/web.md`).

pub mod app;
pub mod hub;

use crate::config::ApiKey;
use crate::sim::{SimClock, Simulation};
use hub::{load_registry, ReadySocket, WebFetcher, WebHub};
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

/// Connects to the hub, loads the frame registry of the build, and starts
/// drawing on the canvas with id `canvas_id`.
///
/// `build_id` may be empty for the active build of the space. `time_scale`
/// is simulation seconds per wall-clock second. The returned promise
/// resolves once the frame loop runs, and rejects with a message if the
/// settings are incomplete, the hub refuses, or the registry is invalid or
/// never becomes ready.
#[wasm_bindgen]
pub async fn start(
    canvas_id: String,
    hub_url: String,
    api_key: String,
    space_id: String,
    build_id: String,
    time_scale: f64,
) -> Result<(), JsValue> {
    if hub_url.trim().is_empty() || api_key.is_empty() || space_id.trim().is_empty() {
        return Err(js_error("hub URL, API key, and space id are required"));
    }
    if !time_scale.is_finite() {
        return Err(js_error("the time scale must be a number"));
    }
    let canvas = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(&canvas_id))
        .ok_or_else(|| js_error(format!("no element with id {canvas_id}")))?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| js_error(format!("{canvas_id} is not a canvas")))?;
    let build = build_id.trim();
    let hub = WebHub::connect(
        hub_url.trim(),
        ApiKey::new(api_key),
        space_id.trim(),
        (!build.is_empty()).then_some(build),
    )
    .await
    .map_err(|e| js_error(format!("{e:#}")))?;
    let socket = ReadySocket::open(&hub.ready_url());
    let system = load_registry(&hub, socket.as_ref())
        .await
        .map_err(|e| js_error(format!("{e:#}")))?;
    let sim = Simulation::new(
        system,
        SimClock::new(gx_core::units::Seconds::new(0.0), time_scale),
    );
    let fetcher = WebFetcher::new(Rc::new(hub), socket);
    let event_loop = EventLoop::new().map_err(js_error)?;
    // Wait, not Poll: on the web Poll is a busy loop that starves the page.
    // Each frame requests the next redraw, which the browser delivers once
    // per display refresh.
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.spawn_app(app::WebApp::new(sim, fetcher, canvas));
    Ok(())
}
