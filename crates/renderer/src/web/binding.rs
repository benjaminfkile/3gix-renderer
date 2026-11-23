//! The web binding of the renderer API: the wasm-bindgen class `Renderer`
//! (`docs/api.md` section 1 item 2, sections 3 to 8). Every method
//! converts its values ([`super::value`]) and calls the same
//! [`crate::api::Api`] the native binding uses; no logic lives here.
//! Calls are made from the page's thread between ticks, and events are
//! delivered to the `on` listeners after each tick ([`Shared::deliver`]).

use super::value::{self, js_error};
use crate::api::{Api, ApiError, Controls, Event};
use js_sys::{Array, Function, Object, Promise, Reflect};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// The event names of `docs/api.md` section 8.
pub const EVENTS: [&str; 7] = [
    "ready",
    "error",
    "camera",
    "frameChanged",
    "settled",
    "hit",
    "stats",
];

/// The event names in the generated TypeScript, for `on` and the surface
/// check (`scripts/api-surface-check.mjs`).
#[wasm_bindgen(typescript_custom_section)]
const EVENT_TYPE: &str = r#"
/** The events of `Renderer.on` (`docs/api.md` section 8). */
export type RendererEvent = "ready" | "error" | "camera" | "frameChanged" | "settled" | "hit" | "stats";
"#;

/// What the class and the frame loop share: the API, the listeners, and
/// the canvas's pixel ratio.
pub struct Shared {
    /// The API both sides call.
    pub api: RefCell<Api>,
    /// Physical pixels per CSS pixel of the canvas: queries take and give
    /// CSS pixels on the web (`docs/api.md` section 6).
    pub scale: Cell<f64>,
    listeners: RefCell<Vec<(u32, String, Function)>>,
    next_listener: Cell<u32>,
    /// `create`'s promise functions until the loop runs.
    pending: RefCell<Option<(Function, Function)>>,
}

impl Shared {
    /// Shared state over `api`.
    pub fn new(api: Api) -> Shared {
        Shared {
            api: RefCell::new(api),
            scale: Cell::new(1.0),
            listeners: RefCell::new(Vec::new()),
            next_listener: Cell::new(0),
            pending: RefCell::new(None),
        }
    }

    /// Drains the API's events and calls the listeners of each, in queue
    /// order. The first `ready` resolves `create`'s promise. Called by the
    /// frame loop after each tick.
    pub fn deliver(&self) {
        let events = self.api.borrow_mut().events();
        for event in events {
            let (name, payload) = match self.event_to_js(&event) {
                Ok(e) => e,
                Err(e) => ("error", e),
            };
            if name == "ready" {
                if let Some((resolve, _)) = self.pending.borrow_mut().take() {
                    let _ = resolve.call0(&JsValue::NULL);
                }
            }
            let listeners: Vec<Function> = self
                .listeners
                .borrow()
                .iter()
                .filter(|(_, n, _)| n == name)
                .map(|(_, _, f)| f.clone())
                .collect();
            for f in listeners {
                if let Err(e) = f.call1(&JsValue::NULL, &payload) {
                    web_sys::console::error_2(&"gx-renderer: an event listener threw".into(), &e);
                }
            }
        }
    }

    /// The loop could not start or stopped: rejects `create`'s promise
    /// while it waits, an `error` event after.
    pub fn fail(&self, message: &str) {
        if let Some((_, reject)) = self.pending.borrow_mut().take() {
            let _ = reject.call1(&JsValue::NULL, &js_error(message));
            return;
        }
        self.emit("error", &JsValue::from_str(message));
    }

    fn emit(&self, name: &str, payload: &JsValue) {
        let listeners: Vec<Function> = self
            .listeners
            .borrow()
            .iter()
            .filter(|(_, n, _)| n == name)
            .map(|(_, _, f)| f.clone())
            .collect();
        for f in listeners {
            let _ = f.call1(&JsValue::NULL, payload);
        }
    }

    fn event_to_js(&self, event: &Event) -> Result<(&'static str, JsValue), JsValue> {
        Ok(match event {
            Event::Ready => ("ready", JsValue::UNDEFINED),
            Event::Error(m) => ("error", JsValue::from_str(m)),
            Event::Camera(p) => ("camera", value::pose_to_js(p)?),
            Event::FrameChanged { from, to } => {
                let o = Object::new();
                Reflect::set(&o, &"from".into(), &value::id_to_js(*from)?)?;
                Reflect::set(&o, &"to".into(), &value::id_to_js(*to)?)?;
                ("frameChanged", o.into())
            }
            Event::Settled { seconds } => {
                let o = Object::new();
                Reflect::set(&o, &"seconds".into(), &JsValue::from_f64(*seconds))?;
                ("settled", o.into())
            }
            Event::Hit { hit, x, y, button } => {
                let scale = self.scale.get();
                let o = Object::new();
                Reflect::set(&o, &"hit".into(), &value::hit_to_js(*hit)?)?;
                Reflect::set(&o, &"x".into(), &JsValue::from_f64(x / scale))?;
                Reflect::set(&o, &"y".into(), &JsValue::from_f64(y / scale))?;
                Reflect::set(&o, &"button".into(), &JsValue::from(*button))?;
                ("hit", o.into())
            }
            Event::Stats(s) => ("stats", value::stats_to_js(s)),
        })
    }
}

fn api_error(e: ApiError) -> JsValue {
    js_error(e)
}

/// The renderer on a canvas (`docs/api.md` sections 3 to 8): created with
/// `Renderer.create`, driven by its methods between ticks, observed with
/// `on`.
#[wasm_bindgen]
pub struct Renderer {
    shared: Rc<Shared>,
}

impl Renderer {
    fn api(&self) -> std::cell::RefMut<'_, Api> {
        self.shared.api.borrow_mut()
    }
}

#[wasm_bindgen]
impl Renderer {
    /// `create(target, connection, options)`: connects to the hub
    /// (`connection` is `{ hubUrl, apiKey, spaceId, buildId }`, `buildId`
    /// may be empty), loads the registry, and starts the loop on `target`,
    /// a canvas element or its id. `options` holds the view and time
    /// options by camelCase name (`{ startOn: "3:0,0,1", timeScale: 60 }`)
    /// or as flattened name and value pairs, parsed by `crate::options`.
    /// Resolves to the renderer once the loop runs; rejects with the
    /// message when anything is missing or invalid or the hub refuses.
    pub async fn create(
        target: JsValue,
        connection: JsValue,
        options: JsValue,
    ) -> Result<Renderer, JsValue> {
        let shared = super::start(target, connection, options).await?;
        let promise = Promise::new(&mut |resolve, reject| {
            *shared.pending.borrow_mut() = Some((resolve, reject));
        });
        wasm_bindgen_futures::JsFuture::from(promise).await?;
        Ok(Renderer { shared })
    }

    /// `destroy()`: stops the loop, the module workers, and uploads, and
    /// frees GPU memory. Any later call is an error.
    pub fn destroy(&self) -> Result<(), JsValue> {
        self.api().destroy().map_err(api_error)
    }

    /// `pause()`: no drawing, streaming, or computing; the time holds.
    pub fn pause(&self) -> Result<(), JsValue> {
        self.api().pause().map_err(api_error)
    }

    /// `resume()`: runs again from the time it was paused at.
    pub fn resume(&self) -> Result<(), JsValue> {
        self.api().resume().map_err(api_error)
    }

    /// `resize(width, height, pixelRatio)`: the surface size in physical
    /// pixels, for hosts that size the canvas themselves.
    pub fn resize(&self, width: f64, height: f64, pixel_ratio: f64) -> Result<(), JsValue> {
        let w = value::count(width, "width")?;
        let h = value::count(height, "height")?;
        self.api().resize(w, h, pixel_ratio).map_err(api_error)
    }

    /// `getCamera()`: the pose as of the last completed tick.
    #[wasm_bindgen(js_name = getCamera)]
    pub fn get_camera(&self) -> Result<JsValue, JsValue> {
        value::pose_to_js(&self.api().get_camera())
    }

    /// `setCamera(pose)`: puts the camera exactly at the pose next tick.
    #[wasm_bindgen(js_name = setCamera)]
    pub fn set_camera(&self, pose: JsValue) -> Result<(), JsValue> {
        let pose = value::pose(&pose)?;
        self.api().set_camera(pose).map_err(api_error)
    }

    /// `placeOver(frameId, direction, height, pitchDeg)`: the placement of
    /// `--start-on <frame>:<x>,<y>,<z>@+<height>^<pitch>`.
    #[wasm_bindgen(js_name = placeOver)]
    pub fn place_over(
        &self,
        frame_id: JsValue,
        direction: JsValue,
        height: f64,
        pitch_deg: f64,
    ) -> Result<(), JsValue> {
        let id = value::frame_id(&frame_id)?;
        let d = value::numbers(&direction, "direction")?;
        self.api()
            .place_over(id, d, height, pitch_deg)
            .map_err(api_error)
    }

    /// `setControls("builtin" | "none")`: who flies the camera from the
    /// keyboard and mouse.
    #[wasm_bindgen(js_name = setControls)]
    pub fn set_controls(&self, controls: &str) -> Result<(), JsValue> {
        let c = match controls {
            "builtin" => Controls::Builtin,
            "none" => Controls::None,
            _ => {
                return Err(js_error(format!(
                    "controls {controls:?} are not \"builtin\" or \"none\""
                )))
            }
        };
        self.api().set_controls(c);
        Ok(())
    }

    /// `setFlightInput({ move, look, boost } | null)`: the free camera's
    /// held input; null clears it.
    #[wasm_bindgen(js_name = setFlightInput)]
    pub fn set_flight_input(&self, input: JsValue) -> Result<(), JsValue> {
        let input = value::flight_input(&input)?;
        self.api().set_flight_input(input);
        Ok(())
    }

    /// `setSpeed(mps)`: the set flight speed.
    #[wasm_bindgen(js_name = setSpeed)]
    pub fn set_speed(&self, mps: f64) -> Result<(), JsValue> {
        self.api().set_speed(mps).map_err(api_error)
    }

    /// `setFov(deg)`: the vertical field of view, degrees.
    #[wasm_bindgen(js_name = setFov)]
    pub fn set_fov(&self, deg: f64) -> Result<(), JsValue> {
        self.api().set_fov(deg).map_err(api_error)
    }

    /// `getFov()`: the vertical field of view, degrees.
    #[wasm_bindgen(js_name = getFov)]
    pub fn get_fov(&self) -> f64 {
        self.api().get_fov()
    }

    /// `getTime()`: `{ utcMs, originUtcMs, nowUtcMs, timeScale }` as of
    /// the last completed tick.
    #[wasm_bindgen(js_name = getTime)]
    pub fn get_time(&self) -> JsValue {
        self.api()
            .get_time()
            .map_or(JsValue::NULL, |t| value::time_to_js(&t))
    }

    /// `setTime(utcMs)`: jumps to that instant next tick.
    #[wasm_bindgen(js_name = setTime)]
    pub fn set_time(&self, utc_ms: f64) -> Result<(), JsValue> {
        let t = value::utc_ms(utc_ms)?;
        self.api().set_time(t).map_err(api_error)
    }

    /// `setTimeScale(x)`: the playback rate; negative runs back.
    #[wasm_bindgen(js_name = setTimeScale)]
    pub fn set_time_scale(&self, x: f64) -> Result<(), JsValue> {
        self.api().set_time_scale(x).map_err(api_error)
    }

    /// `frames()`: every frame of the registry.
    pub fn frames(&self) -> Result<Array, JsValue> {
        self.api().frames().iter().map(value::frame_to_js).collect()
    }

    /// `transform(point, fromFrame, toFrame, kind)`: a point or direction
    /// from one frame's axes into another's at the instant drawn.
    pub fn transform(
        &self,
        point: JsValue,
        from_frame: JsValue,
        to_frame: JsValue,
        kind: &str,
    ) -> Result<Array, JsValue> {
        let p = value::numbers(&point, "point")?;
        let from = value::frame_id(&from_frame)?;
        let to = value::frame_id(&to_frame)?;
        let kind = value::kind(kind)?;
        let v = self.api().transform(p, from, to, kind).map_err(api_error)?;
        Ok(v.iter().map(|&x| JsValue::from_f64(x)).collect())
    }

    /// `surfaceHeight(frameId, direction)`: `{ height, level, finer }` or
    /// null.
    #[wasm_bindgen(js_name = surfaceHeight)]
    pub fn surface_height(
        &self,
        frame_id: JsValue,
        direction: JsValue,
    ) -> Result<JsValue, JsValue> {
        let id = value::frame_id(&frame_id)?;
        let d = value::numbers(&direction, "direction")?;
        let h = self.api().surface_height(id, d).map_err(api_error)?;
        Ok(value::surface_height_to_js(h))
    }

    /// `raycast(x, y)`: from a CSS pixel, `{ frameId, position, distance,
    /// kind }` or null.
    pub fn raycast(&self, x: f64, y: f64) -> Result<JsValue, JsValue> {
        let s = self.shared.scale.get();
        let hit = self.api().raycast(x * s, y * s).map_err(api_error)?;
        value::hit_to_js(hit)
    }

    /// `project(frameId, point)`: `{ x, y, depth }` in CSS pixels, or null.
    pub fn project(&self, frame_id: JsValue, point: JsValue) -> Result<JsValue, JsValue> {
        let id = value::frame_id(&frame_id)?;
        let p = value::numbers(&point, "point")?;
        let projected = self.api().project(id, p).map_err(api_error)?;
        Ok(value::projected_to_js(projected, self.shared.scale.get()))
    }

    /// `layers()`: `[{ id, overlay, exclusive, shown }]`.
    pub fn layers(&self) -> Array {
        self.api().layers().iter().map(value::layer_to_js).collect()
    }

    /// `setShown(layerId, shown)`: shows or hides a layer.
    #[wasm_bindgen(js_name = setShown)]
    pub fn set_shown(&self, layer_id: &str, shown: bool) -> Result<(), JsValue> {
        self.api().set_shown(layer_id, shown).map_err(api_error)
    }

    /// `setExposure({ mode: "auto", biasStops } | { mode: "fixed", stops })`.
    #[wasm_bindgen(js_name = setExposure)]
    pub fn set_exposure(&self, exposure: JsValue) -> Result<(), JsValue> {
        let e = value::exposure(&exposure)?;
        self.api().set_exposure(e).map_err(api_error)
    }

    /// `setHud(on)`: the built-in text overlay on or off.
    #[wasm_bindgen(js_name = setHud)]
    pub fn set_hud(&self, on: bool) -> Result<(), JsValue> {
        self.api().set_hud(on).map_err(api_error)
    }

    /// `setMeshMode("surfaces" | "vertices" | "edges")`.
    #[wasm_bindgen(js_name = setMeshMode)]
    pub fn set_mesh_mode(&self, mode: &str) -> Result<(), JsValue> {
        let m = value::mesh_mode(mode)?;
        self.api().set_mesh_mode(m).map_err(api_error)
    }

    /// `setMaxFps(n | null)`: the frame cap, null for none.
    #[wasm_bindgen(js_name = setMaxFps)]
    pub fn set_max_fps(&self, fps: Option<f64>) -> Result<(), JsValue> {
        let fps = fps.map(|n| value::count(n, "max fps")).transpose()?;
        self.api().set_max_fps(fps).map_err(api_error)
    }

    /// `setComputeBudget(workers)`: module workers this client may run.
    #[wasm_bindgen(js_name = setComputeBudget)]
    pub fn set_compute_budget(&self, workers: f64) -> Result<(), JsValue> {
        let n = value::count(workers, "the compute budget")?;
        self.api().set_compute_budget(n as usize).map_err(api_error)
    }

    /// `getComputeBudget()`: module workers this client may run.
    #[wasm_bindgen(js_name = getComputeBudget)]
    pub fn get_compute_budget(&self) -> f64 {
        self.api().get_compute_budget() as f64
    }

    /// `on(name, fn)`: calls `listener` with each event `name` carries
    /// (section 8), after each tick. Returns the function that
    /// unsubscribes it.
    pub fn on(&self, name: &str, listener: Function) -> Result<Function, JsValue> {
        if !EVENTS.contains(&name) {
            return Err(js_error(format!("there is no event {name:?}")));
        }
        let id = self.shared.next_listener.get();
        self.shared.next_listener.set(id + 1);
        (self.shared.listeners.borrow_mut()).push((id, name.to_string(), listener));
        let shared = Rc::downgrade(&self.shared);
        let off = Closure::<dyn FnMut()>::new(move || {
            if let Some(s) = shared.upgrade() {
                s.listeners.borrow_mut().retain(|(i, _, _)| *i != id);
            }
        });
        Ok(off.into_js_value().unchecked_into())
    }
}
