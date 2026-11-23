//! Values across the web binding (`docs/api.md` section 2): plain JS
//! objects and arrays to and from the [`crate::api`] types, shaped as the
//! contract says. Only conversion and shape checks live here; every rule
//! about the values themselves (known frames, unit quaternions, limits) is
//! [`crate::api::Api`]'s.

use crate::api::{Exposure, FlightInput, FrameInfo, Hit, HitKind, Layer, Pose, Projected};
use crate::api::{Kind, MeshMode, Stats, SurfaceHeight, Time};
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

/// The first integer JS numbers cannot all hold: 2^53. A frame id at or
/// above it is an error in both directions.
pub const MAX_SAFE_ID: f64 = 9_007_199_254_740_992.0;

/// A JS `Error` with `message`, as every method of the binding throws.
pub fn js_error(message: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}

fn get(value: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    if !value.is_object() {
        return Err(js_error(format!("expected an object with {key}")));
    }
    Reflect::get(value, &JsValue::from_str(key))
}

fn set(object: &Object, key: &str, value: impl Into<JsValue>) {
    // Setting a plain property on a fresh object cannot fail.
    let _ = Reflect::set(object, &JsValue::from_str(key), &value.into());
}

/// A JS number as `f64`; `what` names it in the error.
pub fn number(value: &JsValue, what: &str) -> Result<f64, JsValue> {
    value
        .as_f64()
        .ok_or_else(|| js_error(format!("{what} is not a number")))
}

/// A frame id from a JS number: a whole number from 0 to below 2^53.
pub fn frame_id(value: &JsValue) -> Result<u64, JsValue> {
    let v = number(value, "the frame id")?;
    if !(v.is_finite() && v >= 0.0 && v.fract() == 0.0) {
        return Err(js_error(format!("frame id {v} is not a whole number")));
    }
    if v >= MAX_SAFE_ID {
        return Err(js_error(format!("frame id {v} is at or above 2^53")));
    }
    Ok(v as u64)
}

/// A frame id as a JS number; an error at or above 2^53.
pub fn id_to_js(id: u64) -> Result<JsValue, JsValue> {
    if id as f64 >= MAX_SAFE_ID {
        return Err(js_error(format!("frame id {id} is at or above 2^53")));
    }
    Ok(JsValue::from_f64(id as f64))
}

/// A JS array of `N` numbers.
pub fn numbers<const N: usize>(value: &JsValue, what: &str) -> Result<[f64; N], JsValue> {
    if !Array::is_array(value) {
        return Err(js_error(format!("{what} is not an array of {N} numbers")));
    }
    let a = Array::from(value);
    if a.length() as usize != N {
        return Err(js_error(format!("{what} is not an array of {N} numbers")));
    }
    let mut out = [0.0; N];
    for (i, v) in out.iter_mut().enumerate() {
        *v = number(&a.get(i as u32), what)?;
    }
    Ok(out)
}

fn array(values: &[f64]) -> Array {
    values.iter().map(|&v| JsValue::from_f64(v)).collect()
}

/// A pose from `{ frameId, position, orientation, speed }`.
pub fn pose(value: &JsValue) -> Result<Pose, JsValue> {
    Ok(Pose {
        frame_id: frame_id(&get(value, "frameId")?)?,
        position: numbers(&get(value, "position")?, "position")?,
        orientation: numbers(&get(value, "orientation")?, "orientation")?,
        speed: number(&get(value, "speed")?, "speed")?,
    })
}

/// A pose as `{ frameId, position, orientation, speed }`.
pub fn pose_to_js(p: &Pose) -> Result<JsValue, JsValue> {
    let o = Object::new();
    set(&o, "frameId", id_to_js(p.frame_id)?);
    set(&o, "position", array(&p.position));
    set(&o, "orientation", array(&p.orientation));
    set(&o, "speed", p.speed);
    Ok(o.into())
}

/// A flight input from `{ move, look, boost }`, or `None` for null or
/// undefined.
pub fn flight_input(value: &JsValue) -> Result<Option<FlightInput>, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    Ok(Some(FlightInput {
        r#move: numbers(&get(value, "move")?, "move")?,
        look: numbers(&get(value, "look")?, "look")?,
        boost: get(value, "boost")?.is_truthy(),
    }))
}

/// An exposure from `{ mode: "auto", biasStops }` or `{ mode: "fixed",
/// stops }`.
pub fn exposure(value: &JsValue) -> Result<Exposure, JsValue> {
    match get(value, "mode")?.as_string().as_deref() {
        Some("auto") => Ok(Exposure::Auto {
            bias_stops: number(&get(value, "biasStops")?, "biasStops")?,
        }),
        Some("fixed") => Ok(Exposure::Fixed {
            stops: number(&get(value, "stops")?, "stops")?,
        }),
        _ => Err(js_error("the exposure mode is not \"auto\" or \"fixed\"")),
    }
}

/// A mesh mode from `"surfaces"`, `"vertices"`, or `"edges"`.
pub fn mesh_mode(value: &str) -> Result<MeshMode, JsValue> {
    crate::options::parse_mesh_mode("meshMode", value).map_err(js_error)
}

/// A transform kind from `"point"` or `"direction"`.
pub fn kind(value: &str) -> Result<Kind, JsValue> {
    match value {
        "point" => Ok(Kind::Point),
        "direction" => Ok(Kind::Direction),
        _ => Err(js_error(format!(
            "kind {value:?} is not \"point\" or \"direction\""
        ))),
    }
}

/// A UTC instant, milliseconds, from a JS number: finite, rounded to
/// the millisecond.
pub fn utc_ms(value: f64) -> Result<i64, JsValue> {
    if !value.is_finite() || value.abs() >= MAX_SAFE_ID {
        return Err(js_error("the time is not a finite number of milliseconds"));
    }
    Ok(value.round() as i64)
}

/// A count from a JS number: a whole number from 0 below 2^32.
pub fn count(value: f64, what: &str) -> Result<u32, JsValue> {
    if !(value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < 4_294_967_296.0) {
        return Err(js_error(format!("{what} {value} is not a whole number")));
    }
    Ok(value as u32)
}

/// The time as `{ utcMs, originUtcMs, nowUtcMs, timeScale }`.
pub fn time_to_js(t: &Time) -> JsValue {
    let o = Object::new();
    set(&o, "utcMs", t.utc_ms as f64);
    set(&o, "originUtcMs", t.origin_utc_ms as f64);
    set(&o, "nowUtcMs", t.now_utc_ms as f64);
    set(&o, "timeScale", t.time_scale);
    o.into()
}

/// A frame as `{ id, parentId, mass, angularVelocity, hasSurface, surface }`.
pub fn frame_to_js(f: &FrameInfo) -> Result<JsValue, JsValue> {
    let o = Object::new();
    set(&o, "id", id_to_js(f.id)?);
    set(
        &o,
        "parentId",
        f.parent_id
            .map(id_to_js)
            .transpose()?
            .unwrap_or(JsValue::NULL),
    );
    set(&o, "mass", f.mass);
    set(&o, "angularVelocity", array(&f.angular_velocity));
    set(&o, "hasSurface", f.has_surface);
    let surface = match f.surface {
        Some(s) => {
            let so = Object::new();
            set(&so, "radiusEquatorial", s.radius_equatorial);
            set(&so, "radiusPolar", s.radius_polar);
            set(&so, "heightFloor", s.height_floor);
            so.into()
        }
        None => JsValue::NULL,
    };
    set(&o, "surface", surface);
    Ok(o.into())
}

/// A layer as `{ id, overlay, exclusive, shown }`.
pub fn layer_to_js(l: &Layer) -> JsValue {
    let o = Object::new();
    set(&o, "id", l.id.as_str());
    set(&o, "overlay", l.overlay);
    set(&o, "exclusive", l.exclusive);
    set(&o, "shown", l.shown);
    o.into()
}

/// A surface height as `{ height, level, finer }`, or null.
pub fn surface_height_to_js(h: Option<SurfaceHeight>) -> JsValue {
    let Some(h) = h else {
        return JsValue::NULL;
    };
    let o = Object::new();
    set(&o, "height", h.height);
    set(&o, "level", h.level);
    set(&o, "finer", h.finer);
    o.into()
}

/// A raycast hit as `{ frameId, position, distance, kind }`, or null.
pub fn hit_to_js(h: Option<Hit>) -> Result<JsValue, JsValue> {
    let Some(h) = h else {
        return Ok(JsValue::NULL);
    };
    let o = Object::new();
    set(&o, "frameId", id_to_js(h.frame_id)?);
    set(&o, "position", array(&h.position));
    set(&o, "distance", h.distance);
    let kind = match h.kind {
        HitKind::Surface => "surface",
        HitKind::Cells => "cells",
    };
    set(&o, "kind", kind);
    Ok(o.into())
}

/// A projected point as `{ x, y, depth }`, or null; `scale` divides the
/// pixels (physical to CSS).
pub fn projected_to_js(p: Option<Projected>, scale: f64) -> JsValue {
    let Some(p) = p else {
        return JsValue::NULL;
    };
    let o = Object::new();
    set(&o, "x", p.x / scale);
    set(&o, "y", p.y / scale);
    set(&o, "depth", p.depth);
    o.into()
}

/// The stats as `{ fps, tilesDrawn, triangles, cellsDrawn, streaming,
/// computing, uploading }`.
pub fn stats_to_js(s: &Stats) -> JsValue {
    let o = Object::new();
    set(&o, "fps", s.fps);
    set(&o, "tilesDrawn", s.tiles_drawn as f64);
    set(&o, "triangles", s.triangles as f64);
    set(&o, "cellsDrawn", s.cells_drawn as f64);
    set(&o, "streaming", s.streaming as f64);
    set(&o, "computing", s.computing as f64);
    set(&o, "uploading", s.uploading as f64);
    o.into()
}

/// Checks a pose's shape and frame id as `setCamera` converts it, and
/// returns it as the binding would hand it back. Exported so the
/// conversions can be checked under Node, without a GPU or a hub.
#[wasm_bindgen(js_name = checkPose)]
pub fn check_pose(value: JsValue) -> Result<JsValue, JsValue> {
    pose_to_js(&pose(&value)?)
}

/// Checks a frame id as every method taking one converts it: a whole
/// number from 0 to below 2^53. Exported for the Node checks.
#[wasm_bindgen(js_name = checkFrameId)]
pub fn check_frame_id(value: JsValue) -> Result<f64, JsValue> {
    frame_id(&value).map(|id| id as f64)
}
