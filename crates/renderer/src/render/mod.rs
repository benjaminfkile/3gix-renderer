//! Rendering: camera-relative scene data, the wgpu frame pass, the overlay,
//! and the headless offscreen target.
//!
//! This task draws no matter yet, only what moves: a marker at every frame
//! origin, a line from every frame to its parent, and the debug overlay.
//! Space itself is not rendered (`space-model.md` section 1, rule 3); the
//! markers are a debug aid, not matter.
//!
//! - [`scene`]: positions relative to the camera in `f32`, the projection,
//!   and the depth strategy (`space-model.md` section 5).
//! - [`overlay`]: the overlay text.
//! - [`gpu`]: the pipelines and the one render pass.
//! - [`headless`]: the offscreen target and PNG output (native only).

pub mod gpu;
pub mod overlay;
pub mod scene;

#[cfg(not(target_arch = "wasm32"))]
pub mod headless;
