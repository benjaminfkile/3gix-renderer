//! Rendering: camera-relative scene data, the wgpu frame pass, the overlay,
//! and the headless offscreen target.
//!
//! Matter is drawn as lit surfaces extracted from the density field
//! (`space-model.md` section 2), lit by point lights derived from hot
//! matter, exposed automatically, and tone mapped. On top go a marker at
//! every frame origin, a line from every frame to its parent, and the debug
//! overlay. Space itself is not rendered (`space-model.md` section 1,
//! rule 3); the markers are a debug aid, not matter.
//!
//! - [`scene`]: positions relative to the camera in `f32`, the projection,
//!   and the depth strategy (`space-model.md` section 5).
//! - [`overlay`]: the overlay text.
//! - [`gpu`]: the pipelines, the surface pass, exposure, and the display
//!   pass. Shaders are in `src/shaders/` (`surface.wgsl`, `exposure.wgsl`)
//!   and `src/render/markers.wgsl`.
//! - [`headless`]: the offscreen target and PNG output (native only).

pub mod gpu;
pub mod overlay;
pub mod scene;

#[cfg(not(target_arch = "wasm32"))]
pub mod headless;
