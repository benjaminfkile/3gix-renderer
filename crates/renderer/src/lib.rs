//! The 3GIX renderer engine.
//!
//! The renderer applies the laws to compiled matter and draws whatever is
//! near the camera (`space-model.md` section 2, "The renderer runs the
//! laws"). The one rule: the renderer may know physics, never objects. No
//! module, identifier, string, or comment here names anything in the
//! universe; frames are integers, and a frame is "frame 4".
//!
//! This crate is the engine as a library, so the desktop binary, the
//! headless tests, and the later browser build share one code path:
//!
//! - [`config`]: environment and command line configuration.
//! - [`hub`]: the hub client for chunks, the readiness WebSocket, and the
//!   frame registry (native only).
//! - [`mock_hub`]: a small in-process hub for tests and local runs (native
//!   only).
//! - [`sim`]: the simulation clock and the integration of every frame to the
//!   simulation time (`space-model.md` section 6).
//! - [`camera`]: the free camera parented to its nearest frame, the floating
//!   origin of `space-model.md` section 5.
//! - [`render`]: camera-relative scene data, the wgpu renderer, the overlay,
//!   and the headless offscreen target.
//! - [`app`]: the desktop window loop and the headless screenshot run
//!   (native only).
//!
//! All world math stays in `gx-core`'s `f64` types. `f32` appears only in
//! data handed to the GPU, and only after the camera position has been
//! subtracted with [`gx_core::frames::FrameSystem::relative`].

#![warn(missing_docs)]

pub mod camera;
pub mod config;
pub mod render;
pub mod sim;

#[cfg(not(target_arch = "wasm32"))]
pub mod app;
#[cfg(not(target_arch = "wasm32"))]
pub mod hub;
#[cfg(not(target_arch = "wasm32"))]
pub mod mock_hub;
