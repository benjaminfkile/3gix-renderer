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
//! - [`protocol`]: the hub's wire protocol without I/O, shared by the
//!   native and browser clients.
//! - [`hub`]: the hub client for chunks, the readiness WebSocket, and the
//!   frame registry (native only).
//! - [`mock_hub`]: a small in-process hub for tests and local runs (native
//!   only).
//! - [`sim`]: the simulation clock between the origin and now, and the
//!   integration of every frame to it with the relativistic laws and
//!   checkpoints (`space-model.md` sections 6.1, 6.2, and 7).
//! - [`seen`]: every frame at its retarded state for the observer and every
//!   light from its emitter's retarded position (`space-model.md` section
//!   6.4).
//! - [`clock`]: the monotonic clock and the step durations of the frame
//!   thread's matter work (`rendering.md` section 4).
//! - [`camera`]: the free camera parented to its nearest frame, the floating
//!   origin of `space-model.md` section 5.
//! - [`spawn`]: spawning the camera just above a frame's matter and
//!   carrying it with that frame.
//! - [`controls`]: keyboard and mouse bindings, shared by the window and the
//!   browser canvas (native, or with the `web` feature).
//! - [`stream`]: cell selection, asking ahead, the cell cache, decoding and
//!   compositing (`space-model.md` sections 2, 5, 7, 8; `rendering.md`
//!   section 3).
//! - [`selector`]: the selection on its own thread natively, inline in the
//!   browser (`rendering.md` sections 3.4 and 4).
//! - [`extract`]: Transvoxel extraction of drawn cells on a worker pool.
//! - [`volume`]: gas and plasma drawn by ray marching.
//! - [`farfield`]: far frames drawn as point sprites.
//! - [`light`]: lights and emission from hot matter (`matter-format.md`
//!   section 3.3).
//! - [`world`]: the matter pipeline from selection to drawable meshes and
//!   lights, one bounded step of the frame thread (`rendering.md` section
//!   4).
//! - [`render`]: camera-relative scene data, the wgpu renderer, the overlay,
//!   and the headless offscreen target.
//! - [`app`]: the desktop window loop and the headless screenshot run
//!   (native only).
//! - `web`: the browser build on a canvas with WebGPU (`wasm32` with the
//!   `web` feature; see `docs/web.md`).
//!
//! All world math stays in `gx-core`'s `f64` types. `f32` appears only in
//! data handed to the GPU, and only after the camera position has been
//! subtracted with [`gx_core::frames::FrameSystem::relative`].

#![warn(missing_docs)]

pub mod camera;
pub mod clock;
pub mod config;
#[cfg(any(not(target_arch = "wasm32"), feature = "web"))]
pub mod controls;
pub mod directions;
pub mod extract;
pub mod farfield;
pub mod light;
pub mod protocol;
pub mod redraw;
pub mod render;
pub mod seen;
pub mod selector;
pub mod sim;
pub mod spawn;
pub mod stream;
pub mod tiles;
pub mod transvoxel;
pub mod volume;
pub mod world;

#[cfg(not(target_arch = "wasm32"))]
pub mod app;
#[cfg(not(target_arch = "wasm32"))]
pub mod hub;
#[cfg(not(target_arch = "wasm32"))]
pub mod mock_hub;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod web;
