# 3gix-renderer

The renderer of the 3GIX space runtime. It applies the laws to compiled matter and draws whatever is near the camera. It runs natively on the desktop and in the browser from one Rust codebase through wgpu.

## The one rule

The renderer may know physics. It may never know objects.

Nothing in this repository knows what any piece of matter is. There is no code path, file, identifier, string, or comment named after any object in the universe. The vocabulary lint from the core library enforces the banned word list, and a failed lint fails the build.

## What it does

- Fetches the frame registry for a build and integrates every frame forward from the epoch under Newtonian gravity.
- Fetches matter cells near the camera at a depth chosen by distance, through the hub's chunk endpoint and readiness WebSocket.
- Composites overlapping sections, relaxes fluid matter toward the local equipotential, derives emission from temperature.
- Extracts surfaces or ray marches the density field and draws with camera-relative single precision over a floating origin.
- Lets the camera go anywhere, at any time scale.

## What it depends on

- `gx-core`, the shared library: matter format decoder and validator, units, laws, chunk keys, the frame registry.
- `wgpu` and `winit` for graphics and windowing. Headless tests run on a software Vulkan driver with no window.

## Specification

- Architecture: `3GIXHub/docs/architecture/space-model.md`
- Matter format v1: `3GIXHub/docs/architecture/matter-format.md`

## Contributing rules

- No word from the core library's `scripts/vocab-banned.txt`, and no proper noun for any body, anywhere in this repository.
- No infrastructure identifiers, secrets, hostnames, or LAN addresses in code, history, docs, or CI output. This repository is public.
- No em or en dashes in any text.
- `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`, and the vocabulary lint pass on every change.
