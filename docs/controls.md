# Controls

Every key and mouse binding of the desktop renderer and the browser build
(the bindings live in `src/controls.rs`, shared by both). Keys are matched by
physical position, so they sit in the same place on every keyboard layout.
In the browser, click the canvas first so it has the keyboard focus.

## Camera

| Input | Action |
|---|---|
| `W` / `S` | Fly forward / backward along the view direction |
| `A` / `D` | Fly left / right |
| `E` / `Q` | Fly up / down along the camera's up axis |
| Right mouse drag | Look around (yaw and pitch about the camera's own axes) |
| Mouse wheel | Scale flight speed: each notch multiplies or divides it by 1.25 |
| `1` to `9`, `0` | Jump to a view of the frame with that index (see below) |
| `Home` | Jump to a view of the whole system from above the root frame |

Flight speed also scales on its own with the distance from the camera to its
frame's `root_extent / 8` region: one second covers the remaining distance to
that region, with a floor of 1 m/s inside it. Flying near a frame is slow and
flying in deep space is fast. The wheel multiplier applies on top.

The number keys index the frames in registry order: frame ids sorted
ascending, skipping the root. `1` is the first, `9` the ninth, `0` the tenth.
A jump places the camera at `3 * root_extent / 8` from the frame origin,
above it the way `Home` sees the root: along root `+z`, looking along root
`-z` with root `+y` up. Because that direction is fixed in root axes, the
frame shows a lit side and a dark side, and the lit side turns as the light
moves over simulation time. A key past the last frame does nothing.

## Simulation time

| Input | Action |
|---|---|
| `Space` | Pause or resume |
| `[` | Halve the time scale |
| `]` | Double the time scale |
| `,` | Step one day (86400 s) backwards, while paused |
| `.` | Step one day (86400 s) forwards, while paused |
| `R` | Reset the simulation time to the launch offset |

The time scale stays between 1/1024 and 1e12 in magnitude. When the scale
asks for more than 20000 integration steps of 600 s in one rendered frame,
the overlay shows `sim lag` and the frames catch up over the following
rendered frames; the integrator is never skipped.

## Exposure

| Input | Action |
|---|---|
| `=` / numpad `+` | Brighten: exposure bias up half a stop |
| `-` / numpad `-` | Darken: exposure bias down half a stop |

Exposure follows the log-average luminance of the lit pixels on screen with
a 0.5 s adaptation; the bias, between -16 and +16 stops, applies on top. See
`docs/shading.md`.

## Window

| Input | Action |
|---|---|
| `Esc` | Quit |
| Close button | Quit |
