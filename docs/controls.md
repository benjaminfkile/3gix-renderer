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
| Mouse wheel | Set the flight speed: each notch multiplies or divides it by 1.25 (about ten notches per factor of ten) |
| `Shift` (held) | Boost: fly at ten times the set speed while held |
| `1` to `9`, `0` | Jump to a view of the frame with that index (see below) |
| `Home` | Jump to a view of the whole system from above the root frame |
| `M` | Hide or show the frame markers and the frame-to-parent lines (drawing aids, not matter) |

The camera behaves like an observer floating free in space: simulation time
moves and turns the frames, but not the camera, which keeps its place and
view direction in root coordinates. Bodies move past it on their orbits. The
frame the camera is parented to is only a precision device (the camera
re-parents to the nearest frame as it flies) and never carries it along.

W flies along the view direction, which is always the screen center; a
crosshair drawn with the overlay marks it. S flies the opposite way, A and D
slide sideways, Q and E slide down and up, and a right drag turns the view, so
whatever is under the crosshair is what W flies toward.

The camera is an observer that nothing in space acts on. Holding a movement
key flies a straight line at the flight speed, in meters per second, and only
your input changes that speed: the wheel sets it (1e-3 to 1e15 m/s, 1e11 m/s
for a newly placed camera, kept across number-key jumps) and Shift boosts it
tenfold while held. The flight speed glides to a new setting with a 0.15 s
time constant (evenly per factor of ten), so a burst of notches or a Shift
press does not jump. Matter, gravity, and the frame the
camera happens to be parented to play no part: it flies through a star or a
planet the same way it flies through empty space. The overlay's camera line
shows the current flight speed.

The number keys index the frames in registry order: frame ids sorted
ascending, skipping the root. `1` is the first, `9` the ninth, `0` the tenth.
A jump places the camera at `3 * root_extent / 8` from the frame origin,
above it the way `Home` sees the root: along root `+z`, looking along root
`-z` with root `+y` up. Because that direction is fixed in root axes, the
frame shows a lit side and a dark side, and the lit side turns as the light
moves over simulation time. The camera floats free where it lands, and the
body moves on along its orbit. A key past the last frame does nothing.

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
| `=` / numpad `+` | Brighten: exposure up half a stop |
| `-` / numpad `-` | Darken: exposure down half a stop |

By default exposure follows the log-average luminance of the lit pixels on
screen with a 0.5 s adaptation, and the keys move a bias, between -16 and
+16 stops, that applies on top. With `--exposure-stops <s>` the exposure is
fixed instead (no adaptation) and the keys move that fixed value, between
-64 and +64 stops. The overlay shows `exposure auto` or `exposure fixed <s>`.
See `docs/shading.md`.

## Window

| Input | Action |
|---|---|
| `Esc` | Quit |
| Close button | Quit |
