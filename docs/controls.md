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
| `1` to `9`, `0` | Jump to a view of the frame with that index (see below); with the spawn chooser open, spawn instead (see Spawning) |
| `G` | Open or close the spawn chooser (see Spawning) |
| `L` | Open or close the layer chooser (see Layers) |
| `Home` | Jump to a view of the whole system from above the root frame |
| `M` | Hide or show the frame markers and the frame-to-parent lines (drawing aids, not matter) |
| `V` | Type a speed in meters per second (see Typing a speed) |
| `X` | Cycle how surfaces are drawn: lit surfaces, then (for debugging) every mesh vertex as a point, then every triangle edge as a line, unlit and seen from both sides. `--mesh-mode surfaces\|vertices\|edges` sets it at start, also headless |

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
camera happens to be parented to play no part: it flies through the densest matter
the same way it flies through empty space. The overlay's camera line
shows the current flight speed.

The number keys, on the top row or the keypad, index the frames in
registry order: frame ids sorted ascending, skipping the root. `1` is the first, `9` the ninth, `0` the tenth.
A jump places the camera at `3 * root_extent / 8` from the frame origin,
above it the way `Home` sees the root: along root `+z`, looking along root
`-z` with root `+y` up. Because that direction is fixed in root axes, the
frame shows a lit side and a dark side, and the lit side turns as the light
moves over simulation time. The camera floats free where it lands, and the
body moves on along its orbit. A key past the last frame does nothing.

## Typing a speed

`V` opens a speed entry on the overlay's last line: type a number of
meters per second (digits, the number row or the numpad, and one decimal
point), `Backspace` to correct, `Enter` to set it, `Escape` to cancel.
It sets the flight speed (clamped to 1e-3 to 1e15 m/s). While the entry
is open every other key waits.

## Layers

| Input | Action |
|---|---|
| `L` | Open or close the layer chooser |
| `1` to `9`, `0` (chooser open) | Show or hide the layer with that number in the chooser |

`L` lists the build's layers in layer order with their number keys and
whether each is shown or hidden, on the overlay's last line:
`layers: 1:frames shown  2:base hidden`. The chooser stays open while
layers are toggled, and `L` closes it (`G` opens the spawn chooser
instead). While any layer is hidden, an overlay line names them:
`hidden layers: base`.

A toggle takes effect on the next frame (`rendering.md` section 7 item 7).
A hidden overlay layer's material is not composited. A hidden plain
layer's surface tiles are not drawn: where it gave a frame's base surface,
the next non-empty surface tile in layer order is drawn, and where there is
none the frame has no surface while it is hidden. A hidden plain layer's
cells are not drawn either: each cached cell keeps every layer's decoded
section and is composited again from the shown layers, with nothing
fetched again, and its mesh and far-field values follow within a frame or
two. Nothing computed changes, and input resolution is unaffected. An
overlay hidden at start (by the build's `shown` or by `--hide-layers`) is
not fetched while it is hidden; once shown it is wanted at once for every
tile the view draws and drawn on them as it arrives.

The layers hidden at start are the build's `shown: false` layers, or the
ones listed by `--hide-layers <id,id,...>` instead (`hideLayers` in the
browser). An id the build does not have is reported and ignored; an empty
list shows every layer.

## Compute budget

`--compute-budget <n>` (`computeBudget` in the browser) sets how many
module workers the client may run at start, 0 to the default (natively
cores - 2, 3/4 of them; one Web Worker in the browser); a larger value is
taken as the default. At 0 the client computes, checks, and uploads
nothing: it streams only what the hub already stores, and a layer the hub
does not store stays missing. `setComputeBudget` changes it while the
client runs, without restarting: a job already running finishes and is
kept, waiting jobs are dropped, and no upload is ever sent in part. No key
sets it.

## Spawning

| Input | Action |
|---|---|
| `G` | Open or close the spawn chooser |
| `1` to `9`, `0` (chooser open) | Spawn the camera onto the frame with that number in the chooser |
| `Home`, or a number key with the chooser closed | Free the camera from the frame again |

`G` lists the frames that have matter, in ascending frame id, with their
number keys, on the overlay's last line: `spawn: 1 frame 3  2 frame 5`. A
frame is listed once its depth-0 cell has arrived with matter in it.
Choosing one puts the camera on the line from that frame's origin toward
where the camera was, 2 m above the outermost matter the loaded cells show
on that line, looking level. As finer cells arrive (for up to 20 s, until
you fly) the point is found again with them.

A spawned camera is attached to its frame: it moves and turns with it, so
the surface stays under it, and the flight keys fly it relative to the
frame. Nothing else acts on it: no gravity, no collision, it flies through
matter like the free camera. The overlay says `camera on frame <id>`.
`Home` or a number-key view frees it, and it floats free in space again.

`--start-on <frame>:<x>,<y>,<z>` spawns the camera onto that frame along
the direction `x, y, z` of its axes as soon as the frame is listed. With
a distance, `--start-on <frame>:<x>,<y>,<z>@<m>`, it only places the free
camera `<m>` meters from the frame origin on that line, looking back at the
origin; `@<m>^<deg>` looks `<deg>` degrees below level instead (level is
toward the frame's `+z`). A headless run does the same (`--view-distance-scale` moves a
placed camera along the line).

## Simulation time

| Input | Action |
|---|---|
| `N` | Jump to now and follow it |
| `,` | Step one hour (3600 s) back, at most to the origin |
| `.` | Step one hour (3600 s) forward, at most to now |
| `[` | Halve the playback rate |
| `]` | Double the playback rate (does nothing at now) |
| `Backspace` | Play backwards at the current rate, or forwards again (a toggle) |
| `Space` | Pause or resume |

Time runs between the origin of the space (the moment its universe began)
and now (the hub's clock, measured at start and every minute so it does not
drift with this machine's clock). The client enters at now, and at now the
simulation time follows now at one second per second: the rate is 1 and
`]` does nothing. Any other key leaves now for the past: `,` steps back an
hour, `Backspace` runs time backwards, `Space` pauses, and `[` plays at half
speed, so the time falls behind now. In the past, `[` and `]` set the rate
between 1/1024 and 1048576 (2^20) simulated seconds per second, forwards or backwards.
Time never goes past now: playing forward into now, or stepping forward
past it, locks the clock to now again at rate 1. Going back stops at the
origin. `N` returns to now from anywhere.

Every jump costs at most one checkpoint interval (86 400 simulated seconds) of
integration, from the nearest checkpoint the renderer keeps.

The overlay's first line shows the simulation time as a UTC date and time,
followed by `now` at now or by the offset from now in seconds (`now -3600
s`). The second shows the rate, `backwards` while rewinding, and `(paused)`.
The third shows the light-travel time from the frame nearest the crosshair:
every frame is drawn where it was when the light now reaching the camera
left it, so that is how far in the past that frame is seen.

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
