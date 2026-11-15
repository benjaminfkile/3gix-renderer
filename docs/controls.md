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
| `1` to `9`, `0` | Jump to a view of the frame with that index (see below); with the spawn chooser open, spawn instead (see Walk) |
| `Home` | Jump to a view of the whole system from above the root frame |
| `Tab` | Switch between fly and walk (see Walk) |
| `G` | Open or close the spawn chooser (see Spawning) |
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
the same way it flies through empty space. After leaving walk mode it also coasts at the
velocity it had there, since nothing acts on it to change that (see Walk). The overlay's camera line
shows the current flight speed.

The number keys index the frames in registry order: frame ids sorted
ascending, skipping the root. `1` is the first, `9` the ninth, `0` the tenth.
A jump places the camera at `3 * root_extent / 8` from the frame origin,
above it the way `Home` sees the root: along root `+z`, looking along root
`-z` with root `+y` up. Because that direction is fixed in root axes, the
frame shows a lit side and a dark side, and the lit side turns as the light
moves over simulation time. The camera floats free where it lands, and the
body moves on along its orbit. A key past the last frame does nothing.

## Walk

| Input | Action in walk mode |
|---|---|
| `W` / `S` | Walk forward / backward along the ground, in the view's heading |
| `A` / `D` | Walk left / right |
| `Shift` (held) | Run: twice the walking speed, so twice the push from rest (at most Mach 4) |
| Mouse wheel | Set the walking speed: each notch multiplies or divides it by 1.25, from 0.1 m/s up to Mach 4 (1,372 m/s) |
| `V` | Type the walking speed (see Typing a speed) |
| `Space` | Jump: 3 m/s along up, only while standing on solid matter |
| Right mouse drag | Look around: yaw about up, pitch up and down, never past straight up or down |
| `Tab` | Back to fly mode |

`Tab` switches between the two modes. In fly mode nothing acts on the
camera (above). In walk mode the camera is a body the laws act on: an 80
kg, 1.75 m observer with a density of 985 kg/m^3 (the core's default
observer), pulled by the gravity of every massive frame, held up by solid
matter and carried along with it by friction (so it turns and orbits with
whatever it stands on), slowed by drag in fluid, gas, and plasma, and
pushed up by their buoyancy. The camera sits at eye height, 0.94 times the
observer's height, above its feet along the local up (against gravity), and
mouse look is relative to that up. Its heading turns with the matter it
stands on.

The walking keys push along the surface the observer stands on, toward
the set walking speed over the ground (twice it running), reaching it in
about a quarter of a second; friction stops the observer when they are
released. The walking speed starts at 1.4 m/s, a person's pace, and the
wheel or a typed speed sets it anywhere from 0.1 m/s to Mach 4 (4 times
343 m/s, the speed of sound in air at sea level and 20 degrees C). It is
kept across fly and walk. The laws still act: at high speed over uneven
ground the observer leaves the surface on the next rise and comes down
under gravity, and it outruns the cells being fetched under its feet, so
it waits (`waiting for matter under the observer`) until they arrive. Off the
ground (falling, jumping, or floating in gas) the walking keys do nothing.
`E` and `Q` do nothing in walk mode. `Space` jumps instead of
pausing; the other time keys work as in fly mode, and the observer stays
where it is on its frame across any jump in time. At a fast playback rate
the observer is stepped by at most 0.25 s of simulation time per frame and
carried with its frame for the rest.

Switching keeps the eye where it is and keeps the velocity: going to walk,
the camera's velocity (what it kept from an earlier walk plus the held
flight keys at the flight speed) becomes the observer's; going back to fly,
the observer's velocity becomes the camera's, and the camera coasts at it
over simulation time (the overlay's mode line shows it as `drift`). A
camera placed by a view key (`Home`, a number key) is at rest. Those keys
also leave walk mode.

The observer belongs to the frame the camera was parented to when walk
mode began, or the frame it spawned onto, and only that frame's matter
holds it up. The cells around its feet are fetched first, at every depth
down to the deepest any layer offers there. Until the deepest of them have
arrived the observer is held in place on its frame, neither falling nor
walking, and the overlay says `waiting for matter under the observer`.

The overlay's last lines show the mode: `mode fly`, or `mode walk frame
<id>` with `contact` (standing on solid matter) or `no contact`, the
speed over the matter, and the set walking speed (with its Mach number
from Mach 0.5 up).

## Typing a speed

`V` opens a speed entry on the overlay's last line: type a number of
meters per second (digits, the number row or the numpad, and one decimal
point), `Backspace` to correct, `Enter` to set it, `Escape` to cancel.
In walk mode it sets the walking speed (clamped to 0.1 m/s to Mach 4), in
fly mode the flight speed (clamped to 1e-3 to 1e15 m/s). While the entry
is open every other key waits.

## Spawning

| Input | Action |
|---|---|
| `G` | Open or close the spawn chooser |
| `1` to `9`, `0` (chooser open) | Spawn onto the frame with that number in the chooser |

`G` lists the frames that have matter, in ascending frame id, with their
number keys, on the overlay's last line: `spawn: 1 frame 3  2 frame 5`. A
frame is listed once any of its depth 0 or depth 1 cells has arrived with
matter in it; opening the chooser fetches those cells for every frame.
Choosing a frame closes the chooser and puts the observer, in walk mode,
on the line from that frame's origin to the camera, 2 m above the outermost
matter on that line, moving with that matter. If finer cells arrive while
it waits for them, the point is found again with them, so it starts above
the finest matter available. Then the laws take over: on solid matter the
observer comes to rest on the surface; in fluid, gas, or plasma it sinks
until buoyancy holds it, where the surrounding density is its own.

With the chooser closed the number keys are the fly views described above.

## Simulation time

| Input | Action |
|---|---|
| `N` | Jump to now and follow it |
| `,` | Step one hour (3600 s) back, at most to the origin |
| `.` | Step one hour (3600 s) forward, at most to now |
| `[` | Halve the playback rate |
| `]` | Double the playback rate (does nothing at now) |
| `Backspace` | Play backwards at the current rate, or forwards again (a toggle) |
| `Space` | Pause or resume (in walk mode `Space` jumps instead) |

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

Every jump costs at most one checkpoint interval (one simulated day) of
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
