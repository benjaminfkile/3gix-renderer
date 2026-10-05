# End to end run against the real hub and compiler

`scripts/e2e.sh` runs the renderer against the real hub and the real system
compiler inside one container. It copies the hub repository to `/tmp/hub`
and starts it with the hub's own harness (`scripts/e2e/hub-up.sh`: Redis, a
SeaweedFS S3 gateway, Postgres, the hub, and its seed), clones the system
compiler to `/tmp/compiler`, builds it and runs `system-compiler serve`
with the credentials from `/tmp/gx-e2e.env`, builds `gx-renderer` and
`tools/png-stats` in release mode, takes four headless screenshots at
640 x 360, checks them, and stops everything. The hub database and object
store are reset first, so every chunk is compiled during the run. Every
service listens on loopback only; the ids the hub prints (compiler, build)
are random per run.

The renderer learns nothing from the run: the only inputs are the view
flags below, and every other number comes from the registry the hub serves.

## Recorded run

| | |
|---|---|
| Date | 2026-10-05 |
| Hub commit | `88679d183ead901aa0d8b704bbcb0371e9cd95b9` |
| Compiler commit | `02e9e959d4fdcb2514fb85905a5511bddd492687` (branch `grunt`) |
| Core revision | `331ae53b12af71ea72d6e1d5dfe64f104df1a05b` (`Cargo.lock`) |
| Renderer | `6f047a60321dcd58418c0a9cfb89c7dc821f60d1` plus the change that adds this file |
| Graphics adapter | llvmpipe (software Vulkan) |
| Command | `sh scripts/e2e.sh` |
| Exit status | 1: 12 assertions passed, 1 failed (`home.png` blob count, see `Blocked`) |

Three runs from a reset hub gave byte-identical screenshots and the same
assertion results; only the timings differ. The statistics below are from
the last of them.

## Screenshots

Each screenshot is matter only (`--no-overlay`: no overlay text, frame
markers, or lines), waits until every cell it selected and every pinned far
field cell is ready (`--wait-ready-seconds 300`, which exits non-zero on
timeout), and writes its overlay statistics (`--stats-json`, the `.json`
file beside each image).

| File | Flags | What it is |
|---|---|---|
| `home.png` | `--view home` | the `Home` view of the whole system at launch time |
| `frame-3.png` | `--view 3` | the view number key 3 jumps to (the third non-root frame by ascending id) at launch time |
| `frame-3-plus-90d.png` | `--view 3 --start-offset-seconds 7776000` | the same view 90 days (7776000 s) after the epoch |
| `frame-1-close.png` | `--view 1 --view-distance-scale 0.25` | number key 1's view at a quarter of its distance |

A number key view looks at its frame from above, along root `-z` with root
`+y` up, from `3 * root_extent / 8` (`docs/controls.md`). Because that
direction is fixed in root axes, the lit side of frame 3 turns as the frame
moves around the hot frame over the 90 days.

![home](home.png)
![frame 3](frame-3.png)
![frame 3, 90 days later](frame-3-plus-90d.png)
![frame 1 close](frame-1-close.png)

## Assertions

`tools/png-stats` measures luma (`0.2126 R + 0.7152 G + 0.0722 B` of the
8-bit values, 0 to 255). Output of the recorded run:

```
PASS screenshot home.png: every selected cell ready
PASS screenshot frame-3.png: every selected cell ready
PASS screenshot frame-3-plus-90d.png: every selected cell ready
PASS screenshot frame-1-close.png: every selected cell ready
FAIL blobs home.png: 1 components above luma 32 of at least 1 px (need 9), sizes [32]
PASS disc frame-3.png: largest of 15 components above luma 32 covers 5.83 percent (need 5 to 60), centroid (379.6, 190.2), brighter side mean 184.3, darker side mean 142.4, ratio 1.29 (need 1.2), lit toward (0.77, -0.64)
PASS disc frame-3-plus-90d.png: largest of 17 components above luma 32 covers 11.13 percent (need 5 to 60), centroid (295.5, 161.2), brighter side mean 180.6, darker side mean 119.4, ratio 1.51 (need 1.2), lit toward (-0.99, -0.16)
PASS moved frame-3.png -> frame-3-plus-90d.png: bright centroid (372.5, 183.3) over 16019 px -> (297.6, 162.4) over 26375 px, shift 77.8 px (need 16)
PASS smooth frame-1-close.png: luma p1 77.4 p99 190.2, range 112.8 (need 32), largest neighbor step 1.7 at (350, 81), 0.015 of the range (need at most 0.25)
PASS size home.png: 1779 bytes (limit 307200)
PASS size frame-3.png: 31156 bytes (limit 307200)
PASS size frame-3-plus-90d.png: 40869 bytes (limit 307200)
PASS size frame-1-close.png: 42677 bytes (limit 307200)
e2e: 12 passed, 1 failed
```

What each check means:

- **blobs**: 8-connected components of pixels above luma 32; at least 9.
- **disc**: the largest component above luma 32 covers 5 to 60 percent of
  the image, and it leans to one side. It is split by the line through its
  centroid perpendicular to the offset of its luma-weighted centroid, and the
  brighter half must be at least 1.2 times as bright as the darker half. A
  disc lit face on has no lean and a ratio of 1. The unlit half of the disc
  is black like the background, so the component is the lit part. The dark
  creases where cells meet (each cell's mesh closes in a wall at the cell
  boundary, `src/extract.rs`) split the lit part into a few components; the
  largest still passes.
- **moved**: the centroid of the pixels above luma 32 moves at least 16 px
  between the two views of frame 3. It moved 77.8 px: the lit side turned
  from the right (`lit toward (0.77, -0.64)`) to the left
  (`(-0.99, -0.16)`), so simulation time advanced the frames.
- **smooth**: the luma falls by at least 32 levels across the image and no
  step between neighboring pixels exceeds a quarter of that fall. The close
  view of frame 1 is drawn by 198 volumes and no surface; its largest step is
  1.7 levels over a range of 112.8. For comparison, `frame-3.png` (a
  surface against black) has a largest step of 211.9 over a range of 202.6.

## Statistics

From `--stats-json` (the overlay statistics of the rendered frame).
`selected` is the cell selection; `pinned` are the depth-0 cells kept for
far field sprites; `requests` counts every chunk request, including the
`202` answers; `fetched` counts requests answered `200` with a chunk body;
`bytes` is the body bytes received; the round trips are of the first and of
the latest request; `streamed` is the wall time from the selection to the
draw.

| Screenshot | Selected | Pinned | Requests | Fetched | Bytes | First round trip | Last round trip | Streamed |
|---|---|---|---|---|---|---|---|---|
| `home.png` | 1 | 10 | 22 | 11 | 2289 | 0.0141 s | 0.0344 s | 10.2 s |
| `frame-3.png` | 513 | 9 | 1034 | 522 | 97865 | 0.0175 s | 0.0057 s | 10.4 s |
| `frame-3-plus-90d.png` | 513 | 9 | 942 | 522 | 97865 | 0.0062 s | 0.0039 s | 10.4 s |
| `frame-1-close.png` | 513 | 9 | 1034 | 522 | 958810 | 0.0053 s | 0.0384 s | 19.0 s |

Draw counts: `home.png` 5 sprites and 1 light; both views of frame 3 have 8
meshes (41008 triangles), 9 sprites, and 1 light; `frame-1-close.png` has
198 volumes, 9 sprites, and 1 light.

Every screenshot streamed for about 10 s, and the requests are about twice
the fetched cells. A cell the hub had not compiled yet is answered `202`.
The renderer then subscribes to it on the readiness socket, but the
compiler stores most sections within 100 ms (its log shows 3 to 84 ms
per section), often before the
subscription arrives. Those pushes never come, so the cell is polled once
its 10 s push wait (`stream::PENDING_POLL_AFTER_SECONDS`) is over. The
compiler stored 1456 sections during the run.

## Blocked

### `home.png`: at least 9 distinct bright blobs

This assertion fails, and the screenshot cannot pass it. Neither the hub nor
the compiler misbehaved: both served everything the renderer asked for, and
every selected and pinned cell was ready before the draw. The `Home` view
cannot show 9 separate bright blobs at 640 x 360, for two independent
reasons.

**Geometry.** `Home` looks down root `-z` from the height that fits every
frame's origin and `root_extent / 8` region into the field of view. The
script prints where each frame's origin projects to, from the registry the hub
served (positions at the epoch, a child's added to its parent's):

```
home: frame 1 projects to (-0.04, +0.01) px from the image center, 0.04 px away
home: frame 2 projects to (-0.68, +2.00) px from the image center, 2.11 px away
home: frame 3 projects to (-3.61, +0.24) px from the image center, 3.62 px away
home: frame 4 projects to (-0.92, -4.43) px from the image center, 4.53 px away
home: frame 5 projects to (+6.89, +0.01) px from the image center, 6.89 px away
home: frame 6 projects to (+20.24, -13.85) px from the image center, 24.53 px away
home: frame 7 projects to (+33.07, -31.90) px from the image center, 45.95 px away
home: frame 8 projects to (+65.86, +57.11) px from the image center, 87.17 px away
home: frame 9 projects to (+72.34, +98.93) px from the image center, 122.56 px away
home: frame 10 projects to (-0.93, -4.42) px from the image center, 4.52 px away
```

Frames 1 to 5 and 10 all project within 7 px of the center, and frame 1's
sprite is 6 px across. Frames 4 and 10 project about 0.01 px apart. Even if every
frame were bright, at most frames 6, 7, 8, 9, and a few small groups near
the center could be separate: fewer than 9.

**Brightness.** Frame 1 is the only hot frame (its depth-0 cell has the
only emitter, about 5.4e26 W over the three bands). At the `Home` camera,
about 9.37e12 m above the root, it delivers about 0.49 W m^-2. Every other
frame only reflects that light (`docs/shading.md`, far field): even with
albedo 1, fully lit, and its whole `root_extent / 8` disc facing the camera,
the brightest of them (frame 6) delivers about 4.5e-9 W m^-2, 9e-9 of frame
1's, and the others less. The exposure adapts to frame 1, so their sprites
quantize to 0 in the 8-bit image: above luma 0 there are 32 pixels, all in
frame 1's sprite. The overlay counts 5 sprites drawn.

Making the blobs appear would mean drawing cold frames brighter than the
light they reflect, or a camera that is not a perspective camera, and the
renderer would then show something other than what the hub serves. The
assertion is left as written and failing, and `sh scripts/e2e.sh` exits 1
because of it.

### Upstream observation, not blocking

The hub logs an unhandled exception each time a renderer process exits
with its readiness socket open (twice in the recorded run):

```
fail: Microsoft.AspNetCore.Diagnostics.DeveloperExceptionPageMiddleware[1]
      An unhandled exception has occurred while executing the request.
      System.Net.WebSockets.WebSocketException (0x80004005): The WebSocket is in an invalid state ('Aborted') for this operation. Valid states are: 'Open, CloseReceived, CloseSent'
         at System.Net.WebSockets.ManagedWebSocket.CloseAsync(WebSocketCloseStatus closeStatus, String statusDescription, CancellationToken cancellationToken)
         at ThreeGixHub.Controllers.ChunkSubscriptionController.ReceiveLoopAsync(WebSocket socket, Guid buildId, CancellationToken cancellationToken) in /tmp/hub/3GIXHub/Controllers/ChunkSubscriptionController.cs:line 75
```

The receive loop answers a close frame with `CloseAsync` after the client
has already gone. It only affects the hub's log; every request in the run
was answered.
