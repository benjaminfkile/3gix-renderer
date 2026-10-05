# Shading

How radiance from matter becomes pixels. The code is
`src/shaders/surface.wgsl`, `src/shaders/volume.wgsl`,
`src/shaders/sprite.wgsl`, `src/shaders/exposure.wgsl`, `src/light.rs`,
`src/volume.rs`, `src/farfield.rs`, and `src/render/gpu.rs`. Everything here
is physics applied to samples; nothing depends on what the matter is.

## Bands to channels

Format version 1 carries three wavelength bands (`gx_core::radiance::BAND_EDGES`,
long to short wavelength). Every radiance in the renderer is per band, and
the bands map to the output channels in that order:

| Band | Wavelengths | Channel |
|---|---|---|
| 0 | 600 to 700 nm | red |
| 1 | 500 to 600 nm | green |
| 2 | 400 to 500 nm | blue |

Albedo uses the same order, so albedo band 0 is the red reflectance.

## Lighting model

The surface pass writes radiance in W m^-2 sr^-1 per band into an
`Rgba32Float` target. For a surface point with unit normal `n`, view
direction `v`, albedo `a`, and roughness `r`:

```
L = emission + sum over lights of (a / pi + specular) * E
E = I / d^2 * max(0, n . l)
```

- **Lights.** Point lights from hot matter (`docs/architecture.md`, step 5):
  radiant intensity `I = band_power / (4 pi)` W/sr per band. `d` and `l` come
  from the light position relative to the camera, computed in `f64` on the
  CPU and handed to the shader as `f32`. At most eight lights; no shadows in
  v1.
- **Diffuse.** Lambert, `a / pi` per band.
- **Specular.** GGX microfacet: `D * G * F / (4 (n . l) (n . v))` with
  `alpha = r^2` (at least 0.002 so a perfectly smooth surface stays finite),
  `D = alpha^2 / (pi ((n . h)^2 (alpha^2 - 1) + 1)^2)`, Smith-Schlick
  `G` with `k = alpha / 2`, and Schlick `F` with a fixed `F0` of 0.04. The
  specular term is the same in every band.
- **Emission.** A hot surface glows whether or not any light reaches it:
  `emitted_band_radiance(T, a) = band_radiance(T) * (1 - a)` from `gx-core`.
  `gx-core` is `f64` CPU code, so the renderer evaluates `band_radiance` once
  per 1 K temperature bucket that appears in a drawn mesh and uploads the
  values as a small lookup texture (`light::EmissionTable`, 1024 texels per
  row, append only). Each vertex carries the index of its bucket; the vertex
  shader looks the radiance up and multiplies by `1 - albedo`.

Surface normals are the negated density gradient, so they point from dense
to thin matter. Faces seen from inside (from denser matter) are culled.

## Surfaces and volumes

The state of a sample decides how it is drawn:

| State | Drawn as |
|---|---|
| solid, fluid | a surface: marching cubes at half the largest solid or fluid density of the cell (`docs/architecture.md`) |
| gas, plasma | a volume: ray marched |

Extraction counts gas and plasma as vacuum and the volume counts solid and
fluid as vacuum, so a cell whose densest sample is gas or plasma is drawn
as a volume, and a cell that holds both kinds draws both: the mesh of its
solid and fluid samples and the volume of the rest.

## Volume model

The model is the emission and extinction of `matter-format.md` section 3.3,
with the extinction exactly as in `gx-core`'s `extinction.rs`:

- Extinction per meter `k = density * attenuation`, from the sample whose
  sub-cube contains the point (nearest sample, the lookup of
  `gx_core::extinction::optical_depth_along`).
- Transmittance over a step `ds`: `exp(-k ds)` (Beer-Lambert,
  `gx_core::extinction::transmittance`).
- Emission `E = band_radiance(T) * (1 - albedo)`, with `band_radiance` from
  the same emission table as surfaces (1 K buckets) and the albedo of the
  sample.

Each cell with gas or plasma uploads its density, attenuation, emission
table index (from the temperature), temperature, and albedo as two
`Rgba32Float` 3D textures of `n^3` texels (`n` up to 64), and draws the back
faces of the box around its gas and plasma samples, so every covered pixel
runs one ray whether the camera is outside the box or inside it. The ray
from the camera is clipped to that box (outside it the cell is vacuum and
adds nothing) and to the opaque surface depth from the surface pass, then
split into 64 equal steps and marched front to back from `L = 0`, `T = 1`,
sampling at the middle of each step:

```
L += T * E * k * ds
T *= exp(-k * ds)
```

The fragment is `(w L, w (1 - T))` with `w` the far field weight (1 unless
the frame is fading in, below), blended as `dst = src + (1 - src.a) dst`:
the volume's glow plus what is behind it, dimmed by its transmittance. The
alpha channel then holds the coverage, so the exposure pass counts volume
pixels.

For uniform matter of depth `D` the march tends to `E (1 - exp(-k D))`.
With 64 steps the midpoint sum is high by about `k ds / 2`: 0.8 percent at
an optical depth of 1. `tests/volume.rs` renders a uniform cell of plasma
at 6000 K headless and checks the center pixel of the float target against
that analytic result within 2 percent, and against the CPU mirror of the
march (`VolumeGrid::march`) within 0.1 percent.

The nearest-sample lookup makes a volume blocky at the scale of its samples;
the compiler's resolution decides how fine it is, as for surfaces.

## Far field

A frame with mass above 0 whose `root_extent / 8` region projects to `p`
pixels (region over distance to its origin, times the focal length in
pixels) is drawn as follows:

| `p` | Drawn as |
|---|---|
| under 2 | one point sprite; none of its cells are selected |
| 2 to 8 | the sprite with weight `(8 - p) / 6`, its cells with the rest |
| 8 and up | its cells only |

The cells' weight multiplies the radiance of their surfaces and the glow
and opacity of their volumes, so the sprite blends out as the depth-0 mesh
or volume blends in.

The sprite's brightness comes from the frame's depth-0 cell, which the cell
cache fetches once for this purpose and keeps (pinned, never evicted):

- **Hot** (the cell has an emitter, see "Lights" above): the irradiance of
  a point source of intensity `I = band_power / (4 pi)` at the camera
  distance `D`: `E = I / D^2` per band. The emitter of a sprite-only frame
  also stays a point light, so a far hot frame keeps lighting the rest.
- **Cold**: the active lights reflected by a Lambertian disc of radius
  `R = root_extent / 8` facing each light, with the mass-weighted mean
  albedo `a` of the cell: `E = sum a E_l R^2 max(0, cos theta) / D^2`, with
  `E_l` the irradiance from the light at the frame and `theta` the angle at
  the frame between the light and the camera. A frame seen from its lit side
  is a dot, one seen from its dark side or with no light is nothing.

The sprite is a disc `s` pixels across, `s = 2 + log10(Y / 1e-8 W m^-2)`
clamped to 2 to 6 (`Y` the luminance-weighted irradiance), with a flat
radiance `L = E f^2 / (pi (s / 2)^2)` per band, `f` the focal length in
pixels: each pixel subtends `1 / f^2` steradians, so the disc delivers `E`.
Sprites are added on top of surfaces and volumes (additive blending),
tested against the surface depth, and mark their pixels as covered for the
exposure.

## Float blending

Volumes and sprites blend into the `Rgba32Float` radiance target, which
needs the `FLOAT32_BLENDABLE` device feature. Desktop drivers, the software
Vulkan driver, and current WebGPU browsers have it, and the device is opened
with it when the adapter offers it. Without it the two passes write without
blending, so a volume hides what is behind it and sprites replace the pixel
under them.

## Exposure

The radiance target spans many orders of magnitude, so exposure is
automatic:

1. A compute pass sums `ln(Y)` over every covered pixel whose luminance `Y`
   is above 1e-12 W m^-2 sr^-1, in 16 x 16 blocks, with
   `Y = 0.2126 R + 0.7152 G + 0.0722 B` (Rec. 709 weights on the bands).
   Empty space is not drawn and unlit matter carries no information about
   the exposure; counting either would drag the average toward black.
2. A second pass adds the block sums in a fixed order:
   `Y_avg = exp(sum / count)`, the log-average luminance.
3. The adapted luminance moves toward `Y_avg` in log space by
   `alpha = 1 - exp(-dt / 0.5 s)` each frame (`dt` is the wall-clock frame
   time), so it settles with a 0.5 s time constant. Headless renders jump
   straight to `Y_avg`, so one state always gives one image.
4. The exposure is `0.18 * 2^bias / Y_adapted`: the log-average maps to
   middle grey, and the user bias (`+` and `-`, half a stop per press,
   between -16 and +16 stops) shifts it.

## Tone curve

The exposed radiance goes through a filmic curve per channel, the rational
fit of the ACES reference rendering transform by Krzysztof Narkowicz (2015):

```
f(x) = x (2.51 x + 0.03) / (x (2.43 x + 0.59) + 0.14), clamped to [0, 1]
```

| x | f(x) |
|---|---|
| 0.05 | 0.044 |
| 0.18 | 0.27 |
| 0.5 | 0.62 |
| 1 | 0.80 |
| 4 | 0.97 |

It has a slight toe that deepens shadows, runs close to linear through the
mid tones, and rolls the highlights off toward 1 instead of clipping, so a
hot surface many times brighter than its lit surroundings still reads as a
bright, unclipped shape. The display target is sRGB, so the hardware applies
the sRGB transfer function after the curve.

## Depth

Surfaces use the reversed-z infinite projection of `docs/architecture.md`
(`Depth32Float`, clear 0, greater-or-equal test). The surface pass writes
depth; the display pass loads it so markers and lines are hidden behind
matter.

## Order of the frame

1. Surface pass: opaque meshes with depth, one draw per cell, each with a
   dynamic-offset uniform holding the frame rotation, the camera-relative
   cell origin, and the far field weight.
2. Volume pass: volumes sorted back to front by the camera distance to the
   center of their gas box, blended over the surfaces, each ray stopping at
   the surface depth (read as a texture: depth testing against the mesh
   depth buffer, no depth writes).
3. Sprite pass: far field sprites, added, tested against the surface depth,
   no depth writes.
4. Exposure compute passes.
5. Display pass: the tone-mapped radiance as a full-screen triangle, then
   lines and markers against the surface depth, then the overlay text.
