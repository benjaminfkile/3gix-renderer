# Shading

How radiance from matter becomes pixels. The code is
`src/shaders/surface.wgsl`, `src/shaders/exposure.wgsl`, `src/light.rs`, and
`src/render/gpu.rs`. Everything here is physics applied to samples; nothing
depends on what the matter is.

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

1. Surface pass: meshes, one draw per cell, each with a dynamic-offset
   uniform holding the frame rotation and the camera-relative cell origin.
2. Exposure compute passes.
3. Display pass: the tone-mapped radiance as a full-screen triangle, then
   lines and markers against the surface depth, then the overlay text.
