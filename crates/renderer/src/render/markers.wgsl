// Frame origin markers and frame-to-parent lines.
//
// Every position arrives relative to the camera, in root axes, as f32.
// `view_proj` is the reversed-z infinite projection times the rotation from
// root axes into camera axes; there is no translation anywhere on the GPU.

struct Globals {
    view_proj: mat4x4<f32>,
    // Render target size in pixels.
    viewport: vec2<f32>,
    pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct MarkerOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec4<f32>,
};

// Two triangles covering the billboard square, corners at -1 and +1.
const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_marker(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec3<f32>,
    @location(1) size_px: f32,
    @location(2) color: vec4<f32>,
) -> MarkerOut {
    let corner = CORNERS[vi];
    var clip = globals.view_proj * vec4<f32>(center, 1.0);
    // Offset in screen space: half the diameter in pixels, scaled to clip
    // units at this w.
    let offset = corner * (0.5 * size_px) * 2.0 / globals.viewport;
    clip = vec4<f32>(clip.xy + offset * clip.w, clip.z, clip.w);
    var out: MarkerOut;
    out.clip = clip;
    out.corner = corner;
    out.color = color;
    return out;
}

@fragment
fn fs_marker(in: MarkerOut) -> @location(0) vec4<f32> {
    if dot(in.corner, in.corner) > 1.0 {
        discard;
    }
    return in.color;
}

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_line(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> LineOut {
    var out: LineOut;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return in.color;
}

// The crosshair at the screen center: the view direction, where the forward
// flight key moves the camera. Four arms around an open center, each a light
// bar over a slightly larger dark bar so it shows on bright and dark matter.
// No vertex buffer: 48 vertices, 8 bars of 6.

// Gap from the center to each arm, arm end, and bar half thicknesses, pixels.
const CROSS_GAP: f32 = 5.0;
const CROSS_END: f32 = 16.0;
const CROSS_HALF_LIGHT: f32 = 1.0;
const CROSS_HALF_DARK: f32 = 2.0;

struct CrossOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_crosshair(@builtin(vertex_index) vi: u32) -> CrossOut {
    let bar = vi / 6u;
    let dark = bar < 4u;
    let arm = bar % 4u;
    let pad = select(0.0, CROSS_HALF_DARK - CROSS_HALF_LIGHT, dark);
    let half = select(CROSS_HALF_LIGHT, CROSS_HALF_DARK, dark);
    // The bar along +x from the center, then turned to its arm.
    let c = CORNERS[vi % 6u] * 0.5 + vec2<f32>(0.5, 0.5);
    let along = mix(CROSS_GAP - pad, CROSS_END + pad, c.x);
    let across = mix(-half, half, c.y);
    var d = vec2<f32>(along, across);
    if arm == 1u {
        d = vec2<f32>(-d.y, d.x);
    } else if arm == 2u {
        d = -d;
    } else if arm == 3u {
        d = vec2<f32>(d.y, -d.x);
    }
    var out: CrossOut;
    out.clip = vec4<f32>(d * 2.0 / globals.viewport, 0.0, 1.0);
    out.color = select(vec4<f32>(0.9, 0.9, 0.9, 1.0), vec4<f32>(0.0, 0.0, 0.0, 1.0), dark);
    return out;
}

@fragment
fn fs_crosshair(in: CrossOut) -> @location(0) vec4<f32> {
    return in.color;
}
