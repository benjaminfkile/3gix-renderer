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
