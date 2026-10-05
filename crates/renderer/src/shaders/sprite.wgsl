// Point sprites of far frames.
//
// Each sprite is a disc `size_px` across at a camera-relative position, with
// a flat radiance in W m^-2 sr^-1 per band chosen so the disc delivers the
// frame's irradiance to the camera (src/farfield.rs). Sprites are added to
// the float target (additive blending) after volumes, tested against the
// surface depth without writing it. Alpha 1 marks a covered pixel for the
// exposure pass.

struct Globals {
    view_proj: mat4x4<f32>,
    // Render target size in pixels.
    viewport: vec2<f32>,
    pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct SpriteOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) radiance: vec3<f32>,
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
fn vs_sprite(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec3<f32>,
    @location(1) size_px: f32,
    @location(2) radiance: vec4<f32>,
) -> SpriteOut {
    let corner = CORNERS[vi];
    var clip = globals.view_proj * vec4<f32>(center, 1.0);
    let offset = corner * size_px / globals.viewport;
    clip = vec4<f32>(clip.xy + offset * clip.w, clip.z, clip.w);
    var out: SpriteOut;
    out.clip = clip;
    out.corner = corner;
    out.radiance = radiance.rgb;
    return out;
}

@fragment
fn fs_sprite(in: SpriteOut) -> @location(0) vec4<f32> {
    if dot(in.corner, in.corner) > 1.0 {
        discard;
    }
    return vec4<f32>(in.radiance, 1.0);
}
