// Volumetric gas and plasma, ray marched per cell.
//
// The emission and extinction model of gx-core (matter-format.md section
// 3.3, gx_core::extinction): extinction per meter is k = density *
// attenuation, looked up from the nearest sample; emission is the blackbody
// band radiance of the sample's 1 K temperature bucket (from the emission
// table) times 1 - albedo. Output is radiance in W m^-2 sr^-1 per band into
// the float target, red = band 0.
//
// Each cell draws the back faces of the box around its gas and plasma
// samples, so every covered pixel runs exactly one ray whether the camera
// is outside or inside the box. The ray from the camera is clipped to the
// box and to the opaque surface depth (read from the surface pass's depth
// buffer; nothing here writes depth), split into STEPS equal steps, and
// marched front to back:
//
//   L += T * E * k * ds
//   T *= exp(-k * ds)
//
// The fragment is (w * L, w * (1 - T)) with w the far field weight, blended
// as dst = src + (1 - src.a) * dst: the volume's glow plus what is behind
// it, dimmed by its transmittance. Volumes are drawn back to front.
//
// Every position is relative to the camera, in root axes. The box arrives
// relative to its cell origin, and the model adds the frame rotation and
// the camera-relative cell origin, as for surfaces.

const STEPS: u32 = 64u;
const EMISSION_TABLE_WIDTH: u32 = 1024u;
// Direction components smaller than this are treated as this, so the slab
// test never divides by zero.
const MIN_DIR: f32 = 1.0e-30;

struct Globals {
    view_proj: mat4x4<f32>,
    // xyz: the view direction (camera -z) in root axes; w: the near plane,
    // meters.
    forward: vec4<f32>,
};

struct Volume {
    // Columns of the rotation from frame axes to root axes.
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    // xyz: camera-relative cell origin, root axes, meters; w: weight.
    offset: vec4<f32>,
    // xyz: smallest corner of the gas box relative to the cell origin,
    // frame axes, meters; w: cell edge, meters.
    box_min: vec4<f32>,
    // xyz: largest corner of the gas box; w: samples per axis.
    box_max: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
// Band radiance per 1 K temperature bucket, W m^-2 sr^-1.
@group(0) @binding(1) var emission_table: texture_2d<f32>;
// Reversed-z depth of the opaque surfaces: near / distance, 0 where none.
@group(0) @binding(2) var surface_depth: texture_depth_2d;
@group(1) @binding(0) var<uniform> volume: Volume;
// Per sample: density (kg m^-3), attenuation (m^2 kg^-1), emission table
// index, temperature (K).
@group(1) @binding(1) var params: texture_3d<f32>;
// Per sample: albedo per band.
@group(1) @binding(2) var albedo: texture_3d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
};

@vertex
fn vs_volume(@location(0) corner: vec3<f32>) -> VsOut {
    let rot = mat3x3<f32>(volume.c0.xyz, volume.c1.xyz, volume.c2.xyz);
    let local = mix(volume.box_min.xyz, volume.box_max.xyz, corner);
    let rel = rot * local + volume.offset.xyz;
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(rel, 1.0);
    out.rel = rel;
    return out;
}

fn emitted(index: u32, a: vec3<f32>) -> vec3<f32> {
    let band = textureLoad(
        emission_table,
        vec2<u32>(index % EMISSION_TABLE_WIDTH, index / EMISSION_TABLE_WIDTH),
        0
    ).rgb;
    return band * (vec3<f32>(1.0) - a);
}

fn safe_dir(d: f32) -> f32 {
    if abs(d) < MIN_DIR {
        return select(-MIN_DIR, MIN_DIR, d >= 0.0);
    }
    return d;
}

@fragment
fn fs_volume(in: VsOut) -> @location(0) vec4<f32> {
    let rot = mat3x3<f32>(volume.c0.xyz, volume.c1.xyz, volume.c2.xyz);
    let to_frame = transpose(rot);
    let dir_root = normalize(in.rel);
    // The camera and the ray in the cell's frame, relative to its origin.
    let eye = to_frame * (-volume.offset.xyz);
    let dir = to_frame * dir_root;
    let inv = vec3<f32>(1.0 / safe_dir(dir.x), 1.0 / safe_dir(dir.y), 1.0 / safe_dir(dir.z));
    let u = (volume.box_min.xyz - eye) * inv;
    let v = (volume.box_max.xyz - eye) * inv;
    let lo = min(u, v);
    let hi = max(u, v);
    var t0 = max(max(lo.x, lo.y), max(lo.z, 0.0));
    var t1 = min(hi.x, min(hi.y, hi.z));

    // Stop at the opaque surface in front, if any.
    let depth = textureLoad(surface_depth, vec2<i32>(in.clip.xy), 0);
    let along = dot(dir_root, globals.forward.xyz);
    if depth > 0.0 && along > 0.0 {
        t1 = min(t1, globals.forward.w / depth / along);
    }
    if t1 <= t0 {
        discard;
    }

    let n = i32(volume.box_max.w);
    let cell = volume.box_min.w / volume.box_max.w;
    let ds = (t1 - t0) / f32(STEPS);
    var radiance = vec3<f32>(0.0);
    var t = 1.0;
    for (var i = 0u; i < STEPS; i++) {
        let p = eye + dir * (t0 + (f32(i) + 0.5) * ds);
        let idx = clamp(vec3<i32>(floor(p / cell)), vec3<i32>(0), vec3<i32>(n - 1));
        let s = textureLoad(params, idx, 0);
        let k = s.x * s.y;
        if k <= 0.0 {
            continue;
        }
        let a = textureLoad(albedo, idx, 0).rgb;
        radiance += t * emitted(u32(s.z), a) * k * ds;
        t *= exp(-k * ds);
    }
    let w = volume.offset.w;
    return vec4<f32>(w * radiance, w * (1.0 - t));
}
