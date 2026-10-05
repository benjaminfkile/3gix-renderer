// Lit surfaces of extracted matter.
//
// Output is radiance in W m^-2 sr^-1 per band into a float target: red is
// band 0 (600 to 700 nm), green band 1 (500 to 600 nm), blue band 2 (400 to
// 500 nm), the order of BAND_EDGES in gx-core (long to short wavelength).
// Alpha 1 marks a covered pixel for the exposure pass.
//
// Radiance leaving a surface point toward the camera, per band:
//
//   L = emission + sum over lights of (albedo / pi + specular) * E
//   E = I / d^2 * max(0, n . l)                       (irradiance, W m^-2)
//   emission = band_radiance(T) * (1 - albedo)        (from the lookup table)
//
// Specular is GGX: D * G * F / (4 (n . l) (n . v)) with alpha = roughness^2
// (at least MIN_ALPHA), Smith-Schlick G with k = alpha / 2, and Schlick F
// with a fixed F0 of 0.04. No shadows in v1.
//
// Every position is relative to the camera, in root axes. A mesh arrives
// relative to its cell origin and the model transform adds the rotation of
// its frame and the camera-relative cell origin.

const PI: f32 = 3.14159265358979;
const F0: f32 = 0.04;
const MIN_ALPHA: f32 = 0.002;
const MAX_LIGHTS: u32 = 8u;
const EMISSION_TABLE_WIDTH: u32 = 1024u;

struct Light {
    // Camera-relative position, root axes, meters; w unused.
    position: vec4<f32>,
    // Radiant intensity per band, W sr^-1; w unused.
    intensity: vec4<f32>,
};

struct Globals {
    view_proj: mat4x4<f32>,
    // x: number of lights in use.
    counts: vec4<u32>,
    lights: array<Light, 8>,
};

struct Model {
    // Columns of the rotation from frame axes to root axes.
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    // Camera-relative cell origin, root axes, meters.
    offset: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
// Band radiance per 1 K temperature bucket, W m^-2 sr^-1.
@group(0) @binding(1) var emission_table: texture_2d<f32>;
@group(1) @binding(0) var<uniform> model: Model;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) albedo: vec3<f32>,
    @location(3) roughness: f32,
    @location(4) emission_index: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) albedo: vec3<f32>,
    @location(3) roughness: f32,
    @location(4) emission: vec3<f32>,
};

@vertex
fn vs_surface(in: VsIn) -> VsOut {
    let rot = mat3x3<f32>(model.c0.xyz, model.c1.xyz, model.c2.xyz);
    let rel = rot * in.position + model.offset.xyz;
    let i = in.emission_index;
    let band = textureLoad(
        emission_table,
        vec2<u32>(i % EMISSION_TABLE_WIDTH, i / EMISSION_TABLE_WIDTH),
        0
    ).rgb;
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(rel, 1.0);
    out.rel = rel;
    out.normal = rot * in.normal;
    out.albedo = in.albedo;
    out.roughness = in.roughness;
    out.emission = band * (vec3<f32>(1.0) - in.albedo);
    return out;
}

@fragment
fn fs_surface(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let v = normalize(-in.rel);
    let alpha = max(in.roughness * in.roughness, MIN_ALPHA);
    let a2 = alpha * alpha;
    let k = 0.5 * alpha;
    let ndv = max(dot(n, v), 1.0e-4);
    let gv = ndv / (ndv * (1.0 - k) + k);
    var color = in.emission;
    let count = min(globals.counts.x, MAX_LIGHTS);
    for (var i = 0u; i < count; i++) {
        let light = globals.lights[i];
        let to_light = light.position.xyz - in.rel;
        let d2 = dot(to_light, to_light);
        if d2 <= 0.0 {
            continue;
        }
        let l = to_light * inverseSqrt(d2);
        let ndl = dot(n, l);
        if ndl <= 0.0 {
            continue;
        }
        let irradiance = light.intensity.rgb / d2 * ndl;
        let h = normalize(l + v);
        let ndh = max(dot(n, h), 0.0);
        let vdh = max(dot(v, h), 0.0);
        let dd = ndh * ndh * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * dd * dd);
        let g = gv * ndl / (ndl * (1.0 - k) + k);
        let f = F0 + (1.0 - F0) * pow(1.0 - vdh, 5.0);
        let specular = d * g * f / (4.0 * ndl * ndv);
        color += (in.albedo / PI + vec3<f32>(specular)) * irradiance;
    }
    return vec4<f32>(color, 1.0);
}
