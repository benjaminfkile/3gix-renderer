// Automatic exposure and the filmic tone curve.
//
// 1. cs_partial: every 16 x 16 block of the float target sums the natural
//    log of the luminance of its covered, lit pixels (alpha 1 and luminance
//    above MIN_LUMINANCE) and counts them. Uncovered pixels are empty space,
//    which is not drawn, and unlit ones carry no information about the
//    exposure; both would only drag the average toward black.
// 2. cs_adapt: one workgroup adds the partial sums in a fixed order. The
//    log-average luminance is exp(sum / count). The adapted luminance moves
//    toward it in log space by params.alpha (1 - exp(-dt / 0.5 s), or 1 to
//    jump straight there), and the exposure is KEY * 2^bias / adapted.
// 3. fs_tonemap: scales the float target by the exposure and applies the
//    filmic curve. The output target encodes sRGB.
//
// Luminance uses the Rec. 709 weights on the three bands (red = band 0).
// Every reduction runs in a fixed order, so the same image always gives
// the same exposure on one device.

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
// W m^-2 sr^-1. Far below anything visible; above thermal glow at room
// temperature in the visible bands.
const MIN_LUMINANCE: f32 = 1.0e-12;
// Middle grey: the log-average luminance maps here before the bias.
const KEY: f32 = 0.18;

struct State {
    // Adapted luminance, W m^-2 sr^-1; 0 until the first lit frame.
    adapted: f32,
    // Multiplier applied before the tone curve.
    exposure: f32,
    // Lit pixels counted this frame.
    pixels: f32,
    pad: f32,
};

struct Params {
    bias_stops: f32,
    alpha: f32,
    partial_count: u32,
    pad: u32,
};

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> state: State;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> applied: State;

var<workgroup> sums: array<f32, 256>;
var<workgroup> counts: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn cs_partial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let dims = textureDimensions(hdr);
    var s = 0.0;
    var c = 0.0;
    if gid.x < dims.x && gid.y < dims.y {
        let p = textureLoad(hdr, vec2<i32>(gid.xy), 0);
        let lum = dot(p.rgb, LUMA);
        if p.a > 0.0 && lum > MIN_LUMINANCE {
            s = log(lum);
            c = 1.0;
        }
    }
    sums[li] = s;
    counts[li] = c;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if li < stride {
            sums[li] += sums[li + stride];
            counts[li] += counts[li + stride];
        }
        workgroupBarrier();
    }
    if li == 0u {
        partials[wid.x + wid.y * groups.x] = vec2<f32>(sums[0], counts[0]);
    }
}

@compute @workgroup_size(64)
fn cs_adapt(@builtin(local_invocation_index) li: u32) {
    var s = 0.0;
    var c = 0.0;
    for (var i = li; i < params.partial_count; i += 64u) {
        s += partials[i].x;
        c += partials[i].y;
    }
    sums[li] = s;
    counts[li] = c;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if li < stride {
            sums[li] += sums[li + stride];
            counts[li] += counts[li + stride];
        }
        workgroupBarrier();
    }
    if li == 0u {
        var adapted = state.adapted;
        let total = counts[0];
        if total > 0.0 {
            let mean_log = sums[0] / total;
            if adapted > 0.0 && params.alpha < 1.0 {
                adapted = exp(mix(log(adapted), mean_log, params.alpha));
            } else {
                adapted = exp(mean_log);
            }
        }
        state.adapted = adapted;
        if adapted > 0.0 {
            state.exposure = KEY * exp2(params.bias_stops) / adapted;
        } else {
            state.exposure = exp2(params.bias_stops);
        }
        state.pixels = total;
        state.pad = 0.0;
    }
}

// The filmic curve: the rational fit of the ACES reference rendering
// transform by Krzysztof Narkowicz (2015),
//   f(x) = x (2.51 x + 0.03) / (x (2.43 x + 0.59) + 0.14),
// clamped to [0, 1]. It has a toe that deepens shadows slightly, is nearly
// linear through the mid tones (f(0.18) is about 0.27), and rolls off
// highlights so f(1) is about 0.80 and f(4) about 0.97, approaching 1.
fn filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // One triangle covering the target.
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_tonemap(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = textureLoad(hdr, vec2<i32>(pos.xy), 0);
    return vec4<f32>(filmic(p.rgb * applied.exposure), 1.0);
}
