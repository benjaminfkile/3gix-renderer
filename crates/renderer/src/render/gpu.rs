//! The wgpu renderer: lit matter surfaces into a float target, automatic
//! exposure, the tone curve, then frame markers, frame-to-parent lines, and
//! the overlay text.
//!
//! One frame is:
//!
//! 1. the surface pass: every drawn cell's mesh with `shaders/surface.wgsl`
//!    into an `Rgba32Float` target holding radiance per band, with the
//!    reversed-z `Depth32Float` depth buffer cleared to 0 and a
//!    greater-or-equal test (see [`crate::render::scene`]);
//! 2. the exposure compute passes of `shaders/exposure.wgsl`: log-average
//!    luminance and adaptation;
//! 3. the display pass into the color target (a window surface or the
//!    offscreen texture of [`crate::render::headless`]): the tone-mapped
//!    float target, then markers and lines tested against the surface
//!    pass's depth, then the overlay text.
//!
//! The lighting model, exposure, and tone curve are described in
//! `docs/shading.md`.

use super::overlay::{MARGIN_PX, TEXT_COLOR, TEXT_PX};
use super::scene::Scene;
use crate::extract::SurfaceMesh;
use crate::light::{EmissionTable, EMISSION_TABLE_WIDTH, MAX_LIGHTS};
use bytemuck::{Pod, Zeroable};
use gx_core::key::CellKey;
use std::collections::BTreeMap;
use wgpu::util::DeviceExt;
use wgpu_text::glyph_brush::ab_glyph::FontRef;
use wgpu_text::glyph_brush::{Section, Text};
use wgpu_text::{BrushBuilder, TextBrush};

/// Format of the depth buffer: 32-bit float, reversed-z.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Format of the surface pass target: radiance per band, W m^-2 sr^-1.
pub const RADIANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The value the depth buffer clears to: the far end in reversed-z.
pub const DEPTH_CLEAR: f32 = 0.0;

/// The color the target clears to. Space itself is not drawn
/// (`space-model.md` section 1, rule 3), so it stays black.
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color::BLACK;

/// The overlay typeface, embedded so every machine draws the same glyphs.
pub const OVERLAY_FONT: &[u8] = epaint_default_fonts::HACK_REGULAR;

/// Time constant of the exposure adaptation, seconds.
pub const ADAPTATION_SECONDS: f64 = 0.5;

/// Exposure bias change per `+` or `-` key press, stops.
pub const EXPOSURE_STEP_STOPS: f64 = 0.5;

/// Pixels per side of one block of the luminance reduction.
const LUMINANCE_BLOCK: u32 = 16;

/// Bytes between per-draw model transforms in the dynamic uniform buffer.
const MODEL_STRIDE: u64 = 256;

/// Frames a mesh may go undrawn before its GPU buffers are dropped.
const MESH_IDLE_FRAMES: u64 = 600;

/// How one frame is exposed.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Exposure {
    /// User exposure bias, stops (`+` and `-` keys).
    pub bias_stops: f64,
    /// Wall-clock seconds since the previous frame, for the adaptation; or
    /// `None` to jump straight to this frame's luminance (headless renders,
    /// so one render of one state is always the same image).
    pub adapt_seconds: Option<f64>,
}

impl Exposure {
    /// The fraction of the way, in log luminance, the adapted value moves
    /// toward this frame's: `1 - exp(-dt / 0.5 s)`, or 1 without a delta.
    pub fn alpha(&self) -> f32 {
        match self.adapt_seconds {
            None => 1.0,
            Some(dt) => (1.0 - (-dt.max(0.0) / ADAPTATION_SECONDS).exp()) as f32,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LightGpu {
    position: [f32; 4],
    intensity: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SurfaceGlobals {
    view_proj: [[f32; 4]; 4],
    counts: [u32; 4],
    lights: [LightGpu; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ModelGpu {
    columns: [[f32; 4]; 3],
    offset: [f32; 4],
}

/// One vertex of a surface mesh as the GPU reads it.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SurfaceGpuVertex {
    /// Relative to the cell origin, frame axes, meters.
    position: [f32; 3],
    normal: [f32; 3],
    albedo: [f32; 3],
    roughness: f32,
    emission_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ExposureParams {
    bias_stops: f32,
    alpha: f32,
    partial_count: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MarkerInstance {
    center: [f32; 3],
    size_px: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LineGpuVertex {
    position: [f32; 3],
    color: [f32; 4],
}

/// A mesh uploaded to the GPU.
struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    last_drawn: u64,
}

/// What the last frame drew.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Meshes drawn.
    pub meshes: usize,
    /// Triangles drawn.
    pub triangles: usize,
    /// Point lights in use.
    pub lights: usize,
}

/// The size-dependent targets: depth, the float radiance target, and the
/// bind groups that read it.
struct Targets {
    depth: wgpu::TextureView,
    radiance: wgpu::TextureView,
    partial_count: u32,
    groups: (u32, u32),
    exposure_bind: wgpu::BindGroup,
    tonemap_bind: wgpu::BindGroup,
}

/// The pipelines and buffers of the frame.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    frame: u64,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    marker_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    markers: GrowBuffer,
    lines: GrowBuffer,
    surface_pipeline: wgpu::RenderPipeline,
    surface_globals: wgpu::Buffer,
    surface_layout: wgpu::BindGroupLayout,
    surface_bind: wgpu::BindGroup,
    model_layout: wgpu::BindGroupLayout,
    models: wgpu::Buffer,
    model_bind: wgpu::BindGroup,
    emission: EmissionTable,
    emission_texture: wgpu::Texture,
    emission_uploaded: usize,
    meshes: BTreeMap<CellKey, GpuMesh>,
    exposure_layout: wgpu::BindGroupLayout,
    partial_pipeline: wgpu::ComputePipeline,
    adapt_pipeline: wgpu::ComputePipeline,
    exposure_state: wgpu::Buffer,
    exposure_params: wgpu::Buffer,
    exposure_applied: wgpu::Buffer,
    tonemap_layout: wgpu::BindGroupLayout,
    tonemap_pipeline: wgpu::RenderPipeline,
    targets: Targets,
    stats: FrameStats,
    brush: TextBrush<FontRef<'static>>,
}

/// A vertex buffer that grows to fit what is written to it.
struct GrowBuffer {
    buffer: wgpu::Buffer,
    label: &'static str,
    len: u32,
}

impl GrowBuffer {
    fn new(device: &wgpu::Device, label: &'static str) -> GrowBuffer {
        GrowBuffer {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 256,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            label,
            len: 0,
        }
    }

    fn write<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, items: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(items);
        if bytes.len() as u64 > self.buffer.size() {
            self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(self.label),
                contents: bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        } else if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytes);
        }
        self.len = items.len() as u32;
    }
}

fn texture_2d(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    dynamic: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

fn float_texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn emission_texture(device: &wgpu::Device, rows: u32) -> wgpu::Texture {
    texture_2d(
        device,
        "emission table",
        EMISSION_TABLE_WIDTH,
        rows.max(1),
        wgpu::TextureFormat::Rgba32Float,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    )
}

impl Renderer {
    /// Builds the pipelines for color targets of `format` at the given size.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Renderer {
        let (width, height) = (width.max(1), height.max(1));
        let shader = device.create_shader_module(wgpu::include_wgsl!("markers.wgsl"));
        let surface_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/surface.wgsl"));
        let exposure_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/exposure.wgsl"));
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, false)],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("markers"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let depth_test = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let no_depth = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let pipeline = |label: &str,
                        vs: &str,
                        fs: &str,
                        buffer: wgpu::VertexBufferLayout<'_>,
                        topology: wgpu::PrimitiveTopology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    buffers: &[Some(buffer)],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: Some(depth_test.clone()),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                cache: None,
                multiview_mask: None,
            })
        };
        let marker_pipeline = pipeline(
            "markers",
            "vs_marker",
            "fs_marker",
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<MarkerInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x4],
            },
            wgpu::PrimitiveTopology::TriangleList,
        );
        let line_pipeline = pipeline(
            "lines",
            "vs_line",
            "fs_line",
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineGpuVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
            },
            wgpu::PrimitiveTopology::LineList,
        );

        // Surfaces.
        let surface_globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface globals"),
            size: std::mem::size_of::<SurfaceGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface globals"),
            entries: &[
                uniform_entry(
                    0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    false,
                ),
                float_texture_entry(1, wgpu::ShaderStages::VERTEX),
            ],
        });
        let model_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, true)],
        });
        let emission_tex = emission_texture(device, 1);
        let surface_bind =
            Self::surface_bind_group(device, &surface_layout, &surface_globals, &emission_tex);
        let models = Self::model_buffer(device, 64);
        let model_bind = Self::model_bind_group(device, &model_layout, &models);
        let surface_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("surfaces"),
                bind_group_layouts: &[Some(&surface_layout), Some(&model_layout)],
                immediate_size: 0,
            });
        let surface_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("surfaces"),
            layout: Some(&surface_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &surface_shader,
                entry_point: Some("vs_surface"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SurfaceGpuVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x3,
                        3 => Float32,
                        4 => Uint32
                    ],
                })],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_test.clone()),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &surface_shader,
                entry_point: Some("fs_surface"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: RADIANCE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            cache: None,
            multiview_mask: None,
        });

        // Exposure.
        let exposure_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exposure"),
            entries: &[
                float_texture_entry(0, wgpu::ShaderStages::COMPUTE),
                storage_entry(1),
                storage_entry(2),
                uniform_entry(3, wgpu::ShaderStages::COMPUTE, false),
            ],
        });
        let exposure_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("exposure"),
                bind_group_layouts: &[Some(&exposure_layout)],
                immediate_size: 0,
            });
        let compute = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&exposure_pipeline_layout),
                module: &exposure_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let partial_pipeline = compute("log luminance", "cs_partial");
        let adapt_pipeline = compute("adaptation", "cs_adapt");
        let exposure_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure state"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let exposure_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure params"),
            size: std::mem::size_of::<ExposureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exposure_applied = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure applied"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tonemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tone curve"),
            entries: &[
                float_texture_entry(0, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(4, wgpu::ShaderStages::FRAGMENT, false),
            ],
        });
        let tonemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tone curve"),
                bind_group_layouts: &[Some(&tonemap_layout)],
                immediate_size: 0,
            });
        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tone curve"),
            layout: Some(&tonemap_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &exposure_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(no_depth.clone()),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &exposure_shader,
                entry_point: Some("fs_tonemap"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            cache: None,
            multiview_mask: None,
        });

        // The text shares the display pass and its depth buffer, drawn last,
        // on top of everything, without touching depth.
        let brush = BrushBuilder::using_font_bytes(OVERLAY_FONT)
            .expect("the embedded overlay font parses")
            .with_depth_stencil(Some(no_depth))
            .build(device, width, height, format);
        let targets = Self::targets(
            device,
            width,
            height,
            &exposure_layout,
            &tonemap_layout,
            &exposure_state,
            &exposure_params,
            &exposure_applied,
        );
        Renderer {
            device: device.clone(),
            queue: queue.clone(),
            width,
            height,
            frame: 0,
            globals,
            bind_group,
            marker_pipeline,
            line_pipeline,
            markers: GrowBuffer::new(device, "markers"),
            lines: GrowBuffer::new(device, "lines"),
            surface_pipeline,
            surface_globals,
            surface_layout,
            surface_bind,
            model_layout,
            models,
            model_bind,
            emission: EmissionTable::new(),
            emission_texture: emission_tex,
            emission_uploaded: 0,
            meshes: BTreeMap::new(),
            exposure_layout,
            partial_pipeline,
            adapt_pipeline,
            exposure_state,
            exposure_params,
            exposure_applied,
            tonemap_layout,
            tonemap_pipeline,
            targets,
            stats: FrameStats::default(),
            brush,
        }
    }

    fn surface_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        emission: &wgpu::Texture,
    ) -> wgpu::BindGroup {
        let view = emission.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface globals"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        })
    }

    fn model_buffer(device: &wgpu::Device, draws: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("models"),
            size: draws.max(1) * MODEL_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn model_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        models: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: models,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ModelGpu>() as u64),
                }),
            }],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn targets(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        exposure_layout: &wgpu::BindGroupLayout,
        tonemap_layout: &wgpu::BindGroupLayout,
        state: &wgpu::Buffer,
        params: &wgpu::Buffer,
        applied: &wgpu::Buffer,
    ) -> Targets {
        let depth = texture_2d(
            device,
            "depth",
            width,
            height,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());
        let radiance = texture_2d(
            device,
            "radiance",
            width,
            height,
            RADIANCE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());
        let groups = (
            width.div_ceil(LUMINANCE_BLOCK),
            height.div_ceil(LUMINANCE_BLOCK),
        );
        let partial_count = groups.0 * groups.1;
        let partials = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("luminance partials"),
            size: u64::from(partial_count) * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let exposure_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exposure"),
            layout: exposure_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&radiance),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let tonemap_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone curve"),
            layout: tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&radiance),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: applied.as_entire_binding(),
                },
            ],
        });
        Targets {
            depth,
            radiance,
            partial_count,
            groups,
            exposure_bind,
            tonemap_bind,
        }
    }

    /// The current target size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// What the last frame drew.
    pub fn stats(&self) -> FrameStats {
        self.stats
    }

    /// Resizes the depth buffer, the radiance target, and the overlay
    /// projection.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        self.targets = Self::targets(
            &self.device,
            width,
            height,
            &self.exposure_layout,
            &self.tonemap_layout,
            &self.exposure_state,
            &self.exposure_params,
            &self.exposure_applied,
        );
        self.brush
            .resize_view(width as f32, height as f32, &self.queue);
    }

    /// Uploads a mesh: positions relative to the cell origin in `f32`, and
    /// each vertex's temperature turned into an emission table index.
    fn upload_mesh(&mut self, mesh: &SurfaceMesh) -> GpuMesh {
        let o = [mesh.origin.x, mesh.origin.y, mesh.origin.z];
        let vertices: Vec<SurfaceGpuVertex> = mesh
            .positions
            .iter()
            .zip(&mesh.vertices)
            .map(|(p, v)| SurfaceGpuVertex {
                position: [
                    (p[0] - o[0]) as f32,
                    (p[1] - o[1]) as f32,
                    (p[2] - o[2]) as f32,
                ],
                normal: v.normal,
                albedo: v.albedo,
                roughness: v.roughness,
                emission_index: self.emission.index_for(f64::from(v.temperature)),
            })
            .collect();
        GpuMesh {
            vertices: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("surface vertices"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            indices: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("surface indices"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            index_count: mesh.indices.len() as u32,
            last_drawn: self.frame,
        }
    }

    /// Writes new emission table entries, growing the texture when needed.
    fn sync_emission(&mut self) {
        let len = self.emission.len();
        if len == self.emission_uploaded {
            return;
        }
        let width = EMISSION_TABLE_WIDTH as usize;
        let rows = len.div_ceil(width) as u32;
        if rows > self.emission_texture.height() {
            self.emission_texture = emission_texture(&self.device, rows.next_power_of_two());
            self.surface_bind = Self::surface_bind_group(
                &self.device,
                &self.surface_layout,
                &self.surface_globals,
                &self.emission_texture,
            );
        }
        let mut texels = self.emission.texels().to_vec();
        texels.resize(rows as usize * width, [0.0; 4]);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.emission_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(EMISSION_TABLE_WIDTH * 16),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: EMISSION_TABLE_WIDTH,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        self.emission_uploaded = len;
    }

    /// Records and submits one frame into `target`: lit surfaces, exposure,
    /// the tone curve, markers, lines, and the overlay lines when given.
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        scene: &Scene,
        overlay: Option<&str>,
        exposure: Exposure,
    ) {
        self.frame += 1;
        let aspect = self.width as f32 / self.height as f32;
        let view_proj = scene.view_projection(aspect).to_cols_array_2d();
        let globals = Globals {
            view_proj,
            viewport: [self.width as f32, self.height as f32],
            pad: [0.0; 2],
        };
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));

        // Surfaces: lights, meshes, model transforms.
        let mut lights = [LightGpu::zeroed(); MAX_LIGHTS];
        let light_count = scene.lights.len().min(MAX_LIGHTS);
        for (slot, l) in lights.iter_mut().zip(&scene.lights) {
            slot.position = [l.position[0], l.position[1], l.position[2], 0.0];
            slot.intensity = [l.intensity[0], l.intensity[1], l.intensity[2], 0.0];
        }
        let surface_globals = SurfaceGlobals {
            view_proj,
            counts: [light_count as u32, 0, 0, 0],
            lights,
        };
        self.queue.write_buffer(
            &self.surface_globals,
            0,
            bytemuck::bytes_of(&surface_globals),
        );
        for draw in &scene.surfaces {
            let key = draw.mesh.key;
            if !self.meshes.contains_key(&key) {
                let gpu = self.upload_mesh(&draw.mesh);
                self.meshes.insert(key, gpu);
            }
            if let Some(m) = self.meshes.get_mut(&key) {
                m.last_drawn = self.frame;
            }
        }
        let frame = self.frame;
        self.meshes
            .retain(|_, m| frame - m.last_drawn <= MESH_IDLE_FRAMES);
        self.sync_emission();
        let draws = scene.surfaces.len() as u64;
        if draws * MODEL_STRIDE > self.models.size() {
            self.models = Self::model_buffer(&self.device, draws.next_power_of_two());
            self.model_bind =
                Self::model_bind_group(&self.device, &self.model_layout, &self.models);
        }
        let mut model_bytes = vec![0u8; (draws * MODEL_STRIDE) as usize];
        for (i, draw) in scene.surfaces.iter().enumerate() {
            let r = draw.rotation;
            let m = ModelGpu {
                columns: [
                    [r[0][0], r[0][1], r[0][2], 0.0],
                    [r[1][0], r[1][1], r[1][2], 0.0],
                    [r[2][0], r[2][1], r[2][2], 0.0],
                ],
                offset: [draw.offset[0], draw.offset[1], draw.offset[2], 0.0],
            };
            let at = i * MODEL_STRIDE as usize;
            model_bytes[at..at + std::mem::size_of::<ModelGpu>()]
                .copy_from_slice(bytemuck::bytes_of(&m));
        }
        if !model_bytes.is_empty() {
            self.queue.write_buffer(&self.models, 0, &model_bytes);
        }
        let params = ExposureParams {
            bias_stops: exposure.bias_stops as f32,
            alpha: exposure.alpha(),
            partial_count: self.targets.partial_count,
            pad: 0,
        };
        self.queue
            .write_buffer(&self.exposure_params, 0, bytemuck::bytes_of(&params));

        // Markers and lines.
        let instances: Vec<MarkerInstance> = scene
            .markers
            .iter()
            .map(|m| MarkerInstance {
                center: m.position,
                size_px: m.size_px,
                color: m.color,
            })
            .collect();
        self.markers.write(&self.device, &self.queue, &instances);
        let vertices: Vec<LineGpuVertex> = scene
            .lines
            .iter()
            .map(|v| LineGpuVertex {
                position: v.position,
                color: v.color,
            })
            .collect();
        self.lines.write(&self.device, &self.queue, &vertices);

        let text = overlay.unwrap_or("");
        let section = Section::default()
            .with_screen_position((MARGIN_PX, MARGIN_PX))
            .add_text(Text::new(text).with_scale(TEXT_PX).with_color(TEXT_COLOR));
        self.brush
            .queue(&self.device, &self.queue, [&section])
            .expect("the overlay fits the glyph cache");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        let mut stats = FrameStats {
            lights: light_count,
            ..FrameStats::default()
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.radiance,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.surface_pipeline);
            pass.set_bind_group(0, &self.surface_bind, &[]);
            for (i, draw) in scene.surfaces.iter().enumerate() {
                let Some(m) = self.meshes.get(&draw.mesh.key) else {
                    continue;
                };
                pass.set_bind_group(1, &self.model_bind, &[(i as u64 * MODEL_STRIDE) as u32]);
                pass.set_vertex_buffer(0, m.vertices.slice(..));
                pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.index_count, 0, 0..1);
                stats.meshes += 1;
                stats.triangles += m.index_count as usize / 3;
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("exposure"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.targets.exposure_bind, &[]);
            pass.set_pipeline(&self.partial_pipeline);
            pass.dispatch_workgroups(self.targets.groups.0, self.targets.groups.1, 1);
            pass.set_pipeline(&self.adapt_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.exposure_state, 0, &self.exposure_applied, 0, 16);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, &self.targets.tonemap_bind, &[]);
            pass.draw(0..3, 0..1);
            pass.set_bind_group(0, &self.bind_group, &[]);
            if self.lines.len > 0 {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_vertex_buffer(0, self.lines.buffer.slice(..));
                pass.draw(0..self.lines.len, 0..1);
            }
            if self.markers.len > 0 {
                pass.set_pipeline(&self.marker_pipeline);
                pass.set_vertex_buffer(0, self.markers.buffer.slice(..));
                pass.draw(0..6, 0..self.markers.len);
            }
            if overlay.is_some() {
                self.brush.draw(&mut pass);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.stats = stats;
    }
}
