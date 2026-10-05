//! The wgpu renderer: one render pass with frame markers, frame-to-parent
//! lines, and the overlay text.
//!
//! The renderer draws into any color target view of the format it was built
//! for: a window surface on the desktop or the offscreen texture of
//! [`crate::render::headless`]. It owns a reversed-z `Depth32Float` depth
//! buffer cleared to 0 with a greater-or-equal test (see
//! [`crate::render::scene`] for the depth strategy).

use super::overlay::{MARGIN_PX, TEXT_COLOR, TEXT_PX};
use super::scene::Scene;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu_text::glyph_brush::ab_glyph::FontRef;
use wgpu_text::glyph_brush::{Section, Text};
use wgpu_text::{BrushBuilder, TextBrush};

/// Format of the depth buffer: 32-bit float, reversed-z.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The value the depth buffer clears to: the far end in reversed-z.
pub const DEPTH_CLEAR: f32 = 0.0;

/// The color the target clears to. Space itself is not drawn
/// (`space-model.md` section 1, rule 3), so it stays black.
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color::BLACK;

/// The overlay typeface, embedded so every machine draws the same glyphs.
pub const OVERLAY_FONT: &[u8] = epaint_default_fonts::HACK_REGULAR;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    pad: [f32; 2],
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

/// The pipelines and buffers of the frame pass.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    marker_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    markers: GrowBuffer,
    lines: GrowBuffer,
    depth: wgpu::TextureView,
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

fn depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
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
        let shader = device.create_shader_module(wgpu::include_wgsl!("markers.wgsl"));
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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
            label: Some("frame pass"),
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
        // The text shares the pass and its depth buffer, drawn last, on top
        // of everything, without touching depth.
        let brush = BrushBuilder::using_font_bytes(OVERLAY_FONT)
            .expect("the embedded overlay font parses")
            .with_depth_stencil(Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }))
            .build(device, width, height, format);
        Renderer {
            device: device.clone(),
            queue: queue.clone(),
            width,
            height,
            globals,
            bind_group,
            marker_pipeline,
            line_pipeline,
            markers: GrowBuffer::new(device, "markers"),
            lines: GrowBuffer::new(device, "lines"),
            depth: depth_view(device, width, height),
            brush,
        }
    }

    /// The current target size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Resizes the depth buffer and the overlay projection.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        self.depth = depth_view(&self.device, width, height);
        self.brush
            .resize_view(width as f32, height as f32, &self.queue);
    }

    /// Records and submits the frame pass into `target`: markers, lines, and
    /// the overlay lines when given.
    pub fn render(&mut self, target: &wgpu::TextureView, scene: &Scene, overlay: Option<&str>) {
        let aspect = self.width as f32 / self.height as f32;
        let globals = Globals {
            view_proj: scene.view_projection(aspect).to_cols_array_2d(),
            viewport: [self.width as f32, self.height as f32],
            pad: [0.0; 2],
        };
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
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
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame pass"),
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
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
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
    }
}
