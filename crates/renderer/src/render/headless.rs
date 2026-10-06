//! Headless rendering: an offscreen RGBA8 texture, read back to memory and
//! written as PNG.
//!
//! Used by `--headless`, `--screenshot`, and the integration tests. It runs
//! on any wgpu adapter, including a software Vulkan driver, and needs no
//! window or display. `WGPU_BACKEND` and the other wgpu environment
//! variables select the adapter as usual.

use super::gpu::{describe_adapter, pick_adapter, wanted_features, Exposure, FrameStats, Renderer};
use super::scene::Scene;
use anyhow::{anyhow, Context, Result};
use std::path::Path;

/// Format of the offscreen target: 8-bit RGBA in sRGB encoding.
pub const HEADLESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// An image read back from the GPU: tightly packed RGBA8 rows, top row
/// first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes.
    pub rgba: Vec<u8>,
}

impl Image {
    /// The RGBA value of one pixel.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }

    /// Writes the image as an 8-bit RGBA PNG.
    pub fn write_png(&self, path: &Path) -> Result<()> {
        let file =
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&self.rgba)?;
        writer.finish()?;
        Ok(())
    }
}

/// The float radiance target read back: W m^-2 sr^-1 per band in red,
/// green, blue, and the coverage in alpha, top row first.
#[derive(Clone, Debug, PartialEq)]
pub struct RadianceImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height` texels.
    pub texels: Vec<[f32; 4]>,
}

impl RadianceImage {
    /// The texel of one pixel.
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 4] {
        self.texels[(y * self.width + x) as usize]
    }
}

/// A device, a renderer, and an offscreen texture to render into.
pub struct Headless {
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    renderer: Renderer,
    width: u32,
    height: u32,
    adapter_name: String,
}

impl Headless {
    /// Opens the adapter [`pick_adapter`] chooses (software drivers
    /// included, so this works without a GPU) and creates an offscreen
    /// target of the given size.
    pub fn new(width: u32, height: u32, adapter: Option<&str>) -> Result<Headless> {
        pollster::block_on(Headless::new_async(width, height, adapter))
    }

    async fn new_async(width: u32, height: u32, wanted: Option<&str>) -> Result<Headless> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pick_adapter(&instance, None, wanted)
            .await
            .context("choosing the graphics adapter")?;
        let adapter_name = describe_adapter(&adapter);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("headless"),
                required_features: wanted_features(&adapter),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .context("opening the graphics device")?;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HEADLESS_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let renderer = Renderer::new(&device, &queue, HEADLESS_FORMAT, width, height);
        Ok(Headless {
            device,
            queue,
            texture,
            view,
            renderer,
            width,
            height,
            adapter_name,
        })
    }

    /// The name of the adapter in use, for logs.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Renders one frame with the exposure jumping straight to the frame's
    /// luminance and no bias, and reads it back. The same scene always
    /// gives the same image.
    pub fn render(&mut self, scene: &Scene, overlay: Option<&str>) -> Result<Image> {
        self.render_with(scene, overlay, Exposure::default())
    }

    /// Renders one frame with the given exposure and reads it back.
    pub fn render_with(
        &mut self,
        scene: &Scene,
        overlay: Option<&str>,
        exposure: Exposure,
    ) -> Result<Image> {
        self.renderer.render(&self.view, scene, overlay, exposure);
        self.read_back()
    }

    /// What the last frame drew.
    pub fn stats(&self) -> FrameStats {
        self.renderer.stats()
    }

    /// Whether the device blends into the float radiance target (see
    /// [`wanted_features`]).
    pub fn float_blending(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::FLOAT32_BLENDABLE)
    }

    /// Reads back the float radiance target of the last frame: radiance
    /// before exposure and the tone curve.
    pub fn read_radiance(&self) -> Result<RadianceImage> {
        let bytes = self.copy_out(self.renderer.radiance_texture(), 16)?;
        Ok(RadianceImage {
            width: self.width,
            height: self.height,
            texels: bytes
                .as_chunks::<16>()
                .0
                .iter()
                .map(|t| {
                    core::array::from_fn(|i| {
                        f32::from_le_bytes([t[4 * i], t[4 * i + 1], t[4 * i + 2], t[4 * i + 3]])
                    })
                })
                .collect(),
        })
    }

    fn read_back(&self) -> Result<Image> {
        let rgba = self.copy_out(&self.texture, 4)?;
        Ok(Image {
            width: self.width,
            height: self.height,
            rgba,
        })
    }

    /// Copies a texture of the target size with `texel_bytes` per texel to
    /// memory, tightly packed rows, top row first.
    fn copy_out(&self, texture: &wgpu::Texture, texel_bytes: u32) -> Result<Vec<u8>> {
        let unpadded = self.width * texel_bytes;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded) * u64::from(self.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("waiting for the GPU: {e}"))?;
        rx.recv()
            .context("readback callback dropped")?
            .map_err(|e| anyhow!("mapping the readback buffer: {e}"))?;
        let mut rgba = Vec::with_capacity((unpadded * self.height) as usize);
        {
            let data = buffer
                .get_mapped_range(..)
                .map_err(|e| anyhow!("reading the readback buffer: {e:?}"))?;
            for row in data.chunks(padded as usize) {
                rgba.extend_from_slice(&row[..unpadded as usize]);
            }
        }
        buffer.unmap();
        Ok(rgba)
    }
}
