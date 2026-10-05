//! Pixel statistics on rendered screenshots, for the end to end run
//! (`scripts/e2e.sh`, recorded in `docs/e2e/README.md`).
//!
//! Every measure works on the luma of the 8-bit encoded pixel values,
//! `0.2126 R + 0.7152 G + 0.0722 B` (Rec. 709 weights, the weights of the
//! renderer's exposure pass), from 0 to 255. Alpha is ignored. Nothing here
//! knows what the image shows: the checks are about blobs, coverage, sides,
//! centroids, and steps between neighboring pixels.
//!
//! - [`components`]: 8-connected groups of pixels above a luma threshold,
//!   in scan order of their first pixel.
//! - [`disc_sides`]: splits a component into a brighter and a darker side
//!   (see its docs).
//! - [`bright_centroid`]: the centroid of the pixels above a threshold.
//! - [`falloff`]: the luma range and the largest step between two
//!   neighboring pixels, which tells a smooth falloff from a hard edge.

use std::path::Path;

/// Rec. 709 luma weights on red, green, blue.
pub const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// A luma image, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Gray {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height` luma values, 0 to 255.
    pub luma: Vec<f64>,
}

impl Gray {
    /// Luma of tightly packed 8-bit pixels with `channels` values each:
    /// 1 (gray), 2 (gray, alpha), 3 (RGB), or 4 (RGBA).
    pub fn from_pixels(width: usize, height: usize, channels: usize, data: &[u8]) -> Gray {
        assert!((1..=4).contains(&channels), "1 to 4 channels");
        assert_eq!(data.len(), width * height * channels, "pixel data length");
        let luma = data
            .chunks_exact(channels)
            .map(|p| {
                if channels < 3 {
                    f64::from(p[0])
                } else {
                    LUMA[0] * f64::from(p[0])
                        + LUMA[1] * f64::from(p[1])
                        + LUMA[2] * f64::from(p[2])
                }
            })
            .collect();
        Gray {
            width,
            height,
            luma,
        }
    }

    /// Reads an 8-bit PNG (palette and 16-bit images are expanded and
    /// stripped to 8 bits).
    pub fn load(path: &Path) -> Result<Gray, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut decoder = png::Decoder::new(std::io::BufReader::new(file));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder
            .read_info()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let size = reader
            .output_buffer_size()
            .ok_or_else(|| format!("{}: image too large", path.display()))?;
        let mut buf = vec![0; size];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let channels = info.color_type.samples();
        let (w, h) = (info.width as usize, info.height as usize);
        let mut data = Vec::with_capacity(w * h * channels);
        for row in buf[..info.buffer_size()].chunks_exact(info.line_size) {
            data.extend_from_slice(&row[..w * channels]);
        }
        Ok(Gray::from_pixels(w, h, channels, &data))
    }

    /// The luma at a pixel.
    pub fn at(&self, x: usize, y: usize) -> f64 {
        self.luma[y * self.width + x]
    }

    /// Number of pixels.
    pub fn len(&self) -> usize {
        self.luma.len()
    }

    /// Returns `true` for an image with no pixels.
    pub fn is_empty(&self) -> bool {
        self.luma.is_empty()
    }
}

/// One 8-connected group of pixels above a threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    /// Pixel indices (`y * width + x`), in scan order.
    pub pixels: Vec<usize>,
    /// Centroid of the pixels, `(x, y)`, pixel centers at whole numbers.
    pub centroid: (f64, f64),
    /// Mean luma of the pixels.
    pub mean_luma: f64,
}

/// The 8-connected components of the pixels whose luma is above
/// `threshold`, ordered by their first pixel in scan order.
pub fn components(img: &Gray, threshold: f64) -> Vec<Component> {
    let (w, h) = (img.width, img.height);
    let mut label = vec![false; img.len()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..img.len() {
        if label[start] || img.luma[start] <= threshold {
            continue;
        }
        label[start] = true;
        stack.push(start);
        let mut pixels = Vec::new();
        while let Some(i) = stack.pop() {
            pixels.push(i);
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if !label[j] && img.luma[j] > threshold {
                        label[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        pixels.sort_unstable();
        out.push(component(img, pixels));
    }
    out
}

fn component(img: &Gray, pixels: Vec<usize>) -> Component {
    let n = pixels.len() as f64;
    let (mut sx, mut sy, mut sl) = (0.0, 0.0, 0.0);
    for &i in &pixels {
        sx += (i % img.width) as f64;
        sy += (i / img.width) as f64;
        sl += img.luma[i];
    }
    Component {
        pixels,
        centroid: (sx / n, sy / n),
        mean_luma: sl / n,
    }
}

/// The two sides of a component.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sides {
    /// Mean luma of the brighter side.
    pub bright_mean: f64,
    /// Mean luma of the darker side.
    pub dark_mean: f64,
    /// Unit direction from the darker side to the brighter side, image axes
    /// (`+x` right, `+y` down); `(0, 0)` when the component has no
    /// brightness gradient at all.
    pub direction: (f64, f64),
}

impl Sides {
    /// `bright_mean / dark_mean`, infinite when the darker side is black.
    pub fn ratio(&self) -> f64 {
        if self.dark_mean > 0.0 {
            self.bright_mean / self.dark_mean
        } else {
            f64::INFINITY
        }
    }
}

/// Splits a component into two sides by the line through its centroid
/// perpendicular to the offset of its luma-weighted centroid from its
/// plain centroid (the direction its brightness leans), and returns the
/// mean luma of each side. A component of uniform brightness has no lean:
/// both sides get its mean and the ratio is 1.
pub fn disc_sides(img: &Gray, c: &Component) -> Sides {
    let (cx, cy) = c.centroid;
    let (mut wx, mut wy, mut wl) = (0.0, 0.0, 0.0);
    for &i in &c.pixels {
        let l = img.luma[i];
        wx += l * (i % img.width) as f64;
        wy += l * (i / img.width) as f64;
        wl += l;
    }
    let (dx, dy) = (wx / wl - cx, wy / wl - cy);
    let len = dx.hypot(dy);
    if len.is_nan() || len <= 1.0e-9 {
        return Sides {
            bright_mean: c.mean_luma,
            dark_mean: c.mean_luma,
            direction: (0.0, 0.0),
        };
    }
    let (ux, uy) = (dx / len, dy / len);
    let (mut bs, mut bn, mut ds, mut dn) = (0.0, 0usize, 0.0, 0usize);
    for &i in &c.pixels {
        let side = ((i % img.width) as f64 - cx) * ux + ((i / img.width) as f64 - cy) * uy;
        if side >= 0.0 {
            bs += img.luma[i];
            bn += 1;
        } else {
            ds += img.luma[i];
            dn += 1;
        }
    }
    let mean = |s: f64, n: usize| if n > 0 { s / n as f64 } else { 0.0 };
    Sides {
        bright_mean: mean(bs, bn),
        dark_mean: mean(ds, dn),
        direction: (ux, uy),
    }
}

/// The centroid `(x, y)` of the pixels with luma above `threshold`, and how
/// many there are; `None` when there are none.
pub fn bright_centroid(img: &Gray, threshold: f64) -> Option<((f64, f64), usize)> {
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0usize);
    for (i, &l) in img.luma.iter().enumerate() {
        if l > threshold {
            sx += (i % img.width) as f64;
            sy += (i / img.width) as f64;
            n += 1;
        }
    }
    (n > 0).then(|| ((sx / n as f64, sy / n as f64), n))
}

/// How brightness changes across an image.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Falloff {
    /// The 1st percentile of luma.
    pub low: f64,
    /// The 99th percentile of luma.
    pub high: f64,
    /// The largest luma difference between two horizontally or vertically
    /// neighboring pixels.
    pub max_step: f64,
    /// Where that step is, `(x, y)` of its first pixel.
    pub max_step_at: (usize, usize),
}

impl Falloff {
    /// `high - low`: how far the brightness falls across the image.
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// `max_step / range`: near 0 for a smooth falloff, near 1 for a hard
    /// edge that drops the whole range from one pixel to the next.
    pub fn step_fraction(&self) -> f64 {
        if self.range() > 0.0 {
            self.max_step / self.range()
        } else {
            f64::INFINITY
        }
    }
}

/// The percentiles and the largest neighbor step of an image.
pub fn falloff(img: &Gray) -> Falloff {
    let mut sorted = img.luma.clone();
    sorted.sort_by(f64::total_cmp);
    let pct = |p: f64| {
        let i = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[i]
    };
    let (mut max_step, mut at) = (0.0, (0, 0));
    for y in 0..img.height {
        for x in 0..img.width {
            let l = img.at(x, y);
            if x + 1 < img.width {
                let s = (img.at(x + 1, y) - l).abs();
                if s > max_step {
                    max_step = s;
                    at = (x, y);
                }
            }
            if y + 1 < img.height {
                let s = (img.at(x, y + 1) - l).abs();
                if s > max_step {
                    max_step = s;
                    at = (x, y);
                }
            }
        }
    }
    Falloff {
        low: pct(0.01),
        high: pct(0.99),
        max_step,
        max_step_at: at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Gray {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.push(f(x, y));
            }
        }
        Gray::from_pixels(w, h, 1, &data)
    }

    #[test]
    fn luma_weights_rgb() {
        let g = Gray::from_pixels(2, 1, 4, &[255, 0, 0, 255, 0, 255, 0, 0]);
        assert!((g.luma[0] - 0.2126 * 255.0).abs() < 1e-9);
        assert!((g.luma[1] - 0.7152 * 255.0).abs() < 1e-9);
    }

    #[test]
    fn counts_separate_and_diagonal_blobs() {
        // Three blobs: a 2x2 square, a diagonal pair (8-connected, one
        // blob), and a single pixel.
        let on = [(1, 1), (2, 1), (1, 2), (2, 2), (6, 1), (7, 2), (4, 6)];
        let g = image(10, 8, |x, y| if on.contains(&(x, y)) { 200 } else { 0 });
        let c = components(&g, 100.0);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].pixels.len(), 4);
        assert_eq!(c[0].centroid, (1.5, 1.5));
        assert_eq!(c[1].pixels.len(), 2);
        assert_eq!(c[2].pixels.len(), 1);
        assert!(components(&g, 200.0).is_empty());
    }

    #[test]
    fn half_lit_disc_has_sides_and_uniform_disc_does_not() {
        let r = 20.0;
        let lit = image(64, 64, |x, y| {
            let (dx, dy) = (x as f64 - 32.0, y as f64 - 32.0);
            if dx.hypot(dy) <= r {
                // Brightest on the right, falling to the left.
                (40.0 + 200.0 * (dx + r) / (2.0 * r)) as u8
            } else {
                0
            }
        });
        let c = components(&lit, 10.0);
        assert_eq!(c.len(), 1);
        let s = disc_sides(&lit, &c[0]);
        assert!(s.ratio() > 1.5, "{s:?}");
        assert!(s.direction.0 > 0.99, "{s:?}");

        let flat = image(64, 64, |x, y| {
            if (x as f64 - 32.0).hypot(y as f64 - 32.0) <= r {
                150
            } else {
                0
            }
        });
        let c = components(&flat, 10.0);
        let s = disc_sides(&flat, &c[0]);
        assert!((s.ratio() - 1.0).abs() < 1e-9, "{s:?}");
    }

    #[test]
    fn centroid_of_bright_pixels() {
        let g = image(10, 10, |x, y| if x >= 5 && y < 2 { 255 } else { 10 });
        let ((x, y), n) = bright_centroid(&g, 100.0).unwrap();
        assert_eq!((x, y, n), (7.0, 0.5, 10));
        assert!(bright_centroid(&g, 255.0).is_none());
    }

    #[test]
    fn falloff_tells_gradient_from_edge() {
        let ramp = image(200, 10, |x, _| x as u8);
        let f = falloff(&ramp);
        assert!(f.range() > 190.0);
        assert!(f.step_fraction() < 0.01, "{f:?}");
        let edge = image(200, 10, |x, _| if x < 100 { 0 } else { 200 });
        let f = falloff(&edge);
        assert!((f.step_fraction() - 1.0).abs() < 1e-9, "{f:?}");
        assert_eq!(f.max_step_at, (99, 0));
    }
}
