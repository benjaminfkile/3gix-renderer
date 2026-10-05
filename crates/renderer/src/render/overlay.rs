//! The debug overlay: what it says and where it sits.
//!
//! The overlay is drawn in the same render pass as the markers, in the top
//! left corner, with `wgpu_text` and the Hack typeface embedded through
//! `epaint_default_fonts` (see `docs/architecture.md` for why). This module
//! only formats the text so it can be tested without a GPU.
//!
//! Simulation time is shown as days since J2000, the reference of the
//! registry epoch (`matter-format.md` section 5.1, `space-model.md`
//! section 6).

use crate::sim::SECONDS_PER_DAY;
use gx_core::units::Seconds;

/// Left and top margin of the overlay, pixels.
pub const MARGIN_PX: f32 = 8.0;

/// Text size, pixels.
pub const TEXT_PX: f32 = 16.0;

/// Text color, linear RGBA.
pub const TEXT_COLOR: [f32; 4] = [0.9, 0.92, 0.95, 1.0];

/// The matter pipeline values the overlay reports.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct MatterStats {
    /// Cells in the latest selection.
    pub cells_selected: usize,
    /// Selected cells fetched and composited (or known empty).
    pub cells_ready: usize,
    /// Selected cells requested or waiting for the hub.
    pub cells_pending: usize,
    /// Meshes drawn this frame.
    pub meshes_drawn: usize,
    /// Triangles drawn this frame.
    pub triangles_drawn: usize,
    /// Volumes drawn this frame.
    pub volumes_drawn: usize,
    /// Far field point sprites drawn this frame.
    pub sprites_drawn: usize,
    /// Point lights in use.
    pub lights_active: usize,
    /// Chunk body bytes received this session.
    pub bytes_fetched: u64,
    /// Round-trip time of the latest chunk request, seconds.
    pub last_round_trip: Option<f64>,
}

/// The values the overlay reports.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OverlayInfo {
    /// Simulation time, seconds since J2000.
    pub sim_time: Seconds,
    /// Simulation seconds per wall-clock second.
    pub time_scale: f64,
    /// `true` while the clock is paused.
    pub paused: bool,
    /// The frame the camera is parented to.
    pub camera_frame: u64,
    /// Distance from the camera to its frame origin, meters.
    pub camera_distance: f64,
    /// Number of frames in the registry.
    pub frame_count: usize,
    /// Rendered frames per second, or `None` when not measured (headless).
    pub fps: Option<f64>,
    /// `true` when the integrator was clamped this frame.
    pub sim_lag: bool,
    /// The matter pipeline.
    pub matter: MatterStats,
}

impl OverlayInfo {
    /// The overlay lines, top to bottom.
    pub fn lines(&self) -> Vec<String> {
        let days = self.sim_time.value() / SECONDS_PER_DAY;
        let mut lines = vec![
            format!("J2000 + {days:.4} d"),
            format!(
                "time scale {}x{}",
                format_scale(self.time_scale),
                if self.paused { " (paused)" } else { "" }
            ),
            format!(
                "camera frame {}  |p| {:.4e} m",
                self.camera_frame, self.camera_distance
            ),
            format!("frames {}", self.frame_count),
            match self.fps {
                Some(fps) => format!("fps {fps:.1}"),
                None => "fps -".to_string(),
            },
        ];
        let m = &self.matter;
        lines.push(format!(
            "cells selected {}  ready {}  pending {}",
            m.cells_selected, m.cells_ready, m.cells_pending
        ));
        lines.push(format!(
            "meshes {}  triangles {}  lights {}",
            m.meshes_drawn, m.triangles_drawn, m.lights_active
        ));
        lines.push(format!(
            "volumes {}  sprites {}",
            m.volumes_drawn, m.sprites_drawn
        ));
        lines.push(format!(
            "fetched {} B  rtt {}",
            m.bytes_fetched,
            match m.last_round_trip {
                Some(s) => format!("{s:.3} s"),
                None => "-".to_string(),
            }
        ));
        if self.sim_lag {
            lines.push("sim lag".to_string());
        }
        lines
    }

    /// The overlay as one block of text.
    pub fn text(&self) -> String {
        self.lines().join("\n")
    }
}

/// A time scale without trailing noise: integers as integers, fractions with
/// up to six significant digits.
fn format_scale(scale: f64) -> String {
    if scale.fract() == 0.0 && scale.abs() < 1.0e15 {
        format!("{scale:.0}")
    } else {
        let s = format!("{scale:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> OverlayInfo {
        OverlayInfo {
            sim_time: Seconds::new(8.1e8),
            time_scale: 86400.0,
            paused: false,
            camera_frame: 4,
            camera_distance: 1.5e8,
            frame_count: 6,
            fps: Some(59.94),
            sim_lag: false,
            matter: MatterStats {
                cells_selected: 12,
                cells_ready: 9,
                cells_pending: 3,
                meshes_drawn: 4,
                triangles_drawn: 5120,
                volumes_drawn: 2,
                sprites_drawn: 7,
                lights_active: 1,
                bytes_fetched: 1_048_576,
                last_round_trip: Some(0.0425),
            },
        }
    }

    #[test]
    fn lines_report_every_value() {
        let l = info().lines();
        assert_eq!(l[0], "J2000 + 9375.0000 d");
        assert_eq!(l[1], "time scale 86400x");
        assert_eq!(l[2], "camera frame 4  |p| 1.5000e8 m");
        assert_eq!(l[3], "frames 6");
        assert_eq!(l[4], "fps 59.9");
        assert_eq!(l[5], "cells selected 12  ready 9  pending 3");
        assert_eq!(l[6], "meshes 4  triangles 5120  lights 1");
        assert_eq!(l[7], "volumes 2  sprites 7");
        assert_eq!(l[8], "fetched 1048576 B  rtt 0.043 s");
        assert_eq!(l.len(), 9);
    }

    #[test]
    fn lag_paused_and_fractions() {
        let l = OverlayInfo {
            time_scale: 0.25,
            paused: true,
            fps: None,
            sim_lag: true,
            matter: MatterStats::default(),
            ..info()
        }
        .lines();
        assert_eq!(l[1], "time scale 0.25x (paused)");
        assert_eq!(l[4], "fps -");
        assert_eq!(l[8], "fetched 0 B  rtt -");
        assert_eq!(l[9], "sim lag");
    }
}
