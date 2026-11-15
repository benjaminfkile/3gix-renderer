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
use crate::stream::FetcherStats;
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
    /// Cell requests finished this session.
    pub requests: u64,
    /// Cell requests that ended with decoded sections this session.
    pub cells_fetched: u64,
    /// Section bytes received for cells this session.
    pub bytes_fetched: u64,
    /// Time to complete the first cell request, seconds.
    pub first_round_trip: Option<f64>,
    /// Time to complete the latest cell request, seconds.
    pub last_round_trip: Option<f64>,
    /// Sections streamed from the edge.
    pub sections_streamed: u64,
    /// Sections read from the local cache.
    pub sections_cached: u64,
    /// Sections computed here by the worker.
    pub sections_computed: u64,
    /// Module calls the worker made.
    pub module_runs: u64,
    /// Distinct modules the worker has loaded.
    pub modules_loaded: u64,
    /// Uploads started and not yet answered.
    pub uploads_pending: u64,
    /// Uploads the hub accepted.
    pub uploads_done: u64,
    /// Uploads that failed (logged; drawing never waits for them).
    pub uploads_failed: u64,
    /// Directions requests sent.
    pub directions_requests: u64,
}

impl MatterStats {
    /// The fetch and worker values from a fetcher's counts, every other
    /// value zero.
    pub fn from_stats(s: FetcherStats) -> MatterStats {
        let t = s.tally;
        MatterStats {
            requests: t.requests,
            cells_fetched: t.bodies,
            bytes_fetched: t.bytes,
            first_round_trip: t.first_round_trip.map(|d| d.as_secs_f64()),
            last_round_trip: t.last_round_trip.map(|d| d.as_secs_f64()),
            sections_streamed: s.directions.sections_streamed,
            sections_cached: s.directions.sections_cached,
            sections_computed: s.worker.sections_computed,
            module_runs: s.worker.module_runs,
            modules_loaded: s.worker.modules_loaded,
            uploads_pending: s.worker.uploads_pending,
            uploads_done: s.worker.uploads_done,
            uploads_failed: s.worker.uploads_failed,
            directions_requests: s.directions.directions_requests,
            ..MatterStats::default()
        }
    }
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
    /// Flight speed, meters per second.
    pub camera_speed: f64,
    /// Number of frames in the registry.
    pub frame_count: usize,
    /// Rendered frames per second, or `None` when not measured (headless).
    pub fps: Option<f64>,
    /// `true` when the integrator was clamped this frame.
    pub sim_lag: bool,
    /// The fixed exposure, stops, or `None` for the automatic exposure.
    pub exposure_fixed: Option<f64>,
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
                "camera frame {}  |p| {:.4e} m  speed {:.3e} m/s",
                self.camera_frame, self.camera_distance, self.camera_speed,
            ),
            format!("frames {}", self.frame_count),
            match self.fps {
                Some(fps) => format!("fps {fps:.1}"),
                None => "fps -".to_string(),
            },
            match self.exposure_fixed {
                Some(stops) => format!("exposure fixed {}", format_stops(stops)),
                None => "exposure auto".to_string(),
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
        let rtt = |t: Option<f64>| match t {
            Some(s) => format!("{s:.3} s"),
            None => "-".to_string(),
        };
        lines.push(format!(
            "cells done {}  with sections {}  {} B",
            m.requests, m.cells_fetched, m.bytes_fetched
        ));
        lines.push(format!(
            "rtt first {}  last {}",
            rtt(m.first_round_trip),
            rtt(m.last_round_trip)
        ));
        lines.push(format!(
            "sections streamed {} (cache {})  computed here {}",
            m.sections_streamed + m.sections_cached,
            m.sections_cached,
            m.sections_computed
        ));
        lines.push(format!(
            "uploads pending {}  modules loaded {}",
            m.uploads_pending, m.modules_loaded
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

/// Stops with up to two decimals and no trailing zeros: `12.81`, `-3`,
/// `0.5`.
fn format_stops(stops: f64) -> String {
    let s = format!("{stops:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".to_string()
    } else {
        s.to_string()
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
            camera_speed: 2.5e4,
            frame_count: 6,
            fps: Some(59.94),
            sim_lag: false,
            exposure_fixed: None,
            matter: MatterStats {
                cells_selected: 12,
                cells_ready: 9,
                cells_pending: 3,
                meshes_drawn: 4,
                triangles_drawn: 5120,
                volumes_drawn: 2,
                sprites_drawn: 7,
                lights_active: 1,
                requests: 20,
                cells_fetched: 11,
                bytes_fetched: 1_048_576,
                first_round_trip: Some(0.25),
                last_round_trip: Some(0.0425),
                sections_streamed: 30,
                sections_cached: 6,
                sections_computed: 4,
                module_runs: 8,
                modules_loaded: 2,
                uploads_pending: 1,
                uploads_done: 3,
                uploads_failed: 0,
                directions_requests: 5,
            },
        }
    }

    #[test]
    fn lines_report_every_value() {
        let l = info().lines();
        assert_eq!(l[0], "J2000 + 9375.0000 d");
        assert_eq!(l[1], "time scale 86400x");
        assert_eq!(l[2], "camera frame 4  |p| 1.5000e8 m  speed 2.500e4 m/s");
        assert_eq!(l[3], "frames 6");
        assert_eq!(l[4], "fps 59.9");
        assert_eq!(l[5], "exposure auto");
        assert_eq!(l[6], "cells selected 12  ready 9  pending 3");
        assert_eq!(l[7], "meshes 4  triangles 5120  lights 1");
        assert_eq!(l[8], "volumes 2  sprites 7");
        assert_eq!(l[9], "cells done 20  with sections 11  1048576 B");
        assert_eq!(l[10], "rtt first 0.250 s  last 0.043 s");
        assert_eq!(l[11], "sections streamed 36 (cache 6)  computed here 4");
        assert_eq!(l[12], "uploads pending 1  modules loaded 2");
        assert_eq!(l.len(), 13);
    }

    #[test]
    fn lag_paused_and_fractions() {
        let l = OverlayInfo {
            time_scale: 0.25,
            paused: true,
            fps: None,
            sim_lag: true,
            exposure_fixed: Some(12.8125),
            matter: MatterStats::default(),
            ..info()
        }
        .lines();
        assert_eq!(l[1], "time scale 0.25x (paused)");
        assert_eq!(l[4], "fps -");
        assert_eq!(l[5], "exposure fixed 12.81");
        assert_eq!(l[9], "cells done 0  with sections 0  0 B");
        assert_eq!(l[10], "rtt first -  last -");
        assert_eq!(l[11], "sections streamed 0 (cache 0)  computed here 0");
        assert_eq!(l[12], "uploads pending 0  modules loaded 0");
        assert_eq!(l[13], "sim lag");
        assert_eq!(format_stops(-3.0), "-3");
        assert_eq!(format_stops(0.5), "0.5");
        assert_eq!(format_stops(-0.001), "0");
    }
}
