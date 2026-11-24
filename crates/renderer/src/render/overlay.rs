//! The debug overlay: what it says and where it sits.
//!
//! The overlay is drawn in the same render pass as the markers, in the top
//! left corner, with `wgpu_text` and the Hack typeface embedded through
//! `epaint_default_fonts` (see `docs/architecture.md` for why). This module
//! only formats the text so it can be tested without a GPU.
//!
//! Simulation time is shown as a UTC date and time (converted from TDB with
//! [`gx_core::timescale`]), with its offset from now in seconds, or "now"
//! when the clock follows now (`space-model.md` section 7). One line gives
//! the light-travel time to the frame nearest the crosshair
//! (`space-model.md` section 6.4).

use crate::stream::FetcherStats;
use gx_core::timescale::tdb_seconds_to_utc_ms;
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
    /// Offers in the upload lane or being sent.
    pub uploads_pending: u64,
    /// Sections offered to the upload lane.
    pub uploads_offered: u64,
    /// Offers the hub stored (`201`).
    pub uploads_stored: u64,
    /// Offers that matched a stored copy (`200`).
    pub uploads_matched: u64,
    /// Offers that differed from a stored copy (`202`).
    pub uploads_disputed: u64,
    /// Offers dropped from a full lane.
    pub uploads_dropped: u64,
    /// Offers that failed (counted, never retried; drawing never waits).
    pub uploads_failed: u64,
    /// Downloads chosen and queued for a check.
    pub checks_queued: u64,
    /// Checks whose recompute ran.
    pub checks_run: u64,
    /// Checks skipped (a full check lane, or not runnable here).
    pub checks_skipped: u64,
    /// Checks the hub answered `match`.
    pub checks_matched: u64,
    /// Checks the hub answered `disputed`.
    pub checks_disputed: u64,
    /// Directions requests sent.
    pub directions_requests: u64,
    /// 95th percentile of the frame thread's matter work per step over the
    /// latest steps, milliseconds (`rendering.md` section 4), or `None`
    /// before the first step.
    pub frame_step_p95_ms: Option<f64>,
    /// Bytes the allocator holds and bytes of surface tile geometry
    /// (the memory report, `docs/architecture.md`), or `None` where not
    /// measured.
    #[cfg(feature = "memory-report")]
    pub memory_bytes: Option<(u64, u64)>,
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
            uploads_offered: s.worker.uploads_offered,
            uploads_stored: s.worker.uploads_stored,
            uploads_matched: s.worker.uploads_matched,
            uploads_disputed: s.worker.uploads_disputed,
            uploads_dropped: s.worker.uploads_dropped,
            uploads_failed: s.worker.uploads_failed,
            checks_queued: s.worker.checks_queued,
            checks_run: s.worker.checks_run,
            checks_skipped: s.worker.checks_skipped,
            checks_matched: s.worker.checks_matched,
            checks_disputed: s.worker.checks_disputed,
            directions_requests: s.directions.directions_requests,
            ..MatterStats::default()
        }
    }
}

/// The values the overlay reports.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OverlayInfo {
    /// Simulation time, seconds of TDB since J2000.
    pub sim_time: Seconds,
    /// Simulation time minus now, seconds: 0 at now.
    pub offset_from_now: Seconds,
    /// `true` while the clock follows now.
    pub at_now: bool,
    /// Simulated seconds per wall-clock second, above 0.
    pub rate: f64,
    /// `true` while time runs backwards.
    pub rewinding: bool,
    /// `true` while the clock is paused.
    pub paused: bool,
    /// The frame nearest the crosshair and the light-travel time from it,
    /// seconds, or `None` when no frame is in front of the camera.
    pub crosshair: Option<(u64, f64)>,
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
    /// The fixed exposure, stops, or `None` for the automatic exposure.
    pub exposure_fixed: Option<f64>,
    /// The matter pipeline.
    pub matter: MatterStats,
}

impl OverlayInfo {
    /// The overlay lines, top to bottom.
    pub fn lines(&self) -> Vec<String> {
        let utc = format_utc(tdb_seconds_to_utc_ms(self.sim_time.value()));
        let mut lines = vec![
            if self.at_now {
                format!("{utc}  now")
            } else {
                format!(
                    "{utc}  now {} s",
                    format_offset(self.offset_from_now.value())
                )
            },
            format!(
                "rate {}x{}{}",
                format_scale(self.rate),
                if self.rewinding { " backwards" } else { "" },
                if self.paused { " (paused)" } else { "" }
            ),
            match self.crosshair {
                Some((frame, seconds)) => {
                    format!("light time frame {frame} {} s", format_light_time(seconds))
                }
                None => "light time -".to_string(),
            },
            format!(
                "camera frame {}  |p| {:.4e} m  speed {:.3e} m/s",
                self.camera_frame, self.camera_distance, self.camera_speed,
            ),
            format!("frames {}", self.frame_count),
            match (self.fps, self.matter.frame_step_p95_ms) {
                (Some(fps), Some(step)) => format!("fps {fps:.1}  matter step p95 {step:.2} ms"),
                (Some(fps), None) => format!("fps {fps:.1}"),
                (None, Some(step)) => format!("fps -  matter step p95 {step:.2} ms"),
                (None, None) => "fps -".to_string(),
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
        #[cfg(feature = "memory-report")]
        if let Some((live, tiles)) = m.memory_bytes {
            lines.push(format!(
                "memory live {:.1} MB  tile geometry {:.1} MB",
                crate::memory::megabytes(live),
                crate::memory::megabytes(tiles)
            ));
        }
        lines
    }

    /// The overlay as one block of text.
    pub fn text(&self) -> String {
        self.lines().join("\n")
    }
}

/// A UTC time in milliseconds since the Unix epoch as
/// `YYYY-MM-DD HH:MM:SS.mmm UTC` on the proleptic Gregorian calendar.
pub fn format_utc(utc_ms: i64) -> String {
    let days = utc_ms.div_euclid(86_400_000);
    let ms = utc_ms.rem_euclid(86_400_000);
    // Days to a civil date (the algorithm of H. Hinnant's date library).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}.{:03} UTC",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

/// An offset from now, seconds, with its sign and to the millisecond
/// below 1000 s, whole seconds above: `-3600`, `-0.250`.
fn format_offset(seconds: f64) -> String {
    if seconds.abs() < 1000.0 {
        format!("{seconds:+.3}")
    } else {
        format!("{seconds:+.0}")
    }
}

/// A light-travel time, seconds: to the microsecond below 1 s, the
/// millisecond below 1e6 s, and in exponent form above.
fn format_light_time(seconds: f64) -> String {
    if seconds < 1.0 {
        format!("{seconds:.6}")
    } else if seconds < 1.0e6 {
        format!("{seconds:.3}")
    } else {
        format!("{seconds:.4e}")
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
    use gx_core::timescale::utc_ms_to_tdb_seconds;

    /// 2026-11-15 12:34:56.789 UTC.
    const UTC_MS: i64 = 1_794_746_096_789;

    fn info() -> OverlayInfo {
        OverlayInfo {
            sim_time: Seconds::new(utc_ms_to_tdb_seconds(UTC_MS)),
            offset_from_now: Seconds::new(0.0),
            at_now: true,
            rate: 1.0,
            rewinding: false,
            paused: false,
            crosshair: Some((3, 499.0123)),
            camera_frame: 4,
            camera_distance: 1.5e8,
            camera_speed: 2.5e4,
            frame_count: 6,
            fps: Some(59.94),
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
                uploads_offered: 3,
                uploads_stored: 3,
                uploads_matched: 0,
                uploads_disputed: 0,
                checks_queued: 2,
                checks_run: 2,
                checks_skipped: 0,
                checks_matched: 2,
                checks_disputed: 0,
                uploads_dropped: 0,
                uploads_failed: 0,
                directions_requests: 5,
                frame_step_p95_ms: None,
                #[cfg(feature = "memory-report")]
                memory_bytes: None,
            },
        }
    }

    #[cfg(feature = "memory-report")]
    #[test]
    fn the_memory_line_gives_live_and_tile_geometry_megabytes() {
        let mut i = info();
        i.matter.memory_bytes = Some((2_500_000, 1_000_000));
        let l = i.lines();
        assert_eq!(
            l.last().unwrap(),
            "memory live 2.5 MB  tile geometry 1.0 MB"
        );
    }

    #[test]
    fn lines_report_every_value() {
        let l = info().lines();
        assert_eq!(l[0], "2026-11-15 12:34:56.789 UTC  now");
        assert_eq!(l[1], "rate 1x");
        assert_eq!(l[2], "light time frame 3 499.012 s");
        assert_eq!(l[3], "camera frame 4  |p| 1.5000e8 m  speed 2.500e4 m/s");
        assert_eq!(l[4], "frames 6");
        assert_eq!(l[5], "fps 59.9");
        let mut stepped = info();
        stepped.matter.frame_step_p95_ms = Some(1.234);
        assert_eq!(stepped.lines()[5], "fps 59.9  matter step p95 1.23 ms");
        assert_eq!(l[6], "exposure auto");
        assert_eq!(l[7], "cells selected 12  ready 9  pending 3");
        assert_eq!(l[8], "meshes 4  triangles 5120  lights 1");
        assert_eq!(l[9], "volumes 2  sprites 7");
        assert_eq!(l[10], "cells done 20  with sections 11  1048576 B");
        assert_eq!(l[11], "rtt first 0.250 s  last 0.043 s");
        assert_eq!(l[12], "sections streamed 36 (cache 6)  computed here 4");
        assert_eq!(l[13], "uploads pending 1  modules loaded 2");
        assert_eq!(l.len(), 14);
    }

    #[test]
    fn past_paused_backwards_and_fractions() {
        let l = OverlayInfo {
            sim_time: Seconds::new(utc_ms_to_tdb_seconds(UTC_MS - 3_600_000)),
            offset_from_now: Seconds::new(-3600.0),
            at_now: false,
            rate: 0.25,
            rewinding: true,
            paused: true,
            crosshair: None,
            fps: None,
            exposure_fixed: Some(12.8125),
            matter: MatterStats::default(),
            ..info()
        }
        .lines();
        assert_eq!(l[0], "2026-11-15 11:34:56.789 UTC  now -3600 s");
        assert_eq!(l[1], "rate 0.25x backwards (paused)");
        assert_eq!(l[2], "light time -");
        assert_eq!(l[5], "fps -");
        assert_eq!(l[6], "exposure fixed 12.81");
        assert_eq!(l[10], "cells done 0  with sections 0  0 B");
        assert_eq!(l[11], "rtt first -  last -");
        assert_eq!(l[12], "sections streamed 0 (cache 0)  computed here 0");
        assert_eq!(l[13], "uploads pending 0  modules loaded 0");
        assert_eq!(format_stops(-3.0), "-3");
        assert_eq!(format_stops(0.5), "0.5");
        assert_eq!(format_stops(-0.001), "0");
        assert_eq!(format_offset(-0.25), "-0.250");
        assert_eq!(format_light_time(0.0334), "0.033400");
        assert_eq!(format_light_time(2.5e7), "2.5000e7");
    }

    #[test]
    fn utc_dates() {
        assert_eq!(format_utc(0), "1970-01-01 00:00:00.000 UTC");
        assert_eq!(format_utc(951_782_400_000), "2000-02-29 00:00:00.000 UTC");
        assert_eq!(format_utc(-1), "1969-12-31 23:59:59.999 UTC");
        assert_eq!(format_utc(UTC_MS), "2026-11-15 12:34:56.789 UTC");
    }
}
