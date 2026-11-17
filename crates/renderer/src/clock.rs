//! A monotonic wall clock for measuring and time-slicing the frame thread's
//! matter work (`rendering.md` section 4).
//!
//! Natively it is [`std::time::Instant`]; in the browser it is the page's
//! `performance.now()`. A `wasm32` build without the `web` feature has no
//! clock and reads 0 throughout, which turns every time slice into "do
//! everything now". Nothing that is drawn or sent depends on this clock;
//! it only decides how much work one frame does and what the statistics
//! report.

/// Seconds since an arbitrary fixed moment of this process.
#[cfg(not(target_arch = "wasm32"))]
pub fn now_seconds() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Seconds since the page started (`performance.now()`), or 0 where the
/// page has no performance clock.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub fn now_seconds() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now() / 1000.0)
}

/// No clock: always 0.
#[cfg(all(target_arch = "wasm32", not(feature = "web")))]
pub fn now_seconds() -> f64 {
    0.0
}

/// Frame-thread matter work per step, milliseconds: every step's duration
/// since the session started, for the percentiles of `--stats-json` and
/// the overlay (`rendering.md` sections 4 and 5.2).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepTimes {
    samples: Vec<f64>,
    next: usize,
    count: u64,
    max: f64,
}

/// Steps the overlay's frame step percentile covers: the latest few
/// seconds of a window at display rate.
pub const OVERLAY_STEPS: usize = 300;

/// Most step durations kept; past this the oldest are overwritten, so the
/// percentiles describe the latest steps and the maximum covers them all.
pub const STEP_SAMPLES: usize = 1 << 16;

impl StepTimes {
    /// No steps yet.
    pub fn new() -> StepTimes {
        StepTimes::default()
    }

    /// Records one step of `ms` milliseconds.
    pub fn record(&mut self, ms: f64) {
        if self.samples.len() < STEP_SAMPLES {
            self.samples.push(ms);
        } else {
            self.samples[self.next] = ms;
            self.next = (self.next + 1) % STEP_SAMPLES;
        }
        self.count += 1;
        self.max = self.max.max(ms);
    }

    /// Steps recorded.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The `p`-th percentile (0 to 100, nearest rank) of the kept steps,
    /// milliseconds, or `None` before the first step.
    pub fn percentile(&self, p: f64) -> Option<f64> {
        percentile(&self.samples, p)
    }

    /// The longest step, milliseconds, or `None` before the first step.
    pub fn max(&self) -> Option<f64> {
        (self.count > 0).then_some(self.max)
    }

    /// The `p`-th percentile of the latest `n` steps, milliseconds.
    pub fn recent_percentile(&self, n: usize, p: f64) -> Option<f64> {
        let len = self.samples.len();
        if len < STEP_SAMPLES {
            percentile(&self.samples[len.saturating_sub(n)..], p)
        } else {
            let n = n.min(len);
            let recent: Vec<f64> = (0..n)
                .map(|i| self.samples[(self.next + len - 1 - i) % len])
                .collect();
            percentile(&recent, p)
        }
    }
}

/// The `p`-th percentile (0 to 100) of `values` by the nearest rank rule,
/// or `None` for no values.
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * v.len() as f64).ceil() as usize;
    Some(v[rank.clamp(1, v.len()) - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_by_nearest_rank() {
        let mut t = StepTimes::new();
        assert_eq!(t.percentile(95.0), None);
        for i in 1..=100 {
            t.record(f64::from(i));
        }
        assert_eq!(t.percentile(50.0), Some(50.0));
        assert_eq!(t.percentile(95.0), Some(95.0));
        assert_eq!(t.percentile(100.0), Some(100.0));
        assert_eq!(t.max(), Some(100.0));
        assert_eq!(t.recent_percentile(10, 50.0), Some(95.0));
        assert_eq!(t.count(), 100);
    }
}
