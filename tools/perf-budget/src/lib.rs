//! The comparison behind `scripts/perf-budget.sh`: the renderer's headless
//! statistics (`gx-renderer --headless --stats-json`) for the benchmark
//! views of `rendering.md` section 5.1, checked against the limits of
//! section 5.2 and the committed baseline of section 5.4.
//!
//! # Limits (`rendering.md` 5.2)
//!
//! | Statistic | Limit |
//! |---|---|
//! | `settle_seconds` | B1 and B3 at most 4 s, B2 at most 3 s, B4 at most 2 s; a view that never settled breaches |
//! | `frame_step_ms_p95` | at most 4 ms |
//! | `frame_step_ms_max` | at most 16 ms |
//! | `cells_selected` | at most 4096 |
//! | `triangles_drawn` | at most 2,000,000 |
//!
//! The per-section module limit of 5.2 is measured in the core library,
//! not from these statistics.
//!
//! # Baseline (`rendering.md` 5.4)
//!
//! `perf/baseline.json` holds the last accepted numbers per view:
//!
//! ```json
//! { "B1": { "settle_seconds": 3.1, "bytes_fetched": 231000000 }, ... }
//! }
//! ```
//!
//! A view that settles more than 10 percent slower, or fetches more than 10
//! percent more section bytes, than its baseline breaches. Without a
//! baseline file, or without an entry for a view, the comparison is
//! reported as skipped; nothing is invented in its place.

use serde_json::Value;

/// The benchmark views, in order (`rendering.md` section 5.1).
pub const VIEWS: [&str; 4] = ["B1", "B2", "B3", "B4"];

/// Most frame-thread matter work per step at the 95th percentile,
/// milliseconds.
pub const STEP_P95_MS: f64 = 4.0;

/// Most frame-thread matter work in any one step, milliseconds.
pub const STEP_MAX_MS: f64 = 16.0;

/// Most selected cells in any view.
pub const MAX_CELLS: u64 = 4096;

/// Most triangles drawn in any view.
pub const MAX_TRIANGLES: u64 = 2_000_000;

/// How much a view may exceed its baseline, as a factor (10 percent).
pub const BASELINE_FACTOR: f64 = 1.1;

/// The settle limit of a view, seconds, or `None` for a view the budget
/// does not name.
pub fn settle_limit(view: &str) -> Option<f64> {
    match view {
        "B1" | "B3" => Some(4.0),
        "B2" => Some(3.0),
        "B4" => Some(2.0),
        _ => None,
    }
}

/// What the budget reads from one view's statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewStats {
    /// Seconds until every selected cell was drawn at its selected depth;
    /// `None` if that never happened.
    pub settle_seconds: Option<f64>,
    /// Frame step, 95th percentile, milliseconds.
    pub step_p95_ms: Option<f64>,
    /// Frame step, maximum, milliseconds.
    pub step_max_ms: Option<f64>,
    /// Selected cells.
    pub cells_selected: u64,
    /// Triangles drawn.
    pub triangles_drawn: u64,
    /// Section bytes fetched.
    pub bytes_fetched: u64,
}

impl ViewStats {
    /// Reads the statistics from the renderer's `--stats-json` output.
    pub fn from_json(json: &Value) -> Result<ViewStats, String> {
        let float = |name: &str| -> Result<Option<f64>, String> {
            match &json[name] {
                Value::Null => Ok(None),
                v => v
                    .as_f64()
                    .map(Some)
                    .ok_or_else(|| format!("{name} is not a number")),
            }
        };
        let count = |name: &str| -> Result<u64, String> {
            json[name]
                .as_u64()
                .ok_or_else(|| format!("{name} is missing or not a count"))
        };
        Ok(ViewStats {
            settle_seconds: float("settle_seconds")?,
            step_p95_ms: float("frame_step_ms_p95")?,
            step_max_ms: float("frame_step_ms_max")?,
            cells_selected: count("cells_selected")?,
            triangles_drawn: count("triangles_drawn")?,
            bytes_fetched: count("bytes_fetched")?,
        })
    }
}

/// One view's accepted numbers.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BaselineView {
    /// Accepted settle time, seconds.
    pub settle_seconds: f64,
    /// Accepted section bytes fetched.
    pub bytes_fetched: u64,
}

/// The accepted numbers of every view in the baseline file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Baseline {
    /// By view name, in file order.
    pub views: Vec<(String, BaselineView)>,
}

impl Baseline {
    /// Reads `perf/baseline.json` (crate docs).
    pub fn from_json(json: &Value) -> Result<Baseline, String> {
        let Some(map) = json.as_object() else {
            return Err("the baseline is not an object of views".into());
        };
        let mut views = Vec::new();
        for (name, v) in map {
            let settle_seconds = v["settle_seconds"]
                .as_f64()
                .ok_or_else(|| format!("{name}: settle_seconds is missing"))?;
            let bytes_fetched = v["bytes_fetched"]
                .as_u64()
                .ok_or_else(|| format!("{name}: bytes_fetched is missing"))?;
            views.push((
                name.clone(),
                BaselineView {
                    settle_seconds,
                    bytes_fetched,
                },
            ));
        }
        Ok(Baseline { views })
    }

    /// The accepted numbers of `view`, if any.
    pub fn view(&self, view: &str) -> Option<&BaselineView> {
        self.views.iter().find(|(n, _)| n == view).map(|(_, v)| v)
    }
}

/// The verdict on one view.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewReport {
    /// The view.
    pub view: String,
    /// Its statistics, `None` when the run wrote none.
    pub stats: Option<ViewStats>,
    /// What breaks the budget, empty when nothing does.
    pub breaches: Vec<String>,
    /// The summary line.
    pub line: String,
}

/// The change from `base` to `value` in percent, signed.
fn percent(value: f64, base: f64) -> String {
    if base > 0.0 {
        format!("{:+.1}%", 100.0 * (value / base - 1.0))
    } else {
        "base 0".into()
    }
}

/// Megabytes (10^6 bytes) with one decimal.
fn megabytes(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

/// A value with its unit, or `-` when there is none.
fn or_dash(v: Option<f64>, unit: &str, decimals: usize) -> String {
    match v {
        Some(v) => format!("{v:.decimals$} {unit}"),
        None => "-".into(),
    }
}

/// Checks one view against the limits and, when there is one, its
/// baseline. `stats` is `None` when the run wrote no statistics, which is a
/// breach.
pub fn check_view(
    view: &str,
    stats: Option<&ViewStats>,
    baseline: Option<&BaselineView>,
) -> ViewReport {
    let Some(s) = stats else {
        return ViewReport {
            view: view.into(),
            stats: None,
            breaches: vec!["no statistics".into()],
            line: format!("{view} BREACH: no statistics"),
        };
    };
    let mut breaches = Vec::new();
    match (s.settle_seconds, settle_limit(view)) {
        (None, _) => breaches.push("never settled".to_string()),
        (Some(t), Some(limit)) if t > limit => {
            breaches.push(format!("settle {t:.2} s over {limit} s"))
        }
        _ => {}
    }
    match s.step_p95_ms {
        Some(p) if p > STEP_P95_MS => {
            breaches.push(format!("step p95 {p:.2} ms over {STEP_P95_MS} ms"))
        }
        None => breaches.push("no frame steps".into()),
        _ => {}
    }
    if let Some(m) = s.step_max_ms.filter(|&m| m > STEP_MAX_MS) {
        breaches.push(format!("step max {m:.2} ms over {STEP_MAX_MS} ms"));
    }
    if s.cells_selected > MAX_CELLS {
        breaches.push(format!("{} cells over {MAX_CELLS}", s.cells_selected));
    }
    if s.triangles_drawn > MAX_TRIANGLES {
        breaches.push(format!(
            "{} triangles over {MAX_TRIANGLES}",
            s.triangles_drawn
        ));
    }
    let base = match baseline {
        None => "baseline -".to_string(),
        Some(b) => {
            if let Some(t) = s.settle_seconds {
                if t > b.settle_seconds * BASELINE_FACTOR {
                    breaches.push(format!(
                        "settle {t:.2} s over baseline {:.2} s plus 10%",
                        b.settle_seconds
                    ));
                }
            }
            if s.bytes_fetched as f64 > b.bytes_fetched as f64 * BASELINE_FACTOR {
                breaches.push(format!(
                    "{} fetched over baseline {} plus 10%",
                    megabytes(s.bytes_fetched),
                    megabytes(b.bytes_fetched)
                ));
            }
            format!(
                "baseline settle {}, bytes {}",
                s.settle_seconds
                    .map_or("-".into(), |t| percent(t, b.settle_seconds)),
                percent(s.bytes_fetched as f64, b.bytes_fetched as f64)
            )
        }
    };
    let verdict = if breaches.is_empty() {
        "ok".to_string()
    } else {
        format!("BREACH ({})", breaches.join("; "))
    };
    let line = format!(
        "{view} {verdict}: settle {}, step p95 {} max {}, cells {}, triangles {}, {}, {base}",
        or_dash(s.settle_seconds, "s", 2),
        or_dash(s.step_p95_ms, "ms", 2),
        or_dash(s.step_max_ms, "ms", 2),
        s.cells_selected,
        s.triangles_drawn,
        megabytes(s.bytes_fetched),
    );
    ViewReport {
        view: view.into(),
        stats: Some(s.clone()),
        breaches,
        line,
    }
}

/// The one overall line, for a commit message: the verdict, each view's
/// settle time, the worst step percentiles, the most cells and triangles,
/// the bytes of every view, and whether a baseline was compared.
pub fn overall(reports: &[ViewReport], baseline: Option<&Baseline>) -> String {
    let ok = reports.iter().all(|r| r.breaches.is_empty());
    let settle: Vec<String> = reports
        .iter()
        .map(|r| {
            let t = r.stats.as_ref().and_then(|s| s.settle_seconds);
            format!("{} {}", r.view, or_dash(t, "s", 2))
        })
        .collect();
    let stats: Vec<&ViewStats> = reports.iter().filter_map(|r| r.stats.as_ref()).collect();
    let worst = |f: fn(&ViewStats) -> Option<f64>| {
        stats
            .iter()
            .filter_map(|s| f(s))
            .reduce(f64::max)
            .map_or("-".into(), |v| format!("{v:.2} ms"))
    };
    let bytes: u64 = stats.iter().map(|s| s.bytes_fetched).sum();
    format!(
        "perf-budget {}: settle {}; step p95 {} max {}; cells {}; triangles {}; bytes {}; baseline {}",
        if ok { "ok" } else { "BREACH" },
        settle.join(", "),
        worst(|s| s.step_p95_ms),
        worst(|s| s.step_max_ms),
        stats.iter().map(|s| s.cells_selected).max().unwrap_or(0),
        stats.iter().map(|s| s.triangles_drawn).max().unwrap_or(0),
        megabytes(bytes),
        if baseline.is_some() {
            "compared"
        } else {
            "missing, skipped"
        },
    )
}
