//! `perf-budget`: checks the headless statistics of the benchmark views
//! against the performance budget and the committed baseline, for
//! `scripts/perf-budget.sh`. The rules are in the library docs
//! ([`perf_budget`]).
//!
//! Prints one line per view and one overall line, and exits 0 when nothing
//! breaches, 1 on any breach, and 2 on a usage or read error.

use perf_budget::{check_view, overall, Baseline, ViewStats, VIEWS};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
perf-budget: checks the benchmark views against the performance budget

USAGE:
    perf-budget <stats-dir> [--baseline <baseline.json>]
        reads <stats-dir>/B1.json to B4.json (gx-renderer --stats-json) and
        compares them with rendering.md section 5.2 and, when the file
        exists, the baseline (default perf/baseline.json)
";

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn run() -> Result<bool, String> {
    let mut args = std::env::args().skip(1);
    let mut dir: Option<PathBuf> = None;
    let mut baseline_path = PathBuf::from("perf/baseline.json");
    while let Some(a) = args.next() {
        match a.as_str() {
            "--baseline" => {
                baseline_path = PathBuf::from(args.next().ok_or("--baseline needs a path")?)
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            _ if dir.is_none() && !a.starts_with('-') => dir = Some(PathBuf::from(a)),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let dir = dir.ok_or_else(|| USAGE.to_string())?;
    let baseline = if baseline_path.exists() {
        Some(Baseline::from_json(&read_json(&baseline_path)?)?)
    } else {
        println!(
            "baseline: {} is missing; the baseline comparison is skipped (recorded on the real stack, not here)",
            baseline_path.display()
        );
        None
    };
    let mut reports = Vec::new();
    for view in VIEWS {
        let path = dir.join(format!("{view}.json"));
        let stats = if path.exists() {
            Some(ViewStats::from_json(&read_json(&path)?)?)
        } else {
            None
        };
        let report = check_view(
            view,
            stats.as_ref(),
            baseline.as_ref().and_then(|b| b.view(view)),
        );
        println!("{}", report.line);
        reports.push(report);
    }
    println!("{}", overall(&reports, baseline.as_ref()));
    Ok(reports.iter().all(|r| r.breaches.is_empty()))
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("perf-budget: {e}");
            ExitCode::from(2)
        }
    }
}
