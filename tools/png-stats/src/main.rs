//! `png-stats`: assertions on rendered screenshots for `scripts/e2e.sh`.
//!
//! Each check prints one line, `PASS <check>: <measures>` or
//! `FAIL <check>: <measures>`, and exits 0 when it passed, 1 when it
//! failed, and 2 on a usage or read error. The measures are described in
//! the library docs ([`png_stats`]).

use png_stats::{bright_centroid, components, disc_sides, falloff, Gray};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "\
png-stats: assertions on rendered screenshots

USAGE:
    png-stats summary <png> [--threshold <luma>]
        luma percentiles, bright pixel count and centroid, components
    png-stats blobs <png> --threshold <luma> --min-count <n> [--min-pixels <n>]
        at least n 8-connected components above the threshold, counting
        only components of at least min-pixels pixels (default 1)
    png-stats disc <png> --threshold <luma> --min-percent <p> --max-percent <p>
                   --min-side-ratio <r>
        the largest component above the threshold covers between the two
        percentages of the image, and its brighter side is at least r times
        as bright as its darker side
    png-stats moved <a.png> <b.png> --threshold <luma> --min-shift <px>
        the centroid of the pixels above the threshold moves at least px
        pixels from a to b
    png-stats smooth <png> --min-range <luma> --max-step-fraction <f>
        the luma falls by at least min-range (99th minus 1st percentile)
        and no step between neighboring pixels exceeds f times that range

Luma is 0.2126 R + 0.7152 G + 0.0722 B of the 8-bit values, 0 to 255.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("png-stats: {e}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Positional arguments and `--name value` options.
struct Args {
    positional: Vec<String>,
    options: BTreeMap<String, String>,
}

impl Args {
    fn parse(args: &[String]) -> Result<Args, String> {
        let mut positional = Vec::new();
        let mut options = BTreeMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                let v = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
                options.insert(name.to_string(), v.clone());
            } else {
                positional.push(a.clone());
            }
        }
        Ok(Args {
            positional,
            options,
        })
    }

    fn number(&self, name: &str) -> Result<f64, String> {
        let v = self
            .options
            .get(name)
            .ok_or_else(|| format!("--{name} is required"))?;
        v.parse::<f64>()
            .ok()
            .filter(|x| x.is_finite())
            .ok_or_else(|| format!("--{name}: {v:?} is not a number"))
    }

    fn number_or(&self, name: &str, default: f64) -> Result<f64, String> {
        if self.options.contains_key(name) {
            self.number(name)
        } else {
            Ok(default)
        }
    }

    fn image(&self, index: usize) -> Result<(String, Gray), String> {
        let p = self
            .positional
            .get(index)
            .ok_or_else(|| "missing image path".to_string())?;
        Ok((p.clone(), Gray::load(Path::new(p))?))
    }
}

fn verdict(name: &str, file: &str, ok: bool, detail: String) -> bool {
    println!(
        "{} {name} {file}: {detail}",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn run(args: &[String]) -> Result<bool, String> {
    let Some((cmd, rest)) = args.split_first() else {
        return Err("no command".into());
    };
    let a = Args::parse(rest)?;
    match cmd.as_str() {
        "summary" => {
            let (file, img) = a.image(0)?;
            let t = a.number_or("threshold", 64.0)?;
            let f = falloff(&img);
            let c = components(&img, t);
            let centroid = bright_centroid(&img, t);
            println!(
                "{file}: {}x{}, luma p1 {:.1} p99 {:.1} max step {:.1}, above {t}: {} px in {} components, centroid {}",
                img.width,
                img.height,
                f.low,
                f.high,
                f.max_step,
                centroid.map_or(0, |(_, n)| n),
                c.len(),
                centroid.map_or("-".into(), |((x, y), _)| format!("({x:.1}, {y:.1})")),
            );
            Ok(true)
        }
        "blobs" => {
            let (file, img) = a.image(0)?;
            let t = a.number("threshold")?;
            let min_count = a.number("min-count")?;
            let min_pixels = a.number_or("min-pixels", 1.0)?;
            let all = components(&img, t);
            let sizes: Vec<usize> = all
                .iter()
                .map(|c| c.pixels.len())
                .filter(|&n| n as f64 >= min_pixels)
                .collect();
            let n = sizes.len();
            Ok(verdict(
                "blobs",
                &file,
                n as f64 >= min_count,
                format!(
                    "{n} components above luma {t} of at least {min_pixels} px (need {min_count}), sizes {sizes:?}"
                ),
            ))
        }
        "disc" => {
            let (file, img) = a.image(0)?;
            let t = a.number("threshold")?;
            let (lo, hi) = (a.number("min-percent")?, a.number("max-percent")?);
            let min_ratio = a.number("min-side-ratio")?;
            let all = components(&img, t);
            let Some(big) = all.iter().max_by_key(|c| c.pixels.len()) else {
                return Ok(verdict(
                    "disc",
                    &file,
                    false,
                    format!("no pixel above luma {t}"),
                ));
            };
            let pct = 100.0 * big.pixels.len() as f64 / img.len() as f64;
            let s = disc_sides(&img, big);
            let ok = pct >= lo && pct <= hi && s.ratio() >= min_ratio;
            Ok(verdict(
                "disc",
                &file,
                ok,
                format!(
                    "largest of {} components above luma {t} covers {pct:.2} percent (need {lo} to {hi}), centroid ({:.1}, {:.1}), brighter side mean {:.1}, darker side mean {:.1}, ratio {:.2} (need {min_ratio}), lit toward ({:.2}, {:.2})",
                    all.len(),
                    big.centroid.0,
                    big.centroid.1,
                    s.bright_mean,
                    s.dark_mean,
                    s.ratio(),
                    s.direction.0,
                    s.direction.1
                ),
            ))
        }
        "moved" => {
            let (fa, ia) = a.image(0)?;
            let (fb, ib) = a.image(1)?;
            let t = a.number("threshold")?;
            let min_shift = a.number("min-shift")?;
            let name = format!("{fa} -> {fb}");
            let (Some(((ax, ay), na)), Some(((bx, by), nb))) =
                (bright_centroid(&ia, t), bright_centroid(&ib, t))
            else {
                return Ok(verdict(
                    "moved",
                    &name,
                    false,
                    format!("an image has no pixel above luma {t}"),
                ));
            };
            let shift = (bx - ax).hypot(by - ay);
            Ok(verdict(
                "moved",
                &name,
                shift >= min_shift,
                format!(
                    "bright centroid ({ax:.1}, {ay:.1}) over {na} px -> ({bx:.1}, {by:.1}) over {nb} px, shift {shift:.1} px (need {min_shift})"
                ),
            ))
        }
        "smooth" => {
            let (file, img) = a.image(0)?;
            let min_range = a.number("min-range")?;
            let max_fraction = a.number("max-step-fraction")?;
            let f = falloff(&img);
            let ok = f.range() >= min_range && f.step_fraction() <= max_fraction;
            Ok(verdict(
                "smooth",
                &file,
                ok,
                format!(
                    "luma p1 {:.1} p99 {:.1}, range {:.1} (need {min_range}), largest neighbor step {:.1} at ({}, {}), {:.3} of the range (need at most {max_fraction})",
                    f.low,
                    f.high,
                    f.range(),
                    f.max_step,
                    f.max_step_at.0,
                    f.max_step_at.1,
                    f.step_fraction()
                ),
            ))
        }
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(true)
        }
        other => Err(format!("unknown command {other:?}")),
    }
}
