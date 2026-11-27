//! Configuration from the environment, an optional `.env` file, and the
//! command line.
//!
//! Precedence, strongest first:
//!
//! 1. command line flags,
//! 2. variables in the process environment,
//! 3. variables in a `.env` file in the working directory, if present.
//!
//! The variables are `GX_HUB_URL`, `GX_API_KEY` (a key with the
//! `fetch:chunks` capability), `GX_SPACE_ID`, and the optional
//! `GX_BUILD_ID` and `GX_EDGE_URL`. When no build id is given, the hub
//! client resolves the active build through `GET
//! /space/{spaceId}/build/active` (see [`crate::hub`]). The edge URL, when
//! set, serves the immutable reads (sections, modules, source tiles, and
//! manifests); otherwise the hub serves them.
//!
//! The local cache (`client-compute.md` section 6) defaults to a directory
//! under the platform cache directory ([`gx_worker::cache::default_dir`]
//! natively) limited to 20 GB; `--cache-dir` moves it, `--cache-max-bytes`
//! changes the limit, and `--no-cache` turns it off. `--workers` sets the
//! number of worker threads that run modules (default: cores minus two, at
//! least one).
//!
//! The API key is held in an [`ApiKey`] whose `Debug` output is redacted, so
//! logging a [`Config`] never prints it.
//!
//! Every client enters at now (`space-model.md` section 7). `--start origin`
//! enters at the origin instead, and `--start-offset-seconds` moves the
//! start from either, in seconds; the result is clamped between the origin
//! and now. The time scale is the playback rate of a start in the past, a
//! plain ratio of simulation seconds per wall-clock second (negative plays
//! backwards); at now the rate is always 1.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

pub use crate::options::ViewOptions;
use crate::options::{parse_f64, OPTION_NAMES};

/// Name of the variable holding the hub base URL.
pub const ENV_HUB_URL: &str = "GX_HUB_URL";
/// Name of the variable holding the API key.
pub const ENV_API_KEY: &str = "GX_API_KEY";
/// Name of the variable holding the space id.
pub const ENV_SPACE_ID: &str = "GX_SPACE_ID";
/// Name of the optional variable holding the build id.
pub const ENV_BUILD_ID: &str = "GX_BUILD_ID";
/// Name of the optional variable naming the graphics adapter to use.
pub const ENV_ADAPTER: &str = "GX_ADAPTER";
/// Name of the optional variable holding the edge base URL for immutable
/// reads.
pub const ENV_EDGE_URL: &str = "GX_EDGE_URL";

/// Every variable the renderer reads, in a fixed order.
pub const ENV_KEYS: [&str; 6] = [
    ENV_HUB_URL,
    ENV_API_KEY,
    ENV_SPACE_ID,
    ENV_BUILD_ID,
    ENV_ADAPTER,
    ENV_EDGE_URL,
];

/// Default size limit of the local cache, bytes: 20 GB.
pub const DEFAULT_CACHE_MAX_BYTES: u64 = 20_000_000_000;

/// Default render width in pixels.
pub const DEFAULT_WIDTH: u32 = 1280;
/// Default render height in pixels.
pub const DEFAULT_HEIGHT: u32 = 720;

/// The usage text printed for `--help`.
pub const USAGE: &str = "\
gx-renderer: draws the frames of a 3GIX build

USAGE:
    gx-renderer [FLAGS]

ENVIRONMENT (also read from .env):
    GX_HUB_URL      hub base URL
    GX_API_KEY      API key with the fetch:chunks capability
    GX_SPACE_ID     space id
    GX_BUILD_ID     build id (optional; defaults to the active build)
    GX_EDGE_URL     base URL for sections, modules, and source tiles (optional;
                    defaults to the hub URL)
    GX_ADAPTER      graphics adapter to render with, by a part of its name (optional;
                    defaults to the high performance adapter, so the discrete GPU)

FLAGS:
    --start <now|origin>           enter at now (default) or at the origin
    --start-offset-seconds <f64>   simulation time offset from the start at launch, seconds,
                                   clamped between the origin and now (default 0)
    --time-scale <f64>             playback rate of a start in the past, simulation seconds
                                   per wall second, negative backwards (default 1)
    --headless                     no window; render to an offscreen texture
    --screenshot <path.png>        render one frame headless and write it as PNG
    --width <u32>                  render width in pixels (default 1280)
    --height <u32>                 render height in pixels (default 720)
    --view <home|0-9>              headless view: home (default) or the number key's view
    --view-distance-scale <f64>    multiplies the view's distance from its frame origin (default 1)
    --start-on <frame>:<x>,<y>,<z>[@[+]<m>[^<deg>]]
                                   spawn the camera onto frame <frame> along the direction
                                   x, y, z of its axes (as G does); with @+<m>, <m> meters
                                   above the drawn surface instead of 2 m; with @<m>, only
                                   place it <m> meters from the frame origin, looking at the
                                   origin (with ^<deg>: <deg> degrees below level instead)
    --wait-ready-seconds <f64>     headless: wait until every selected cell is ready, failing
                                   with a non-zero status after this many seconds
    --stats-json <path.json>       headless: write the overlay statistics as JSON
    --no-overlay                   headless: leave the overlay text, frame markers, and lines
                                   out of the image, drawing matter only
    --exposure-stops <f64>         fixed exposure, stops relative to the automatic exposure of a
                                   scene whose log-average luminance is 1 W m^-2 sr^-1; turns
                                   adaptation off, + and - change it (default: automatic)
    --mesh-mode <mode>             surfaces (default), vertices, or edges: how surface meshes
                                   are drawn; X cycles them in the window
    --hide-layers <id,id,...>      layers to hide, overriding the build's shown defaults; L
                                   toggles layers in the window; unknown ids are reported
    --compute-budget <n>           module workers this client may run, 0 to the default
                                   (cores - 2, 3/4 of them); 0 computes and uploads nothing
                                   and only streams what the hub stores
    --hub-url <url>                overrides GX_HUB_URL
    --space-id <id>                overrides GX_SPACE_ID
    --build-id <id>                overrides GX_BUILD_ID
    --adapter <name>               overrides GX_ADAPTER
    --edge-url <url>               overrides GX_EDGE_URL
    --cache-dir <path>             local cache directory (default: gx-renderer under the
                                   platform cache directory)
    --cache-max-bytes <u64>        local cache size limit in bytes (default 20000000000)
    --no-cache                     keep no local cache: every section, module, and tile
                                   comes from the edge or is computed
    --workers <u32>                threads that run modules (default: cores minus two, at
                                   least one)
    --max-fps <u32>                cap the window frame rate below the display refresh
                                   (default 0: one frame per refresh)
    --print-benchmark-start        print the benchmark views' start, <frame>:<x>,<y>,<z>, picked
                                   by rule from the hub's universe, and exit (rendering.md 5.1)
    --help                         print this text
";

/// An API key. Never printed: `Debug` and `Display` show a placeholder.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// Wraps a key.
    pub fn new(key: impl Into<String>) -> ApiKey {
        ApiKey(key.into())
    }

    /// The key itself, for the request header and nothing else.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl fmt::Display for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// Where the simulation time starts.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Start {
    /// At now, following now.
    #[default]
    Now,
    /// At the origin.
    Origin,
}

impl Start {
    /// Parses `now` or `origin`.
    pub fn parse(v: &str) -> Option<Start> {
        match v.trim() {
            "now" => Some(Start::Now),
            "origin" => Some(Start::Origin),
            _ => None,
        }
    }
}

impl fmt::Display for Start {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Start::Now => "now",
            Start::Origin => "origin",
        })
    }
}

/// The camera view a headless run renders.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum View {
    /// The `Home` view of the whole system.
    #[default]
    Home,
    /// The view a number key selects, by its index: key `1` is index 0,
    /// key `0` is index 9 (see [`crate::camera::selectable_frames`]).
    Key(usize),
    /// The view from along a direction of a frame (`--start-on`,
    /// [`crate::camera::Camera::over`]).
    Over(StartOn),
}

impl View {
    /// Parses `home` or a number key `0` to `9`.
    pub fn parse(v: &str) -> Option<View> {
        match v.trim() {
            "home" | "Home" => Some(View::Home),
            "0" => Some(View::Key(9)),
            d if d.len() == 1 => d
                .parse::<usize>()
                .ok()
                .filter(|&n| (1..=9).contains(&n))
                .map(|n| View::Key(n - 1)),
            _ => None,
        }
    }
}

impl fmt::Display for View {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            View::Home => f.write_str("home"),
            View::Key(i) => write!(f, "key {}", (i + 1) % 10),
            View::Over(s) => write!(
                f,
                "over frame {} along {}, {}, {}",
                s.frame_id, s.direction.x, s.direction.y, s.direction.z
            ),
        }
    }
}

/// The resolved configuration of one run.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Hub base URL, without a trailing slash.
    pub hub_url: String,
    /// API key with the `fetch:chunks` capability.
    pub api_key: ApiKey,
    /// The space to draw.
    pub space_id: String,
    /// The build to draw; `None` resolves the active build at startup.
    pub build_id: Option<String>,
    /// A part of the name of the graphics adapter to use; `None` lets the
    /// system pick the high performance adapter.
    pub adapter: Option<String>,
    /// Window frame rate cap, frames per second; 0 means one frame per
    /// display refresh (the present mode already limits it to that).
    pub max_fps: u32,
    /// Edge base URL for immutable reads, without a trailing slash; `None`
    /// reads them from the hub.
    pub edge_url: Option<String>,
    /// The local cache directory; `None` keeps no cache.
    pub cache_dir: Option<PathBuf>,
    /// Local cache size limit, bytes.
    pub cache_max_bytes: u64,
    /// Threads that run modules; `None` for the default.
    pub workers: Option<usize>,
    /// The view and time options (`crate::options`).
    pub options: ViewOptions,
    /// Render offscreen with no window.
    pub headless: bool,
    /// Write one headless frame to this PNG and exit. Implies `headless`.
    pub screenshot: Option<PathBuf>,
    /// Render width in pixels.
    pub width: u32,
    /// Render height in pixels.
    pub height: u32,
    /// Headless: wait at most this long, seconds, for every selected cell
    /// to be ready, and fail if they are not. `None` renders whatever
    /// arrived within [`crate::app::HEADLESS_STREAM_SECONDS`].
    pub wait_ready_seconds: Option<f64>,
    /// Headless: write the overlay statistics to this JSON file.
    pub stats_json: Option<PathBuf>,
    /// Draw the overlay text, the frame markers, and the lines between them
    /// (headless; the window always draws them).
    pub overlay: bool,
}

/// A frame, a direction from its origin in its axes, and optionally a
/// distance from the origin (`--start-on`). The direction is a unit
/// vector.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StartOn {
    /// The frame.
    pub frame_id: u64,
    /// Unit direction from the frame origin, frame axes.
    pub direction: gx_core::units::Vec3,
    /// Distance from the frame origin, meters (`@<m>`); `None` spawns
    /// onto the drawn surface.
    pub distance: Option<f64>,
    /// Height above the drawn surface when spawning, meters (`@+<m>`);
    /// `None` for [`crate::spawn::SPAWN_HEIGHT`].
    pub height: Option<f64>,
    /// With a distance: look this many degrees below level (toward the
    /// frame's `+z` heading) instead of at the frame origin.
    pub pitch_deg: Option<f64>,
}

impl StartOn {
    /// Parses `<frame>:<x>,<y>,<z>[@[+]<meters>[^<pitch degrees>]]`; the
    /// direction must not be zero, the distance or height must be above
    /// zero, and the pitch between -90 and 90.
    pub fn parse(s: &str) -> Option<StartOn> {
        let (place, mut distance, pitch_deg) = match s.split_once('@') {
            Some((p, rest)) => {
                let (d, pitch) = match rest.split_once('^') {
                    Some((d, a)) => {
                        let a: f64 = a.trim().parse().ok()?;
                        if !(-90.0..=90.0).contains(&a) {
                            return None;
                        }
                        (d, Some(a))
                    }
                    None => (rest, None),
                };
                let (above, d) = match d.trim().strip_prefix('+') {
                    Some(h) => (true, h),
                    None => (false, d.trim()),
                };
                let d: f64 = d.parse().ok()?;
                if !(d.is_finite() && d > 0.0) {
                    return None;
                }
                (p, Some((above, d)), pitch)
            }
            None => (s, None, None),
        };
        let height = match distance {
            Some((true, h)) => {
                distance = None;
                Some(h)
            }
            _ => None,
        };
        let distance = distance.map(|(_, d)| d);
        let (frame, dir) = place.split_once(':')?;
        let frame_id = frame.trim().parse().ok()?;
        let v: Vec<f64> = dir
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .ok()?;
        let [x, y, z] = v[..] else {
            return None;
        };
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return None;
        }
        let direction = gx_core::units::Vec3::new(x, y, z).normalized()?;
        Some(StartOn {
            frame_id,
            direction,
            distance,
            height,
            pitch_deg,
        })
    }
}

/// The command line flags, each `None` when not given.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Flags {
    /// The view and time options (`--start`, `--start-offset-seconds`,
    /// `--time-scale`, `--start-on`, `--view`, `--view-distance-scale`,
    /// `--exposure-stops`, `--mesh-mode`).
    pub options: ViewOptions,
    /// `--headless`.
    pub headless: bool,
    /// `--screenshot`.
    pub screenshot: Option<PathBuf>,
    /// `--width`.
    pub width: Option<u32>,
    /// `--height`.
    pub height: Option<u32>,
    /// `--wait-ready-seconds`.
    pub wait_ready_seconds: Option<f64>,
    /// `--stats-json`.
    pub stats_json: Option<PathBuf>,
    /// `--no-overlay`.
    pub no_overlay: bool,
    /// `--hub-url`.
    pub hub_url: Option<String>,
    /// `--space-id`.
    pub space_id: Option<String>,
    /// `--build-id`.
    pub build_id: Option<String>,
    /// `--adapter`.
    pub adapter: Option<String>,
    /// `--max-fps`.
    pub max_fps: Option<u32>,
    /// `--edge-url`.
    pub edge_url: Option<String>,
    /// `--cache-dir`.
    pub cache_dir: Option<PathBuf>,
    /// `--cache-max-bytes`.
    pub cache_max_bytes: Option<u64>,
    /// `--no-cache`.
    pub no_cache: bool,
    /// `--workers`.
    pub workers: Option<u32>,
    /// `--print-benchmark-start`.
    pub print_benchmark_start: bool,
    /// `--help`.
    pub help: bool,
}

/// A configuration problem, reported to the user before anything starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

fn err(msg: impl Into<String>) -> ConfigError {
    ConfigError(msg.into())
}

/// Parses the command line flags, without the program name.
pub fn parse_flags<I, S>(args: I) -> Result<Flags, ConfigError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut flags = Flags::default();
    let mut args = args.into_iter().map(Into::into);
    while let Some(arg) = args.next() {
        // Accept both `--flag value` and `--flag=value`.
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String, ConfigError> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => args
                    .next()
                    .ok_or_else(|| err(format!("{name} needs a value"))),
            }
        };
        if let Some(option) = name.strip_prefix("--") {
            if OPTION_NAMES.contains(&option) {
                let v = value(&name)?;
                flags.options.set(option, &name, &v)?;
                continue;
            }
        }
        match name.as_str() {
            "--headless" => flags.headless = true,
            "--screenshot" => flags.screenshot = Some(PathBuf::from(value(&name)?)),
            "--width" => flags.width = Some(parse_u32(&name, &value(&name)?)?),
            "--height" => flags.height = Some(parse_u32(&name, &value(&name)?)?),
            "--wait-ready-seconds" => {
                let x = parse_f64(&name, &value(&name)?)?;
                if x < 0.0 {
                    return Err(err(format!("{name}: must not be negative")));
                }
                flags.wait_ready_seconds = Some(x)
            }
            "--stats-json" => flags.stats_json = Some(PathBuf::from(value(&name)?)),
            "--no-overlay" => flags.no_overlay = true,
            "--hub-url" => flags.hub_url = Some(value(&name)?),
            "--space-id" => flags.space_id = Some(value(&name)?),
            "--build-id" => flags.build_id = Some(value(&name)?),
            "--adapter" => flags.adapter = Some(value(&name)?),
            "--max-fps" => flags.max_fps = Some(parse_u32(&name, &value(&name)?)?),
            "--edge-url" => flags.edge_url = Some(value(&name)?),
            "--cache-dir" => flags.cache_dir = Some(PathBuf::from(value(&name)?)),
            "--cache-max-bytes" => {
                let v = value(&name)?;
                flags.cache_max_bytes = Some(
                    v.trim()
                        .parse()
                        .map_err(|_| err(format!("{name}: {v:?} is not a byte count")))?,
                )
            }
            "--no-cache" => flags.no_cache = true,
            "--workers" => flags.workers = Some(parse_u32(&name, &value(&name)?)?),
            "--print-benchmark-start" => flags.print_benchmark_start = true,
            "--help" | "-h" => flags.help = true,
            other => return Err(err(format!("unknown argument {other:?}, see --help"))),
        }
    }
    Ok(flags)
}

fn parse_u32(name: &str, v: &str) -> Result<u32, ConfigError> {
    let x: u32 = v
        .trim()
        .parse()
        .map_err(|_| err(format!("{name}: {v:?} is not a positive integer")))?;
    if x == 0 {
        return Err(err(format!("{name}: must be above zero")));
    }
    Ok(x)
}

/// Layers the environment sources: values from a `.env` file, overridden by
/// values from the process environment. Only the [`ENV_KEYS`] are kept.
/// Empty values count as absent.
pub fn layer_env(
    dotenv: impl IntoIterator<Item = (String, String)>,
    process: impl Fn(&str) -> Option<String>,
) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = dotenv
        .into_iter()
        .filter(|(k, v)| ENV_KEYS.contains(&k.as_str()) && !v.trim().is_empty())
        .collect();
    for key in ENV_KEYS {
        if let Some(v) = process(key).filter(|v| !v.trim().is_empty()) {
            env.insert(key.to_string(), v);
        }
    }
    env
}

/// Reads `.env` from the working directory, if present, and layers the
/// process environment over it with [`layer_env`]. Never logs a value.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_env() -> Result<BTreeMap<String, String>, ConfigError> {
    let dotenv: Vec<(String, String)> = match dotenvy::from_path_iter(".env") {
        Ok(iter) => iter
            .collect::<Result<_, _>>()
            .map_err(|e| err(format!(".env: {e}")))?,
        Err(e) if e.not_found() => Vec::new(),
        Err(e) => return Err(err(format!(".env: {e}"))),
    };
    Ok(layer_env(dotenv, |k| std::env::var(k).ok()))
}

impl Config {
    /// Resolves the configuration from the layered environment (see
    /// [`layer_env`]) and the parsed flags. Flags win.
    pub fn resolve(env: &BTreeMap<String, String>, flags: &Flags) -> Result<Config, ConfigError> {
        let pick = |flag: &Option<String>, key: &str| -> Option<String> {
            flag.clone()
                .filter(|v| !v.trim().is_empty())
                .or_else(|| env.get(key).cloned())
        };
        let hub_url = pick(&flags.hub_url, ENV_HUB_URL)
            .ok_or_else(|| err(format!("{ENV_HUB_URL} is not set (or pass --hub-url)")))?;
        let hub_url = hub_url.trim().trim_end_matches('/').to_string();
        if !(hub_url.starts_with("http://") || hub_url.starts_with("https://")) {
            return Err(err(format!(
                "{ENV_HUB_URL} must start with http:// or https://"
            )));
        }
        let api_key = env
            .get(ENV_API_KEY)
            .map(|k| ApiKey::new(k.trim()))
            .ok_or_else(|| err(format!("{ENV_API_KEY} is not set")))?;
        let space_id = pick(&flags.space_id, ENV_SPACE_ID)
            .ok_or_else(|| err(format!("{ENV_SPACE_ID} is not set (or pass --space-id)")))?;
        let build_id = pick(&flags.build_id, ENV_BUILD_ID);
        let edge_url = pick(&flags.edge_url, ENV_EDGE_URL)
            .map(|e| e.trim().trim_end_matches('/').to_string())
            .filter(|e| !e.is_empty());
        if let Some(e) = &edge_url {
            if !(e.starts_with("http://") || e.starts_with("https://")) {
                return Err(err(format!(
                    "{ENV_EDGE_URL} must start with http:// or https://"
                )));
            }
        }
        if flags.no_cache && flags.cache_dir.is_some() {
            return Err(err("--no-cache and --cache-dir contradict each other"));
        }
        let cache_dir = if flags.no_cache {
            None
        } else {
            flags.cache_dir.clone().or_else(default_cache_dir)
        };
        Ok(Config {
            hub_url,
            api_key,
            space_id: space_id.trim().to_string(),
            build_id: build_id.map(|b| b.trim().to_string()),
            options: flags.options.clone(),
            headless: flags.headless || flags.screenshot.is_some(),
            screenshot: flags.screenshot.clone(),
            width: flags.width.unwrap_or(DEFAULT_WIDTH),
            height: flags.height.unwrap_or(DEFAULT_HEIGHT),
            wait_ready_seconds: flags.wait_ready_seconds,
            stats_json: flags.stats_json.clone(),
            overlay: !flags.no_overlay,
            adapter: pick(&flags.adapter, ENV_ADAPTER).filter(|a| !a.trim().is_empty()),
            max_fps: flags.max_fps.unwrap_or(0),
            edge_url,
            cache_dir,
            cache_max_bytes: flags.cache_max_bytes.unwrap_or(DEFAULT_CACHE_MAX_BYTES),
            workers: flags.workers.map(|w| w as usize),
        })
    }
}

/// The default local cache directory: [`gx_worker::cache::default_dir`]
/// natively, none in the browser (which uses the Cache API instead).
fn default_cache_dir() -> Option<PathBuf> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        gx_worker::cache::default_dir()
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn base_env() -> BTreeMap<String, String> {
        layer_env(
            pairs(&[
                (ENV_HUB_URL, "http://hub.invalid/"),
                (ENV_API_KEY, "k"),
                (ENV_SPACE_ID, "s"),
            ]),
            |_| None,
        )
    }

    #[test]
    fn process_env_overrides_dotenv() {
        let env = layer_env(
            pairs(&[
                (ENV_HUB_URL, "http://from-file.invalid"),
                (ENV_SPACE_ID, "file-space"),
                ("UNRELATED", "x"),
            ]),
            |k| (k == ENV_HUB_URL).then(|| "http://from-process.invalid".to_string()),
        );
        assert_eq!(env[ENV_HUB_URL], "http://from-process.invalid");
        assert_eq!(env[ENV_SPACE_ID], "file-space");
        assert!(!env.contains_key("UNRELATED"));
    }

    #[test]
    fn empty_process_value_does_not_override() {
        let env = layer_env(pairs(&[(ENV_SPACE_ID, "file")]), |_| Some(" ".into()));
        assert_eq!(env[ENV_SPACE_ID], "file");
    }

    #[test]
    fn flags_override_env() {
        let mut env = base_env();
        env.insert(ENV_BUILD_ID.into(), "env-build".into());
        let flags = parse_flags([
            "--hub-url",
            "https://flag.invalid",
            "--space-id=flag-space",
            "--build-id",
            "flag-build",
        ])
        .unwrap();
        let c = Config::resolve(&env, &flags).unwrap();
        assert_eq!(c.hub_url, "https://flag.invalid");
        assert_eq!(c.space_id, "flag-space");
        assert_eq!(c.build_id.as_deref(), Some("flag-build"));
    }

    #[test]
    fn defaults_and_env_only() {
        let c = Config::resolve(&base_env(), &Flags::default()).unwrap();
        assert_eq!(c.hub_url, "http://hub.invalid");
        assert_eq!(c.build_id, None);
        assert_eq!(c.options.time_scale, 1.0);
        assert_eq!(c.options.start_offset_seconds, 0.0);
        assert_eq!(c.options.start, Start::Now);
        assert!(!c.headless);
        assert_eq!((c.width, c.height), (DEFAULT_WIDTH, DEFAULT_HEIGHT));
        assert_eq!(c.options.effective_view(), View::Home);
        assert_eq!(c.options.view_distance_scale, 1.0);
        assert_eq!(c.wait_ready_seconds, None);
        assert_eq!(c.stats_json, None);
        assert!(c.overlay);
        assert_eq!(c.options.exposure_stops, None);
        assert_eq!(c.adapter, None);
        assert_eq!(c.max_fps, 0);
        assert_eq!(c.edge_url, None);
        assert_eq!(c.cache_max_bytes, DEFAULT_CACHE_MAX_BYTES);
        assert_eq!(c.workers, None);
    }

    #[test]
    fn cache_edge_and_worker_flags() {
        let flags = parse_flags([
            "--cache-dir",
            "/tmp/c",
            "--cache-max-bytes=1000",
            "--workers",
            "3",
            "--edge-url",
            "https://edge.invalid/",
        ])
        .unwrap();
        let c = Config::resolve(&base_env(), &flags).unwrap();
        assert_eq!(c.cache_dir, Some(PathBuf::from("/tmp/c")));
        assert_eq!(c.cache_max_bytes, 1000);
        assert_eq!(c.workers, Some(3));
        assert_eq!(c.edge_url.as_deref(), Some("https://edge.invalid"));
        let c = Config::resolve(&base_env(), &parse_flags(["--no-cache"]).unwrap()).unwrap();
        assert_eq!(c.cache_dir, None);
        assert!(Config::resolve(
            &base_env(),
            &parse_flags(["--no-cache", "--cache-dir", "x"]).unwrap()
        )
        .is_err());
        assert!(parse_flags(["--cache-max-bytes", "lots"]).is_err());
        assert!(parse_flags(["--workers", "0"]).is_err());
        assert!(Config::resolve(
            &base_env(),
            &parse_flags(["--edge-url", "ftp://x.invalid"]).unwrap()
        )
        .is_err());
    }

    #[test]
    fn adapter_and_frame_cap() {
        let mut env = base_env();
        env.insert(ENV_ADAPTER.into(), "Intel".into());
        let c = Config::resolve(&env, &Flags::default()).unwrap();
        assert_eq!(c.adapter.as_deref(), Some("Intel"), "GX_ADAPTER is read");
        let flags = parse_flags(["--adapter", "NVIDIA", "--max-fps=30"]).unwrap();
        let c = Config::resolve(&env, &flags).unwrap();
        assert_eq!(c.adapter.as_deref(), Some("NVIDIA"), "the flag wins");
        assert_eq!(c.max_fps, 30);
        let flags = parse_flags(["--adapter", "  "]).unwrap();
        let c = Config::resolve(&base_env(), &flags).unwrap();
        assert_eq!(c.adapter, None, "a blank adapter name means no preference");
        assert!(parse_flags(["--max-fps", "fast"]).is_err());
    }

    #[test]
    fn hide_layers_flag() {
        let flags = parse_flags(["--hide-layers", "base,stain"]).unwrap();
        let c = Config::resolve(&base_env(), &flags).unwrap();
        assert_eq!(
            c.options.hide_layers.as_deref(),
            Some(&["base".to_string(), "stain".to_string()][..])
        );
        let c = Config::resolve(&base_env(), &parse_flags(["--hide-layers="]).unwrap()).unwrap();
        assert_eq!(c.options.hide_layers, Some(Vec::new()));
        let c = Config::resolve(&base_env(), &parse_flags(Vec::<String>::new()).unwrap()).unwrap();
        assert_eq!(c.options.hide_layers, None);
    }

    #[test]
    fn view_and_headless_flags() {
        let flags = parse_flags([
            "--view",
            "3",
            "--view-distance-scale=0.25",
            "--wait-ready-seconds",
            "120",
            "--stats-json",
            "s.json",
            "--no-overlay",
            "--exposure-stops=12.75",
        ])
        .unwrap();
        let c = Config::resolve(&base_env(), &flags).unwrap();
        assert_eq!(c.options.exposure_stops, Some(12.75));
        assert!(parse_flags(["--exposure-stops", "bright"]).is_err());
        assert_eq!(
            parse_flags(["--exposure-stops", "-3"])
                .unwrap()
                .options
                .exposure_stops,
            Some(-3.0)
        );
        assert_eq!(c.options.effective_view(), View::Key(2));
        assert_eq!(c.options.view_distance_scale, 0.25);
        assert_eq!(c.wait_ready_seconds, Some(120.0));
        assert_eq!(c.stats_json, Some(PathBuf::from("s.json")));
        assert!(!c.overlay);
        assert_eq!(View::parse("0"), Some(View::Key(9)));
        assert_eq!(View::parse("1"), Some(View::Key(0)));
        assert_eq!(View::parse("home"), Some(View::Home));
        assert_eq!(View::Key(9).to_string(), "key 0");
        assert!(parse_flags(["--view", "10"]).is_err());
        assert!(parse_flags(["--view", "x"]).is_err());
        assert!(parse_flags(["--view-distance-scale", "0"]).is_err());
        assert!(parse_flags(["--wait-ready-seconds", "-1"]).is_err());
        assert!(parse_flags(["--start-on", "4:0,0,0"]).is_err());
        assert!(parse_flags(["--start-on", "4:1,2"]).is_err());
        assert!(parse_flags(["--start-on", "x:1,0,0"]).is_err());
        assert!(parse_flags(["--start-on", "4:1,0,0@0"]).is_err());
    }

    #[test]
    fn start_on_is_a_frame_and_a_unit_direction() {
        let flags = parse_flags(["--start-on", "4:0,3,4"]).unwrap();
        let s = flags.options.start_on.unwrap();
        assert_eq!(s.frame_id, 4);
        assert!((s.direction.y - 0.6).abs() < 1e-12 && (s.direction.z - 0.8).abs() < 1e-12);
        assert_eq!(s.distance, None);
        let s = parse_flags(["--start-on", "4:1,0,0@5e6"])
            .unwrap()
            .options
            .start_on
            .unwrap();
        assert_eq!(s.distance, Some(5e6));
        let s = parse_flags(["--start-on", "4:1,0,0@5e6^-20"])
            .unwrap()
            .options
            .start_on
            .unwrap();
        assert_eq!(s.pitch_deg, Some(-20.0));
        assert!(parse_flags(["--start-on", "4:1,0,0@5e6^-100"]).is_err());
        let s = parse_flags(["--start-on", "4:1,0,0@+1e4^20"])
            .unwrap()
            .options
            .start_on
            .unwrap();
        assert_eq!(
            (s.distance, s.height, s.pitch_deg),
            (None, Some(1e4), Some(20.0))
        );
        assert!(parse_flags(["--start-on", "4:1,0,0@+0"]).is_err());
        assert!(parse_flags(["--start-on", "4:1,0,0@+-5"]).is_err());
    }

    #[test]
    fn numeric_flags() {
        let flags = parse_flags([
            "--time-scale",
            "51200",
            "--start-offset-seconds",
            "-3.5e6",
            "--start",
            "origin",
            "--width",
            "640",
            "--height=360",
            "--screenshot",
            "out.png",
        ])
        .unwrap();
        let c = Config::resolve(&base_env(), &flags).unwrap();
        assert_eq!(c.options.time_scale, 51200.0);
        assert_eq!(c.options.start_offset_seconds, -3.5e6);
        assert_eq!(c.options.start, Start::Origin);
        assert_eq!(Start::Origin.to_string(), "origin");
        assert!(parse_flags(["--start", "later"]).is_err());
        assert_eq!((c.width, c.height), (640, 360));
        assert!(c.headless, "a screenshot implies headless");
        assert_eq!(c.screenshot, Some(PathBuf::from("out.png")));
    }

    #[test]
    fn bad_flags_are_errors() {
        assert!(parse_flags(["--time-scale"]).is_err());
        assert!(parse_flags(["--time-scale", "fast"]).is_err());
        assert!(parse_flags(["--time-scale", "inf"]).is_err());
        assert!(parse_flags(["--width", "0"]).is_err());
        assert!(parse_flags(["--bogus"]).is_err());
        assert!(parse_flags(["--help"]).unwrap().help);
    }

    #[test]
    fn missing_required_values() {
        let env = layer_env(pairs(&[(ENV_HUB_URL, "http://h.invalid")]), |_| None);
        let e = Config::resolve(&env, &Flags::default()).unwrap_err();
        assert!(e.0.contains(ENV_API_KEY));
        let e = Config::resolve(&BTreeMap::new(), &Flags::default()).unwrap_err();
        assert!(e.0.contains(ENV_HUB_URL));
    }

    #[test]
    fn api_key_is_never_printed() {
        let mut env = base_env();
        env.insert(ENV_API_KEY.into(), "very-secret-value".into());
        let c = Config::resolve(&env, &Flags::default()).unwrap();
        let shown = format!("{c:?} {}", c.api_key);
        assert!(!shown.contains("very-secret-value"));
        assert_eq!(c.api_key.expose(), "very-secret-value");
    }
}
