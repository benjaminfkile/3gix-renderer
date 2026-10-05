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
//! `GX_BUILD_ID`. When no build id is given, the hub client resolves the
//! active build through `GET /space/{spaceId}/builds` (see [`crate::hub`]).
//!
//! The API key is held in an [`ApiKey`] whose `Debug` output is redacted, so
//! logging a [`Config`] never prints it.
//!
//! The simulation time offset is in seconds and the time scale is a plain
//! ratio of simulation seconds per wall-clock second (`space-model.md`
//! section 6).

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

/// Name of the variable holding the hub base URL.
pub const ENV_HUB_URL: &str = "GX_HUB_URL";
/// Name of the variable holding the API key.
pub const ENV_API_KEY: &str = "GX_API_KEY";
/// Name of the variable holding the space id.
pub const ENV_SPACE_ID: &str = "GX_SPACE_ID";
/// Name of the optional variable holding the build id.
pub const ENV_BUILD_ID: &str = "GX_BUILD_ID";

/// Every variable the renderer reads, in a fixed order.
pub const ENV_KEYS: [&str; 4] = [ENV_HUB_URL, ENV_API_KEY, ENV_SPACE_ID, ENV_BUILD_ID];

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

FLAGS:
    --time-scale <f64>             simulation seconds per wall second (default 1)
    --start-offset-seconds <f64>   simulation time offset from the epoch at launch (default 0)
    --headless                     no window; render to an offscreen texture
    --screenshot <path.png>        render one frame headless and write it as PNG
    --width <u32>                  render width in pixels (default 1280)
    --height <u32>                 render height in pixels (default 720)
    --view <home|0-9>              headless view: home (default) or the number key's view
    --view-distance-scale <f64>    multiplies the view's distance from its frame origin (default 1)
    --wait-ready-seconds <f64>     headless: wait until every selected cell is ready, failing
                                   with a non-zero status after this many seconds
    --stats-json <path.json>       headless: write the overlay statistics as JSON
    --no-overlay                   headless: leave the overlay text, frame markers, and lines
                                   out of the image, drawing matter only
    --exposure-stops <f64>         fixed exposure, stops relative to the automatic exposure of a
                                   scene whose log-average luminance is 1 W m^-2 sr^-1; turns
                                   adaptation off, + and - change it (default: automatic)
    --hub-url <url>                overrides GX_HUB_URL
    --space-id <id>                overrides GX_SPACE_ID
    --build-id <id>                overrides GX_BUILD_ID
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

/// The camera view a headless run renders.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum View {
    /// The `Home` view of the whole system.
    #[default]
    Home,
    /// The view a number key selects, by its index: key `1` is index 0,
    /// key `0` is index 9 (see [`crate::camera::selectable_frames`]).
    Key(usize),
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
    /// Simulation seconds per wall-clock second.
    pub time_scale: f64,
    /// Simulation time offset from the registry epoch at launch, seconds.
    pub start_offset_seconds: f64,
    /// Render offscreen with no window.
    pub headless: bool,
    /// Write one headless frame to this PNG and exit. Implies `headless`.
    pub screenshot: Option<PathBuf>,
    /// Render width in pixels.
    pub width: u32,
    /// Render height in pixels.
    pub height: u32,
    /// The view a headless run renders.
    pub view: View,
    /// Factor on the view's distance from its frame origin.
    pub view_distance_scale: f64,
    /// Headless: wait at most this long, seconds, for every selected cell
    /// to be ready, and fail if they are not. `None` renders whatever
    /// arrived within [`crate::app::HEADLESS_STREAM_SECONDS`].
    pub wait_ready_seconds: Option<f64>,
    /// Headless: write the overlay statistics to this JSON file.
    pub stats_json: Option<PathBuf>,
    /// Draw the overlay text, the frame markers, and the lines between them
    /// (headless; the window always draws them).
    pub overlay: bool,
    /// A fixed exposure, stops relative to the automatic exposure of the
    /// reference scene ([`crate::render::gpu::fixed_exposure`]); `None`
    /// adapts to the scene.
    pub exposure_stops: Option<f64>,
}

/// The command line flags, each `None` when not given.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Flags {
    /// `--time-scale`.
    pub time_scale: Option<f64>,
    /// `--start-offset-seconds`.
    pub start_offset_seconds: Option<f64>,
    /// `--headless`.
    pub headless: bool,
    /// `--screenshot`.
    pub screenshot: Option<PathBuf>,
    /// `--width`.
    pub width: Option<u32>,
    /// `--height`.
    pub height: Option<u32>,
    /// `--view`.
    pub view: Option<View>,
    /// `--view-distance-scale`.
    pub view_distance_scale: Option<f64>,
    /// `--wait-ready-seconds`.
    pub wait_ready_seconds: Option<f64>,
    /// `--stats-json`.
    pub stats_json: Option<PathBuf>,
    /// `--no-overlay`.
    pub no_overlay: bool,
    /// `--exposure-stops`.
    pub exposure_stops: Option<f64>,
    /// `--hub-url`.
    pub hub_url: Option<String>,
    /// `--space-id`.
    pub space_id: Option<String>,
    /// `--build-id`.
    pub build_id: Option<String>,
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
        match name.as_str() {
            "--time-scale" => flags.time_scale = Some(parse_f64(&name, &value(&name)?)?),
            "--start-offset-seconds" => {
                flags.start_offset_seconds = Some(parse_f64(&name, &value(&name)?)?)
            }
            "--headless" => flags.headless = true,
            "--screenshot" => flags.screenshot = Some(PathBuf::from(value(&name)?)),
            "--width" => flags.width = Some(parse_u32(&name, &value(&name)?)?),
            "--height" => flags.height = Some(parse_u32(&name, &value(&name)?)?),
            "--view" => {
                let v = value(&name)?;
                flags.view = Some(
                    View::parse(&v)
                        .ok_or_else(|| err(format!("{name}: {v:?} is not home or 0 to 9")))?,
                )
            }
            "--view-distance-scale" => {
                let x = parse_f64(&name, &value(&name)?)?;
                if x <= 0.0 {
                    return Err(err(format!("{name}: must be above zero")));
                }
                flags.view_distance_scale = Some(x)
            }
            "--wait-ready-seconds" => {
                let x = parse_f64(&name, &value(&name)?)?;
                if x < 0.0 {
                    return Err(err(format!("{name}: must not be negative")));
                }
                flags.wait_ready_seconds = Some(x)
            }
            "--stats-json" => flags.stats_json = Some(PathBuf::from(value(&name)?)),
            "--no-overlay" => flags.no_overlay = true,
            "--exposure-stops" => flags.exposure_stops = Some(parse_f64(&name, &value(&name)?)?),
            "--hub-url" => flags.hub_url = Some(value(&name)?),
            "--space-id" => flags.space_id = Some(value(&name)?),
            "--build-id" => flags.build_id = Some(value(&name)?),
            "--help" | "-h" => flags.help = true,
            other => return Err(err(format!("unknown argument {other:?}, see --help"))),
        }
    }
    Ok(flags)
}

fn parse_f64(name: &str, v: &str) -> Result<f64, ConfigError> {
    let x: f64 = v
        .trim()
        .parse()
        .map_err(|_| err(format!("{name}: {v:?} is not a number")))?;
    if !x.is_finite() {
        return Err(err(format!("{name}: {v:?} is not finite")));
    }
    Ok(x)
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
        Ok(Config {
            hub_url,
            api_key,
            space_id: space_id.trim().to_string(),
            build_id: build_id.map(|b| b.trim().to_string()),
            time_scale: flags.time_scale.unwrap_or(1.0),
            start_offset_seconds: flags.start_offset_seconds.unwrap_or(0.0),
            headless: flags.headless || flags.screenshot.is_some(),
            screenshot: flags.screenshot.clone(),
            width: flags.width.unwrap_or(DEFAULT_WIDTH),
            height: flags.height.unwrap_or(DEFAULT_HEIGHT),
            view: flags.view.unwrap_or_default(),
            view_distance_scale: flags.view_distance_scale.unwrap_or(1.0),
            wait_ready_seconds: flags.wait_ready_seconds,
            stats_json: flags.stats_json.clone(),
            overlay: !flags.no_overlay,
            exposure_stops: flags.exposure_stops,
        })
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
        assert_eq!(c.time_scale, 1.0);
        assert_eq!(c.start_offset_seconds, 0.0);
        assert!(!c.headless);
        assert_eq!((c.width, c.height), (DEFAULT_WIDTH, DEFAULT_HEIGHT));
        assert_eq!(c.view, View::Home);
        assert_eq!(c.view_distance_scale, 1.0);
        assert_eq!(c.wait_ready_seconds, None);
        assert_eq!(c.stats_json, None);
        assert!(c.overlay);
        assert_eq!(c.exposure_stops, None);
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
        assert_eq!(c.exposure_stops, Some(12.75));
        assert!(parse_flags(["--exposure-stops", "bright"]).is_err());
        assert_eq!(
            parse_flags(["--exposure-stops", "-3"])
                .unwrap()
                .exposure_stops,
            Some(-3.0)
        );
        assert_eq!(c.view, View::Key(2));
        assert_eq!(c.view_distance_scale, 0.25);
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
    }

    #[test]
    fn numeric_flags() {
        let flags = parse_flags([
            "--time-scale",
            "86400",
            "--start-offset-seconds",
            "-3.5e6",
            "--width",
            "640",
            "--height=360",
            "--screenshot",
            "out.png",
        ])
        .unwrap();
        let c = Config::resolve(&base_env(), &flags).unwrap();
        assert_eq!(c.time_scale, 86400.0);
        assert_eq!(c.start_offset_seconds, -3.5e6);
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
