//! The view and time options, shared by the native and browser builds.
//!
//! `rendering.md` section 6 rule 5 (one renderer): both builds start the
//! same way from the same settings. The native build reads these options
//! from its flags (`--start-on`, `--time-scale`, ...), the browser from
//! `web/config.js` and the page's query string under the same names in
//! camelCase (`startOn`, `timeScale`, ...). Either way each value goes
//! through the one parser here ([`ViewOptions::set`]) into one struct
//! ([`ViewOptions`]), and one function ([`launch`]) applies it to the
//! clock and the camera at start.
//!
//! The start time follows `space-model.md` section 7: every client enters
//! at now; `start = origin` enters at the origin instead, and the start
//! offset moves the start from either, in seconds, clamped between the
//! origin and now. The time scale is the playback rate of a start in the
//! past, simulation seconds per wall-clock second (negative plays
//! backwards); at now the rate is always 1.
//!
//! `hide-layers` lists layer ids to hide (`rendering.md` section 7 item
//! 7), overriding the build's `shown` defaults; [`hidden_layer_indices`]
//! resolves it against the build and reports unknown ids.

use crate::camera::Camera;
use crate::config::{ConfigError, Start, StartOn, View};
use crate::render::scene::MeshMode;
use crate::sim::{SimClock, Simulation};
use gx_core::frames::FrameSystem;
use gx_core::units::Seconds;

/// The option names, in the native flags' kebab case, in a fixed order.
pub const OPTION_NAMES: [&str; 9] = [
    "start",
    "start-offset-seconds",
    "time-scale",
    "start-on",
    "view",
    "view-distance-scale",
    "exposure-stops",
    "mesh-mode",
    "hide-layers",
];

/// The view and time options of one start, each at its default until set.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewOptions {
    /// `start`: where the simulation time starts.
    pub start: Start,
    /// `start-offset-seconds`: simulation time offset from the start at
    /// launch, seconds.
    pub start_offset_seconds: f64,
    /// `time-scale`: playback rate of a start in the past, simulation
    /// seconds per wall-clock second; negative plays backwards.
    pub time_scale: f64,
    /// `start-on`: where the camera starts instead of the `Home` view.
    pub start_on: Option<StartOn>,
    /// `view`: the view a headless run renders when there is no
    /// `start-on`.
    pub view: View,
    /// `view-distance-scale`: factor on the view's distance from its frame
    /// origin.
    pub view_distance_scale: f64,
    /// `exposure-stops`: a fixed exposure, stops
    /// ([`crate::render::gpu::fixed_exposure`]); `None` adapts.
    pub exposure_stops: Option<f64>,
    /// `mesh-mode`: how surface meshes are drawn at start.
    pub mesh_mode: MeshMode,
    /// `hide-layers`: layer ids to hide at start, overriding the build's
    /// `shown` defaults; `None` keeps the defaults.
    pub hide_layers: Option<Vec<String>>,
}

impl Default for ViewOptions {
    fn default() -> ViewOptions {
        ViewOptions {
            start: Start::Now,
            start_offset_seconds: 0.0,
            time_scale: 1.0,
            start_on: None,
            view: View::Home,
            view_distance_scale: 1.0,
            exposure_stops: None,
            mesh_mode: MeshMode::Surfaces,
            hide_layers: None,
        }
    }
}

/// The camelCase form of a kebab-case option name: `start-on` is
/// `startOn`.
pub fn camel_name(kebab: &str) -> String {
    let mut out = String::with_capacity(kebab.len());
    let mut upper = false;
    for c in kebab.chars() {
        if c == '-' {
            upper = true;
        } else if upper {
            out.extend(c.to_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn err(msg: impl Into<String>) -> ConfigError {
    ConfigError(msg.into())
}

/// Parses a finite number for the option `name`.
pub fn parse_f64(name: &str, v: &str) -> Result<f64, ConfigError> {
    let x: f64 = v
        .trim()
        .parse()
        .map_err(|_| err(format!("{name}: {v:?} is not a number")))?;
    if !x.is_finite() {
        return Err(err(format!("{name}: {v:?} is not finite")));
    }
    Ok(x)
}

/// Parses `now` or `origin` for `start`.
pub fn parse_start(name: &str, v: &str) -> Result<Start, ConfigError> {
    Start::parse(v).ok_or_else(|| err(format!("{name}: {v:?} is not now or origin")))
}

/// Parses `home` or a number key `0` to `9` for `view`.
pub fn parse_view(name: &str, v: &str) -> Result<View, ConfigError> {
    View::parse(v).ok_or_else(|| err(format!("{name}: {v:?} is not home or 0 to 9")))
}

/// Parses a factor above zero for `view-distance-scale`.
pub fn parse_distance_scale(name: &str, v: &str) -> Result<f64, ConfigError> {
    let x = parse_f64(name, v)?;
    if x <= 0.0 {
        return Err(err(format!("{name}: must be above zero")));
    }
    Ok(x)
}

/// Parses `surfaces`, `vertices`, or `edges` for `mesh-mode`.
pub fn parse_mesh_mode(name: &str, v: &str) -> Result<MeshMode, ConfigError> {
    match v.trim() {
        "surfaces" => Ok(MeshMode::Surfaces),
        "vertices" => Ok(MeshMode::Vertices),
        "edges" => Ok(MeshMode::Edges),
        _ => Err(err(format!(
            "{name}: {v:?} is not surfaces, vertices, or edges"
        ))),
    }
}

/// Parses a comma-separated list of layer ids for `hide-layers`: blanks
/// around ids are dropped, an empty list hides nothing.
pub fn parse_layer_ids(v: &str) -> Vec<String> {
    v.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// The hidden layers, by index in `layer_ids` (the build's layers in
/// order): the ids in `hide` when given, else `default_hidden`. Ids not in
/// the build are returned second, to be reported and ignored.
pub fn hidden_layer_indices(
    layer_ids: &[&str],
    hide: Option<&[String]>,
    default_hidden: std::collections::BTreeSet<usize>,
) -> (std::collections::BTreeSet<usize>, Vec<String>) {
    let Some(hide) = hide else {
        return (default_hidden, Vec::new());
    };
    let mut hidden = std::collections::BTreeSet::new();
    let mut unknown = Vec::new();
    for id in hide {
        match layer_ids.iter().position(|l| l == id) {
            Some(i) => {
                hidden.insert(i);
            }
            None => unknown.push(id.clone()),
        }
    }
    (hidden, unknown)
}

/// Parses `<frame>:<x>,<y>,<z>[@[+]<m>[^<deg>]]` for `start-on`
/// ([`StartOn::parse`]).
pub fn parse_start_on(name: &str, v: &str) -> Result<StartOn, ConfigError> {
    StartOn::parse(v).ok_or_else(|| {
        err(format!(
            "{name}: {v:?} is not <frame>:<x>,<y>,<z> with a direction other than zero"
        ))
    })
}

impl ViewOptions {
    /// Sets the option `name` (kebab case, without dashes in front) from
    /// `value`. Errors name the option as `label` (`--start-on` natively,
    /// `startOn` in the browser). Returns `Ok(false)` when `name` is not a
    /// view option.
    pub fn set(&mut self, name: &str, label: &str, value: &str) -> Result<bool, ConfigError> {
        match name {
            "start" => self.start = parse_start(label, value)?,
            "start-offset-seconds" => self.start_offset_seconds = parse_f64(label, value)?,
            "time-scale" => self.time_scale = parse_f64(label, value)?,
            "start-on" => self.start_on = Some(parse_start_on(label, value)?),
            "view" => self.view = parse_view(label, value)?,
            "view-distance-scale" => self.view_distance_scale = parse_distance_scale(label, value)?,
            "exposure-stops" => self.exposure_stops = Some(parse_f64(label, value)?),
            "mesh-mode" => self.mesh_mode = parse_mesh_mode(label, value)?,
            "hide-layers" => self.hide_layers = Some(parse_layer_ids(value)),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The options from camelCase name and value pairs (the browser's
    /// `config.js` values, then its query string), later pairs overriding
    /// earlier ones. Unknown names are left to the caller; the first
    /// invalid value is the error.
    pub fn from_camel_pairs<'a, I>(pairs: I) -> Result<ViewOptions, ConfigError>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut o = ViewOptions::default();
        for (key, value) in pairs {
            if let Some(name) = OPTION_NAMES.iter().find(|n| camel_name(n) == key) {
                o.set(name, key, value)?;
            }
        }
        Ok(o)
    }

    /// The camera view: `start-on` if given, `view` otherwise.
    pub fn effective_view(&self) -> View {
        self.start_on.map(View::Over).unwrap_or(self.view)
    }
}

/// The start of a run: the simulation at its start time and the camera at
/// its start place, with the spawn still to come.
pub struct Launch {
    /// The simulation at the configured start time.
    pub sim: Simulation,
    /// The camera at `start-on`, or the `Home` view.
    pub camera: Camera,
    /// A `start-on` without a distance, waiting to spawn onto the frame's
    /// matter ([`crate::spawn::Spawner::pending`]).
    pub pending_spawn: Option<StartOn>,
}

/// The clock of a simulation of `system` (at its epoch, the origin) whose
/// now is `now`, seconds of TDB since J2000, started as `options` say.
pub fn launch_clock(system: &FrameSystem, options: &ViewOptions, now: Seconds) -> SimClock {
    let origin = system.tree().epoch();
    let from = match options.start {
        Start::Now => now,
        Start::Origin => origin,
    };
    SimClock::starting_at(
        origin,
        now,
        from + Seconds::new(options.start_offset_seconds),
        options.time_scale,
    )
}

/// The camera `options` start with: over the `start-on` frame (looking at
/// it until a spawn lands), or `Home` when there is none or its frame is
/// not in the registry.
pub fn launch_camera(system: &FrameSystem, options: &ViewOptions) -> Camera {
    options
        .start_on
        .and_then(|s| {
            Camera::over_pitched(
                system,
                s.frame_id,
                s.direction,
                s.distance,
                s.distance.and(s.pitch_deg),
            )
        })
        .unwrap_or_else(|| Camera::home(system))
}

/// Applies `options` to the clock and the camera at start, for both
/// builds.
pub fn launch(system: FrameSystem, options: &ViewOptions, now: Seconds) -> Launch {
    let clock = launch_clock(&system, options, now);
    let camera = launch_camera(&system, options);
    Launch {
        sim: Simulation::new(system, clock),
        camera,
        pending_spawn: options.start_on.filter(|s| s.distance.is_none()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camel_names() {
        let camel: Vec<String> = OPTION_NAMES.iter().map(|n| camel_name(n)).collect();
        assert_eq!(
            camel,
            [
                "start",
                "startOffsetSeconds",
                "timeScale",
                "startOn",
                "view",
                "viewDistanceScale",
                "exposureStops",
                "meshMode",
                "hideLayers"
            ]
        );
    }

    #[test]
    fn hide_layers_overrides_the_shown_defaults() {
        let o = ViewOptions::from_camel_pairs([("hideLayers", " ground , ,stain,nope")]).unwrap();
        let hide = o.hide_layers.clone().unwrap();
        assert_eq!(hide, ["ground", "stain", "nope"]);
        let ids = ["ground", "stain", "haze"];
        let (hidden, unknown) = hidden_layer_indices(&ids, Some(&hide), [2].into());
        assert_eq!(hidden, [0, 1].into());
        assert_eq!(unknown, ["nope"]);
        // Without the option the build's defaults stand; an empty list
        // shows every layer.
        assert_eq!(hidden_layer_indices(&ids, None, [2].into()).0, [2].into());
        let none = ViewOptions::from_camel_pairs([("hideLayers", "")]).unwrap();
        let (hidden, _) = hidden_layer_indices(&ids, none.hide_layers.as_deref(), [2].into());
        assert!(hidden.is_empty());
    }

    #[test]
    fn the_parser_sets_each_option() {
        let mut o = ViewOptions::default();
        for (n, v) in [
            ("start", "origin"),
            ("start-offset-seconds", "-3.5e6"),
            ("time-scale", "86400"),
            ("start-on", "3:0,0,1@+10"),
            ("view", "3"),
            ("view-distance-scale", "0.5"),
            ("exposure-stops", "2"),
            ("mesh-mode", "edges"),
        ] {
            assert!(o.set(n, n, v).unwrap(), "{n}");
        }
        assert_eq!(o.start, Start::Origin);
        assert_eq!(o.start_offset_seconds, -3.5e6);
        assert_eq!(o.time_scale, 86400.0);
        let s = o.start_on.unwrap();
        assert_eq!((s.frame_id, s.height, s.distance), (3, Some(10.0), None));
        assert_eq!(o.view, View::Key(2));
        assert_eq!(o.effective_view(), View::Over(s));
        assert_eq!(o.view_distance_scale, 0.5);
        assert_eq!(o.exposure_stops, Some(2.0));
        assert_eq!(o.mesh_mode, MeshMode::Edges);
        assert!(!o.set("width", "width", "3").unwrap());
    }

    #[test]
    fn invalid_values_name_the_option() {
        let mut o = ViewOptions::default();
        for (n, v) in [
            ("start", "later"),
            ("start-offset-seconds", "x"),
            ("time-scale", "inf"),
            ("start-on", "3:0,0,0"),
            ("view", "11"),
            ("view-distance-scale", "0"),
            ("exposure-stops", ""),
            ("mesh-mode", "wire"),
        ] {
            let e = o.set(n, "label", v).unwrap_err();
            assert!(e.0.starts_with("label: "), "{n}: {e}");
        }
        assert_eq!(o, ViewOptions::default());
    }

    #[test]
    fn camel_pairs_later_override_earlier() {
        let o = ViewOptions::from_camel_pairs([
            ("timeScale", "2"),
            ("hubUrl", "ignored"),
            ("timeScale", "-4"),
            ("startOn", "1:1,0,0@5e6"),
        ])
        .unwrap();
        assert_eq!(o.time_scale, -4.0);
        assert_eq!(o.start_on.unwrap().distance, Some(5e6));
        let e = ViewOptions::from_camel_pairs([("startOn", "1:0,0,0")]).unwrap_err();
        assert!(e.0.starts_with("startOn: "), "{e}");
    }

    #[test]
    fn start_on_puts_the_camera_where_the_native_flag_does() {
        use crate::config::{parse_flags, ApiKey, Config};
        let place = "5:0,1,1@300^-10";
        let flags = parse_flags(["--start-on", place, "--time-scale=-2"]).unwrap();
        let mut env = std::collections::BTreeMap::new();
        env.insert(
            crate::config::ENV_HUB_URL.into(),
            "http://hub.invalid".into(),
        );
        env.insert(crate::config::ENV_API_KEY.into(), "k".into());
        env.insert(crate::config::ENV_SPACE_ID.into(), "s".into());
        let native = Config::resolve(&env, &flags).unwrap();
        assert_eq!(native.api_key, ApiKey::new("k"));
        let web = ViewOptions::from_camel_pairs([("startOn", place), ("timeScale", "-2")]).unwrap();
        assert_eq!(native.options, web);
        let now = Seconds::new(1e6);
        let a = launch(crate::camera::tests::system(), &native.options, now);
        let b = launch(crate::camera::tests::system(), &web, now);
        assert_eq!(a.camera, b.camera);
        assert_eq!(a.camera.frame_id, 5);
        assert!((a.camera.position.length() - 300.0).abs() < 1e-9);
        assert_ne!(a.camera, Camera::home(a.sim.system()));
        assert_eq!(a.sim.time(), b.sim.time());
        assert_eq!(a.pending_spawn, None);
    }
}
