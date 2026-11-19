// Optional local settings for the browser renderer. Copy to web/config.js (gitignored)
// and the panel comes up filled in; with autostart the renderer starts on page load.
// The hub's desktop launcher (3GIXHub/scripts/desktop/web.ps1) writes this file for you.
export default {
  hubUrl: "http://localhost:8080",
  apiKey: "local-dev-3gix-key",
  spaceId: "a1000000-0000-0000-0000-000000000001",
  buildId: "",
  autostart: true,
  // View and time options: the native flags of the same names in camelCase,
  // parsed by the same code. Leave one out (or empty) for its default. The
  // page's query string overrides any of them, e.g. ?startOn=3:0,0,1&timeScale=60.
  //
  // start: "now",                  // "now" (default) or "origin"
  // startOffsetSeconds: 0,         // seconds from the start, clamped between origin and now
  // timeScale: 1,                  // playback rate of a start in the past, negative backwards
  // startOn: "3:0,0,1@+10",        // <frame>:<x>,<y>,<z>[@[+]<m>[^<deg>]], as --start-on
  // view: "home",                  // home or 0 to 9 (used by headless runs)
  // viewDistanceScale: 1,          // factor on the start view's distance from its frame origin
  // exposureStops: 0,              // fixed exposure in stops; leave out for automatic
  // meshMode: "surfaces",          // surfaces, vertices, or edges
  // hideLayers: "a,b",             // layer ids to hide, overriding the build's shown defaults
};
