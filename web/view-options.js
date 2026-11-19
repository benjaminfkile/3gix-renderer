// The view and time options of the browser build (rendering.md section 6
// rule 5, one renderer): the same options as the native flags, named in
// camelCase. The values are only gathered here; the renderer's own parser
// (crates/renderer/src/options.rs) checks and applies them, the same one
// the native flags go through.

// The option names, in the order of the native OPTION_NAMES.
export const VIEW_OPTION_NAMES = [
  "start",
  "startOffsetSeconds",
  "timeScale",
  "startOn",
  "view",
  "viewDistanceScale",
  "exposureStops",
  "meshMode",
];

// The options as flattened name and value pairs for start(): first the
// values in config.js (`config`, an object), then the page's query string
// (`search`, such as location.search), so a link overrides config.js.
// Empty, null, and undefined config values are left out; a query value is
// taken as given. Names that are not view options are ignored.
export function viewOptionPairs(config, search) {
  const pairs = [];
  for (const name of VIEW_OPTION_NAMES) {
    const v = config?.[name];
    if (v !== undefined && v !== null && v !== "") {
      pairs.push(name, String(v));
    }
  }
  const query = new URLSearchParams(search ?? "");
  for (const [name, v] of query) {
    if (VIEW_OPTION_NAMES.includes(name)) {
      pairs.push(name, v);
    }
  }
  return pairs;
}
