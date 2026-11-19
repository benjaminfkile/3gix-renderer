// Checks web/view-options.js under Node: config.js values and the query
// string become the option pairs the browser build hands to start(), with
// the query string after (so overriding) config.js. The names must be the
// native flag names in camelCase (crates/renderer/src/options.rs).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { VIEW_OPTION_NAMES, viewOptionPairs } from "../web/view-options.js";

const source = readFileSync(new URL("../crates/renderer/src/options.rs", import.meta.url), "utf8");
const block = source.match(/pub const OPTION_NAMES: \[&str; \d+\] = \[([^\]]*)\]/);
assert.ok(block, "OPTION_NAMES not found in options.rs");
const native = [...block[1].matchAll(/"([a-z-]+)"/g)].map((m) => m[1]);
const camel = native.map((n) => n.replace(/-([a-z])/g, (_, c) => c.toUpperCase()));
assert.deepEqual(VIEW_OPTION_NAMES, camel);

assert.deepEqual(viewOptionPairs({}, ""), []);
assert.deepEqual(viewOptionPairs(undefined, undefined), []);

const config = {
  hubUrl: "http://hub.invalid",
  startOn: "3:0,0,1@+10",
  timeScale: 60,
  start: "origin",
  exposureStops: null,
  meshMode: "",
};
assert.deepEqual(viewOptionPairs(config, ""), [
  "start", "origin",
  "timeScale", "60",
  "startOn", "3:0,0,1@+10",
]);
assert.deepEqual(
  viewOptionPairs(config, "?timeScale=-2&startOn=4%3A1%2C0%2C0%405e6&apiKey=x&view=3"),
  [
    "start", "origin",
    "timeScale", "60",
    "startOn", "3:0,0,1@+10",
    "timeScale", "-2",
    "startOn", "4:1,0,0@5e6",
    "view", "3",
  ],
);
// An invalid query value is passed on as given, for the parser to reject.
assert.deepEqual(viewOptionPairs({}, "startOffsetSeconds=soon"), ["startOffsetSeconds", "soon"]);

console.log("web options ok");
