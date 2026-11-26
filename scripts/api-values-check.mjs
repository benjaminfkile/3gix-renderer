// Checks the web binding's value conversions under Node, without a GPU or
// a hub (docs/api.md section 2): loads web/pkg (built by web/build.sh) and
// calls its checkPose, checkFrameId, and checkSurface, which convert
// exactly as the Renderer methods do. A pose round-trips as a plain
// object; a frame id that is not a whole number from 0 to below 2^53, and
// a pose with a missing or misshapen field, throw an Error. A surface
// shape in frames() has exactly the fields radiusXy, radiusZ, heightFloor.
//
// Usage: node scripts/api-values-check.mjs

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = await import(join(root, "web/pkg/renderer.js"));
pkg.initSync({ module: readFileSync(join(root, "web/pkg/renderer_bg.wasm")) });
const { checkPose, checkFrameId, checkSurface } = pkg;

const throws = (f, pattern, what) => {
    assert.throws(f, (e) => e instanceof Error && pattern.test(e.message), what);
};

// Frame ids.
assert.equal(checkFrameId(0), 0);
assert.equal(checkFrameId(3), 3);
assert.equal(checkFrameId(2 ** 53 - 1), 2 ** 53 - 1);
throws(() => checkFrameId(2 ** 53), /at or above 2\^53/, "2^53");
throws(() => checkFrameId(2 ** 60), /at or above 2\^53/, "2^60");
throws(() => checkFrameId(-1), /not a whole number/, "negative");
throws(() => checkFrameId(1.5), /not a whole number/, "fraction");
throws(() => checkFrameId(NaN), /not a whole number/, "NaN");
throws(() => checkFrameId(Infinity), /not a whole number/, "Infinity");
throws(() => checkFrameId("3"), /not a number/, "string");
throws(() => checkFrameId(3n), /not a number/, "bigint");

// Poses.
const pose = { frameId: 7, position: [1, -2, 3.5], orientation: [0, 0, 0, 1], speed: 10 };
assert.deepEqual(checkPose(pose), pose);
assert.deepEqual(checkPose({ ...pose, extra: true }), pose);
throws(() => checkPose({ ...pose, frameId: 2 ** 53 }), /at or above 2\^53/, "pose id 2^53");
throws(() => checkPose({ ...pose, frameId: undefined }), /not a number/, "no frame id");
throws(() => checkPose({ ...pose, position: [1, 2] }), /position is not an array of 3/, "short position");
throws(() => checkPose({ ...pose, position: "1,2,3" }), /position is not an array of 3/, "string position");
throws(() => checkPose({ ...pose, orientation: [0, 0, 1] }), /orientation is not an array of 4/, "short quaternion");
throws(() => checkPose({ ...pose, orientation: [0, 0, "0", 1] }), /orientation is not a number/, "string component");
throws(() => checkPose({ ...pose, speed: null }), /speed is not a number/, "no speed");
throws(() => checkPose(null), /expected an object/, "null");
throws(() => checkPose(5), /expected an object/, "number");

// Surface shapes (frames(), docs/api.md section 5).
assert.deepEqual(checkSurface(4, 3.9, -0.5), { radiusXy: 4, radiusZ: 3.9, heightFloor: -0.5 });

console.log("api values ok");
