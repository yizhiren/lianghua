import assert from "node:assert/strict";
import test from "node:test";

import { isStrategyScanning } from "../app/run-status.js";

test("keeps the scan button running after a page refresh", () => {
  assert.equal(isStrategyScanning(false, "running"), true);
});

test("shows the idle button only when neither the request nor backend run is active", () => {
  assert.equal(isStrategyScanning(true, "success"), true);
  assert.equal(isStrategyScanning(false, "success"), false);
  assert.equal(isStrategyScanning(false, undefined), false);
});
