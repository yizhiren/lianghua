import assert from "node:assert/strict";
import test from "node:test";

import { conditionFormula, indicatorFormula } from "../app/formula-format.js";

test("shows lag and rolling-window semantics instead of self comparisons", () => {
  const previousClose = { name: "price", field: "value", params: { lag: 1 }, timeframe: "day" };
  const previousMinimum = { name: "price", field: "value", params: { lag: 1, window: 20, window_op: "min" }, timeframe: "day" };
  assert.equal(
    conditionFormula({ op: "compare", left: previousClose, right: previousMinimum, comparator: "==" }, "day"),
    "T-1日收盘价 == T-1日最近20个交易日最低收盘价",
  );
});

test("shows MACD lag, window, volume scale and relative change", () => {
  assert.equal(
    indicatorFormula({ name: "macd", field: "dif", params: { fast: 12, slow: 26, signal: 9, lag: 1, window: 20, window_op: "min" } }, "day"),
    "T-1日最近20个交易日最低MACD.DIFF(12,26,9)",
  );
  assert.equal(
    indicatorFormula({ name: "volume", field: "value", params: { window: 60, window_op: "mean", scale: 1.2 } }, "day"),
    "T日最近60个交易日平均成交量 × 1.2",
  );
  assert.equal(
    conditionFormula({ op: "relative_change", left: { name: "price", field: "value", params: {} }, right: 5, comparator: "<=" }, "day"),
    "T日收盘价相对前一期的涨跌幅 <= 5%",
  );
});

test("shows cross-series selection semantics", () => {
  assert.equal(
    indicatorFormula({
      name: "macd",
      field: "dif",
      params: { fast: 12, slow: 26, signal: 9, select_by: "price.close", select_op: "min", select_window: 20, select_tie: "last" },
    }, "day"),
    "T日最近20个交易日内收盘价最低日（取最后一次）对应的MACD.DIFF(12,26,9)",
  );
});
