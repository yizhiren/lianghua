function numericParam(params, name, fallback = 0) {
  const value = Number(params[name]);
  return Number.isFinite(value) ? value : fallback;
}

export function timeframeName(timeframe) {
  return timeframe === "week" ? "完整周K" : timeframe === "mixed" ? "跨周期" : "日K";
}

function timePoint(timeframe, lag) {
  if (timeframe === "day") return lag > 0 ? `T-${lag}日` : "T日";
  if (timeframe === "week") return lag > 0 ? `前${lag}个完整周K` : "当前完整周K";
  return lag > 0 ? `前${lag}期跨周期` : "当前跨周期";
}

function windowUnit(timeframe) {
  return timeframe === "week" ? "个完整周K" : timeframe === "day" ? "个交易日" : "期";
}

function baseIndicator(indicator, params) {
  const defaultPeriod = indicator.name === "rsi" ? 14
    : indicator.name === "sma" || indicator.name === "ema" ? 20
      : indicator.name === "volume" ? 5
        : indicator.name === "kdj" ? 9 : 0;
  const period = numericParam(params, "period", defaultPeriod);
  const names = {
    "price.open": "开盘价",
    "price.high": "最高价",
    "price.low": "最低价",
    "price.close": "收盘价",
    "price.value": "收盘价",
    "candle.body_pct": "K线实体百分比",
    "candle.bullish": "阳线标记（阳线=1）",
    "candle.one_word_limit_down": "一字跌停标记（一字跌停=1）",
    "security.is_a_share": "沪深A股标记（A股=1）",
    "security.is_st": "ST标记（ST或*ST=1）",
    "security.trade_status": "交易状态（正常交易=1）",
    "security.listed_trading_days": "上市交易日数",
    "position.pnl_pct": "持仓盈亏百分比",
    "position.add_count": "持仓补仓次数",
    "position.last_add_price": "前一次补仓价格",
    "position.days_since_last_add": "距前一次补仓自然日数",
    "position.days_since_last_buy": "距上次买入或补仓自然日数",
    "position.first_add_20d_low": "首次补仓时20日最低价",
    "position.first_add_post_low_high": "首次补仓时最低点后最高价",
    "position.first_add_rebound_pct": "首次补仓时反弹幅度百分比",
    "position.first_add_rebound_confirmed": "首次补仓时反弹确认（确认=1）",
  };
  const key = `${indicator.name}.${indicator.field}`;
  if (names[key]) return names[key];
  if (indicator.name === "macd") {
    const field = { dif: "DIFF", dea: "DEA", histogram: "柱值" }[indicator.field] || indicator.field;
    return `MACD.${field}(${numericParam(params, "fast", 12)},${numericParam(params, "slow", 26)},${numericParam(params, "signal", 9)})`;
  }
  if (indicator.name === "rsi") return `RSI(${period})`;
  if (indicator.name === "sma") return `SMA(${period})`;
  if (indicator.name === "ema") return `EMA(${period})`;
  if (indicator.name === "volume" && indicator.field === "ratio") return `成交量÷${period}日平均成交量`;
  if (indicator.name === "volume") return "成交量";
  if (indicator.name === "kdj") return `KDJ.${indicator.field.toUpperCase()}(${period || 9})`;
  if (indicator.name === "boll") {
    const field = { upper: "上轨", middle: "中轨", lower: "下轨", bandwidth: "带宽", percent_b: "%B" }[indicator.field] || indicator.field;
    return `BOLL.${field}(${period || 20})`;
  }
  if (indicator.name === "atr") return `ATR${indicator.field === "percent" ? "百分比" : ""}(${period || 14})`;
  if (indicator.name === "adx") return `ADX(${period || 14})`;
  if (indicator.name === "volatility") return `波动率(${period || 20})`;
  return `${indicator.name}.${indicator.field}`;
}

function selectorName(value) {
  const names = {
    "price.open": "开盘价",
    "price.high": "最高价",
    "price.low": "最低价",
    "price.close": "收盘价",
    "price.value": "收盘价",
    "volume.value": "成交量",
  };
  return names[value] || value;
}

export function indicatorFormula(indicator, fallback) {
  if (!indicator) return "缺少指标";
  const timeframe = indicator.timeframe || fallback;
  const params = indicator.params || {};
  const lag = Math.max(0, Math.trunc(numericParam(params, "lag", 0)));
  const window = Math.max(0, Math.trunc(numericParam(params, "window", 0)));
  const base = baseIndicator(indicator, params);
  const staticValue = indicator.name === "position" || indicator.name === "security";
  const point = staticValue ? "" : timePoint(timeframe, lag);
  let expression = base;

  if (params.select_by) {
    const selectWindow = Math.max(1, Math.trunc(numericParam(params, "select_window", window || 1)));
    const selectOp = params.select_op === "max" ? "最高" : "最低";
    const tie = params.select_tie === "last" ? "最后一次" : params.select_tie === "first" ? "第一次" : "";
    expression = `最近${selectWindow}${windowUnit(timeframe)}内${selectorName(String(params.select_by))}${selectOp}日${tie ? `（取${tie}）` : ""}对应的${base}`;
  } else if (window) {
    const operation = { min: "最低", max: "最高", mean: "平均" }[params.window_op] || "窗口内";
    expression = `最近${window}${windowUnit(timeframe)}${operation}${base}`;
  }

  const scale = numericParam(params, "scale", 1);
  if (scale !== 1) expression = `${expression} × ${scale}`;
  return `${point}${expression}`;
}

export function conditionFormula(condition, fallback) {
  const timeframe = condition.timeframe || fallback;
  const children = condition.conditions || [];
  const left = indicatorFormula(condition.left, timeframe);
  const right = typeof condition.right === "number"
    ? String(condition.right)
    : indicatorFormula(condition.right, timeframe);
  const periods = condition.periods || 1;
  switch (condition.op) {
    case "all": return `全部满足（${children.map((item) => conditionFormula(item, timeframe)).join(" 且 ")}）`;
    case "any": return `任一满足（${children.map((item) => conditionFormula(item, timeframe)).join(" 或 ")}）`;
    case "not": return `不满足（${children[0] ? conditionFormula(children[0], timeframe) : "缺少条件"}）`;
    case "compare": return `${left} ${condition.comparator || ">"} ${right}`;
    case "cross_above": return `${left} 上穿 ${right}（前一期≤，当前期>）`;
    case "cross_below": return `${left} 下穿 ${right}（前一期≥，当前期<）`;
    case "rising": return `最近${periods}个${timeframeName(timeframe)}的 ${left} 严格递增`;
    case "falling": return `最近${periods}个${timeframeName(timeframe)}的 ${left} 严格递减`;
    case "breakout": return condition.direction === "low"
      ? `${left} < 前${periods}期最低值`
      : `${left} > 前${periods}期最高值`;
    case "within_pct": return `${condition.lower}% ≤ (${left} ÷ ${right} − 1) × 100 ≤ ${condition.upper}%`;
    case "ratio_pct": return `${condition.lower}% ≤ (${left} ÷ ${right}) × 100 ≤ ${condition.upper}%`;
    case "relative_change": return `${left}相对前一期的涨跌幅 ${condition.comparator} ${right}%`;
    case "fraction_of_recent": return `${left} ÷ 前${periods}期正值峰值 ${condition.comparator} ${right}`;
    case "consecutive": return `连续${periods}个${timeframeName(timeframe)}满足（${children[0] ? conditionFormula(children[0], timeframe) : "缺少条件"}）`;
    default: return `未识别运算：${condition.op}`;
  }
}
