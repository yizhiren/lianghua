from __future__ import annotations

import json
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import httpx
from pydantic import ValidationError

from .config import settings
from .indicators import adx, atr, bollinger, ema, kdj, macd, rolling_std, rsi, sma, weekly_bars
from .schemas import RuleCondition, RuleIndicator, StrategyDSL


DEFAULT_DESCRIPTION = (
    "从全量股票中筛选出周K的MACD柱连续变大的股票放入待定列表；"
    "从待定列表中选出日K的MACD死叉股票并高亮；"
    "从持仓列表中选出日K的MACD金叉股票并高亮。"
)


DEFAULT_DSL: dict[str, Any] = {
    "schema_version": 1,
    "rules": [
        {
            "id": "weekly-macd-rising",
            "scope": "universe",
            "timeframe": "week",
            "condition": {
                "op": "rising",
                "left": {"name": "macd", "field": "histogram", "params": {"fast": 12, "slow": 26, "signal": 9}},
                "periods": 3,
            },
            "action": "add_candidate",
            "label": "周K MACD柱连续3期增大",
        },
        {
            "id": "candidate-daily-death-cross",
            "scope": "candidate",
            "timeframe": "day",
            "condition": {
                "op": "cross_below",
                "left": {"name": "macd", "field": "dif", "params": {}},
                "right": {"name": "macd", "field": "dea", "params": {}},
            },
            "action": "highlight_candidate",
            "label": "日K MACD死叉",
        },
        {
            "id": "position-daily-golden-cross",
            "scope": "position",
            "timeframe": "day",
            "condition": {
                "op": "cross_above",
                "left": {"name": "macd", "field": "dif", "params": {}},
                "right": {"name": "macd", "field": "dea", "params": {}},
            },
            "action": "highlight_position",
            "label": "日K MACD金叉",
        },
    ],
}

DEFAULT_EXPLANATION = [
    "扫描全部普通股；周K MACD柱连续3期严格增大时自动加入待定列表。",
    "待定股票日K DIF向下穿越DEA时高亮。",
    "持仓股票日K DIF向上穿越DEA时高亮。",
]


RIGHT_SIDE_DSL: dict[str, Any] = {
    "schema_version": 1,
    "rules": [
        {
            "id": "weekly-watch-pool",
            "scope": "universe",
            "timeframe": "mixed",
            "condition": {
                "op": "all",
                "conditions": [
                    {
                        "op": "rising",
                        "timeframe": "week",
                        "left": {"name": "macd", "field": "dif", "params": {}},
                        "periods": 2,
                    },
                    {
                        "op": "compare",
                        "timeframe": "day",
                        "left": {"name": "rsi", "field": "value", "params": {"period": 14}},
                        "right": 60,
                        "comparator": "<",
                    },
                    {
                        "op": "within_pct",
                        "timeframe": "week",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": {"name": "sma", "field": "value", "params": {"period": 5}},
                        "lower": 0,
                        "upper": 12,
                    },
                ],
            },
            "action": "add_candidate",
            "label": "完整周K DIFF上升、RSI低于60且站上5周线不超过12%",
        },
        {
            "id": "preferred-low-position-or-strong-background",
            "scope": "candidate",
            "timeframe": "mixed",
            "condition": {
                "op": "any",
                "conditions": [
                    {
                        "op": "ratio_pct",
                        "timeframe": "week",
                        "left": {"name": "macd", "field": "dif", "params": {}},
                        "right": {"name": "price", "field": "close", "params": {}},
                        "lower": -1,
                        "upper": 1,
                    },
                    {
                        "op": "cross_above",
                        "timeframe": "week",
                        "left": {"name": "macd", "field": "dif", "params": {}},
                        "right": 0,
                    },
                    {
                        "op": "compare",
                        "timeframe": "day",
                        "left": {"name": "macd", "field": "dif", "params": {}},
                        "right": 0,
                        "comparator": ">",
                    },
                ],
            },
            "action": "highlight_candidate",
            "label": "优选：周线DIFF靠近或刚上零轴，或日K DIFF在零轴上方",
        },
        {
            "id": "pullback-buy",
            "scope": "candidate",
            "timeframe": "mixed",
            "condition": {
                "op": "all",
                "conditions": [
                    {
                        "op": "within_pct",
                        "timeframe": "day",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": {"name": "sma", "field": "value", "params": {"period": 5}, "timeframe": "week"},
                        "lower": -2,
                        "upper": 2,
                    },
                    {
                        "op": "any",
                        "timeframe": "day",
                        "conditions": [
                            {
                                "op": "cross_below",
                                "left": {"name": "macd", "field": "dif", "params": {}},
                                "right": {"name": "macd", "field": "dea", "params": {}},
                            },
                            {
                                "op": "all",
                                "conditions": [
                                    {
                                        "op": "compare",
                                        "left": {"name": "macd", "field": "histogram", "params": {}},
                                        "right": 0,
                                        "comparator": ">=",
                                    },
                                    {
                                        "op": "fraction_of_recent",
                                        "left": {"name": "macd", "field": "histogram", "params": {}},
                                        "right": 0.3,
                                        "comparator": "<=",
                                        "periods": 5,
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
            "action": "signal_buy",
            "target_position_pct": 45,
            "label": "回踩5周线且日K调整到位",
        },
        {
            "id": "stabilized-add",
            "scope": "position",
            "timeframe": "mixed",
            "condition": {
                "op": "all",
                "timeframe": "day",
                "conditions": [
                    {
                        "op": "within_pct",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": {"name": "sma", "field": "value", "params": {"period": 5}, "timeframe": "week"},
                        "lower": -2,
                        "upper": 2,
                    },
                    {
                        "op": "compare",
                        "left": {"name": "volume", "field": "ratio", "params": {"period": 5}},
                        "right": 0.7,
                        "comparator": "<=",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "candle", "field": "body_pct", "params": {}},
                        "right": 1.5,
                        "comparator": "<=",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "candle", "field": "bullish", "params": {}},
                        "right": 1,
                        "comparator": "==",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "macd", "field": "dif", "params": {}},
                        "right": 0,
                        "comparator": ">",
                    },
                    {
                        "op": "cross_above",
                        "left": {"name": "macd", "field": "dif", "params": {}},
                        "right": {"name": "macd", "field": "dea", "params": {}},
                    },
                    {
                        "op": "rising",
                        "left": {"name": "macd", "field": "histogram", "params": {}},
                        "periods": 2,
                    },
                ],
            },
            "action": "signal_add",
            "target_position_pct": 80,
            "label": "5周线附近缩量企稳且日K重新金叉",
        },
        {
            "id": "weekly-trend-hold",
            "scope": "position",
            "timeframe": "week",
            "condition": {
                "op": "all",
                "conditions": [
                    {"op": "rising", "left": {"name": "macd", "field": "dif", "params": {}}, "periods": 2},
                    {
                        "op": "compare",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": {"name": "sma", "field": "value", "params": {"period": 5}},
                        "comparator": ">=",
                    },
                ],
            },
            "action": "signal_hold",
            "label": "完整周K趋势仍有效，忽略日线噪音",
        },
        {
            "id": "weekly-reduce",
            "scope": "position",
            "timeframe": "mixed",
            "condition": {
                "op": "all",
                "conditions": [
                    {
                        "op": "compare",
                        "timeframe": "week",
                        "left": {"name": "macd", "field": "histogram", "params": {}},
                        "right": 0,
                        "comparator": ">",
                    },
                    {
                        "op": "relative_change",
                        "timeframe": "week",
                        "left": {"name": "macd", "field": "histogram", "params": {}},
                        "right": -30,
                        "comparator": "<=",
                    },
                    {
                        "op": "compare",
                        "timeframe": "day",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": {"name": "sma", "field": "value", "params": {"period": 5}, "timeframe": "week"},
                        "comparator": "<",
                    },
                ],
            },
            "action": "signal_reduce",
            "target_position_pct": 40,
            "label": "周线红柱缩短30%且价格跌破5周线",
        },
        {
            "id": "weekly-exit",
            "scope": "position",
            "timeframe": "week",
            "condition": {
                "op": "falling",
                "left": {"name": "macd", "field": "dif", "params": {}},
                "periods": 2,
            },
            "action": "signal_exit",
            "target_position_pct": 0,
            "label": "完整周K DIFF拐头向下",
        },
        {
            "id": "hard-stop",
            "scope": "position",
            "timeframe": "mixed",
            "condition": {
                "op": "any",
                "conditions": [
                    {
                        "op": "compare",
                        "timeframe": "day",
                        "left": {"name": "position", "field": "pnl_pct", "params": {}},
                        "right": -7,
                        "comparator": "<=",
                    },
                    {
                        "op": "consecutive",
                        "timeframe": "day",
                        "periods": 2,
                        "conditions": [
                            {
                                "op": "compare",
                                "left": {"name": "price", "field": "close", "params": {}},
                                "right": {"name": "sma", "field": "value", "params": {"period": 10}, "timeframe": "week"},
                                "comparator": "<",
                            }
                        ],
                    },
                ],
            },
            "action": "signal_stop",
            "target_position_pct": 0,
            "label": "亏损达到7%或连续2日未收复10周线",
        },
    ],
}


RIGHT_SIDE_EXPLANATION = [
    "每15分钟执行；所有周线指标只使用最近一根已完成周K，本周未收盘周K不会参与判断。",
    "完整周K DIFF上升、日K RSI(14)<60、周收盘站上5周线且偏离不超过12%时加入待定池。",
    "周线DIFF/股价处于±1%以内、周线DIFF刚上零轴，或日K DIFF>0时优选高亮，不作为入池硬条件。",
    "最新价距离5周线-2%至+2%，且日K死叉或红柱缩至近5期峰值30%以内时提示建仓至45%。",
    "5周线附近出现缩量小阳线、日K重新金叉且红柱放大时提示加仓至80%。",
    "完整周K DIFF继续上升且周收盘不破5周线时提示继续持有。",
    "周线红柱缩短至少30%且最新价跌破5周线时提示减仓至40%；完整周K DIFF向下时提示清仓。",
    "亏损达到-7%，或连续2个日K收盘低于当前10周线时提示止损清仓。",
]


# 通用指标变换参数让 DSL 可以表达跨序列定位，而不把某一种背离写死在执行器里：
# window_op/window 计算含当期的滚动值；select_* 按另一序列的窗口极值定位；scale 进行常数缩放。
LEFT_SIDE_DSL: dict[str, Any] = {
    "schema_version": 1,
    "rules": [
        {
            "id": "left-side-bottom-pool",
            "scope": "universe",
            "timeframe": "day",
            "daily_bar_mode": "completed",
            "condition": {
                "op": "all",
                "conditions": [
                    {
                        "op": "compare",
                        "left": {"name": "rsi", "field": "value", "params": {"period": 14}},
                        "right": 25,
                        "comparator": "<",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": {
                            "name": "price",
                            "field": "close",
                            "params": {"window_op": "min", "window": 20, "scale": 1.02},
                        },
                        "comparator": "<=",
                    },
                    {
                        "op": "compare",
                        "left": {
                            "name": "macd",
                            "field": "dif",
                            "params": {"fast": 12, "slow": 26, "signal": 9},
                        },
                        "right": {
                            "name": "macd",
                            "field": "dif",
                            "params": {
                                "fast": 12,
                                "slow": 26,
                                "signal": 9,
                                "select_by": "price.close",
                                "select_op": "min",
                                "select_window": 20,
                                "select_tie": "first",
                            },
                        },
                        "comparator": ">",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": {"name": "sma", "field": "value", "params": {"period": 250, "scale": 0.75}},
                        "comparator": "<",
                    },
                    {
                        "op": "compare",
                        "left": {
                            "name": "volume",
                            "field": "value",
                            "params": {"window_op": "mean", "window": 5},
                        },
                        "right": {
                            "name": "volume",
                            "field": "value",
                            "params": {"window_op": "mean", "window": 20, "scale": 0.65},
                        },
                        "comparator": "<",
                    },
                ],
            },
            "action": "add_candidate",
            "label": "RSI极度超卖、20日价格与DIF底背离、深度跌破年线且地量",
        },
        {
            "id": "entry-day-low-hard-stop",
            "scope": "position",
            "timeframe": "day",
            "daily_bar_mode": "completed",
            "condition": {
                "op": "compare",
                "left": {"name": "price", "field": "close", "params": {}},
                "right": {"name": "position", "field": "entry_day_low", "params": {"scale": 0.97}},
                "comparator": "<",
            },
            "action": "signal_stop",
            "target_position_pct": 0,
            "label": "收盘价跌破建仓日最低价3%",
        },
    ],
}

LEFT_SIDE_EXPLANATION = [
    "仅使用已完成的前复权日K；RSI(14)<25、价格处于20日最低收盘价上方2%以内、当前DIF高于最早最低收盘价日DIF、收盘价低于MA250的75%、VOL_MA5低于VOL_MA20的65%时加入待定池。",
    "最低收盘价重复时取窗口内最早日期；若当日为新低，当前DIF与自身比较，底背离条件自然不成立。",
    "建仓后的已完成日K收盘价低于建仓日最低价的97%时，生成最高优先级止损清仓信号。",
]


TWO_DAY_LEFT_SIDE_DSL: dict[str, Any] = {
    "schema_version": 1,
    "rules": [
        {
            "id": "two-day-left-side-confirmation",
            "scope": "universe",
            "timeframe": "day",
            "daily_bar_mode": "completed",
            "condition": {
                "op": "all",
                "conditions": [
                    {"op": "compare", "left": {"name": "security", "field": "is_a_share", "params": {}}, "right": 1, "comparator": "=="},
                    {"op": "compare", "left": {"name": "security", "field": "is_st", "params": {}}, "right": 0, "comparator": "=="},
                    {"op": "compare", "left": {"name": "security", "field": "listed_trading_days", "params": {}}, "right": 120, "comparator": ">="},
                    {"op": "compare", "left": {"name": "candle", "field": "one_word_limit_down", "params": {}}, "right": 0, "comparator": "=="},
                    {
                        "op": "compare",
                        "left": {"name": "sma", "field": "value", "params": {"period": 60}},
                        "right": {"name": "sma", "field": "value", "params": {"period": 120}},
                        "comparator": "<",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": {"name": "sma", "field": "value", "params": {"period": 60}},
                        "comparator": "<",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "price", "field": "close", "params": {"lag": 1}},
                        "right": {"name": "price", "field": "close", "params": {"window_op": "min", "window": 30, "lag": 1}},
                        "comparator": "==",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "macd", "field": "dif", "params": {"fast": 12, "slow": 26, "signal": 9, "lag": 1}},
                        "right": {
                            "name": "macd",
                            "field": "dif",
                            "params": {"fast": 12, "slow": 26, "signal": 9, "window_op": "min", "window": 30, "lag": 1},
                        },
                        "comparator": ">",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "rsi", "field": "value", "params": {"period": 14, "lag": 1}},
                        "right": 30,
                        "comparator": "<",
                    },
                    {
                        "op": "compare",
                        "left": {"name": "volume", "field": "value", "params": {"lag": 1}},
                        "right": {"name": "volume", "field": "value", "params": {"window_op": "mean", "window": 20, "lag": 1}},
                        "comparator": "<",
                    },
                    {"op": "rising", "left": {"name": "price", "field": "close", "params": {}}, "periods": 2},
                    {
                        "op": "rising",
                        "left": {"name": "macd", "field": "dif", "params": {"fast": 12, "slow": 26, "signal": 9}},
                        "periods": 2,
                    },
                    {
                        "op": "compare",
                        "left": {"name": "volume", "field": "value", "params": {}},
                        "right": {"name": "volume", "field": "value", "params": {"window_op": "mean", "window": 60}},
                        "comparator": "<",
                    },
                    {
                        "op": "relative_change",
                        "left": {"name": "price", "field": "close", "params": {}},
                        "right": 5,
                        "comparator": "<=",
                    },
                ],
            },
            "action": "add_candidate",
            "label": "A股非ST且上市满120日，T-1底背离超卖缩量，T日温和止跌确认",
        }
    ],
}

TWO_DAY_LEFT_SIDE_EXPLANATION = [
    "仅扫描沪深A股，排除ST、*ST、少于120根已完成交易日K及T日一字跌停股票。",
    "T日保持MA60低于MA120且收盘低于MA60；T-1收盘为当时30日最低、DIF高于当时30日最低DIF、RSI(14)低于30且成交量低于当时20日均量。",
    "T日收盘和DIF均高于T-1，成交量低于60日均量，且涨幅大于0并不超过5%时加入待定池。",
]


def _is_right_side_description(description: str) -> bool:
    compact = "".join(description.lower().split())
    return (
        "5周" in compact
        and "diff" in compact
        and "rsi" in compact
        and ("回踩" in compact or "建仓" in compact)
        and ("止损" in compact or "10周" in compact)
    )


def validate_dsl(value: dict[str, Any]) -> StrategyDSL:
    parsed = StrategyDSL.model_validate(value)

    def check_indicator(indicator) -> None:
        if indicator is None:
            return
        fields = {
            "macd": {"dif", "dea", "histogram"},
            "sma": {"value"},
            "ema": {"value"},
            "rsi": {"value"},
            "kdj": {"k", "d", "j"},
            "boll": {"middle", "upper", "lower", "percent_b", "bandwidth"},
            "atr": {"value", "percent"},
            "adx": {"value", "plus_di", "minus_di"},
            "volatility": {"value"},
            "price": {"open", "high", "low", "close", "value"},
            "volume": {"value", "ratio"},
            "candle": {"body_pct", "bullish", "one_word_limit_down"},
            "security": {"is_a_share", "is_st", "trade_status", "listed_trading_days"},
            "position": {
                "pnl_pct",
                "entry_day_low",
                "atr_stop_price",
                "trailing_atr_stop_price",
                "holding_trading_days",
                "add_count",
                "last_add_price",
                "days_since_last_add",
                "days_since_last_buy",
                "first_add_20d_low",
                "first_add_post_low_high",
                "first_add_rebound_pct",
                "first_add_rebound_confirmed",
            },
        }
        if indicator.field not in fields[indicator.name]:
            raise ValueError(f"{indicator.name}.{indicator.field} 不是允许的指标字段")
        params = indicator.params
        period = int(params.get("period", 14))
        if indicator.name in {"sma", "ema", "rsi", "kdj", "boll", "atr", "adx", "volatility"} and not 2 <= period <= 250:
            raise ValueError(f"{indicator.name} period 必须在 2 到 250 之间")
        if indicator.name == "macd":
            fast = int(params.get("fast", 12))
            slow = int(params.get("slow", 26))
            signal = int(params.get("signal", 9))
            if not (2 <= fast < slow <= 250 and 2 <= signal <= 60):
                raise ValueError("MACD 参数必须满足 2 <= fast < slow <= 250 且 2 <= signal <= 60")
        if indicator.name == "volume" and indicator.field == "ratio" and not 2 <= period <= 250:
            raise ValueError("成交量均值周期必须在 2 到 250 之间")
        if indicator.name == "position" and indicator.field in {"atr_stop_price", "trailing_atr_stop_price"}:
            multiple = float(params.get("multiple", 2))
            if not 0.5 <= multiple <= 10:
                raise ValueError("ATR止损倍数必须在0.5到10之间")
        window_op = params.get("window_op")
        window = params.get("window")
        if (window_op is None) != (window is None):
            raise ValueError("滚动计算必须同时提供 window_op 和 window")
        if window_op is not None:
            if window_op not in {"min", "max", "mean"}:
                raise ValueError("window_op 仅支持 min/max/mean")
            if not 2 <= int(window) <= 250:
                raise ValueError("window 必须在 2 到 250 之间")
        select_by = params.get("select_by")
        selector_keys = {"select_op", "select_window", "select_tie"}
        if select_by is None and any(key in params for key in selector_keys):
            raise ValueError("跨序列定位必须提供 select_by")
        if select_by is not None:
            if window_op is not None:
                raise ValueError("同一指标不能同时使用滚动聚合和跨序列定位")
            if select_by not in {"price.open", "price.high", "price.low", "price.close", "volume.value"}:
                raise ValueError("select_by 不是允许的定位序列")
            if params.get("select_op") not in {"min", "max"}:
                raise ValueError("select_op 仅支持 min/max")
            if not 2 <= int(params.get("select_window", 0)) <= 250:
                raise ValueError("select_window 必须在 2 到 250 之间")
            if params.get("select_tie", "first") not in {"first", "last"}:
                raise ValueError("select_tie 仅支持 first/last")
        if "scale" in params and not -1000 <= float(params["scale"]) <= 1000:
            raise ValueError("scale 必须在 -1000 到 1000 之间")
        if "lag" in params and not 0 <= int(params["lag"]) <= 250:
            raise ValueError("lag 必须在 0 到 250 之间")

    def visit(condition) -> None:
        check_indicator(condition.left)
        if hasattr(condition.right, "name"):
            check_indicator(condition.right)
        for child in condition.conditions:
            visit(child)

    for rule in parsed.rules:
        visit(rule.condition)
    return parsed


def _heuristic_draft(description: str) -> tuple[dict[str, Any], list[str], str]:
    lowered = description.lower()
    if _is_right_side_description(description):
        return RIGHT_SIDE_DSL, RIGHT_SIDE_EXPLANATION, "local"
    if any(keyword in lowered for keyword in ("rsi", "kdj", "均线", "ema", "sma", "突破", "成交量")):
        rules: list[dict[str, Any]] = []
        explanations: list[str] = []
        if "rsi" in lowered:
            rules.append({
                "id": "universe-rsi-strength",
                "scope": "universe",
                "timeframe": "day",
                "condition": {
                    "op": "compare",
                    "left": {"name": "rsi", "field": "value", "params": {"period": 14}},
                    "right": 55,
                    "comparator": ">",
                },
                "action": "add_candidate",
                "label": "日K RSI(14)高于55",
            })
            explanations.append("全市场日K RSI(14)高于55时自动加入待定列表。")
        if "kdj" in lowered:
            rules.append({
                "id": "candidate-kdj-cross",
                "scope": "candidate",
                "timeframe": "day",
                "condition": {
                    "op": "cross_above",
                    "left": {"name": "kdj", "field": "k", "params": {"period": 9}},
                    "right": {"name": "kdj", "field": "d", "params": {"period": 9}},
                },
                "action": "highlight_candidate",
                "label": "日K KDJ金叉",
            })
            explanations.append("待定股票日K K线上穿D线时高亮。")
        if any(keyword in lowered for keyword in ("均线", "sma", "ema")):
            indicator_name = "ema" if "ema" in lowered else "sma"
            rules.append({
                "id": "universe-moving-average-cross",
                "scope": "universe",
                "timeframe": "day",
                "condition": {
                    "op": "cross_above",
                    "left": {"name": "price", "field": "close", "params": {}},
                    "right": {"name": indicator_name, "field": "value", "params": {"period": 20}},
                },
                "action": "add_candidate",
                "label": f"收盘价上穿20日{indicator_name.upper()}",
            })
            explanations.append(f"全市场收盘价上穿20日{indicator_name.upper()}时自动加入待定列表。")
        if "成交量" in lowered:
            rules.append({
                "id": "universe-volume-expansion",
                "scope": "universe",
                "timeframe": "day",
                "condition": {
                    "op": "compare",
                    "left": {"name": "volume", "field": "ratio", "params": {"period": 5}},
                    "right": 1.5,
                    "comparator": ">",
                },
                "action": "add_candidate",
                "label": "成交量大于5日均量1.5倍",
            })
            explanations.append("全市场成交量超过5日均量1.5倍时自动加入待定列表。")
        if "突破" in lowered:
            rules.append({
                "id": "universe-price-breakout",
                "scope": "universe",
                "timeframe": "day",
                "condition": {
                    "op": "breakout",
                    "left": {"name": "price", "field": "close", "params": {}},
                    "periods": 20,
                    "direction": "high",
                },
                "action": "add_candidate",
                "label": "收盘价突破20日高点",
            })
            explanations.append("全市场收盘价突破前20日高点时自动加入待定列表。")
        if rules:
            return {"schema_version": 1, "rules": rules}, explanations, "local"
    return DEFAULT_DSL, DEFAULT_EXPLANATION, "local"


async def translate_description(description: str) -> tuple[dict[str, Any], list[str], str]:
    if _is_right_side_description(description):
        validate_dsl(RIGHT_SIDE_DSL)
        return RIGHT_SIDE_DSL, RIGHT_SIDE_EXPLANATION, "local"
    if not settings.ai_enabled:
        dsl, explanation, source = _heuristic_draft(description)
        validate_dsl(dsl)
        return dsl, explanation, source
    prompt = (
        "你是股票策略编译器，只返回一个JSON对象，顶层必须且只能包含dsl和explanation。"
        "必须严格使用下面示例的字段名和结构，禁止使用type、operator、source、period或子对象形式的value等别名："
        '{"dsl":{"schema_version":1,"rules":[{"id":"volume-expansion","scope":"universe",'
        '"timeframe":"day","condition":{"op":"compare","left":{"name":"volume","field":"ratio",'
        '"params":{"period":5}},"right":1.5,"comparator":">"},"action":"add_candidate",'
        '"label":"成交量放大"}]},"explanation":["全市场成交量超过5日均量1.5倍时加入待定。"]}'
        "rules每项必须包含id、scope、timeframe、condition、action、label。"
        "scope仅universe/candidate/position。timeframe可为day/week/mixed，week始终表示最近完整周K；"
        "自动待定池是每轮扫描的当前快照，不会跨日累加。universe/add_candidate的condition"
        "必须包含用户要求的完整入池条件；禁止只把市场、ST、上市日数等基础过滤放在"
        "universe规则，却把同一次选股的关键技术条件拆到candidate规则。"
        "日线只允许收盘后判断时，规则增加daily_bar_mode=completed；否则省略或使用latest。"
        "condition和indicator均可用timeframe指定day或week。"
        "condition.op仅all/any/not/compare/cross_above/cross_below/rising/falling/breakout/within_pct/ratio_pct/"
        "relative_change/fraction_of_recent/consecutive；within_pct用lower和upper表示百分比区间。"
        "action可为add_candidate/highlight_candidate/highlight_position/signal_buy/signal_add/signal_hold/"
        "signal_reduce/signal_exit/signal_stop；仓位动作可用target_position_pct表示目标仓位。"
        "all/any/not的子条件放在conditions数组；指标对象固定为name、field、params三个字段；"
        "指标滚动计算使用params.window_op=min/max/mean与params.window；跨序列极值日定位使用params.select_by、"
        "select_op=min/max、select_window和select_tie=first/last；固定倍数使用params.scale；引用前N根K线使用params.lag=N。"
        "name仅macd/sma/ema/rsi/kdj/boll/atr/adx/volatility/price/volume/candle/security/position。security可用字段为"
        "is_a_share/is_st/trade_status/listed_trading_days；candle还可用one_word_limit_down。position可用字段为pnl_pct/add_count/"
        "entry_day_low/last_add_price/days_since_last_add/days_since_last_buy/first_add_20d_low/first_add_post_low_high/first_add_rebound_pct/"
        "first_add_rebound_confirmed。禁止输出Markdown和代码。待转换策略：" + description
    )
    headers = {"Authorization": f"Bearer {settings.ai_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": settings.ai_model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.ai_timeout_seconds) as client:
            api_root = settings.ai_base_url if settings.ai_base_url.endswith("/v1") else f"{settings.ai_base_url}/v1"
            response = await client.post(f"{api_root}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        dsl = parsed["dsl"]
        explanation = parsed.get("explanation", [])
        validate_dsl(dsl)
        return dsl, explanation, "ai"
    except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError, ValidationError, ValueError):
        dsl, explanation, source = _heuristic_draft(description)
        validate_dsl(dsl)
        return dsl, explanation, source


def _rolling_series(values: list[float | None], period: int, operation: str) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        window = values[index - period + 1 : index + 1]
        if len(window) < period or any(value is None for value in window):
            result.append(None)
            continue
        numeric = [float(value) for value in window if value is not None]
        if operation == "min":
            result.append(min(numeric))
        elif operation == "max":
            result.append(max(numeric))
        else:
            result.append(sum(numeric) / period)
    return result


def _selector_series(selector: str, bars: list[dict]) -> list[float | None]:
    source, field = selector.split(".", 1)
    if source == "price":
        return [float(bar[field]) for bar in bars]
    if source == "volume" and field == "value":
        return [float(bar["volume"]) for bar in bars]
    raise ValueError(f"unsupported selector {selector}")


def _series_at_selector_extreme(
    values: list[float | None],
    selector: list[float | None],
    period: int,
    operation: str,
    tie: str,
) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        start = index - period + 1
        if start < 0:
            result.append(None)
            continue
        candidates = selector[start : index + 1]
        if any(value is None for value in candidates):
            result.append(None)
            continue
        extreme = min(candidates) if operation == "min" else max(candidates)
        matches = [start + relative for relative, value in enumerate(candidates) if value == extreme]
        selected_index = matches[0] if tie == "first" else matches[-1]
        result.append(values[selected_index])
    return result


def _transform_series(
    values: list[float | None], params: dict[str, Any], bars: list[dict]
) -> list[float | None]:
    if "window_op" in params:
        values = _rolling_series(values, int(params["window"]), str(params["window_op"]))
    if "select_by" in params:
        selector = _selector_series(str(params["select_by"]), bars)
        values = _series_at_selector_extreme(
            values,
            selector,
            int(params["select_window"]),
            str(params["select_op"]),
            str(params.get("select_tie", "first")),
        )
    scale = float(params.get("scale", 1))
    if scale != 1:
        values = [None if value is None else float(value) * scale for value in values]
    lag = int(params.get("lag", 0))
    if lag:
        values = [None] * lag + values[:-lag]
    return values


def _is_st_name(name: str) -> bool:
    return bool(re.match(r"^(?:S\*)?\*?ST", name.strip().upper()))


def _daily_limit_ratio(context: dict[str, Any] | None) -> Decimal:
    code = str((context or {}).get("code") or "")
    if code.startswith(("300", "301", "688", "689")):
        return Decimal("0.20")
    # 自2026-07-06起，沪深主板风险警示股也与其他主板股票统一为10%。
    return Decimal("0.10")


def _one_word_limit_down_series(
    bars: list[dict], context: dict[str, Any] | None
) -> list[float | None]:
    if not bars:
        return []
    result: list[float | None] = [None]
    ratio = _daily_limit_ratio(context)
    cent = Decimal("0.01")
    for previous, current in zip(bars, bars[1:]):
        prices = [Decimal(str(current[field])) for field in ("open", "high", "low", "close")]
        previous_close = Decimal(str(previous["close"]))
        limit_price = (previous_close * (Decimal("1") - ratio)).quantize(cent, rounding=ROUND_HALF_UP)
        one_price = all(price.quantize(cent, rounding=ROUND_HALF_UP) == prices[0].quantize(cent, rounding=ROUND_HALF_UP) for price in prices[1:])
        is_limit = prices[3].quantize(cent, rounding=ROUND_HALF_UP) == limit_price
        has_trade = float(current.get("volume", 0)) > 0
        result.append(1.0 if one_price and is_limit and has_trade else 0.0)
    return result


def _series(indicator: RuleIndicator, bars: list[dict], context: dict[str, Any] | None = None) -> list[float | None]:
    closes = [float(bar["close"]) for bar in bars]
    volumes = [float(bar["volume"]) for bar in bars]
    params = indicator.params
    if indicator.name == "price":
        values: list[float | None] = [float(bar.get(indicator.field, bar["close"])) for bar in bars]
    elif indicator.name == "volume":
        if indicator.field == "ratio":
            period = int(params.get("period", 5))
            averages = sma(volumes, period)
            values = [None if avg in (None, 0) else value / avg for value, avg in zip(volumes, averages)]
        else:
            values = volumes
    elif indicator.name == "candle":
        if indicator.field == "body_pct":
            values = [0.0 if float(bar["open"]) == 0 else abs(float(bar["close"]) - float(bar["open"])) / float(bar["open"]) * 100 for bar in bars]
        elif indicator.field == "bullish":
            values = [1.0 if float(bar["close"]) >= float(bar["open"]) else 0.0 for bar in bars]
        else:
            values = _one_word_limit_down_series(bars, context)
    elif indicator.name == "security":
        if indicator.field == "is_a_share":
            scalar = 1.0 if str((context or {}).get("market")) in {"SH", "SZ"} else 0.0
            values = [scalar for _bar in bars]
        elif indicator.field == "is_st":
            name_status = 1.0 if _is_st_name(str((context or {}).get("name") or "")) else 0.0
            values = [name_status if bar.get("is_st") is None else float(bar["is_st"]) for bar in bars]
            # 历史回测仍以每日 is_st 为准；实时扫描还必须尊重当前证券名称。
            # 否则已改名为 ST/*ST 但最后一根历史K线状态滞后的股票会被误选。
            if bool((context or {}).get("live_snapshot")) and name_status:
                values = [1.0 for _bar in bars]
        elif indicator.field == "trade_status":
            values = [1.0 if bar.get("trade_status") is None else float(bar["trade_status"]) for bar in bars]
        else:
            values = [float(index + 1) for index in range(len(bars))]
    elif indicator.name == "position":
        if indicator.field == "pnl_pct":
            average_cost = float((context or {}).get("avg_cost") or 0)
            values = [None if average_cost <= 0 else (close / average_cost - 1) * 100 for close in closes]
        elif indicator.field in {"atr_stop_price", "trailing_atr_stop_price"}:
            average_cost = float((context or {}).get("avg_cost") or 0)
            entry_date = str((context or {}).get("entry_trade_date") or "")
            multiple = float(params.get("multiple", 2))
            atr_values = atr(bars, int(params.get("period", 14)))
            highest = None
            entry_atr = None
            values = []
            for bar, atr_value in zip(bars, atr_values):
                if not entry_date or str(bar.get("trade_date")) < entry_date or atr_value is None:
                    values.append(None)
                    continue
                if entry_atr is None:
                    entry_atr = float(atr_value)
                highest = max(float(bar["high"]), highest or float(bar["high"]))
                anchor = average_cost if indicator.field == "atr_stop_price" else highest
                # Initial risk is fixed with the ATR known on entry.  Only the
                # trailing stop is allowed to adapt to current volatility.
                risk_atr = entry_atr if indicator.field == "atr_stop_price" else float(atr_value)
                values.append(None if anchor <= 0 else anchor - multiple * risk_atr)
        elif indicator.field == "holding_trading_days":
            entry_date = str((context or {}).get("entry_trade_date") or "")
            count = 0
            values = []
            for bar in bars:
                if entry_date and str(bar.get("trade_date")) >= entry_date:
                    count += 1
                    values.append(float(count))
                else:
                    values.append(None)
        else:
            value = (context or {}).get(indicator.field)
            scalar = None if value is None else float(value)
            values = [scalar for _bar in bars]
    elif indicator.name == "macd":
        result = macd(closes, int(params.get("fast", 12)), int(params.get("slow", 26)), int(params.get("signal", 9)))
        values = result.get(indicator.field, result["histogram"])
    elif indicator.name == "sma":
        values = sma(closes, int(params.get("period", 20)))
    elif indicator.name == "ema":
        values = ema(closes, int(params.get("period", 20)))
    elif indicator.name == "rsi":
        values = rsi(closes, int(params.get("period", 14)))
    elif indicator.name == "kdj":
        result = kdj(bars, int(params.get("period", 9)))
        values = result.get(indicator.field, result["k"])
    elif indicator.name == "boll":
        result = bollinger(closes, int(params.get("period", 20)), float(params.get("stddev", 2)))
        values = result.get(indicator.field, result["middle"])
    elif indicator.name == "atr":
        values = atr(bars, int(params.get("period", 14)))
        if indicator.field == "percent":
            values = [None if value is None or close == 0 else value / close * 100 for value, close in zip(values, closes)]
    elif indicator.name == "adx":
        result = adx(bars, int(params.get("period", 14)))
        values = result.get(indicator.field, result["value"])
    elif indicator.name == "volatility":
        period = int(params.get("period", 20))
        returns = [0.0] + [current / previous - 1 if previous else 0.0 for previous, current in zip(closes, closes[1:])]
        deviations = rolling_std(returns, period)
        values = [None if value is None else value * (252 ** 0.5) * 100 for value in deviations]
    else:
        raise ValueError(f"unsupported indicator {indicator.name}")
    return _transform_series(values, params, bars)


def _current(series: list[float | None], offset: int = 0) -> float | None:
    index = len(series) - 1 - offset
    return series[index] if index >= 0 else None


def _indicator_values(
    indicator: RuleIndicator,
    bars_by_timeframe: dict[str, list[dict]],
    default_timeframe: str,
    context: dict[str, Any] | None,
) -> tuple[list[float | None], str]:
    timeframe = indicator.timeframe or default_timeframe
    return _series(indicator, bars_by_timeframe.get(timeframe, []), context), timeframe


def _value_at(
    indicator: RuleIndicator,
    bars_by_timeframe: dict[str, list[dict]],
    default_timeframe: str,
    context: dict[str, Any] | None,
    offset: int,
    anchor_timeframe: str,
    relative: int = 0,
) -> float | None:
    values, indicator_timeframe = _indicator_values(indicator, bars_by_timeframe, default_timeframe, context)
    base_offset = offset if indicator_timeframe == anchor_timeframe else 0
    return _current(values, base_offset + relative)


def _compare(left: float, right: float, comparator: str | None) -> bool:
    comparators = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    return comparators.get(comparator or ">", lambda _a, _b: False)(left, right)


def evaluate_condition(
    condition: RuleCondition,
    bars: list[dict] | dict[str, list[dict]],
    default_timeframe: str = "day",
    context: dict[str, Any] | None = None,
    offset: int = 0,
    anchor_timeframe: str | None = None,
) -> bool:
    bars_by_timeframe = bars if isinstance(bars, dict) else {"day": bars, "week": weekly_bars(bars)}
    timeframe = condition.timeframe or default_timeframe
    anchor = anchor_timeframe or timeframe
    if condition.op == "all":
        return bool(condition.conditions) and all(
            evaluate_condition(item, bars_by_timeframe, timeframe, context, offset, anchor) for item in condition.conditions
        )
    if condition.op == "any":
        return any(evaluate_condition(item, bars_by_timeframe, timeframe, context, offset, anchor) for item in condition.conditions)
    if condition.op == "not":
        return len(condition.conditions) == 1 and not evaluate_condition(
            condition.conditions[0], bars_by_timeframe, timeframe, context, offset, anchor
        )
    if condition.op == "consecutive":
        return len(condition.conditions) == 1 and all(
            evaluate_condition(condition.conditions[0], bars_by_timeframe, timeframe, context, step, timeframe)
            for step in range(condition.periods)
        )
    if condition.left is None:
        return False
    left, left_timeframe = _indicator_values(condition.left, bars_by_timeframe, timeframe, context)
    left_offset = offset if left_timeframe == anchor else 0
    if isinstance(condition.right, RuleIndicator):
        right_series, right_timeframe = _indicator_values(condition.right, bars_by_timeframe, timeframe, context)
        right_offset = offset if right_timeframe == anchor else 0
    elif isinstance(condition.right, (float, int)):
        right_series = [float(condition.right)]
        right_offset = 0
    else:
        right_series = []
        right_offset = 0
    if condition.op in ("cross_above", "cross_below"):
        left_previous, left_current = _current(left, left_offset + 1), _current(left, left_offset)
        if isinstance(condition.right, RuleIndicator):
            right_previous, right_current = _current(right_series, right_offset + 1), _current(right_series, right_offset)
        else:
            right_previous = right_current = _current(right_series)
        if None in (left_previous, left_current, right_previous, right_current):
            return False
        if condition.op == "cross_above":
            return left_previous <= right_previous and left_current > right_current
        return left_previous >= right_previous and left_current < right_current
    if condition.op in ("rising", "falling"):
        count = condition.periods
        end = len(left) - left_offset
        values = left[max(0, end - count) : end]
        if len(values) < count or any(value is None for value in values):
            return False
        pairs = zip(values, values[1:])
        return all(a < b for a, b in pairs) if condition.op == "rising" else all(a > b for a, b in pairs)
    if condition.op == "breakout":
        current = _current(left, left_offset)
        end = len(left) - left_offset - 1
        history = [value for value in left[max(0, end - condition.periods) : end] if value is not None]
        if current is None or len(history) < condition.periods:
            return False
        return current > max(history) if condition.direction == "high" else current < min(history)
    if condition.op == "compare":
        left_value = _current(left, left_offset)
        right_value = _current(right_series, right_offset) if isinstance(condition.right, RuleIndicator) else _current(right_series)
        if left_value is None or right_value is None:
            return False
        return _compare(left_value, right_value, condition.comparator)
    if condition.op == "within_pct" and isinstance(condition.right, RuleIndicator):
        left_value = _current(left, left_offset)
        right_value = _current(right_series, right_offset)
        if left_value is None or right_value in (None, 0):
            return False
        deviation = (left_value / right_value - 1) * 100
        return float(condition.lower) <= deviation <= float(condition.upper)
    if condition.op == "ratio_pct" and isinstance(condition.right, RuleIndicator):
        left_value = _current(left, left_offset)
        right_value = _current(right_series, right_offset)
        if left_value is None or right_value in (None, 0):
            return False
        ratio = left_value / right_value * 100
        return float(condition.lower) <= ratio <= float(condition.upper)
    if condition.op == "relative_change":
        current, previous = _current(left, left_offset), _current(left, left_offset + 1)
        threshold = _current(right_series)
        if current is None or previous in (None, 0) or threshold is None:
            return False
        # Decimal price boundaries such as -2% and +5% should remain inclusive;
        # suppress binary floating-point tails before applying the comparator.
        change_pct = round((current / previous - 1) * 100, 10)
        return _compare(change_pct, threshold, condition.comparator)
    if condition.op == "fraction_of_recent":
        current = _current(left, left_offset)
        threshold = _current(right_series)
        end = len(left) - left_offset - 1
        recent = [value for value in left[max(0, end - condition.periods) : end] if value is not None and value > 0]
        if current is None or threshold is None or not recent:
            return False
        peak = max(recent)
        return peak > 0 and _compare(current / peak, threshold, condition.comparator)
    return False


def evaluate_rule(
    rule,
    daily_bars: list[dict],
    context: dict[str, Any] | None = None,
    market: str = "SH",
) -> bool:
    if rule.daily_bar_mode == "completed":
        daily_bars = [bar for bar in daily_bars if not bool(bar.get("is_provisional", 0))]

    def uses_entry_day_low(condition: RuleCondition) -> bool:
        indicators = [condition.left, condition.right if isinstance(condition.right, RuleIndicator) else None]
        return any(
            indicator is not None and indicator.name == "position" and indicator.field == "entry_day_low"
            for indicator in indicators
        ) or any(uses_entry_day_low(child) for child in condition.conditions)

    if (
        rule.daily_bar_mode == "completed"
        and context
        and context.get("entry_trade_date")
        and uses_entry_day_low(rule.condition)
        and (not daily_bars or str(daily_bars[-1]["trade_date"]) <= str(context["entry_trade_date"]))
    ):
        return False
    bars_by_timeframe = {
        "day": daily_bars,
        "week": weekly_bars(daily_bars, completed_only=True, market=market),
    }
    default_timeframe = "day" if rule.timeframe == "mixed" else rule.timeframe
    return evaluate_condition(rule.condition, bars_by_timeframe, default_timeframe, context)


def validation_errors(value: dict[str, Any]) -> list[str]:
    try:
        validate_dsl(value)
        return []
    except ValidationError as error:
        return [".".join(str(part) for part in item["loc"]) + ": " + item["msg"] for item in error.errors()]
    except ValueError as error:
        return [str(error)]
