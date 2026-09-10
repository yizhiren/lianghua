from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from backend.app.dsl import (
    DEFAULT_DSL,
    LEFT_SIDE_DSL,
    RIGHT_SIDE_DSL,
    TWO_DAY_LEFT_SIDE_DSL,
    _heuristic_draft,
    _series,
    evaluate_condition,
    evaluate_rule,
    validate_dsl,
)
from backend.app.indicators import adx, atr, bollinger, ema, macd, rsi, sma, weekly_bars
from backend.app.schemas import RuleCondition, RuleIndicator, StrategyRule


def test_moving_averages_and_macd_shapes():
    values = [float(value) for value in range(1, 61)]
    assert sma(values, 5)[3] is None
    assert sma(values, 5)[4] == 3.0
    assert len(ema(values, 12)) == len(values)
    result = macd(values)
    assert set(result) == {"dif", "dea", "histogram"}
    assert all(len(line) == len(values) for line in result.values())
    assert result["dif"][-1] > result["dea"][-1]


def test_rsi_rising_market_reaches_100():
    values = [float(value) for value in range(1, 40)]
    result = rsi(values, 14)
    assert result[13] is None
    assert result[-1] == 100.0


def test_daily_bars_resample_to_weeks():
    start = date(2026, 1, 5)
    bars = []
    for index in range(10):
        current = start + timedelta(days=index)
        if current.weekday() < 5:
            bars.append({"trade_date": current.isoformat(), "open": 10 + index, "high": 12 + index, "low": 9 + index, "close": 11 + index, "volume": 100})
    weeks = weekly_bars(bars)
    assert len(weeks) == 2
    assert weeks[0]["volume"] == 500
    assert weeks[0]["open"] == 10


def test_completed_weekly_bars_exclude_current_unfinished_week():
    bars = []
    for current in (date(2026, 7, 13) + timedelta(days=index) for index in range(10)):
        if current.weekday() < 5:
            bars.append({"trade_date": current.isoformat(), "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "is_provisional": 0})
    now = datetime(2026, 7, 22, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    completed = weekly_bars(bars, completed_only=True, now=now)
    assert len(completed) == 1
    assert completed[0]["trade_date"] == "2026-07-17"


def test_rising_condition_uses_strict_order():
    bars = [{"trade_date": f"2026-01-{day:02d}", "open": day, "high": day, "low": day, "close": day, "volume": 1} for day in range(1, 10)]
    condition = RuleCondition.model_validate({"op": "rising", "left": {"name": "price", "field": "close"}, "periods": 3})
    assert evaluate_condition(condition, bars) is True
    bars[-1]["close"] = bars[-2]["close"]
    assert evaluate_condition(condition, bars) is False


def test_relative_change_inclusive_boundaries_ignore_float_tails():
    lower = RuleCondition.model_validate({
        "op": "relative_change",
        "left": {"name": "price", "field": "close"},
        "right": -2,
        "comparator": ">=",
    })
    upper = RuleCondition.model_validate({
        "op": "relative_change",
        "left": {"name": "price", "field": "close"},
        "right": 5,
        "comparator": "<=",
    })

    def matches(close: float) -> bool:
        bars = [
            {"trade_date": "2026-08-19", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"trade_date": "2026-08-20", "open": close, "high": close, "low": close, "close": close, "volume": 1},
        ]
        return evaluate_condition(lower, bars) and evaluate_condition(upper, bars)

    assert matches(98) is True
    assert matches(105) is True
    assert matches(97.99) is False
    assert matches(105.01) is False


def test_initial_atr_stop_stays_fixed_when_volatility_changes_after_entry():
    bars = [
        {"trade_date": "2026-01-01", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1},
        {"trade_date": "2026-01-02", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1},
        {"trade_date": "2026-01-05", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1},
        {"trade_date": "2026-01-06", "open": 10, "high": 20, "low": 1, "close": 10, "volume": 1},
    ]
    indicator = RuleIndicator.model_validate({
        "name": "position", "field": "atr_stop_price",
        "params": {"period": 2, "multiple": 1},
    })
    values = _series(
        indicator, bars,
        {"avg_cost": 10, "entry_trade_date": "2026-01-05"},
    )
    assert values[2] == 8
    assert values[3] == values[2]


def test_default_dsl_is_valid():
    parsed = validate_dsl(DEFAULT_DSL)
    assert len(parsed.rules) == 3
    assert parsed.rules[0].action == "add_candidate"


def test_right_side_dsl_supports_mixed_timeframes_and_position_actions():
    parsed = validate_dsl(RIGHT_SIDE_DSL)
    assert len(parsed.rules) == 8
    assert parsed.rules[0].timeframe == "mixed"
    assert {rule.action for rule in parsed.rules} >= {"signal_buy", "signal_add", "signal_reduce", "signal_exit", "signal_stop"}


def test_left_side_dsl_is_valid_and_uses_completed_daily_bars():
    parsed = validate_dsl(LEFT_SIDE_DSL)
    assert len(parsed.rules) == 2
    assert all(rule.daily_bar_mode == "completed" for rule in parsed.rules)
    assert parsed.rules[1].condition.right.field == "entry_day_low"


def test_selector_extreme_uses_first_matching_date_and_can_use_last_explicitly():
    bars = []
    for index in range(20):
        close = 5 if index in {3, 10} else 10
        volume = 10 if index == 3 else 100 if index == 10 else 20
        bars.append({"trade_date": f"2026-01-{index + 1:02d}", "open": close, "high": close, "low": close, "close": close, "volume": volume})

    def condition(tie: str) -> RuleCondition:
        return RuleCondition.model_validate({
            "op": "compare",
            "left": {"name": "volume", "field": "value", "params": {}},
            "right": {
                "name": "volume",
                "field": "value",
                "params": {"select_by": "price.close", "select_op": "min", "select_window": 20, "select_tie": tie},
            },
            "comparator": ">",
        })

    assert evaluate_condition(condition("first"), bars) is True
    assert evaluate_condition(condition("last"), bars) is False


def test_current_price_low_makes_dif_compare_with_itself_and_fail():
    bars = [
        {"trade_date": f"2026-02-{index + 1:02d}", "open": 100 - index, "high": 100 - index, "low": 100 - index, "close": 100 - index, "volume": 100}
        for index in range(20)
    ]
    condition = RuleCondition.model_validate({
        "op": "compare",
        "left": {"name": "macd", "field": "dif", "params": {"fast": 12, "slow": 26, "signal": 9}},
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
    })
    assert evaluate_condition(condition, bars) is False


def test_volume_moving_average_comparison_and_scale_are_exact():
    bars = [
        {"trade_date": f"2026-03-{index + 1:02d}", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 50 if index >= 15 else 100}
        for index in range(20)
    ]
    condition = RuleCondition.model_validate({
        "op": "compare",
        "left": {"name": "volume", "field": "value", "params": {"window_op": "mean", "window": 5}},
        "right": {"name": "volume", "field": "value", "params": {"window_op": "mean", "window": 20, "scale": 0.65}},
        "comparator": "<",
    })
    assert evaluate_condition(condition, bars) is True


def test_completed_daily_rule_ignores_intraday_provisional_bar():
    dsl = {
        "schema_version": 1,
        "rules": [{
            "id": "completed-only",
            "scope": "universe",
            "timeframe": "day",
            "daily_bar_mode": "completed",
            "condition": {"op": "compare", "left": {"name": "price", "field": "close"}, "right": 10, "comparator": "=="},
            "action": "add_candidate",
            "label": "仅看收盘",
        }],
    }
    rule = validate_dsl(dsl).rules[0]
    bars = [
        {"trade_date": "2026-07-17", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100, "is_provisional": 0},
        {"trade_date": "2026-07-20", "open": 20, "high": 20, "low": 20, "close": 20, "volume": 100, "is_provisional": 1},
    ]
    assert evaluate_rule(rule, bars) is True


def test_entry_low_stop_only_starts_after_entry_trading_day():
    rule = validate_dsl(LEFT_SIDE_DSL).rules[1]
    prior_bar = {
        "trade_date": "2026-07-17",
        "open": 86,
        "high": 87,
        "low": 84,
        "close": 85,
        "volume": 100,
        "is_provisional": 0,
    }
    entry_bar = {
        "trade_date": "2026-07-20",
        "open": 100,
        "high": 101,
        "low": 90,
        "close": 95,
        "volume": 100,
        "is_provisional": 1,
    }
    context = {"entry_trade_date": "2026-07-20", "entry_day_low": 90}
    assert evaluate_rule(rule, [prior_bar, entry_bar], context=context) is False
    entry_bar = {**entry_bar, "is_provisional": 0}
    next_bar = {**entry_bar, "trade_date": "2026-07-21", "open": 86, "high": 87, "low": 84, "close": 85}
    assert evaluate_rule(rule, [prior_bar, entry_bar, next_bar], context=context) is True


def test_lag_is_applied_after_rolling_window():
    bars = [
        {"trade_date": f"2026-07-{day:02d}", "open": close, "high": close, "low": close, "close": close, "volume": 100}
        for day, close in enumerate((5, 4, 3, 2), start=1)
    ]
    aligned = RuleCondition.model_validate({
        "op": "compare",
        "left": {"name": "price", "field": "close", "params": {"lag": 1}},
        "right": {"name": "price", "field": "close", "params": {"window_op": "min", "window": 3, "lag": 1}},
        "comparator": "==",
    })
    current_window = RuleCondition.model_validate({
        "op": "compare",
        "left": {"name": "price", "field": "close", "params": {"lag": 1}},
        "right": {"name": "price", "field": "close", "params": {"window_op": "min", "window": 3}},
        "comparator": "==",
    })
    assert evaluate_condition(aligned, bars) is True
    assert evaluate_condition(current_window, bars) is False


def test_security_metadata_filters_a_shares_st_and_completed_bar_count():
    bars = [
        {"trade_date": f"2025-{index // 28 + 1:02d}-{index % 28 + 1:02d}", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100}
        for index in range(120)
    ]
    conditions = [
        RuleCondition.model_validate({"op": "compare", "left": {"name": "security", "field": "is_a_share"}, "right": 1, "comparator": "=="}),
        RuleCondition.model_validate({"op": "compare", "left": {"name": "security", "field": "is_st"}, "right": 0, "comparator": "=="}),
        RuleCondition.model_validate({"op": "compare", "left": {"name": "security", "field": "listed_trading_days"}, "right": 120, "comparator": ">="}),
    ]
    normal = {"market": "SH", "code": "600000", "name": "浦发银行"}
    assert all(evaluate_condition(condition, bars, context=normal) for condition in conditions)
    assert evaluate_condition(conditions[0], bars, context={**normal, "market": "HK"}) is False
    assert evaluate_condition(conditions[1], bars, context={**normal, "name": "*ST测试"}) is False
    assert evaluate_condition(conditions[2], bars[:-1], context=normal) is False


@pytest.mark.parametrize(
    ("code", "name", "limit_close"),
    [("600000", "主板测试", 9.0), ("301001", "创业板测试", 8.0), ("600001", "*ST测试", 9.0)],
)
def test_one_word_limit_down_uses_board_and_st_limit(code, name, limit_close):
    bars = [
        {"trade_date": "2026-07-19", "open": 10, "high": 10.2, "low": 9.8, "close": 10, "volume": 100},
        {"trade_date": "2026-07-20", "open": limit_close, "high": limit_close, "low": limit_close, "close": limit_close, "volume": 100},
    ]
    condition = RuleCondition.model_validate({
        "op": "compare",
        "left": {"name": "candle", "field": "one_word_limit_down"},
        "right": 1,
        "comparator": "==",
    })
    assert evaluate_condition(condition, bars, context={"market": "SH", "code": code, "name": name}) is True
    bars[-1]["high"] = limit_close + 0.01
    assert evaluate_condition(condition, bars, context={"market": "SH", "code": code, "name": name}) is False


def test_two_day_left_side_strategy_matches_only_with_t_minus_one_alignment():
    trade_dates = []
    current = date(2025, 1, 1)
    while len(trade_dates) < 150:
        if current.weekday() < 5:
            trade_dates.append(current.isoformat())
        current += timedelta(days=1)
    prefix = [200 - (80 * index / 119) for index in range(120)]
    closes = prefix + [110, 100, 90] + [94.5, 99, 103.5, 108] + [108 - (19 * (index + 1) / 22) for index in range(22)] + [90]
    bars = [
        {
            "trade_date": trade_date,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 100 if index >= 148 else 1000,
            "is_provisional": 0,
        }
        for index, (trade_date, close) in enumerate(zip(trade_dates, closes))
    ]
    rule = validate_dsl(TWO_DAY_LEFT_SIDE_DSL).rules[0]
    context = {"market": "SH", "code": "600000", "name": "测试股份"}
    assert evaluate_rule(rule, bars, context=context) is True
    assert evaluate_rule(rule, bars, context={**context, "name": "ST测试"}) is False
    assert evaluate_rule(rule, bars, context={**context, "market": "HK"}) is False
    bars[-2]["close"] = 91
    assert evaluate_rule(rule, bars, context=context) is False


def test_right_side_description_with_build_position_word_uses_full_template():
    description = "完整周K的DIFF上升，日K RSI小于60，站上5周均线后等待建仓，跌破10周均线止损。"
    dsl, _explanation, source = _heuristic_draft(description)
    assert source == "local"
    assert len(dsl["rules"]) == 8


def test_within_percent_can_compare_daily_price_to_weekly_average():
    daily = [{"trade_date": "2026-07-20", "open": 101, "high": 102, "low": 100, "close": 101, "volume": 100}]
    weekly = [
        {"trade_date": f"2026-{month:02d}-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 500}
        for month in range(1, 6)
    ]
    condition = RuleCondition.model_validate({
        "op": "within_pct",
        "timeframe": "day",
        "left": {"name": "price", "field": "close"},
        "right": {"name": "sma", "field": "value", "params": {"period": 5}, "timeframe": "week"},
        "lower": -2,
        "upper": 2,
    })
    assert evaluate_condition(condition, {"day": daily, "week": weekly}) is True


def test_consecutive_daily_condition_keeps_weekly_reference_fixed():
    daily = [
        {"trade_date": f"2026-07-{day:02d}", "open": close, "high": close, "low": close, "close": close, "volume": 100}
        for day, close in ((18, 105), (19, 99), (20, 98))
    ]
    weekly = [
        {"trade_date": f"2026-{month:02d}-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 500}
        for month in range(1, 11)
    ]
    condition = RuleCondition.model_validate({
        "op": "consecutive",
        "timeframe": "day",
        "periods": 2,
        "conditions": [{
            "op": "compare",
            "left": {"name": "price", "field": "close"},
            "right": {"name": "sma", "field": "value", "params": {"period": 10}, "timeframe": "week"},
            "comparator": "<",
        }],
    })
    assert evaluate_condition(condition, {"day": daily, "week": weekly}) is True


def test_dsl_rejects_unbounded_indicator_parameters():
    invalid = {
        "schema_version": 1,
        "rules": [{
            "id": "bad-macd",
            "scope": "universe",
            "timeframe": "day",
            "condition": {"op": "rising", "left": {"name": "macd", "field": "histogram", "params": {"fast": 30, "slow": 10}}, "periods": 3},
            "action": "add_candidate",
            "label": "非法MACD",
        }],
    }
    with pytest.raises(ValueError, match="MACD"):
        validate_dsl(invalid)


def test_position_add_state_can_be_used_in_dsl_conditions():
    bars = [
        {"trade_date": f"2026-06-{day:02d}", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
        for day in range(1, 21)
    ]
    condition = RuleCondition.model_validate({
        "op": "all",
        "conditions": [
            {"op": "compare", "left": {"name": "position", "field": "add_count"}, "right": 1, "comparator": ">="},
            {"op": "compare", "left": {"name": "price", "field": "close"}, "right": {"name": "position", "field": "last_add_price"}, "comparator": "<"},
            {"op": "compare", "left": {"name": "position", "field": "first_add_rebound_confirmed"}, "right": 1, "comparator": "=="},
            {"op": "compare", "left": {"name": "position", "field": "days_since_last_add"}, "right": 3, "comparator": ">="},
            {"op": "compare", "left": {"name": "position", "field": "days_since_last_buy"}, "right": 3, "comparator": ">="},
        ],
    })
    context = {"add_count": 2, "last_add_price": 105, "first_add_rebound_confirmed": 1, "days_since_last_add": 3.25, "days_since_last_buy": 3.25}
    assert evaluate_condition(condition, bars, context=context) is True
    context["days_since_last_add"] = 2.99
    assert evaluate_condition(condition, bars, context=context) is False


def test_boll_atr_adx_and_volatility_are_available_to_rules():
    start = date(2025, 1, 1)
    bars = []
    for index in range(80):
        close = 100 + index * 0.4 + (index % 5 - 2) * 0.2
        bars.append({
            "trade_date": (start + timedelta(days=index)).isoformat(), "open": close - 0.2,
            "high": close + 1, "low": close - 1, "close": close, "volume": 1000 + index,
            "is_provisional": 0,
        })
    boll = bollinger([bar["close"] for bar in bars], 20, 2)
    assert boll["middle"][-1] is not None and boll["upper"][-1] > boll["lower"][-1]
    assert atr(bars, 14)[-1] > 0
    assert adx(bars, 14)["value"][-1] is not None
    rule = StrategyRule.model_validate({
        "id": "advanced-indicators", "scope": "universe", "timeframe": "day",
        "condition": {"op": "all", "conditions": [
            {"op": "compare", "left": {"name": "boll", "field": "bandwidth", "params": {"period": 20}}, "right": 0, "comparator": ">"},
            {"op": "compare", "left": {"name": "atr", "field": "percent", "params": {"period": 14}}, "right": 0, "comparator": ">"},
            {"op": "compare", "left": {"name": "adx", "field": "value", "params": {"period": 14}}, "right": 0, "comparator": ">="},
            {"op": "compare", "left": {"name": "volatility", "field": "value", "params": {"period": 20}}, "right": 0, "comparator": ">"},
        ]}, "action": "add_candidate", "label": "高级指标可执行",
    })
    assert evaluate_rule(rule, bars) is True


def test_security_status_uses_point_in_time_bar_fields_before_current_name():
    bars = [{
        "trade_date": "2020-06-01", "open": 10, "high": 11, "low": 9,
        "close": 10, "volume": 100, "is_provisional": 0,
        "is_st": 1, "trade_status": 0,
    }]
    rule = StrategyRule.model_validate({
        "id": "historical-status", "scope": "universe", "timeframe": "day",
        "condition": {"op": "all", "conditions": [
            {"op": "compare", "left": {"name": "security", "field": "is_st"}, "right": 1, "comparator": "=="},
            {"op": "compare", "left": {"name": "security", "field": "trade_status"}, "right": 0, "comparator": "=="},
        ]},
        "action": "add_candidate", "label": "使用历史状态",
    })
    assert evaluate_rule(rule, bars, context={"name": "当前正常名称", "market": "SH"}) is True


def test_live_snapshot_rejects_current_st_name_even_when_bar_status_is_stale():
    bars = [{
        "trade_date": "2026-07-20", "open": 10, "high": 11, "low": 9,
        "close": 10, "volume": 100, "is_provisional": 0, "is_st": 0,
    }]
    rule = StrategyRule.model_validate({
        "id": "live-non-st", "scope": "universe", "timeframe": "day",
        "condition": {
            "op": "compare",
            "left": {"name": "security", "field": "is_st"},
            "right": 0,
            "comparator": "==",
        },
        "action": "add_candidate", "label": "排除当前ST",
    })
    assert evaluate_rule(
        rule,
        bars,
        context={"name": "*ST测试", "market": "SH", "live_snapshot": 1},
    ) is False
