from __future__ import annotations

import re
from typing import Any

from .dsl import validate_dsl
from .schemas import RuleCondition, RuleIndicator, StrategyDSL


TIMEFRAME_TO_TEXT = {"day": "日K", "week": "完整周K", "mixed": "跨周期"}
TEXT_TO_TIMEFRAME = {value: key for key, value in TIMEFRAME_TO_TEXT.items()}
SCOPE_TO_TEXT = {"universe": "全市场", "candidate": "待定池", "position": "持仓池"}
TEXT_TO_SCOPE = {value: key for key, value in SCOPE_TO_TEXT.items()}
ACTION_TO_TEXT = {
    "add_candidate": "加入待定",
    "highlight_candidate": "优选关注",
    "highlight_position": "持仓提醒",
    "signal_buy": "提示建仓",
    "signal_add": "提示加仓",
    "signal_hold": "继续持有",
    "signal_reduce": "提示减仓",
    "signal_exit": "提示清仓",
    "signal_stop": "止损清仓",
}
TEXT_TO_ACTION = {value: key for key, value in ACTION_TO_TEXT.items()}
DAILY_BAR_MODE_TO_TEXT = {"latest": "包含盘中临时日K", "completed": "仅使用已完成日K"}
TEXT_TO_DAILY_BAR_MODE = {value: key for key, value in DAILY_BAR_MODE_TO_TEXT.items()}

INDICATOR_TO_TEXT = {
    ("macd", "dif"): "MACD.DIFF",
    ("macd", "dea"): "MACD.DEA",
    ("macd", "histogram"): "MACD.柱",
    ("sma", "value"): "SMA",
    ("ema", "value"): "EMA",
    ("rsi", "value"): "RSI",
    ("kdj", "k"): "KDJ.K",
    ("kdj", "d"): "KDJ.D",
    ("kdj", "j"): "KDJ.J",
    ("boll", "middle"): "BOLL.中轨",
    ("boll", "upper"): "BOLL.上轨",
    ("boll", "lower"): "BOLL.下轨",
    ("boll", "percent_b"): "BOLL.%B",
    ("boll", "bandwidth"): "BOLL.带宽",
    ("atr", "value"): "ATR",
    ("atr", "percent"): "ATR百分比",
    ("adx", "value"): "ADX",
    ("adx", "plus_di"): "ADX.+DI",
    ("adx", "minus_di"): "ADX.-DI",
    ("volatility", "value"): "年化波动率",
    ("price", "open"): "开盘价",
    ("price", "high"): "最高价",
    ("price", "low"): "最低价",
    ("price", "close"): "收盘价",
    ("price", "value"): "收盘价",
    ("volume", "value"): "成交量",
    ("volume", "ratio"): "成交量比",
    ("candle", "body_pct"): "K线实体百分比",
    ("candle", "bullish"): "阳线",
    ("candle", "one_word_limit_down"): "一字跌停",
    ("security", "is_a_share"): "沪深A股",
    ("security", "is_st"): "ST股票",
    ("security", "trade_status"): "当日正常交易状态",
    ("security", "listed_trading_days"): "上市交易日数",
    ("position", "pnl_pct"): "盈亏百分比",
    ("position", "entry_day_low"): "建仓日最低价",
    ("position", "add_count"): "补仓次数",
    ("position", "last_add_price"): "前一次补仓价格",
    ("position", "days_since_last_add"): "距前一次补仓自然日数",
    ("position", "days_since_last_buy"): "距上次买入或补仓自然日数",
    ("position", "holding_trading_days"): "持仓交易日数",
    ("position", "atr_stop_price"): "ATR初始止损价",
    ("position", "trailing_atr_stop_price"): "ATR移动止损价",
    ("position", "first_add_20d_low"): "首次补仓时20日最低价",
    ("position", "first_add_post_low_high"): "首次补仓时最低点后最高价",
    ("position", "first_add_rebound_pct"): "首次补仓时反弹幅度百分比",
    ("position", "first_add_rebound_confirmed"): "首次补仓时反弹已确认",
}
TEXT_TO_INDICATOR = {value: key for key, value in INDICATOR_TO_TEXT.items()}
TEXT_TO_INDICATOR.update(
    {
        "前次补仓价格": ("position", "last_add_price"),
        "首次加仓时20日最低价": ("position", "first_add_20d_low"),
        "首次加仓时最低点后最高价": ("position", "first_add_post_low_high"),
        "首次加仓时反弹幅度百分比": ("position", "first_add_rebound_pct"),
        "首次加仓时反弹已确认": ("position", "first_add_rebound_confirmed"),
    }
)


def _number(value: float | int) -> str:
    return f"{float(value):g}"


def _params_text(params: dict[str, Any]) -> str:
    if not params:
        return ""
    return "[" + ";".join(f"{key}={value}" for key, value in sorted(params.items())) + "]"


def _indicator_text(indicator: RuleIndicator, default_timeframe: str) -> str:
    name = INDICATOR_TO_TEXT[(indicator.name, indicator.field)]
    if indicator.name == "position":
        return f"持仓.{name}{_params_text(indicator.params)}"
    timeframe = indicator.timeframe or default_timeframe
    return f"{TIMEFRAME_TO_TEXT[timeframe]}.{name}{_params_text(indicator.params)}"


def _condition_text(condition: RuleCondition, default_timeframe: str) -> str:
    timeframe = condition.timeframe or default_timeframe
    children = [_condition_text(item, timeframe) for item in condition.conditions]
    left = _indicator_text(condition.left, timeframe) if condition.left else ""
    if isinstance(condition.right, RuleIndicator):
        right = _indicator_text(condition.right, timeframe)
    elif isinstance(condition.right, (float, int)):
        right = _number(condition.right)
    else:
        right = ""
    if condition.op == "all":
        return f"全部满足({','.join(children)})"
    if condition.op == "any":
        return f"任一满足({','.join(children)})"
    if condition.op == "not":
        return f"不满足({children[0]})"
    if condition.op == "consecutive":
        return f"连续满足({TIMEFRAME_TO_TEXT[timeframe]},{condition.periods},{children[0]})"
    compare_names = {">": "大于", ">=": "大于等于", "<": "小于", "<=": "小于等于", "==": "等于", "!=": "不等于"}
    if condition.op == "compare":
        return f"{compare_names[condition.comparator or '>']}({left},{right})"
    if condition.op == "cross_above":
        return f"上穿({left},{right})"
    if condition.op == "cross_below":
        return f"下穿({left},{right})"
    if condition.op == "rising":
        return f"连续上升({left},{condition.periods})"
    if condition.op == "falling":
        return f"连续下降({left},{condition.periods})"
    if condition.op == "breakout":
        name = "突破近高" if condition.direction == "high" else "跌破近低"
        return f"{name}({left},{condition.periods})"
    if condition.op == "within_pct":
        return f"偏离率介于({left},{right},{_number(condition.lower)},{_number(condition.upper)})"
    if condition.op == "ratio_pct":
        return f"比值百分比介于({left},{right},{_number(condition.lower)},{_number(condition.upper)})"
    if condition.op == "relative_change":
        return f"相对前期变化({left},{condition.comparator},{right})"
    if condition.op == "fraction_of_recent":
        return f"最近峰值占比({left},{condition.periods},{condition.comparator},{right})"
    raise ValueError(f"不支持的条件操作：{condition.op}")


def render_controlled_text(name: str, dsl: dict[str, Any] | StrategyDSL) -> str:
    parsed = dsl if isinstance(dsl, StrategyDSL) else validate_dsl(dsl)
    lines = [f"策略名称：{name.strip()}", "执行周期：每15分钟", "周K口径：仅使用最近完整周K"]
    for rule in parsed.rules:
        default_timeframe = "day" if rule.timeframe == "mixed" else rule.timeframe
        lines.extend(
            [
                "",
                f"规则：{rule.id}",
                f"对象：{SCOPE_TO_TEXT[rule.scope]}",
                f"默认周期：{TIMEFRAME_TO_TEXT[rule.timeframe]}",
                f"日K口径：{DAILY_BAR_MODE_TO_TEXT[rule.daily_bar_mode]}",
                f"条件：{_condition_text(rule.condition, default_timeframe)}",
                f"动作：{ACTION_TO_TEXT[rule.action]}",
                f"标签：{rule.label}",
            ]
        )
        if rule.target_position_pct is not None:
            lines.append(f"目标仓位：{_number(rule.target_position_pct)}%")
    return "\n".join(lines).strip()


def _split_args(value: str) -> list[str]:
    result: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(value):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    return [item for item in result if item]


def _call(expression: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r"([^()]+)\((.*)\)", expression.strip())
    if not match:
        raise ValueError(f"条件格式错误：{expression}")
    return match.group(1).strip(), _split_args(match.group(2))


def _parse_scalar(value: str) -> float:
    try:
        return float(value.rstrip("%"))
    except ValueError as error:
        raise ValueError(f"数字格式错误：{value}") from error


def _parse_indicator(value: str) -> dict[str, Any]:
    value = value.strip()
    params: dict[str, float | int | str] = {}
    params_match = re.search(r"\[([^]]*)\]$", value)
    if params_match:
        for pair in params_match.group(1).split(";"):
            key, raw = pair.split("=", 1)
            if re.fullmatch(r"-?\d+", raw):
                number: float | int | str = int(raw)
            elif re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", raw):
                number = float(raw)
            else:
                number = raw
            params[key] = number
        value = value[: params_match.start()]
    timeframe = None
    if value.startswith("日K."):
        timeframe, name = "day", value[3:]
    elif value.startswith("完整周K."):
        timeframe, name = "week", value[5:]
    elif value.startswith("持仓."):
        name = value[3:]
    else:
        raise ValueError(f"指标必须声明日K、完整周K或持仓：{value}")
    if name not in TEXT_TO_INDICATOR:
        raise ValueError(f"未知指标：{name}")
    indicator_name, field = TEXT_TO_INDICATOR[name]
    result: dict[str, Any] = {"name": indicator_name, "field": field, "params": params}
    if timeframe:
        result["timeframe"] = timeframe
    return result


def _indicator_or_number(value: str) -> dict[str, Any] | float:
    if value.startswith(("日K.", "完整周K.", "持仓.")):
        return _parse_indicator(value)
    return _parse_scalar(value)


def _parse_condition(expression: str) -> dict[str, Any]:
    name, args = _call(expression)
    if name in {"全部满足", "任一满足"}:
        return {"op": "all" if name == "全部满足" else "any", "conditions": [_parse_condition(item) for item in args]}
    if name == "不满足" and len(args) == 1:
        return {"op": "not", "conditions": [_parse_condition(args[0])]}
    if name == "连续满足" and len(args) == 3:
        if args[0] not in TEXT_TO_TIMEFRAME or args[0] == "跨周期":
            raise ValueError("连续满足的周期只能是日K或完整周K")
        return {"op": "consecutive", "timeframe": TEXT_TO_TIMEFRAME[args[0]], "periods": int(args[1]), "conditions": [_parse_condition(args[2])]}
    comparator_by_name = {"大于": ">", "大于等于": ">=", "小于": "<", "小于等于": "<=", "等于": "==", "不等于": "!="}
    if name in comparator_by_name and len(args) == 2:
        return {"op": "compare", "left": _parse_indicator(args[0]), "right": _indicator_or_number(args[1]), "comparator": comparator_by_name[name]}
    if name in {"上穿", "下穿"} and len(args) == 2:
        return {"op": "cross_above" if name == "上穿" else "cross_below", "left": _parse_indicator(args[0]), "right": _indicator_or_number(args[1])}
    if name in {"连续上升", "连续下降"} and len(args) == 2:
        return {"op": "rising" if name == "连续上升" else "falling", "left": _parse_indicator(args[0]), "periods": int(args[1])}
    if name in {"突破近高", "跌破近低"} and len(args) == 2:
        return {"op": "breakout", "left": _parse_indicator(args[0]), "periods": int(args[1]), "direction": "high" if name == "突破近高" else "low"}
    if name in {"偏离率介于", "比值百分比介于"} and len(args) == 4:
        return {"op": "within_pct" if name == "偏离率介于" else "ratio_pct", "left": _parse_indicator(args[0]), "right": _parse_indicator(args[1]), "lower": _parse_scalar(args[2]), "upper": _parse_scalar(args[3])}
    if name == "相对前期变化" and len(args) == 3:
        return {"op": "relative_change", "left": _parse_indicator(args[0]), "comparator": args[1], "right": _parse_scalar(args[2])}
    if name == "最近峰值占比" and len(args) == 4:
        return {"op": "fraction_of_recent", "left": _parse_indicator(args[0]), "periods": int(args[1]), "comparator": args[2], "right": _parse_scalar(args[3])}
    raise ValueError(f"未知或参数数量错误的条件：{expression}")


def compile_controlled_text(text: str) -> tuple[str, dict[str, Any]]:
    clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not clean_lines or not clean_lines[0].startswith("策略名称："):
        raise ValueError("第一行必须是“策略名称：...”")
    name = clean_lines[0].split("：", 1)[1].strip()
    if not name:
        raise ValueError("策略名称不能为空")
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in clean_lines[1:]:
        if line.startswith("规则："):
            current = {"规则": line.split("：", 1)[1].strip()}
            blocks.append(current)
            continue
        if current is None:
            continue
        if "：" not in line:
            raise ValueError(f"规则行缺少中文冒号：{line}")
        key, value = line.split("：", 1)
        current[key] = value.strip()
    if not blocks:
        raise ValueError("至少需要一条规则")
    rules = []
    for block in blocks:
        missing = [key for key in ("规则", "对象", "默认周期", "条件", "动作", "标签") if not block.get(key)]
        if missing:
            raise ValueError(f"规则{block.get('规则', '')}缺少字段：{','.join(missing)}")
        if block["对象"] not in TEXT_TO_SCOPE:
            raise ValueError(f"未知对象：{block['对象']}")
        if block["默认周期"] not in TEXT_TO_TIMEFRAME:
            raise ValueError(f"未知默认周期：{block['默认周期']}")
        if block["动作"] not in TEXT_TO_ACTION:
            raise ValueError(f"未知动作：{block['动作']}")
        rule: dict[str, Any] = {
            "id": block["规则"],
            "scope": TEXT_TO_SCOPE[block["对象"]],
            "timeframe": TEXT_TO_TIMEFRAME[block["默认周期"]],
            "condition": _parse_condition(block["条件"]),
            "action": TEXT_TO_ACTION[block["动作"]],
            "label": block["标签"],
        }
        if block.get("日K口径"):
            if block["日K口径"] not in TEXT_TO_DAILY_BAR_MODE:
                raise ValueError(f"未知日K口径：{block['日K口径']}")
            rule["daily_bar_mode"] = TEXT_TO_DAILY_BAR_MODE[block["日K口径"]]
        if block.get("目标仓位"):
            rule["target_position_pct"] = _parse_scalar(block["目标仓位"])
        rules.append(rule)
    dsl = {"schema_version": 1, "rules": rules}
    validate_dsl(dsl)
    return name, dsl


def explanations_from_dsl(dsl: dict[str, Any]) -> list[str]:
    parsed = validate_dsl(dsl)
    return [f"{SCOPE_TO_TEXT[rule.scope]}：{rule.label} → {ACTION_TO_TEXT[rule.action]}。" for rule in parsed.rules]
