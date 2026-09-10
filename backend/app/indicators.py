from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import math
from typing import Iterable, Optional
from zoneinfo import ZoneInfo


Number = Optional[float]


def sma(values: list[float], period: int) -> list[Number]:
    result: list[Number] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        result.append(running / period if index >= period - 1 else None)
    return result


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2 / (period + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, list[float]]:
    fast_line = ema(values, fast)
    slow_line = ema(values, slow)
    dif = [fast_value - slow_value for fast_value, slow_value in zip(fast_line, slow_line)]
    dea = ema(dif, signal)
    histogram = [2 * (dif_value - dea_value) for dif_value, dea_value in zip(dif, dea)]
    return {"dif": dif, "dea": dea, "histogram": histogram}


def rsi(values: list[float], period: int = 14) -> list[Number]:
    if not values:
        return []
    gains = [0.0]
    losses = [0.0]
    for previous, current in zip(values, values[1:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    result: list[Number] = [None] * len(values)
    if len(values) <= period:
        return result
    average_gain = sum(gains[1 : period + 1]) / period
    average_loss = sum(losses[1 : period + 1]) / period
    result[period] = 100.0 if average_loss == 0 else 100 - 100 / (1 + average_gain / average_loss)
    for index in range(period + 1, len(values)):
        average_gain = (average_gain * (period - 1) + gains[index]) / period
        average_loss = (average_loss * (period - 1) + losses[index]) / period
        result[index] = 100.0 if average_loss == 0 else 100 - 100 / (1 + average_gain / average_loss)
    return result


def kdj(bars: list[dict], period: int = 9, smooth_k: int = 3, smooth_d: int = 3) -> dict[str, list[Number]]:
    k_values: list[Number] = []
    d_values: list[Number] = []
    j_values: list[Number] = []
    k = 50.0
    d = 50.0
    for index, bar in enumerate(bars):
        if index < period - 1:
            k_values.append(None)
            d_values.append(None)
            j_values.append(None)
            continue
        window = bars[index - period + 1 : index + 1]
        lowest = min(float(item["low"]) for item in window)
        highest = max(float(item["high"]) for item in window)
        rsv = 50.0 if highest == lowest else (float(bar["close"]) - lowest) / (highest - lowest) * 100
        k = (smooth_k - 1) / smooth_k * k + rsv / smooth_k
        d = (smooth_d - 1) / smooth_d * d + k / smooth_d
        k_values.append(k)
        d_values.append(d)
        j_values.append(3 * k - 2 * d)
    return {"k": k_values, "d": d_values, "j": j_values}


def rolling_std(values: list[float], period: int) -> list[Number]:
    result: list[Number] = []
    for index in range(len(values)):
        window = values[index - period + 1 : index + 1]
        if len(window) < period:
            result.append(None)
            continue
        mean = sum(window) / period
        result.append(math.sqrt(sum((value - mean) ** 2 for value in window) / period))
    return result


def bollinger(values: list[float], period: int = 20, stddev: float = 2.0) -> dict[str, list[Number]]:
    middle = sma(values, period)
    deviations = rolling_std(values, period)
    upper: list[Number] = []
    lower: list[Number] = []
    percent_b: list[Number] = []
    bandwidth: list[Number] = []
    for value, mean, deviation in zip(values, middle, deviations):
        if mean is None or deviation is None:
            upper.append(None); lower.append(None); percent_b.append(None); bandwidth.append(None)
            continue
        high, low = mean + stddev * deviation, mean - stddev * deviation
        upper.append(high); lower.append(low)
        width = high - low
        percent_b.append(None if width == 0 else (value - low) / width)
        bandwidth.append(None if mean == 0 else width / mean * 100)
    return {"middle": middle, "upper": upper, "lower": lower, "percent_b": percent_b, "bandwidth": bandwidth}


def true_range(bars: list[dict]) -> list[float]:
    result: list[float] = []
    for index, bar in enumerate(bars):
        high, low = float(bar["high"]), float(bar["low"])
        if index == 0:
            result.append(high - low)
        else:
            previous_close = float(bars[index - 1]["close"])
            result.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return result


def wilder(values: list[float], period: int) -> list[Number]:
    result: list[Number] = [None] * len(values)
    if len(values) < period:
        return result
    average = sum(values[:period]) / period
    result[period - 1] = average
    for index in range(period, len(values)):
        average = (average * (period - 1) + values[index]) / period
        result[index] = average
    return result


def atr(bars: list[dict], period: int = 14) -> list[Number]:
    return wilder(true_range(bars), period)


def adx(bars: list[dict], period: int = 14) -> dict[str, list[Number]]:
    plus_dm = [0.0]
    minus_dm = [0.0]
    for previous, current in zip(bars, bars[1:]):
        up = float(current["high"]) - float(previous["high"])
        down = float(previous["low"]) - float(current["low"])
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    ranges = wilder(true_range(bars), period)
    plus_smooth, minus_smooth = wilder(plus_dm, period), wilder(minus_dm, period)
    plus_di: list[Number] = []
    minus_di: list[Number] = []
    dx: list[float] = []
    for tr, plus, minus in zip(ranges, plus_smooth, minus_smooth):
        if tr in (None, 0) or plus is None or minus is None:
            plus_di.append(None); minus_di.append(None); dx.append(0.0)
            continue
        p, m = plus / tr * 100, minus / tr * 100
        plus_di.append(p); minus_di.append(m)
        dx.append(0.0 if p + m == 0 else abs(p - m) / (p + m) * 100)
    return {"value": wilder(dx, period), "plus_di": plus_di, "minus_di": minus_di}


def weekly_bars(
    daily_bars: list[dict],
    *,
    completed_only: bool = False,
    market: str = "SH",
    now: datetime | None = None,
) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for bar in daily_bars:
        trade_date = date.fromisoformat(str(bar["trade_date"])[:10])
        iso = trade_date.isocalendar()
        groups[(iso.year, iso.week)].append(bar)
    result = []
    local_now = (now or datetime.now(ZoneInfo("Asia/Shanghai"))).astimezone(ZoneInfo("Asia/Shanghai"))
    current_iso = local_now.date().isocalendar()
    current_week = (current_iso.year, current_iso.week)
    close_minute = 16 * 60 if market == "HK" else 15 * 60
    for week_key, group in groups.items():
        ordered = sorted(group, key=lambda item: item["trade_date"])
        if completed_only and week_key == current_week:
            last_date = date.fromisoformat(str(ordered[-1]["trade_date"])[:10])
            after_week_close = (
                local_now.weekday() > 4
                or (
                    local_now.weekday() == 4
                    and local_now.hour * 60 + local_now.minute >= close_minute
                    and last_date.weekday() == 4
                    and not bool(ordered[-1].get("is_provisional", 0))
                )
            )
            if not after_week_close:
                continue
        result.append(
            {
                "trade_date": ordered[-1]["trade_date"],
                "open": ordered[0]["open"],
                "high": max(item["high"] for item in ordered),
                "low": min(item["low"] for item in ordered),
                "close": ordered[-1]["close"],
                "volume": sum(item["volume"] for item in ordered),
            }
        )
    return result


def compact(values: Iterable[Number]) -> list[float]:
    return [float(value) for value in values if value is not None]
