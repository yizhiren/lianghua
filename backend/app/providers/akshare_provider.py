from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from .base import MarketDataProvider, Quote, SecurityInfo


EXCLUDED_NAME_PARTS = ("ETF", "基金", "债", "权证", "REIT", "优先", "认购", "认沽")
TEMPORARY_NAME_SUFFIXES = ("（新）", "(新)", "－新", "-新", "（旧）", "(旧)", "－旧", "-旧")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value) in ("nan", "None", "-"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class AkshareProvider(MarketDataProvider):
    def __init__(self, retries: int = 3):
        try:
            import akshare as ak
        except ImportError as error:
            raise RuntimeError("AKShare 未安装，请先安装 requirements.txt") from error
        self.ak = ak
        self.retries = max(1, retries)

    def _retry(self, operation):
        last_error = None
        for attempt in range(self.retries):
            try:
                return operation()
            except Exception as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(0.5 * (2 ** attempt))
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _is_stock(name: str) -> bool:
        normalized = name.upper()
        return (
            bool(name)
            and not name.endswith(TEMPORARY_NAME_SUFFIXES)
            and not any(part.upper() in normalized for part in EXCLUDED_NAME_PARTS)
        )

    def _parse_spot(self, frame, market: str, currency: str) -> tuple[list[SecurityInfo], list[Quote]]:
        securities: list[SecurityInfo] = []
        quotes: list[Quote] = []
        now = datetime.now().astimezone().isoformat()
        for record in frame.to_dict("records"):
            raw_code = str(record.get("代码", "")).strip().lower()
            item_market = market
            if market == "A":
                if raw_code.startswith(("sh", "sz")):
                    item_market, raw_code = raw_code[:2].upper(), raw_code[2:]
                elif raw_code.startswith(("600", "601", "603", "605", "688", "689")):
                    item_market = "SH"
                elif raw_code.startswith(("000", "001", "002", "003", "300", "301")):
                    item_market = "SZ"
                else:
                    continue
            code = raw_code.zfill(5 if item_market == "HK" else 6)
            name = str(record.get("名称") or record.get("中文名称") or "").strip()
            if not code.isdigit() or not self._is_stock(name):
                continue
            security_id = f"{item_market}.{code}"
            securities.append(SecurityInfo(security_id, item_market, code, name, currency))
            price = _number(record.get("最新价"))
            if price <= 0:
                continue
            quotes.append(
                Quote(
                    security_id=security_id,
                    market=item_market,
                    price=price,
                    open=_number(record.get("今开"), price),
                    high=_number(record.get("最高"), price),
                    low=_number(record.get("最低"), price),
                    previous_close=_number(record.get("昨收"), price),
                    volume=_number(record.get("成交量")),
                    quote_time=now,
                )
            )
        return securities, quotes

    def fetch_universe_and_quotes(self) -> tuple[list[SecurityInfo], list[Quote]]:
        result_securities: list[SecurityInfo] = []
        result_quotes: list[Quote] = []

        def first_available(label: str, sources):
            errors = []
            for source_name, operation in sources:
                try:
                    frame = self._retry(operation)
                    if frame is not None and not frame.empty:
                        return frame
                    errors.append(f"{source_name}: 空数据")
                except Exception as error:
                    errors.append(f"{source_name}: {error}")
            raise ConnectionError(f"{label}行情源全部失败：" + "；".join(errors))

        a_frame = first_available("A股", [
            ("东财沪深A股", self.ak.stock_zh_a_spot_em),
            ("东财异步沪深A股", self.ak.stock_zh_a_spot_em_async),
            ("新浪沪深A股", self.ak.stock_zh_a_spot),
        ])
        hk_frame = first_available("港股", [
            ("东财港股主板", self.ak.stock_hk_main_board_spot_em),
            ("新浪港股", self.ak.stock_hk_spot),
        ])
        for frame, market, currency in ((a_frame, "A", "CNY"), (hk_frame, "HK", "HKD")):
            securities, quotes = self._parse_spot(frame, market, currency)
            result_securities.extend(securities)
            result_quotes.extend(quotes)
        if not result_securities or not result_quotes:
            raise ConnectionError("行情源未返回可用股票报价")
        return result_securities, result_quotes

    def fetch_hk_delisted_master(self) -> list[dict[str, Any]]:
        import pandas as pd

        url = "https://di.hkex.com.hk/di/NSDelistedStockList.aspx?lang=EN"
        tables = self._retry(lambda: pd.read_html(url))
        table = max(tables, key=len)
        records: dict[str, dict[str, Any]] = {}
        for raw_code, raw_name in table.iloc[1:, :2].itertuples(index=False, name=None):
            code = str(raw_code).strip().split(".")[0].zfill(5)
            name = str(raw_name).strip()
            if not code.isdigit() or not name:
                continue
            records.setdefault(code, {
                "id": f"HK.{code}", "market": "HK", "code": code, "name": name,
                "currency": "HKD", "listing_date": None, "delisting_date": None,
                "is_active": 0,
            })
        return list(records.values())

    def fetch_daily_bars(self, security: SecurityInfo, start_date: str, end_date: str) -> list[dict]:
        compact_start = start_date.replace("-", "")
        compact_end = end_date.replace("-", "")
        if security.market in ("SH", "SZ"):
            sources = [
                (lambda: self.ak.stock_zh_a_hist(symbol=security.code, period="daily", start_date=compact_start, end_date=compact_end, adjust="qfq"), "akshare_eastmoney_qfq"),
                (lambda: self.ak.stock_zh_a_daily(symbol=f"{security.market.lower()}{security.code}", start_date=compact_start, end_date=compact_end, adjust="qfq"), "akshare_sina_qfq"),
            ]
        else:
            sources = [
                (lambda: self.ak.stock_hk_hist(symbol=security.code, period="daily", start_date=compact_start, end_date=compact_end, adjust="qfq"), "akshare_hk_eastmoney_qfq"),
                (lambda: self.ak.stock_hk_daily(symbol=security.code, adjust="qfq"), "akshare_hk_sina_qfq"),
            ]
        frame = None
        source_name = ""
        errors = []
        for operation, candidate_name in sources:
            try:
                frame = self._retry(operation)
                source_name = candidate_name
                if frame is not None and not frame.empty:
                    break
            except Exception as error:
                errors.append(f"{candidate_name}: {error}")
        if frame is None:
            raise ConnectionError("；".join(errors) or "AKShare未返回日K")
        mapping = {
            "日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume",
            "date": "trade_date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
        }
        frame = frame.rename(columns=mapping)
        records = []
        for item in frame.to_dict("records"):
            trade_date = str(item["trade_date"])[:10]
            if start_date <= trade_date <= end_date:
                records.append(
                    {
                        "trade_date": trade_date,
                        "open": _number(item["open"]),
                        "high": _number(item["high"]),
                        "low": _number(item["low"]),
                        "close": _number(item["close"]),
                        "volume": _number(item.get("volume")),
                        "trade_status": 1,
                        "is_st": None,
                        "data_source": source_name,
                    }
                )
        return records
