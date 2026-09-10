from __future__ import annotations

import threading
from typing import Any

from .base import MarketDataProvider, Quote, SecurityInfo


_SESSION_LOCK = threading.RLock()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _is_a_share(market: str, code: str) -> bool:
    if market == "SH":
        return code.startswith(("600", "601", "603", "605", "688", "689"))
    return code.startswith(("000", "001", "002", "003", "300", "301"))


class BaoStockProvider(MarketDataProvider):
    """Long-history A-share source with daily point-in-time ST/trading status."""

    def __init__(self) -> None:
        try:
            import baostock as bs
        except ImportError as error:
            raise RuntimeError("BaoStock 未安装，请先安装 requirements.txt") from error
        self.bs = bs
        self._entered = False

    def _login(self) -> None:
        response = self.bs.login()
        if response.error_code != "0":
            raise ConnectionError(f"BaoStock登录失败：{response.error_msg}")

    def __enter__(self) -> "BaoStockProvider":
        _SESSION_LOCK.acquire()
        try:
            self._login()
        except Exception:
            _SESSION_LOCK.release()
            raise
        self._entered = True
        return self

    def __exit__(self, *_args) -> None:
        try:
            if self._entered:
                self.bs.logout()
        finally:
            self._entered = False
            _SESSION_LOCK.release()

    def _require_session(self) -> None:
        if not self._entered:
            raise RuntimeError("BaoStockProvider 必须在上下文管理器中使用")

    @staticmethod
    def _session_expired(response: Any) -> bool:
        message = str(getattr(response, "error_msg", ""))
        return getattr(response, "error_code", "0") != "0" and "未登录" in message

    def _retry_after_relogin(self, query) -> Any:
        """BaoStock sessions can expire during a long batch; renew once in place."""
        response = query()
        if self._session_expired(response):
            try:
                self.bs.logout()
            except Exception:
                pass
            self._login()
            response = query()
        return response

    def fetch_security_master(self) -> list[dict[str, Any]]:
        self._require_session()
        response = self._retry_after_relogin(self.bs.query_stock_basic)
        if response.error_code != "0":
            raise ConnectionError(f"BaoStock证券主数据失败：{response.error_msg}")
        records: list[dict[str, Any]] = []
        while response.next():
            item = dict(zip(response.fields, response.get_row_data()))
            raw_code = str(item.get("code") or "")
            if "." not in raw_code or str(item.get("type")) != "1":
                continue
            prefix, code = raw_code.split(".", 1)
            market = "SH" if prefix.lower() == "sh" else "SZ"
            if not _is_a_share(market, code):
                continue
            records.append({
                "id": f"{market}.{code}", "market": market, "code": code,
                "name": str(item.get("code_name") or code), "currency": "CNY",
                "listing_date": str(item.get("ipoDate") or "") or None,
                "delisting_date": str(item.get("outDate") or "") or None,
                "is_active": 1 if str(item.get("status")) == "1" else 0,
            })
        return records

    def fetch_universe_and_quotes(self) -> tuple[list[SecurityInfo], list[Quote]]:
        master = self.fetch_security_master()
        securities = [
            SecurityInfo(item["id"], item["market"], item["code"], item["name"], "CNY")
            for item in master
        ]
        return securities, []

    def fetch_daily_bars(self, security: SecurityInfo, start_date: str, end_date: str) -> list[dict]:
        self._require_session()
        code = f"{security.market.lower()}.{security.code}"
        fields = "date,open,high,low,close,volume,tradestatus,isST"
        response = self._retry_after_relogin(lambda: self.bs.query_history_k_data_plus(
            code, fields, start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2",
        ))
        if response.error_code != "0":
            raise ConnectionError(f"BaoStock日K失败：{response.error_msg}")
        records: list[dict[str, Any]] = []
        while response.next():
            item = dict(zip(response.fields, response.get_row_data()))
            close = _number(item.get("close"))
            if close <= 0:
                continue
            records.append({
                "trade_date": str(item["date"]),
                "open": _number(item.get("open")), "high": _number(item.get("high")),
                "low": _number(item.get("low")), "close": close,
                "volume": _number(item.get("volume")),
                "trade_status": int(_number(item.get("tradestatus"), 1)),
                "is_st": int(_number(item.get("isST"), 0)),
                "data_source": "baostock_qfq",
            })
        return records
