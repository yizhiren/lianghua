from __future__ import annotations

import pandas as pd

from backend.app.providers.akshare_provider import AkshareProvider


class _FallbackAkshare:
    @staticmethod
    def _failed():
        raise ConnectionError("主源不可用")

    stock_zh_a_spot_em = _failed
    stock_zh_a_spot_em_async = _failed
    stock_hk_main_board_spot_em = _failed

    @staticmethod
    def stock_zh_a_spot():
        return pd.DataFrame([
            {"代码": "sh600000", "名称": "浦发银行", "最新价": 10, "今开": 9.8, "最高": 10.2, "最低": 9.7, "昨收": 9.9, "成交量": 1000},
            {"代码": "sz000001", "名称": "平安银行", "最新价": 11, "今开": 10.8, "最高": 11.2, "最低": 10.7, "昨收": 10.9, "成交量": 2000},
        ])

    @staticmethod
    def stock_hk_spot():
        return pd.DataFrame([
            {"代码": "00700", "中文名称": "腾讯控股", "最新价": 500, "今开": 495, "最高": 505, "最低": 493, "昨收": 498, "成交量": 3000},
        ])


def test_realtime_quotes_fall_back_to_sina_and_normalize_codes():
    provider = object.__new__(AkshareProvider)
    provider.ak = _FallbackAkshare()
    provider.retries = 1

    securities, quotes = provider.fetch_universe_and_quotes()

    assert {item.id for item in securities} == {"SH.600000", "SZ.000001", "HK.00700"}
    assert {item.security_id for item in quotes} == {"SH.600000", "SZ.000001", "HK.00700"}
