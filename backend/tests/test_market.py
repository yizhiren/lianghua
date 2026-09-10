from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from backend.app import market as market_module
from backend.app.database import Database
from backend.app.market import MarketService


def test_backfill_batch_rotates_across_markets_and_recounts_progress(tmp_path: Path):
    class RecordingMarketService(MarketService):
        def __init__(self, database):
            super().__init__(database)
            self.seen: list[str] = []

        def backfill_security(self, security_id: str, refresh_progress: bool = True) -> int:
            self.seen.append(security_id)
            first_day = date(2024, 1, 1)
            rows = [
                (
                    security_id,
                    (first_day + timedelta(days=index)).isoformat(),
                    10,
                    11,
                    9,
                    10,
                    1000,
                    0,
                )
                for index in range(60)
            ]
            self.db.executemany(
                """INSERT INTO bars(security_id,trade_date,open,high,low,close,volume,is_provisional)
                   VALUES(?,?,?,?,?,?,?,?)""",
                rows,
            )
            self.db.execute(
                """INSERT INTO market_backfill_state(
                       security_id,desired_start,actual_start,actual_end,bar_count,status,attempts,source,updated_at
                   ) VALUES(?,?,?,?,60,'complete',1,'test',?)""",
                (security_id, self._desired_start(), rows[0][1], rows[-1][1], "2026-07-20T00:00:00+00:00"),
            )
            if refresh_progress:
                self._refresh_backfill_counts()
            return len(rows)

    database = Database(tmp_path / "market.db")
    database.initialize()
    for market in ("SH", "SZ", "HK"):
        database.execute(
            """INSERT INTO market_status(market,state,data_status,initialized,backfilled,total)
               VALUES(?,'closed','live',1,0,3)""",
            (market,),
        )
        for index in range(1, 4):
            code = f"{index:05d}" if market == "HK" else f"{index:06d}"
            database.execute(
                """INSERT INTO securities(id,market,code,name,currency,is_active)
                   VALUES(?,?,?,?,?,1)""",
                (f"{market}.{code}", market, code, f"{market}-{index}", "HKD" if market == "HK" else "CNY"),
            )

    original_mode = market_module.settings.market_data_mode
    object.__setattr__(market_module.settings, "market_data_mode", "akshare")
    try:
        service = RecordingMarketService(database)
        result = service.backfill_next_batch(limit=6)
    finally:
        object.__setattr__(market_module.settings, "market_data_mode", original_mode)

    assert sorted(service.seen) == sorted(
        ["SH.000001", "SZ.000001", "HK.00001", "SH.000002", "SZ.000002", "HK.00002"]
    )
    assert result == {"processed": 6, "bars": 360}
    progress = database.all("SELECT market,backfilled FROM market_status ORDER BY market")
    assert {row["market"]: row["backfilled"] for row in progress} == {"HK": 2, "SH": 2, "SZ": 2}


def test_finalize_daily_bars_only_completes_requested_market(tmp_path: Path):
    database = Database(tmp_path / "finalize.db")
    database.initialize()
    for market, code in (("SH", "600000"), ("HK", "09988")):
        security_id = f"{market}.{code}"
        database.execute(
            "INSERT INTO securities(id,market,code,name,currency,is_active) VALUES(?,?,?,?,?,1)",
            (security_id, market, code, security_id, "CNY" if market == "SH" else "HKD"),
        )
        database.execute(
            """INSERT INTO bars(security_id,trade_date,open,high,low,close,volume,is_provisional)
               VALUES(?,?,?,?,?,?,?,1)""",
            (security_id, "2026-07-20", 10, 11, 9, 10, 100),
        )
    service = MarketService(database)
    assert service.finalize_daily_bars(("SH", "SZ"), "2026-07-20") == 1
    states = {row["security_id"]: row["is_provisional"] for row in database.all("SELECT security_id,is_provisional FROM bars")}
    assert states == {"SH.600000": 0, "HK.09988": 1}


def test_incremental_backfill_replaces_raw_gap_and_rebases_qfq_prefix(tmp_path: Path):
    class IncrementalProvider:
        def __init__(self):
            self.requested_start = None

        def fetch_daily_bars(self, _security, start_date: str, _end_date: str):
            self.requested_start = start_date
            return [
                {"trade_date": "2026-07-21", "open": 50, "high": 51, "low": 49, "close": 50,
                 "volume": 1000, "trade_status": 1, "is_st": 0, "data_source": "test_qfq"},
                {"trade_date": "2026-07-22", "open": 51, "high": 56, "low": 50, "close": 55,
                 "volume": 1100, "trade_status": 1, "is_st": 0, "data_source": "test_qfq"},
                {"trade_date": "2026-08-21", "open": 59, "high": 61, "low": 58, "close": 60,
                 "volume": 1200, "trade_status": 1, "is_st": 0, "data_source": "test_qfq"},
            ]

    database = Database(tmp_path / "incremental.db")
    database.initialize()
    database.execute(
        """INSERT INTO securities(id,market,code,name,currency,is_active)
           VALUES('HK.00001','HK','00001','测试股份','HKD',1)"""
    )
    database.executemany(
        """INSERT INTO bars(
               security_id,trade_date,open,high,low,close,volume,is_provisional,
               trade_status,is_st,data_source
           ) VALUES('HK.00001',?,?,?,?,?,?,0,1,0,?)""",
        [
            ("2026-07-20", 100, 102, 98, 100, 900, "old_qfq"),
            ("2026-07-21", 100, 102, 98, 100, 1000, "old_qfq"),
            ("2026-08-21", 300, 305, 295, 300, 1200, None),
        ],
    )
    database.execute(
        """INSERT INTO market_backfill_state(
               security_id,desired_start,actual_start,actual_end,bar_count,status,attempts,source,updated_at
           ) VALUES('HK.00001','2011-07-21','2026-07-20','2026-07-21',2,'complete',1,'old_qfq',?)""",
        ("2026-08-21T00:00:00+00:00",),
    )
    provider = IncrementalProvider()
    service = MarketService(database)
    assert service.backfill_security("HK.00001", hk_provider=provider) == 3
    assert provider.requested_start == "2026-07-21"
    bars = database.all(
        "SELECT trade_date,close,data_source FROM bars WHERE security_id='HK.00001' ORDER BY trade_date"
    )
    assert bars == [
        {"trade_date": "2026-07-20", "close": 50.0, "data_source": "old_qfq"},
        {"trade_date": "2026-07-21", "close": 50.0, "data_source": "test_qfq"},
        {"trade_date": "2026-07-22", "close": 55.0, "data_source": "test_qfq"},
        {"trade_date": "2026-08-21", "close": 60.0, "data_source": "test_qfq"},
    ]
    state = database.one("SELECT * FROM market_backfill_state WHERE security_id='HK.00001'")
    assert state["actual_start"] == "2026-07-20"
    assert state["actual_end"] == "2026-08-21"
    assert state["bar_count"] == 4


def test_manual_strategy_refresh_syncs_once_and_finalizes_after_close(tmp_path: Path):
    class RefreshMarketService(MarketService):
        def __init__(self, database):
            super().__init__(database)
            self.sync_count = 0

        def sync_universe_and_quotes(self):
            self.sync_count += 1
            now = "2026-08-21T09:01:00+00:00"
            with self.db.transaction() as connection:
                for market in ("SH", "SZ", "HK"):
                    connection.execute(
                        """INSERT INTO market_status(market,state,quote_time,last_sync_at,data_status)
                           VALUES(?,'closed',?,?,'live')
                           ON CONFLICT(market) DO UPDATE SET quote_time=excluded.quote_time,
                             last_sync_at=excluded.last_sync_at""",
                        (market, "2026-08-21T17:01:00+08:00", now),
                    )
                connection.execute(
                    """INSERT INTO securities(id,market,code,name,currency,is_active)
                       VALUES('SH.600000','SH','600000','浦发银行','CNY',1)"""
                )
                connection.execute(
                    """INSERT INTO bars(security_id,trade_date,open,high,low,close,volume,is_provisional)
                       VALUES('SH.600000','2026-08-21',10,11,9,10,100,1)"""
                )
            return {"quotes": 1}

    database = Database(tmp_path / "manual-refresh.db")
    database.initialize()
    service = RefreshMarketService(database)
    original_mode = market_module.settings.market_data_mode
    object.__setattr__(market_module.settings, "market_data_mode", "akshare")
    try:
        now = datetime(2026, 8, 21, 17, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        first = service.refresh_for_strategy_scan(now)
        second = service.refresh_for_strategy_scan(now + timedelta(minutes=30))
    finally:
        object.__setattr__(market_module.settings, "market_data_mode", original_mode)

    assert first["refreshed"] is True
    assert first["finalized"] == 1
    assert second["refreshed"] is False
    assert second["finalized"] == 0
    assert service.sync_count == 1
    bar = database.one("SELECT is_provisional FROM bars WHERE security_id='SH.600000'")
    assert bar["is_provisional"] == 0
