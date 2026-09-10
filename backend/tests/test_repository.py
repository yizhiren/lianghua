from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import pytest

from backend.app.database import Database
from backend.app.dsl import RIGHT_SIDE_DSL, RIGHT_SIDE_EXPLANATION
from backend.app.repository import Repository
from backend.app.seed import seed_demo
from backend.app.strategy_service import StrategyService


@pytest.fixture()
def services(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    seed_demo(database, force=True)
    repository = Repository(database)
    service = StrategyService(database, repository)
    return database, repository, service


def test_seeded_dashboard_has_two_strategy_columns(services):
    _database, repository, _service = services
    dashboard = repository.dashboard()
    assert len(dashboard["strategies"]) == 2
    assert {item["market"] for item in dashboard["market_status"]} == {"SH", "SZ", "HK"}
    assert dashboard["strategies"][0]["candidates"]
    assert dashboard["strategies"][0]["positions"]


def test_candidate_position_round_trip_is_transactional(services):
    _database, repository, _service = services
    strategy_id = "strategy-momentum"
    security_id = "HK.09988"
    repository.add_position(strategy_id, security_id, 100, 150)
    strategy = repository.get_strategy(strategy_id)
    assert any(item["security_id"] == security_id for item in strategy["positions"])
    assert not any(item["security_id"] == security_id for item in strategy["candidates"])
    assert repository.delete_position(strategy_id, security_id, True)
    strategy = repository.get_strategy(strategy_id)
    assert any(item["security_id"] == security_id for item in strategy["candidates"])


def test_position_records_entry_trade_date_and_uses_that_days_low(services):
    database, repository, service = services
    strategy_id = "strategy-momentum"
    security_id = "HK.09988"
    database.execute("DELETE FROM bars WHERE security_id=?", (security_id,))
    database.execute(
        """INSERT INTO bars(security_id,trade_date,open,high,low,close,volume,is_provisional)
           VALUES(?,?,?,?,?,?,?,0)""",
        (security_id, "2026-07-20", 100, 105, 91, 99, 1000),
    )
    strategy = repository.add_position(
        strategy_id, security_id, 100, 99, "2026-07-20T16:30:00+08:00"
    )
    position = next(item for item in strategy["positions"] if item["security_id"] == security_id)
    context = next(item for item in service._position_items(strategy_id) if item["security_id"] == security_id)
    assert position["entry_trade_date"] == "2026-07-20"
    assert position["entry_day_low"] == 91
    assert context["entry_day_low"] == 91


def test_duplicate_between_lists_is_rejected(services):
    _database, repository, _service = services
    with pytest.raises(ValueError, match="持仓"):
        repository.add_candidate("strategy-momentum", "SH.600519")


def test_strategy_execution_persists_run(services):
    database, _repository, service = services
    run = service.execute("strategy-momentum", "test-slot")
    assert run["status"] == "success"
    assert run["securities_scanned"] == 8
    assert database.one("SELECT * FROM strategy_runs WHERE id=?", (run["id"],)) is not None


def test_strategy_execution_replaces_auto_candidates_with_current_snapshot(services):
    database, repository, service = services
    dsl = {
        "schema_version": 1,
        "rules": [
            {
                "id": "a-share-snapshot",
                "scope": "universe",
                "timeframe": "day",
                "condition": {
                    "op": "compare",
                    "left": {"name": "security", "field": "is_a_share", "params": {}},
                    "right": 1,
                    "comparator": "==",
                },
                "action": "add_candidate",
                "label": "当前A股快照",
            }
        ],
    }
    strategy = repository.create_strategy("快照策略", "只保留本轮命中", dsl, ["当前A股快照"])
    repository.add_candidate(strategy["id"], "HK.09988", "auto")
    repository.add_candidate(strategy["id"], "HK.01810", "manual")

    first_run = service.execute(strategy["id"], "snapshot-a-share")
    first_candidates = database.all(
        "SELECT security_id,added_by FROM candidates WHERE strategy_id=? ORDER BY security_id",
        (strategy["id"],),
    )
    assert first_run["candidates_added"] == 4
    assert {item["security_id"] for item in first_candidates if item["added_by"] == "auto"} == {
        "SH.600036",
        "SH.600519",
        "SZ.000858",
        "SZ.300750",
    }
    assert {item["security_id"] for item in first_candidates if item["added_by"] == "manual"} == {"HK.01810"}

    hk_dsl = {
        **dsl,
        "rules": [
            {
                **dsl["rules"][0],
                "id": "hk-snapshot",
                "condition": {
                    "op": "compare",
                    "left": {"name": "security", "field": "is_a_share", "params": {}},
                    "right": 0,
                    "comparator": "==",
                },
                "label": "当前港股快照",
            }
        ],
    }
    repository.activate_version(strategy["id"], "切换为港股快照", hk_dsl, ["当前港股快照"])
    second_run = service.execute(strategy["id"], "snapshot-hk")
    second_candidates = database.all(
        "SELECT security_id,added_by FROM candidates WHERE strategy_id=? ORDER BY security_id",
        (strategy["id"],),
    )
    assert second_run["candidates_added"] == 3
    assert {item["security_id"] for item in second_candidates if item["added_by"] == "auto"} == {
        "HK.00700",
        "HK.03690",
        "HK.09988",
    }
    assert {item["security_id"] for item in second_candidates if item["added_by"] == "manual"} == {"HK.01810"}


def test_strategy_execution_excludes_stale_completed_bar_from_snapshot(services):
    database, repository, service = services
    latest_date = database.one(
        """SELECT MAX(b.trade_date) AS value FROM bars b
           JOIN securities s ON s.id=b.security_id
           WHERE s.market='HK' AND b.is_provisional=0"""
    )["value"]
    database.execute(
        "UPDATE market_status SET state='closed',quote_time=? WHERE market='HK'",
        (f"{latest_date}T16:00:00+08:00",),
    )
    stale_security = "HK.09988"
    database.execute(
        "DELETE FROM bars WHERE security_id=? AND trade_date=?",
        (stale_security, latest_date),
    )
    strategy = repository.create_strategy(
        "日K新鲜度测试",
        "仅允许最新完整日K入池",
        {
            "schema_version": 1,
            "rules": [{
                "id": "latest-hk-snapshot",
                "scope": "universe",
                "timeframe": "day",
                "daily_bar_mode": "completed",
                "condition": {
                    "op": "compare",
                    "left": {"name": "security", "field": "is_a_share", "params": {}},
                    "right": 0,
                    "comparator": "==",
                },
                "action": "add_candidate",
                "label": "最新港股快照",
            }],
        },
        ["只使用与港股市场最新交易日一致的完整日K"],
    )
    repository.add_candidate(strategy["id"], stale_security, "auto")

    service.execute(strategy["id"], "stale-completed-bar")

    candidates = {
        row["security_id"]
        for row in database.all(
            "SELECT security_id FROM candidates WHERE strategy_id=?",
            (strategy["id"],),
        )
    }
    assert stale_security not in candidates
    assert "HK.00700" in candidates


def test_strategy_snapshot_excludes_past_delisting_date(services):
    database, repository, service = services
    strategy = repository.create_strategy(
        "退市过滤",
        "全市场价格有效时入池",
        {
            "schema_version": 1,
            "rules": [{
                "id": "positive-price",
                "scope": "universe",
                "timeframe": "day",
                "condition": {
                    "op": "compare",
                    "left": {"name": "price", "field": "close", "params": {}},
                    "right": 0,
                    "comparator": ">",
                },
                "action": "add_candidate",
                "label": "价格有效",
            }],
        },
        ["价格有效"],
    )
    database.execute(
        "UPDATE securities SET is_active=1,delisting_date='2020-01-01' WHERE id='SH.600519'"
    )
    service.execute(strategy["id"], "exclude-delisted")
    candidates = {
        item["security_id"]
        for item in database.all(
            "SELECT security_id FROM candidates WHERE strategy_id=?",
            (strategy["id"],),
        )
    }
    assert "SH.600519" not in candidates
    assert len(candidates) == 7


def test_archived_strategy_is_hidden_but_history_remains(services):
    database, repository, _service = services
    assert repository.archive_strategy("strategy-breakout")
    assert repository.get_strategy("strategy-breakout") is None
    assert database.one("SELECT * FROM strategy_versions WHERE strategy_id='strategy-breakout'") is not None


def test_right_side_strategy_executes_with_mixed_timeframes(services):
    _database, repository, service = services
    strategy = repository.create_strategy(
        "周线趋势回踩右侧交易",
        "5周均线、周线DIFF、RSI、回踩和止损",
        RIGHT_SIDE_DSL,
        RIGHT_SIDE_EXPLANATION,
    )
    repository.add_position(strategy["id"], "SH.600036", 100, 40)
    run = service.execute(strategy["id"], "right-side-test-slot")
    assert run["status"] == "success"
    assert run["securities_scanned"] == 8


def _replace_snapshot_bars(database, security_id: str, post_low_high: float) -> None:
    database.execute("DELETE FROM bars WHERE security_id=?", (security_id,))
    start = date(2026, 6, 1)
    rows = []
    for index in range(20):
        current = start + timedelta(days=index)
        if index < 10:
            low, high = 100.0, 110.0
        elif index == 10:
            low, high = 90.0, 91.0
        else:
            low, high = 92.0, post_low_high
        rows.append((security_id, current.isoformat(), 100.0, high, low, 100.0, 1000.0, 0))
    database.executemany(
        "INSERT INTO bars(security_id,trade_date,open,high,low,close,volume,is_provisional) VALUES(?,?,?,?,?,?,?,?)",
        rows,
    )


def test_first_position_add_freezes_temporally_ordered_rebound_snapshot(services):
    database, repository, _service = services
    strategy_id = "strategy-momentum"
    security_id = "HK.09988"
    _replace_snapshot_bars(database, security_id, 94.0)
    repository.add_position(strategy_id, security_id, 100, 150)

    strategy = repository.add_to_position(
        strategy_id, security_id, 50, 100, "2026-07-01T17:00:00+08:00"
    )
    position = next(item for item in strategy["positions"] if item["security_id"] == security_id)
    assert position["quantity"] == 150
    assert position["avg_cost"] == pytest.approx(133.333333)
    assert position["add_count"] == 1
    assert position["last_add_price"] == 100
    assert position["first_add_20d_low"] == 90
    assert position["first_add_20d_low_date"] == "2026-06-11"
    assert position["first_add_post_low_high"] == 94
    assert position["first_add_rebound_pct"] == pytest.approx(4.444444)
    # 110出现在最低点之前，不能用它误判为最低点后的5%反弹。
    assert position["first_add_rebound_confirmed"] is False

    strategy = repository.add_to_position(
        strategy_id, security_id, 50, 120, "2026-07-02T17:00:00+08:00"
    )
    position = next(item for item in strategy["positions"] if item["security_id"] == security_id)
    assert position["quantity"] == 200
    assert position["avg_cost"] == pytest.approx(130)
    assert position["add_count"] == 2
    assert position["last_add_price"] == 120
    assert position["first_add_20d_low"] == 90
    assert position["first_add_post_low_high"] == 94
    assert len(position["add_events"]) == 2


def test_first_position_add_confirms_rebound_after_low(services):
    database, repository, _service = services
    strategy_id = "strategy-momentum"
    security_id = "HK.09988"
    _replace_snapshot_bars(database, security_id, 95.0)
    repository.add_position(strategy_id, security_id, 100, 150)
    strategy = repository.add_to_position(
        strategy_id, security_id, 10, 95, "2026-07-01T17:00:00+08:00"
    )
    position = next(item for item in strategy["positions"] if item["security_id"] == security_id)
    assert position["first_add_rebound_pct"] == pytest.approx(5.555556)
    assert position["first_add_rebound_confirmed"] is True


def test_first_position_add_rejects_incomplete_real_history(services):
    database, repository, _service = services
    strategy_id = "strategy-momentum"
    security_id = "HK.09988"
    _replace_snapshot_bars(database, security_id, 95.0)
    database.execute("DELETE FROM bars WHERE security_id=? AND trade_date>'2026-06-10'", (security_id,))
    repository.add_position(strategy_id, security_id, 100, 150)
    with pytest.raises(ValueError, match="20个已完成交易日"):
        repository.add_to_position(strategy_id, security_id, 10, 95, "2026-07-01T17:00:00+08:00")
    position = database.one(
        "SELECT * FROM positions WHERE strategy_id=? AND security_id=?", (strategy_id, security_id)
    )
    assert position["quantity"] == 100
    assert position["add_count"] == 0


def test_same_security_position_add_state_is_isolated_between_strategies(services):
    database, repository, _service = services
    security_id = "HK.09988"
    _replace_snapshot_bars(database, security_id, 95.0)
    repository.add_position("strategy-momentum", security_id, 100, 150)
    repository.add_position("strategy-breakout", security_id, 200, 140)

    repository.add_to_position(
        "strategy-momentum", security_id, 10, 95, "2026-07-01T17:00:00+08:00"
    )
    repository.add_to_position(
        "strategy-breakout", security_id, 20, 105, "2026-07-01T17:00:00+08:00"
    )

    momentum = repository.get_strategy("strategy-momentum")
    breakout = repository.get_strategy("strategy-breakout")
    momentum_position = next(item for item in momentum["positions"] if item["security_id"] == security_id)
    breakout_position = next(item for item in breakout["positions"] if item["security_id"] == security_id)
    assert momentum_position["last_add_price"] == 95
    assert breakout_position["last_add_price"] == 105
    assert momentum_position["quantity"] == 110
    assert breakout_position["quantity"] == 220
    assert [event["price"] for event in momentum_position["add_events"]] == [95]
    assert [event["price"] for event in breakout_position["add_events"]] == [105]


def test_deleting_position_clears_add_state_before_reopening(services):
    database, repository, _service = services
    strategy_id = "strategy-momentum"
    security_id = "HK.09988"
    _replace_snapshot_bars(database, security_id, 95.0)
    repository.add_position(strategy_id, security_id, 100, 150)
    repository.add_to_position(
        strategy_id, security_id, 10, 95, "2026-07-01T17:00:00+08:00"
    )

    assert repository.delete_position(strategy_id, security_id)
    assert database.one(
        """SELECT 1 FROM position_events
           WHERE strategy_id=? AND security_id=? AND event_type='added'""",
        (strategy_id, security_id),
    ) is None

    reopened = repository.add_position(strategy_id, security_id, 50, 110)
    position = next(item for item in reopened["positions"] if item["security_id"] == security_id)
    assert position["add_count"] == 0
    assert position["last_add_price"] is None
    assert position["last_add_at"] is None
    assert position["first_add_20d_low"] is None
    assert position["first_add_rebound_confirmed"] is None
    assert position["add_events"] == []


def test_last_buy_interval_falls_back_to_open_time_until_first_add(services):
    database, repository, service = services
    strategy_id = "strategy-momentum"
    security_id = "HK.09988"
    _replace_snapshot_bars(database, security_id, 95.0)
    opened_at = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    repository.add_position(strategy_id, security_id, 100, 150, opened_at)

    before_add = next(item for item in service._position_items(strategy_id) if item["security_id"] == security_id)
    assert before_add["days_since_last_add"] is None
    assert 3.9 <= before_add["days_since_last_buy"] <= 4.1

    repository.add_to_position(strategy_id, security_id, 10, 95)
    after_add = next(item for item in service._position_items(strategy_id) if item["security_id"] == security_id)
    assert 0 <= after_add["days_since_last_add"] < 0.01
    assert 0 <= after_add["days_since_last_buy"] < 0.01
