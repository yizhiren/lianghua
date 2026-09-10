from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.database import Database
from backend.app.seed import seed_demo
from backend.app.trading import ACCOUNT_ID, TradingService


@pytest.fixture()
def trading(tmp_path: Path):
    database = Database(tmp_path / "trading.db")
    database.initialize()
    seed_demo(database, force=True)
    service = TradingService(database)
    service.initialize()
    return database, service


def rule(action: str, target: float | None, rule_id: str = "trade-rule"):
    return SimpleNamespace(id=rule_id, action=action, target_position_pct=target)


def align_quote_to_completed_bar(database: Database, security_id: str) -> float:
    bar = database.one(
        "SELECT close FROM bars WHERE security_id=? AND is_provisional=0 ORDER BY trade_date DESC LIMIT 1",
        (security_id,),
    )
    assert bar
    database.execute("UPDATE securities SET latest_price=? WHERE id=?", (bar["close"], security_id))
    return float(bar["close"])


def eligible_time(intent: dict) -> datetime:
    return datetime.fromisoformat(intent["eligible_at"]) + timedelta(minutes=1)


def test_buy_intent_is_idempotent_requires_approval_and_fills(trading):
    database, service = trading
    service.configure_allocation("strategy-momentum", "HKD", 200_000, True)
    align_quote_to_completed_bar(database, "HK.09988")

    first = service.create_intent(
        "strategy-momentum", rule("signal_buy", 10), "HK.09988", "底背离确认"
    )
    second = service.create_intent(
        "strategy-momentum", rule("signal_buy", 10), "HK.09988", "底背离确认"
    )

    assert first and second and first["id"] == second["id"]
    assert first["status"] == "pending_approval"
    assert database.one("SELECT COUNT(*) AS count FROM order_intents")["count"] == 1

    approved = service.approve(first["id"])
    assert approved["status"] == "approved"
    result = service.process(eligible_time(approved))
    assert result["processed"] == 1
    assert service.intent(first["id"])["status"] == "filled"
    assert database.one(
        "SELECT quantity FROM positions WHERE strategy_id=? AND security_id=?",
        ("strategy-momentum", "HK.09988"),
    )["quantity"] > 0
    assert database.one(
        "SELECT 1 FROM candidates WHERE strategy_id=? AND security_id=?",
        ("strategy-momentum", "HK.09988"),
    ) is None


def test_completed_bar_can_create_intent_when_weekend_quote_is_missing(trading):
    database, service = trading
    service.configure_allocation("strategy-momentum", "HKD", 200_000, True)
    database.execute(
        "UPDATE securities SET latest_price=NULL,quote_time=NULL WHERE id=?",
        ("HK.09988",),
    )
    latest_bar = database.one(
        """SELECT trade_date,close FROM bars WHERE security_id=? AND is_provisional=0
           ORDER BY trade_date DESC LIMIT 1""",
        ("HK.09988",),
    )
    assert latest_bar

    intent = service.create_intent(
        "strategy-momentum", rule("signal_buy", 10), "HK.09988", "收盘后选股"
    )

    assert intent and intent["status"] == "pending_approval"
    assert intent["signal_date"] == latest_bar["trade_date"]
    assert intent["signal_price"] == pytest.approx(latest_bar["close"])


def test_strategy_and_single_security_caps_are_enforced(trading):
    database, service = trading
    align_quote_to_completed_bar(database, "HK.09988")
    service.configure_allocation("strategy-momentum", "HKD", 400_000, True)
    intent = service.create_intent(
        "strategy-momentum", rule("signal_buy", 100, "risk-rule"), "HK.09988", "风险测试"
    )
    assert intent
    service.approve(intent["id"])
    service.process(eligible_time(intent))
    blocked = service.intent(intent["id"])
    assert blocked["status"] == "blocked"
    assert "30" in blocked["blocked_reason"]

    database.execute(
        "UPDATE strategy_allocations SET capital_limit=300000 WHERE strategy_id=? AND currency='HKD'",
        ("strategy-momentum",),
    )
    retry = service.create_intent(
        "strategy-momentum", rule("signal_buy", 100, "security-cap-rule"), "HK.09988", "单股上限测试"
    )
    assert retry
    service.approve(retry["id"])
    service.process(eligible_time(retry))
    filled = service.intent(retry["id"])
    equity = service._currency_equity("HKD")
    position = database.one(
        "SELECT quantity FROM broker_positions WHERE account_id=? AND security_id=?", (ACCOUNT_ID, "HK.09988")
    )
    quote = database.one("SELECT latest_price FROM securities WHERE id='HK.09988'")["latest_price"]
    assert filled["status"] == "filled"
    assert position["quantity"] * quote <= equity * 0.10 + quote * 100


def test_a_share_stop_retries_through_t_plus_one_then_fills(trading):
    database, service = trading
    service.configure_allocation("strategy-momentum", "CNY", 200_000, True)
    align_quote_to_completed_bar(database, "SH.600519")
    intent = service.create_intent(
        "strategy-momentum", rule("signal_stop", 0, "hard-stop"), "SH.600519", "收盘跌破硬止损"
    )
    assert intent and intent["status"] == "approved"
    database.execute(
        "UPDATE broker_positions SET acquired_date=?,available_quantity=0 WHERE account_id=? AND security_id=?",
        (date.today().isoformat(), ACCOUNT_ID, "SH.600519"),
    )

    service.process(eligible_time(intent))
    waiting = service.intent(intent["id"])
    assert waiting["status"] == "approved"
    assert "T+1" in waiting["blocked_reason"]

    database.execute(
        "UPDATE broker_positions SET acquired_date=? WHERE account_id=? AND security_id=?",
        ((date.today() - timedelta(days=1)).isoformat(), ACCOUNT_ID, "SH.600519"),
    )
    service.process(eligible_time(intent) + timedelta(seconds=15))
    assert service.intent(intent["id"])["status"] == "filled"
    assert database.one(
        "SELECT 1 FROM positions WHERE strategy_id=? AND security_id=?",
        ("strategy-momentum", "SH.600519"),
    ) is None


def test_emergency_stop_cancels_orders_and_reconciliation_pauses_entries(trading):
    database, service = trading
    service.configure_allocation("strategy-momentum", "HKD", 200_000, True)
    align_quote_to_completed_bar(database, "HK.09988")
    intent = service.create_intent(
        "strategy-momentum", rule("signal_buy", 10, "emergency-rule"), "HK.09988", "紧急停止测试"
    )
    assert intent
    service.approve(intent["id"])
    summary = service.set_controls(emergency_stop=True)
    assert summary["emergency_stop"] == 1
    assert service.intent(intent["id"])["status"] == "canceled"

    service.set_controls(emergency_stop=False, entry_paused=False)
    database.execute(
        "UPDATE broker_positions SET quantity=quantity+1 WHERE account_id=? AND security_id=?",
        (ACCOUNT_ID, "HK.00700"),
    )
    reconciliation = service.reconcile()
    assert reconciliation["ok"] is False
    assert service.account_summary()["entry_paused"] == 1
