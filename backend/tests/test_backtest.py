from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.backtest import BacktestService
from backend.app.database import Database
from backend.app.repository import Repository
from backend.app.schemas import BacktestRequest
from backend.app.seed import seed_demo


@pytest.fixture()
def backtesting(tmp_path: Path):
    database = Database(tmp_path / "backtest.db")
    database.initialize()
    seed_demo(database, force=True)
    repository = Repository(database)
    strategy = repository.create_strategy(
        "回测测试策略",
        "全市场加入待定，次日买入，持仓次日退出",
        {
            "schema_version": 1,
            "rules": [
                {
                    "id": "universe",
                    "scope": "universe",
                    "timeframe": "day",
                    "daily_bar_mode": "completed",
                    "condition": {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": 0, "comparator": ">"},
                    "action": "add_candidate",
                    "label": "有效价格",
                },
                {
                    "id": "buy",
                    "scope": "candidate",
                    "timeframe": "day",
                    "daily_bar_mode": "completed",
                    "condition": {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": 0, "comparator": ">"},
                    "action": "signal_buy",
                    "target_position_pct": 10,
                    "label": "收盘生成买入信号",
                },
                {
                    "id": "exit",
                    "scope": "position",
                    "timeframe": "day",
                    "daily_bar_mode": "completed",
                    "condition": {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": 0, "comparator": ">"},
                    "action": "signal_exit",
                    "target_position_pct": 0,
                    "label": "收盘生成退出信号",
                },
            ],
        },
        [],
    )
    return database, BacktestService(database), strategy


def request_for(strategy_id: str, **overrides):
    values = {
        "strategy_id": strategy_id,
        "start_date": (date.today() - timedelta(days=90)).isoformat(),
        "end_date": date.today().isoformat(),
        "initial_cny": 1_000_000,
        "initial_hkd": 1_000_000,
        "max_holding_days": None,
        "stop_loss_pct": None,
        "force_close": True,
    }
    values.update(overrides)
    return BacktestRequest(**values)


def test_backtest_replays_at_close_and_fills_on_later_bar(backtesting):
    database, service, strategy = backtesting
    result = service.run(request_for(strategy["id"]))

    assert result["status"] == "success"
    assert result["summary"]["signals"] > 0
    assert result["summary"]["trades"] > 0
    assert result["summary"]["trading_days"] > 20
    assert result["equity_curve"]
    assert database.one("SELECT 1 FROM backtest_runs WHERE id=?", (result["id"],))
    for trade in result["trades"]:
        if trade["action"] != "signal_exit" or trade["reason"] != "回测期末平仓":
            assert trade["trade_date"] > trade["signal_date"]


def test_backtest_applies_costs_and_persists_signal_horizons(backtesting):
    _database, service, strategy = backtesting
    result = service.run(request_for(strategy["id"], commission_pct=0.5, min_commission=10, slippage_pct=0.2))

    assert result["summary"]["total_fees"] > 0
    assert set(result["summary"]["signal_analysis"]) == {"5", "10", "20", "60"}
    assert result["summary"]["survivorship_warning"]
    listed = service.list_runs()
    assert listed[0]["id"] == result["id"]


def test_backtest_rejects_invalid_range(backtesting):
    _database, service, strategy = backtesting
    with pytest.raises(ValueError, match="结束日期"):
        service.run(request_for(strategy["id"], start_date=date.today().isoformat(), end_date=date.today().isoformat()))
