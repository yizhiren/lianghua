from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest

import backend.app.training as training_module
from backend.app.database import Database
from backend.app.dsl import validate_dsl
from backend.app.seed import seed_demo
from backend.app.training import TrainingService


@pytest.fixture()
def training(tmp_path: Path, monkeypatch):
    database = Database(tmp_path / "training.db")
    database.initialize()
    seed_demo(database, force=True)
    service = TrainingService(database)
    monkeypatch.setattr(
        training_module,
        "settings",
        replace(training_module.settings, training_snapshot_dir=tmp_path / "snapshots"),
    )
    return database, service


def test_data_quality_blocks_promotion_when_history_is_short(training):
    _database, service = training
    quality = service.data_quality()
    assert set(quality["markets"]) == {"A", "HK"}
    assert quality["ready_for_promotion"] is False
    assert quality["markets"]["A"]["history_gate"] is False
    assert quality["markets"]["A"]["ready_for_training"] is False
    assert "data_history" in quality["markets"]["A"]["blocking_reasons"]
    assert quality["markets"]["A"]["warnings"]


def test_formal_campaign_waits_for_data_but_smoke_remains_available(training):
    database, service = training
    with pytest.raises(RuntimeError, match="正式训练暂缓"):
        service.start_campaigns(["left-A"], 4, "manual")
    assert database.one("SELECT COUNT(*) AS count FROM training_campaigns")["count"] == 0


def test_ready_tracks_and_dashboard_expose_data_building_phase(training, monkeypatch):
    _database, service = training
    original = service.data_quality()
    original["markets"]["A"].update({
        "ready_for_training": True,
        "blocking_reasons": [],
        "history_gate": True,
        "coverage_gate": True,
        "point_in_time_status": True,
    })
    monkeypatch.setattr(service, "data_quality", lambda: original)
    assert service.ready_tracks() == ["left-A", "right-A"]
    assert service.formal_training_due_tracks() == ["left-A", "right-A"]
    dashboard = service.dashboard()
    assert dashboard["champions"][0]["data_ready"] is True
    assert dashboard["champions"][2]["evaluation_status"] == "pending_data"


def test_smoke_campaign_persists_snapshot_trials_events_and_research_champion(training):
    database, service = training
    campaigns = service.start_campaigns(["left-A"], 4, "smoke")
    campaign_id = campaigns[0]["id"]
    thread = service._threads[campaign_id]
    thread.join(timeout=30)
    assert not thread.is_alive()
    campaign = service.campaign(campaign_id)
    assert campaign["status"] == "research_only"
    assert campaign["completed_trials"] == 4
    assert len(campaign["trials"]) == 4
    assert campaign["events"]
    champion = service.champion("left-A")
    assert champion["status"] == "research_only"
    assert champion["gates"]["passed"] is False
    snapshot = database.one("SELECT * FROM training_data_snapshots WHERE id=?", (campaign["snapshot_id"],))
    assert snapshot and Path(snapshot["storage_path"]).exists()


def test_same_market_tracks_in_one_batch_reuse_snapshot(training):
    _database, service = training
    campaigns = service.start_campaigns(["left-A", "right-A"], 4, "smoke")
    service._threads[campaigns[0]["id"]].join(timeout=30)
    left = service.campaign(campaigns[0]["id"])
    right = service.campaign(campaigns[1]["id"])
    assert left["snapshot_id"] == right["snapshot_id"]
    assert any(event["event_type"] == "snapshot_reused" for event in right["events"])
    assert any(event["event_type"] == "feature_cache_reused" for event in right["events"])


def test_research_champion_cannot_be_approved(training):
    _database, service = training
    campaigns = service.start_campaigns(["left-A"], 4, "smoke")
    service._threads[campaigns[0]["id"]].join(timeout=30)
    with pytest.raises(ValueError, match="未通过"):
        service.approve("left-A")


def test_validation_tiers_keep_strict_gate_separate_from_observation():
    assert TrainingService._validation_tier({"passed": True, "reasons": []}) == "strict_champion"
    assert TrainingService._validation_tier({
        "passed": False, "reasons": ["deflated_sharpe"],
    }) == "paper_observation"
    assert TrainingService._validation_tier({
        "passed": False, "reasons": ["sharpe", "deflated_sharpe"],
    }) == "research_only"


def test_research_result_can_be_published_as_non_trading_scanner(training):
    database, service = training
    campaigns = service.start_campaigns(["left-A"], 4, "smoke")
    service._threads[campaigns[0]["id"]].join(timeout=30)
    champion = service.champion("left-A")
    gates = dict(champion["gates"])
    for key in training_module.DATA_GATE_NAMES:
        gates[key] = True
    service._refresh_gate_outcome(gates)
    database.execute(
        "UPDATE strategy_champions SET gates_json=? WHERE track='left-A'",
        (database.dump(gates),),
    )

    published = service.publish_research("left-A")
    assert published["status"] == "research_only"
    assert published["gates"]["passed"] is False
    strategy = service.repo.get_strategy(published["strategy_id"])
    assert strategy and strategy["name"] == "训练研究·左侧·A股"
    assert "不生成交易订单" in strategy["description"]
    assert database.one(
        "SELECT 1 FROM strategy_allocations WHERE strategy_id=?",
        (strategy["id"],),
    ) is None

    republished = service.publish_research("left-A")
    assert republished["strategy_id"] == strategy["id"]
    assert service.repo.get_strategy(strategy["id"])["current_version"] == 2


def test_research_strategy_is_renamed_when_it_later_passes_and_is_approved(training):
    database, service = training
    campaigns = service.start_campaigns(["left-A"], 4, "smoke")
    service._threads[campaigns[0]["id"]].join(timeout=30)
    champion = service.champion("left-A")
    gates = dict(champion["gates"])
    for key, value in list(gates.items()):
        if isinstance(value, bool) and key != "passed":
            gates[key] = True
    gates.update({"passed": True, "reasons": []})
    database.execute(
        "UPDATE strategy_champions SET gates_json=? WHERE track='left-A'",
        (database.dump({**gates, "positive_return": False, "passed": False, "reasons": ["positive_return"]}),),
    )
    research = service.publish_research("left-A")
    database.execute(
        "UPDATE strategy_champions SET gates_json=? WHERE track='left-A'",
        (database.dump(gates),),
    )

    approved = service.approve("left-A")
    assert approved["strategy_id"] == research["strategy_id"]
    assert approved["status"] == "paper"
    assert service.repo.get_strategy(approved["strategy_id"])["name"] == "训练冠军·左侧·A股"


def test_trained_left_and_right_templates_compile_to_deployable_dsl():
    left = {
        "rsi_max": 38.0, "low_proximity_pct": 4.0,
        "boll_b_max": 0.1, "divergence_window": 40,
        "volume_ratio_max": 1.8,
        "confirm_return_max": 4.0, "trend_floor_ratio": 0.88,
        "trend_slope_floor": 0.95, "adx_max": 35.0, "stop_atr": 2.3,
        "exit_rsi": 58.0, "max_hold": 20, "low_window": 60,
    }
    right = {
        "rsi_min": 48.0, "rsi_max": 70.0,
        "adx_min": 22.0, "volume_ratio_min": 1.2, "stop_atr": 2.3,
        "signal_return_max": 5.0, "trail_atr": 3.0, "max_hold": 90, "trend_slope_days": 40,
        "atr_pct_max": 7.0, "breakout_window": 120, "channel_exit_window": 20,
        "require_weekly": True,
    }
    left_dsl = TrainingService._strategy_dsl("left", "A", left)
    assert validate_dsl(left_dsl).rules
    serialized_left = str(left_dsl)
    assert "histogram" in serialized_left
    assert "'field': 'dif'" not in serialized_left
    right_hk_dsl = TrainingService._strategy_dsl("right", "HK", right)
    right_a_dsl = TrainingService._strategy_dsl("right", "A", right)
    assert validate_dsl(right_hk_dsl).rules
    assert validate_dsl(right_a_dsl).rules
    # 自动待定池只包含完整入场条件命中的股票，不能先把整个市场池加入待定。
    for dsl in (left_dsl, right_hk_dsl, right_a_dsl):
        entry_condition = dsl["rules"][1]["condition"]
        assert entry_condition in dsl["rules"][0]["condition"]["conditions"]
        assert dsl["rules"][1]["target_position_pct"] == 33.33
    assert left_dsl["rules"][1]["condition"]["conditions"][0]["right"]["params"]["window"] == 60
    left_conditions = left_dsl["rules"][1]["condition"]["conditions"]
    assert left_conditions[0]["upper"] == 4.0
    assert left_conditions[1]["left"]["params"]["period"] == 14
    left_volume = next(condition for condition in left_conditions if condition["op"] == "ratio_pct")
    assert left_volume["left"]["params"] == {}
    quality_filter = next(condition for condition in left_conditions if condition["op"] == "any")
    assert {item["left"]["name"] for item in quality_filter["conditions"]} == {"boll", "macd"}
    histogram_turn = next(
        condition for condition in left_conditions
        if condition.get("left", {}).get("field") == "histogram"
        and isinstance(condition.get("right"), dict)
    )
    assert histogram_turn["right"]["params"] == {"lag": 1}
    left_target = next(rule for rule in left_dsl["rules"] if rule["id"] == "trained-left-target")
    assert left_target["condition"]["conditions"][0]["left"]["params"]["period"] == 6
    hk_volume = right_hk_dsl["rules"][1]["condition"]["conditions"][0]["conditions"][1]["left"]
    a_volume = right_a_dsl["rules"][1]["condition"]["conditions"][0]["conditions"][1]["left"]
    assert hk_volume["params"] == {"window_op": "mean", "window": 5}
    assert a_volume["params"] == {}
    assert right_a_dsl["rules"][1]["condition"]["conditions"][0]["conditions"][0]["periods"] == 120
    assert any(rule["id"] == "trained-right-channel-exit" for rule in right_a_dsl["rules"])
    assert not any(rule["id"] == "trained-right-channel-exit" for rule in right_hk_dsl["rules"])
    assert any(
        condition.get("left", {}).get("params") == {"period": 200}
        and condition.get("right", {}).get("params") == {"period": 200, "lag": 60}
        for condition in right_a_dsl["rules"][1]["condition"]["conditions"]
    )


def test_market_specific_search_ranges_use_daily_volume_for_a_shares():
    left_a = TrainingService._distributions("left", "A")
    left_hk = TrainingService._distributions("left", "HK")
    right_a = TrainingService._distributions("right", "A")
    right_hk = TrainingService._distributions("right", "HK")
    assert left_a["volume_ratio_max"].low == 1.2
    assert left_hk["volume_ratio_max"].low == 0.6
    assert left_a["boll_b_max"].low == 0.05
    assert "boll_b_max" not in left_hk
    assert left_a["divergence_window"].choices == (20, 40)
    assert "divergence_window" not in left_hk
    assert left_a["low_window"].choices == (20, 60)
    assert left_a["low_proximity_pct"].low == 3
    assert left_hk["max_hold"].choices[0] == 5
    assert left_a["rsi_max"].low == 30
    assert left_hk["rsi_max"].low == 32
    assert left_a["trend_floor_ratio"].high == 0.98
    assert left_hk["trend_slope_floor"].high == 1.02
    assert right_a["volume_ratio_min"].high == 2.5
    assert right_a["atr_pct_max"].low == 2.5
    assert right_a["max_hold"].choices == (30, 60, 90, 120, 150, 180)
    assert right_a["breakout_window"].choices == (20, 55, 120, 250)
    assert right_a["channel_exit_window"].choices == (20, 40, 60)
    assert right_hk["volume_ratio_min"].high == 1.6
    assert right_hk["atr_pct_max"].low == 5
    assert right_hk["breakout_window"].choices == (20, 55, 120)


def test_legacy_params_outside_tightened_range_cannot_reenter_new_generation():
    distributions = TrainingService._distributions("right", "A")
    valid = {
        "rsi_min": 45.0, "rsi_max": 75.0, "adx_min": 20.0,
        "volume_ratio_min": 1.2, "signal_return_max": 5.0,
        "stop_atr": 2.0, "trail_atr": 3.0, "max_hold": 60,
        "breakout_window": 55, "channel_exit_window": 20,
        "trend_slope_days": 40, "atr_pct_max": 4.0,
        "require_weekly": False,
    }
    assert TrainingService._params_match_distributions(valid, distributions)
    assert not TrainingService._params_match_distributions(
        {**valid, "atr_pct_max": 1.9}, distributions,
    )


@pytest.mark.parametrize("exchange", ["SH", "SZ"])
def test_a_share_replay_normalizes_exchange_to_logical_market(exchange):
    dates = pd.bdate_range("2025-01-01", periods=223)
    frame = pd.DataFrame({
        "security_id": [f"{exchange}.600000"] * len(dates),
        "market": [exchange] * len(dates),
        "trade_date": dates,
        "open": np.full(len(dates), 100.0),
        "high": np.full(len(dates), 102.0),
        "low": np.full(len(dates), 99.0),
        "close": np.full(len(dates), 100.0),
        "trade_status": np.ones(len(dates)),
        "is_st": np.zeros(len(dates)),
        "rsi": np.full(len(dates), 55.0),
        "atr": np.full(len(dates), 1.0),
        "dif": np.full(len(dates), 1.0),
        "dea": np.zeros(len(dates)),
        "hist": np.full(len(dates), 2.0),
        "adx": np.full(len(dates), 25.0),
        "atr_pct": np.full(len(dates), 1.0),
        "recent_abs_return_max": np.ones(len(dates)),
        "recent_calendar_gap_max": np.full(len(dates), 3.0),
        "vol_ratio": np.full(len(dates), 0.5),
        "day_vol_ratio": np.full(len(dates), 2.0),
        "return1": np.ones(len(dates)),
        "return20": np.full(len(dates), 5.0),
        "return60": np.full(len(dates), 8.0),
        "ma60": np.full(len(dates), 95.0),
        "ma120": np.full(len(dates), 90.0),
        "ma200": np.full(len(dates), 80.0),
        "low20_prev": np.full(len(dates), 90.0),
        "high55": np.full(len(dates), 100.0),
        "week_trend": np.ones(len(dates), dtype=bool),
    })
    frame.loc[160, "ma200"] = 79.0
    frame.loc[180, "ma120"] = 89.0
    frame.loc[220, "close"] = 101.0
    frame.loc[221:, "open"] = 101.0
    params = {
        "rsi_min": 45.0, "rsi_max": 70.0, "adx_min": 20.0,
        "volume_ratio_min": 1.5, "signal_return_max": 5.0,
        "stop_atr": 2.0, "trail_atr": 3.0, "max_hold": 30,
        "trend_slope_days": 40, "atr_pct_max": 5.0,
        "require_weekly": False,
    }
    assert len(TrainingService._trades(frame, "right", params)) == 1


def test_new_generation_retests_incumbent_and_records_champion_challenge(training):
    _database, service = training
    first = service.start_campaigns(["left-A"], 4, "smoke")[0]
    service._threads[first["id"]].join(timeout=30)
    first_result = service.campaign(first["id"])
    second = service.start_campaigns(["left-A"], 4, "smoke")[0]
    service._threads[second["id"]].join(timeout=30)
    campaign = service.campaign(second["id"])
    assert campaign["config"]["generation"] == 2
    assert campaign["snapshot_id"] == first_result["snapshot_id"]
    assert any(
        event["event_type"] == "snapshot_reused" and "数据未变化" in event["message"]
        for event in campaign["events"]
    )
    assert campaign["summary"]["champion_challenge"]["incumbent_trial_id"]
    assert any(event["event_type"] in {"champion_replaced", "champion_revalidated", "challenger_rejected"} for event in campaign["events"])


def test_fold_ranges_are_exactly_three_contiguous_development_periods():
    folds, holdout = TrainingService._fold_ranges("2011-07-21", "2026-08-21")
    assert len(folds) == 3
    assert folds[-1][1] == "2023-08-14"
    assert holdout == ("2023-08-15", "2026-08-21")
    for previous, current in zip(folds, folds[1:]):
        assert pd.Timestamp(previous[1]) + pd.Timedelta(days=1) == pd.Timestamp(current[0])


def test_snapshot_filter_removes_hk_temporary_products_and_post_delisting_rows():
    frame = pd.DataFrame([
        {"security_id": "HK.02900", "market": "HK", "code": "02900", "trade_date": "2026-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "listing_date": None, "delisting_date": None},
        {"security_id": "HK.04335", "market": "HK", "code": "04335", "trade_date": "2026-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "listing_date": None, "delisting_date": None},
        {"security_id": "HK.02866", "market": "HK", "code": "02866", "trade_date": "2026-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "listing_date": None, "delisting_date": None},
        {"security_id": "HK.00209", "market": "HK", "code": "00209", "name": "万维智能科技－新", "trade_date": "2026-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "listing_date": None, "delisting_date": None},
        {"security_id": "HK.03308", "market": "HK", "code": "03308", "trade_date": "2023-09-18", "open": 1, "high": 1, "low": 1, "close": 1, "listing_date": "2011-01-01", "delisting_date": "2023-09-18"},
        {"security_id": "HK.03308", "market": "HK", "code": "03308", "trade_date": "2026-01-02", "open": 100, "high": 100, "low": 100, "close": 100, "listing_date": "2011-01-01", "delisting_date": "2023-09-18"},
    ])
    prepared, stats = TrainingService._prepare_snapshot_frame(frame)
    assert prepared[["security_id", "trade_date"]].astype(str).values.tolist() == [
        ["HK.02866", "2026-01-02"],
        ["HK.03308", "2023-09-18"],
    ]
    assert stats == {"excluded_rows": 4, "excluded_securities": 3}


def test_portfolio_metrics_mark_interim_drawdown_instead_of_only_exit_pnl():
    dates = np.array(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
    trade = {
        "security_id": "HK.00001", "entry_date": "2026-01-05", "exit_date": "2026-01-07",
        "return": 0.0, "duration": 2, "mfe": 0.0, "mae": -0.5, "reason": "max_hold",
        "_dates": dates, "_close": np.array([100.0, 50.0, 100.0]),
        "_entry_index": 0, "_exit_index": 2, "_entry_price": 100.0,
    }
    metrics = TrainingService._portfolio_metrics([trade], "2026-01-05", "2026-01-09")
    assert metrics["total_return_pct"] == pytest.approx(0)
    assert metrics["max_drawdown_pct"] == pytest.approx(-5)


def test_portfolio_and_benchmark_share_thirty_percent_strategy_exposure():
    trades = [
        {
            "security_id": f"HK.{index:05d}", "entry_date": "2026-01-05",
            "exit_date": "2026-01-06", "return": 0.10, "duration": 1,
            "mfe": 0.10, "mae": 0.0, "reason": "target",
        }
        for index in range(4)
    ]
    metrics = TrainingService._portfolio_metrics(trades, "2026-01-05", "2026-01-06")
    assert metrics["trades"] == 3
    assert metrics["total_return_pct"] == pytest.approx(3)
    assert metrics["strategy_exposure_cap_pct"] == pytest.approx(30)

    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
        "return1": [0.0, 10.0],
    })
    benchmark = TrainingService._benchmark_metrics([frame], "2026-01-05", "2026-01-06")
    assert benchmark["total_return_pct"] == pytest.approx(3)
    assert benchmark["exposure_pct"] == pytest.approx(30)
    matched = TrainingService._benchmark_at_exposure(benchmark, 10)
    assert matched["total_return_pct"] == pytest.approx(1)
    assert matched["exposure_pct"] == pytest.approx(10)


def test_portfolio_uses_signal_strength_instead_of_ticker_order_when_capacity_is_full():
    trades = [
        {
            "security_id": security_id, "entry_date": "2026-01-05",
            "exit_date": "2026-01-06", "return": trade_return,
            "signal_score": score, "duration": 1, "mfe": max(0, trade_return),
            "mae": min(0, trade_return), "reason": "target",
        }
        for security_id, trade_return, score in [
            ("HK.00001", -0.10, 1),
            ("HK.00002", 0.10, 2),
            ("HK.00003", 0.20, 3),
            ("HK.99999", 0.50, 100),
        ]
    ]
    metrics = TrainingService._portfolio_metrics(trades, "2026-01-05", "2026-01-06")
    assert metrics["trades"] == 3
    assert metrics["total_return_pct"] == pytest.approx(8)


def test_development_score_penalizes_one_bad_fold_instead_of_hiding_it_with_median():
    def fold(annual, sharpe, drawdown):
        return {
            "trades": 50, "annual_return_pct": annual, "sharpe": sharpe,
            "sortino": sharpe * 1.2, "calmar": annual / abs(drawdown),
            "max_drawdown_pct": drawdown, "turnover": 4,
        }

    consistent = [fold(8, 0.8, -8), fold(9, 0.9, -9), fold(7, 0.7, -10)]
    unstable = [fold(30, 2.0, -10), fold(35, 2.2, -12), fold(-20, -1.0, -35)]
    assert TrainingService._development_score(consistent, {}) > TrainingService._development_score(unstable, {})


def test_development_score_penalizes_candidates_below_formal_trade_gate():
    def fold(trades):
        return {
            "trades": trades, "annual_return_pct": 8, "sharpe": 0.8,
            "sortino": 1.0, "calmar": 1.0, "max_drawdown_pct": -8,
            "turnover": 4,
        }

    adequately_sampled = [fold(60), fold(60), fold(60)]
    sparse = [fold(40), fold(40), fold(40)]
    assert (
        TrainingService._development_score(adequately_sampled, {})
        > TrainingService._development_score(sparse, {}) + 10
    )


def test_evaluate_counts_zero_trade_fold_as_failed_regime(training, monkeypatch):
    _database, service = training

    def metrics(trades, total_return):
        return {
            "trades": trades, "total_return_pct": total_return,
            "annual_return_pct": total_return, "max_drawdown_pct": -5,
            "sharpe": 1 if total_return > 0 else 0, "sortino": 1,
            "calmar": 1, "turnover": 2, "observations": 500,
            "return_skewness": 0, "return_kurtosis": 3,
        }

    results = iter([
        metrics(30, 5), metrics(0, 0), metrics(30, 5),
        metrics(60, 5), metrics(30, 5), metrics(30, 4),
    ])
    monkeypatch.setattr(service, "_trades", lambda *_args: [{"entry_date": "2025-01-01"}])
    monkeypatch.setattr(service, "_portfolio_metrics", lambda *_args, **_kwargs: next(results))
    _score, summary, _folds, gates = service._evaluate(
        [pd.DataFrame()], "left", {}, "2011-01-01", "2026-01-01",
        {
            "history_gate": True, "coverage_gate": True,
            "freshness_gate": True, "point_in_time_status": True,
        },
        {"annual_return_pct": 0, "total_return_pct": 0},
    )
    assert summary["profitable_fold_ratio"] == pytest.approx(2 / 3)
    assert gates["profitable_folds"] is False
    assert gates["trade_count"] is False


def test_deflated_sharpe_probability_is_monotonic_and_bounded():
    base = {"observations": 756, "return_skewness": 0.0, "return_kurtosis": 3.0}
    tried = np.linspace(-0.2, 0.8, 200)
    low = TrainingService._deflated_sharpe_probability({**base, "sharpe": 0.5}, tried, 50)
    high = TrainingService._deflated_sharpe_probability({**base, "sharpe": 2.0}, tried, 50)
    assert 0 <= low < high <= 1


def test_deflated_sharpe_uses_empirical_trial_variance():
    metrics = {
        "observations": 1500, "return_skewness": 0.0,
        "return_kurtosis": 3.0, "sharpe": 1.2,
    }
    tight = np.linspace(0.1, 0.3, 100)
    wide = np.linspace(-1.0, 1.0, 100)
    tight_probability = TrainingService._deflated_sharpe_probability(metrics, tight, 50)
    wide_probability = TrainingService._deflated_sharpe_probability(metrics, wide, 50)
    assert tight_probability > wide_probability


def test_effective_trial_count_accounts_for_correlated_return_paths():
    base = np.linspace(-0.02, 0.03, 24)
    correlated = [base * scale for scale in (0.8, 0.9, 1.0, 1.1, 1.2)]
    effective, average_correlation = TrainingService._effective_trial_count(correlated)
    assert average_correlation == pytest.approx(0.99)
    assert effective == 2


def test_stability_report_moves_one_parameter_at_a_time(training, monkeypatch):
    _database, service = training
    base = {"first": 1.0, "second": 2.0, "count": 10, "flag": True}
    distributions = {
        "first": training_module.optuna.distributions.FloatDistribution(0, 2),
        "second": training_module.optuna.distributions.FloatDistribution(0, 4),
        "count": training_module.optuna.distributions.IntDistribution(5, 15, step=5),
        "flag": training_module.optuna.distributions.CategoricalDistribution([True, False]),
    }
    seen = []

    def fake_evaluate(_frames, _style, params, *_args, **_kwargs):
        seen.append(params)
        return 10.0, {}, [], {}

    monkeypatch.setattr(service, "_evaluate", fake_evaluate)
    report = service._stability_report(
        [], "left", base, "2020-01-01", "2026-01-01", {}, 10.0,
        distributions=distributions,
    )
    assert report["passed"] is True
    assert report["method"] == "one_parameter_at_a_time_5pct_range"
    assert len(seen) == 7
    assert all(sum(value != base[key] for key, value in candidate.items()) <= 1 for candidate in seen)


def test_initialize_requeues_interrupted_campaign(training, monkeypatch):
    database, service = training
    now = datetime.now(timezone.utc).isoformat()
    campaign_id = "interrupted-campaign"
    database.execute(
        """INSERT INTO training_campaigns(id,track,market,style,status,trigger_type,budget,created_at)
           VALUES(?,?,?,?,?,'manual',4,?)""",
        (campaign_id, "left-A", "A", "left", "running", now),
    )
    database.execute(
        """INSERT INTO training_trials(id,campaign_id,trial_number,status,params_json,created_at)
           VALUES('interrupted-trial',?,0,'running','{}',?)""",
        (campaign_id, now),
    )
    called = threading.Event()
    monkeypatch.setattr(service, "_run_queue", lambda campaign_ids: called.set())
    service.initialize()
    assert called.wait(2)
    assert database.one("SELECT status FROM training_campaigns WHERE id=?", (campaign_id,))["status"] == "queued"
    assert database.one("SELECT status FROM training_trials WHERE id='interrupted-trial'")["status"] == "interrupted"


def test_interrupted_campaign_resumes_snapshot_and_remaining_budget(training):
    database, service = training
    campaign = service.start_campaigns(["left-A"], 4, "smoke")[0]
    service._threads[campaign["id"]].join(timeout=30)
    first_trial = database.one(
        "SELECT params_json FROM training_trials WHERE campaign_id=? ORDER BY trial_number LIMIT 1",
        (campaign["id"],),
    )
    database.execute(
        """UPDATE training_campaigns
           SET status='running',budget=6,completed_trials=4,progress=66,error=NULL,finished_at=NULL
           WHERE id=?""",
        (campaign["id"],),
    )
    database.execute(
        """INSERT INTO training_trials(id,campaign_id,trial_number,status,params_json,created_at)
           VALUES('half-finished-trial',?,4,'running',?,?)""",
        (campaign["id"], first_trial["params_json"], datetime.now(timezone.utc).isoformat()),
    )
    resumed = TrainingService(database)
    resumed.initialize()
    resumed._threads[campaign["id"]].join(timeout=30)
    result = resumed.campaign(campaign["id"])
    assert result["status"] == "research_only"
    assert result["completed_trials"] == 6
    assert len([trial for trial in result["trials"] if trial["status"] == "success"]) == 6
    assert any(event["event_type"] == "resuming_after_restart" for event in result["events"])
