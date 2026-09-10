from __future__ import annotations

import math
import statistics
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from .database import Database, db, utcnow
from .dsl import evaluate_rule, validate_dsl
from .schemas import BacktestRequest


ACTION_PRIORITY = {
    "signal_stop": 100,
    "signal_exit": 90,
    "signal_reduce": 80,
    "signal_add": 70,
    "signal_buy": 60,
}


@dataclass
class SimPosition:
    quantity: float
    avg_cost: float
    entry_date: str
    entry_day_low: float
    holding_days: int = 0
    add_count: int = 0
    last_add_price: float | None = None
    last_add_date: str | None = None


@dataclass
class PendingOrder:
    security_id: str
    side: str
    action: str
    target_pct: float
    signal_date: str
    signal_price: float
    rule_id: str
    reason: str


class BacktestService:
    def __init__(self, database: Database = db):
        self.db = database
        self._lock = threading.Lock()

    def _validated_dsl(self, request: BacktestRequest):
        try:
            start = date.fromisoformat(request.start_date)
            end = date.fromisoformat(request.end_date)
        except ValueError as error:
            raise ValueError("回测日期格式必须为YYYY-MM-DD") from error
        if start >= end:
            raise ValueError("回测结束日期必须晚于开始日期")
        strategy = self.db.one(
            "SELECT * FROM strategies WHERE id=? AND deleted_at IS NULL", (request.strategy_id,)
        )
        if not strategy:
            raise LookupError("策略不存在")
        version = self.db.one(
            "SELECT dsl_json FROM strategy_versions WHERE strategy_id=? AND version=?",
            (request.strategy_id, strategy["current_version"]),
        )
        if not version:
            raise ValueError("策略缺少可回测版本")
        return validate_dsl(self.db.load(version["dsl_json"]))

    def start(self, request: BacktestRequest) -> tuple[dict[str, Any], Any]:
        dsl = self._validated_dsl(request)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("已有回测正在运行，请稍后重试")
        run_id = str(uuid.uuid4())
        try:
            self.db.execute(
                """INSERT INTO backtest_runs(id,strategy_id,status,start_date,end_date,config_json,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (run_id, request.strategy_id, "running", request.start_date, request.end_date, self.db.dump(request.model_dump()), utcnow()),
            )
            return self.detail(run_id), dsl
        except Exception:
            self._lock.release()
            raise

    def execute(self, run_id: str, request: BacktestRequest, dsl: Any, raise_errors: bool = False) -> None:
        try:
            result = self._simulate(run_id, request, dsl)
            self.db.execute(
                "UPDATE backtest_runs SET status='success',summary_json=?,finished_at=? WHERE id=?",
                (self.db.dump(result["summary"]), utcnow(), run_id),
            )
        except Exception as error:
            self.db.execute(
                "UPDATE backtest_runs SET status='failed',error=?,finished_at=? WHERE id=?",
                (str(error)[:500], utcnow(), run_id),
            )
            if raise_errors:
                raise
        finally:
            self._lock.release()

    def run(self, request: BacktestRequest) -> dict[str, Any]:
        started, dsl = self.start(request)
        self.execute(started["id"], request, dsl, raise_errors=True)
        return self.detail(started["id"])

    def _load_securities(self, request: BacktestRequest) -> list[dict[str, Any]]:
        securities = self.db.all(
            """SELECT s.* FROM securities s
               WHERE EXISTS (
                 SELECT 1 FROM bars b WHERE b.security_id=s.id AND b.trade_date<=? AND b.is_provisional=0
               ) ORDER BY s.market,s.code""",
            (request.end_date,),
        )
        eligible = []
        for security in securities:
            bars = self.db.all(
                """SELECT trade_date,open,high,low,close,volume,0 AS is_provisional,trade_status,is_st
                   FROM bars WHERE security_id=? AND trade_date<=? AND is_provisional=0 ORDER BY trade_date""",
                (security["id"], request.end_date),
            )
            if len(bars) >= 30 and any(bar["trade_date"] >= request.start_date for bar in bars):
                security["bars"] = bars
                security["bar_by_date"] = {bar["trade_date"]: bar for bar in bars}
                security["index_by_date"] = {bar["trade_date"]: index for index, bar in enumerate(bars)}
                eligible.append(security)
        if len(eligible) > request.max_securities:
            raise ValueError(f"可回测股票有{len(eligible)}只，超过当前上限{request.max_securities}只")
        if not eligible:
            raise ValueError("所选日期范围没有足够的已完成日K数据")
        return eligible

    @staticmethod
    def _context(security: dict[str, Any], position: SimPosition | None = None) -> dict[str, Any]:
        context: dict[str, Any] = {
            "security_id": security["id"],
            "market": security["market"],
            "code": security["code"],
            "name": security["name"],
        }
        if position:
            context.update(
                {
                    "avg_cost": position.avg_cost,
                    "entry_trade_date": position.entry_date,
                    "entry_day_low": position.entry_day_low,
                    "add_count": position.add_count,
                    "last_add_price": position.last_add_price,
                    "days_since_last_add": None if not position.last_add_date else position.holding_days,
                    "days_since_last_buy": position.holding_days,
                    "first_add_20d_low": None,
                    "first_add_post_low_high": None,
                    "first_add_rebound_pct": None,
                    "first_add_rebound_confirmed": None,
                }
            )
        return context

    @staticmethod
    def _fee(gross: float, request: BacktestRequest, sell: bool) -> float:
        commission = max(request.min_commission, gross * request.commission_pct / 100) if gross > 0 else 0
        tax = gross * request.sell_tax_pct / 100 if sell else 0
        return commission + tax

    def _simulate(self, run_id: str, request: BacktestRequest, dsl) -> dict[str, Any]:
        securities = self._load_securities(request)
        by_id = {security["id"]: security for security in securities}
        trade_dates = sorted(
            {
                bar["trade_date"]
                for security in securities
                for bar in security["bars"]
                if request.start_date <= bar["trade_date"] <= request.end_date
            }
        )
        if len(trade_dates) < 2:
            raise ValueError("回测区间至少需要两个交易日")

        cash = {"CNY": request.initial_cny, "HKD": request.initial_hkd}
        initial = dict(cash)
        positions: dict[str, SimPosition] = {}
        candidates: set[str] = set()
        pending: dict[str, PendingOrder] = {}
        last_close: dict[str, float] = {}
        trades: list[dict[str, Any]] = []
        signals: list[dict[str, Any]] = []
        curve: list[dict[str, Any]] = []
        peak = 100.0

        universe_rules = [rule for rule in dsl.rules if rule.scope == "universe"]
        candidate_rules = [rule for rule in dsl.rules if rule.scope == "candidate"]
        position_rules = [rule for rule in dsl.rules if rule.scope == "position"]
        if not universe_rules:
            candidates.update(by_id)

        for current_date in trade_dates:
            for security_id, security in by_id.items():
                bar = security["bar_by_date"].get(current_date)
                if not bar:
                    continue
                last_close[security_id] = float(bar["close"])
                position = positions.get(security_id)
                if position:
                    position.holding_days += 1
                order = pending.get(security_id)
                if order and order.signal_date < current_date:
                    self._execute_order(order, security, bar, cash, positions, trades, request)
                    pending.pop(security_id, None)

            for security_id, security in by_id.items():
                bar = security["bar_by_date"].get(current_date)
                if not bar:
                    continue
                index = security["index_by_date"][current_date]
                history = security["bars"][: index + 1]
                if len(history) < 30:
                    continue
                base_context = self._context(security)
                for rule in universe_rules:
                    if rule.action == "add_candidate" and evaluate_rule(rule, history, base_context, security["market"]):
                        if security_id not in positions:
                            candidates.add(security_id)

                if security_id in candidates and security_id not in positions and security_id not in pending:
                    matched = [
                        rule
                        for rule in candidate_rules
                        if evaluate_rule(rule, history, base_context, security["market"])
                    ]
                    buy_rules = [rule for rule in matched if rule.action == "signal_buy"]
                    if buy_rules:
                        winner = max(buy_rules, key=lambda rule: ACTION_PRIORITY[rule.action])
                        self._schedule(run_id, pending, signals, security, winner, current_date, float(bar["close"]), request)

                position = positions.get(security_id)
                if not position or security_id in pending:
                    continue
                context = self._context(security, position)
                forced_rule: Any | None = None
                forced_action = ""
                forced_reason = ""
                if request.stop_loss_pct is not None and float(bar["close"]) < position.entry_day_low * (1 - request.stop_loss_pct / 100):
                    forced_action, forced_reason = "signal_stop", f"收盘价跌破建仓日低点{request.stop_loss_pct:g}%"
                elif request.max_holding_days and position.holding_days >= request.max_holding_days:
                    forced_action, forced_reason = "signal_exit", f"达到最长持有{request.max_holding_days}个交易日"
                if forced_action:
                    forced_rule = type("BacktestRule", (), {
                        "id": f"backtest-{forced_action}", "action": forced_action,
                        "target_position_pct": 0.0, "label": forced_reason,
                    })()
                    self._schedule(run_id, pending, signals, security, forced_rule, current_date, float(bar["close"]), request)
                    continue
                matched = [
                    rule
                    for rule in position_rules
                    if rule.action in ACTION_PRIORITY and evaluate_rule(rule, history, context, security["market"])
                ]
                if matched:
                    winner = max(matched, key=lambda rule: ACTION_PRIORITY[rule.action])
                    self._schedule(run_id, pending, signals, security, winner, current_date, float(bar["close"]), request)

            values = {"CNY": cash["CNY"], "HKD": cash["HKD"]}
            for security_id, position in positions.items():
                security = by_id[security_id]
                values[security["currency"]] += position.quantity * last_close.get(security_id, position.avg_cost)
            normalized = ((values["CNY"] / initial["CNY"] + values["HKD"] / initial["HKD"]) / 2) * 100
            peak = max(peak, normalized)
            curve.append(
                {
                    "trade_date": current_date,
                    "cny_equity": values["CNY"],
                    "hkd_equity": values["HKD"],
                    "normalized_equity": normalized,
                    "drawdown_pct": (normalized / peak - 1) * 100,
                }
            )

        if request.force_close and positions:
            last_date = trade_dates[-1]
            for security_id in list(positions):
                security = by_id[security_id]
                bar = security["bar_by_date"].get(last_date)
                if not bar:
                    continue
                order = PendingOrder(security_id, "sell", "signal_exit", 0, last_date, float(bar["close"]), "period-end", "回测期末平仓")
                self._execute_order(order, security, {**bar, "open": bar["close"]}, cash, positions, trades, request, ignore_protection=True)
            values = {"CNY": cash["CNY"], "HKD": cash["HKD"]}
            for security_id, position in positions.items():
                security = by_id[security_id]
                values[security["currency"]] += position.quantity * last_close.get(security_id, position.avg_cost)
            normalized = ((values["CNY"] / initial["CNY"] + values["HKD"] / initial["HKD"]) / 2) * 100
            peak = max(peak, normalized)
            curve[-1].update(
                cny_equity=values["CNY"], hkd_equity=values["HKD"], normalized_equity=normalized,
                drawdown_pct=(normalized / peak - 1) * 100,
            )

        summary = self._summary(request, securities, signals, trades, curve)
        self._persist(run_id, signals, trades, curve)
        return {"summary": summary}

    def _schedule(self, run_id: str, pending: dict[str, PendingOrder], signals: list[dict[str, Any]], security: dict[str, Any], rule: Any, signal_date: str, signal_price: float, request: BacktestRequest) -> None:
        side = "buy" if rule.action in {"signal_buy", "signal_add"} else "sell"
        target = rule.target_position_pct
        if rule.action in {"signal_exit", "signal_stop"}:
            target = 0.0
        if target is None:
            target = request.default_position_pct if side == "buy" else 0.0
        reason = str(rule.label)
        pending[security["id"]] = PendingOrder(
            security["id"], side, rule.action, float(target), signal_date, signal_price, str(rule.id), reason
        )
        signals.append(
            {
                "id": str(uuid.uuid4()), "security_id": security["id"], "signal_date": signal_date,
                "rule_id": str(rule.id), "action": rule.action, "signal_price": signal_price, "reason": reason,
            }
        )

    def _execute_order(self, order: PendingOrder, security: dict[str, Any], bar: dict[str, Any], cash: dict[str, float], positions: dict[str, SimPosition], trades: list[dict[str, Any]], request: BacktestRequest, ignore_protection: bool = False) -> None:
        currency = security["currency"]
        lot = max(1, int(security.get("lot_size") or 100))
        open_price = float(bar["open"])
        price = open_price * (1 + request.slippage_pct / 100 if order.side == "buy" else 1 - request.slippage_pct / 100)
        if not ignore_protection and order.side == "buy" and price > order.signal_price * 1.01:
            return
        if not ignore_protection and order.side == "sell" and order.action != "signal_stop" and price < order.signal_price * 0.98:
            return
        position = positions.get(order.security_id)
        target_pct = min(order.target_pct, request.max_security_pct) if order.side == "buy" else order.target_pct
        target_value = (request.initial_cny if currency == "CNY" else request.initial_hkd) * target_pct / 100
        current_quantity = position.quantity if position else 0.0
        target_quantity = target_value / price if price > 0 else 0
        if order.side == "buy":
            raw_quantity = max(0.0, target_quantity - current_quantity)
            quantity = math.floor(raw_quantity / lot) * lot
            while quantity > 0:
                gross = quantity * price
                fee = self._fee(gross, request, False)
                if gross + fee <= cash[currency] + 1e-9:
                    break
                quantity -= lot
            if quantity <= 0:
                return
            gross = quantity * price
            fee = self._fee(gross, request, False)
            cash[currency] -= gross + fee
            if position:
                new_quantity = position.quantity + quantity
                position.avg_cost = (position.avg_cost * position.quantity + gross + fee) / new_quantity
                position.quantity = new_quantity
                position.add_count += 1
                position.last_add_price = price
                position.last_add_date = bar["trade_date"]
            else:
                positions[order.security_id] = SimPosition(quantity, (gross + fee) / quantity, bar["trade_date"], float(bar["low"]))
            realized = None
        else:
            if not position:
                return
            raw_quantity = max(0.0, position.quantity - target_quantity)
            quantity = min(position.quantity, math.ceil(raw_quantity / lot) * lot)
            if quantity <= 0:
                return
            gross = quantity * price
            fee = self._fee(gross, request, True)
            cash[currency] += gross - fee
            realized = (price - position.avg_cost) * quantity - fee
            position.quantity -= quantity
            if position.quantity <= 1e-9:
                positions.pop(order.security_id, None)
        trades.append(
            {
                "id": str(uuid.uuid4()), "security_id": order.security_id, "side": order.side,
                "quantity": quantity, "price": price, "fee": fee, "signal_date": order.signal_date,
                "trade_date": bar["trade_date"], "action": order.action, "reason": order.reason,
                "realized_pnl": realized,
            }
        )

    @staticmethod
    def _summary(request: BacktestRequest, securities: list[dict[str, Any]], signals: list[dict[str, Any]], trades: list[dict[str, Any]], curve: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = [float(point["normalized_equity"]) for point in curve]
        returns = [normalized[index] / normalized[index - 1] - 1 for index in range(1, len(normalized)) if normalized[index - 1] != 0]
        sharpe = 0.0
        if len(returns) > 1 and statistics.stdev(returns) > 0:
            sharpe = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(252)
        sells = [trade for trade in trades if trade["side"] == "sell" and trade["realized_pnl"] is not None]
        wins = [trade for trade in sells if float(trade["realized_pnl"]) > 0]
        days = max(1, (date.fromisoformat(request.end_date) - date.fromisoformat(request.start_date)).days)
        total_return = normalized[-1] / 100 - 1 if normalized else 0
        signal_analysis: dict[str, Any] = {}
        for horizon in (5, 10, 20, 60):
            observations = []
            for signal in signals:
                if signal["action"] != "signal_buy":
                    continue
                security = next((item for item in securities if item["id"] == signal["security_id"]), None)
                if not security:
                    continue
                index = security["index_by_date"].get(signal["signal_date"])
                if index is not None and index + horizon < len(security["bars"]):
                    future = float(security["bars"][index + horizon]["close"])
                    observations.append((future / float(signal["signal_price"]) - 1) * 100)
            signal_analysis[str(horizon)] = {
                "count": len(observations),
                "average_return_pct": sum(observations) / len(observations) if observations else None,
                "win_rate_pct": sum(value > 0 for value in observations) / len(observations) * 100 if observations else None,
            }
        return {
            "securities": len(securities), "trading_days": len(curve), "signals": len(signals), "trades": len(trades),
            "closed_trades": len(sells), "win_rate_pct": len(wins) / len(sells) * 100 if sells else None,
            "total_return_pct": total_return * 100,
            "annualized_return_pct": ((1 + total_return) ** (365 / days) - 1) * 100 if total_return > -1 else -100,
            "max_drawdown_pct": min((float(point["drawdown_pct"]) for point in curve), default=0),
            "sharpe": sharpe,
            "cny_return_pct": (float(curve[-1]["cny_equity"]) / request.initial_cny - 1) * 100,
            "hkd_return_pct": (float(curve[-1]["hkd_equity"]) / request.initial_hkd - 1) * 100,
            "total_fees": sum(float(trade["fee"]) for trade in trades),
            "signal_analysis": signal_analysis,
            "survivorship_warning": "证券池使用当前数据库中的股票，未包含完整历史退市和历史ST状态。",
        }

    def _persist(self, run_id: str, signals: list[dict[str, Any]], trades: list[dict[str, Any]], curve: list[dict[str, Any]]) -> None:
        self.db.executemany(
            """INSERT INTO backtest_signals(id,run_id,security_id,signal_date,rule_id,action,signal_price,reason)
               VALUES(?,?,?,?,?,?,?,?)""",
            [(item["id"], run_id, item["security_id"], item["signal_date"], item["rule_id"], item["action"], item["signal_price"], item["reason"]) for item in signals],
        )
        self.db.executemany(
            """INSERT INTO backtest_trades(id,run_id,security_id,side,quantity,price,fee,signal_date,trade_date,action,reason,realized_pnl)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(item["id"], run_id, item["security_id"], item["side"], item["quantity"], item["price"], item["fee"], item["signal_date"], item["trade_date"], item["action"], item["reason"], item["realized_pnl"]) for item in trades],
        )
        self.db.executemany(
            """INSERT INTO backtest_equity_curve(run_id,trade_date,cny_equity,hkd_equity,normalized_equity,drawdown_pct)
               VALUES(?,?,?,?,?,?)""",
            [(run_id, item["trade_date"], item["cny_equity"], item["hkd_equity"], item["normalized_equity"], item["drawdown_pct"]) for item in curve],
        )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.all(
            """SELECT r.*,s.name AS strategy_name FROM backtest_runs r JOIN strategies s ON s.id=r.strategy_id
               ORDER BY r.created_at DESC LIMIT ?""",
            (limit,),
        )
        return [{**row, "config": self.db.load(row.pop("config_json"), {}), "summary": self.db.load(row.pop("summary_json"), {})} for row in rows]

    def detail(self, run_id: str) -> dict[str, Any]:
        row = self.db.one(
            """SELECT r.*,s.name AS strategy_name FROM backtest_runs r JOIN strategies s ON s.id=r.strategy_id WHERE r.id=?""",
            (run_id,),
        )
        if not row:
            raise LookupError("回测记录不存在")
        config = self.db.load(row.pop("config_json"), {})
        summary = self.db.load(row.pop("summary_json"), {})
        return {
            **row, "config": config, "summary": summary,
            "equity_curve": self.db.all("SELECT * FROM backtest_equity_curve WHERE run_id=? ORDER BY trade_date", (run_id,)),
            "trades": self.db.all(
                """SELECT t.*,s.code,s.name AS security_name,s.market,s.currency FROM backtest_trades t
                   JOIN securities s ON s.id=t.security_id WHERE t.run_id=? ORDER BY t.trade_date,t.id""",
                (run_id,),
            ),
            "signals": self.db.all(
                """SELECT x.*,s.code,s.name AS security_name,s.market FROM backtest_signals x
                   JOIN securities s ON s.id=x.security_id WHERE x.run_id=? ORDER BY x.signal_date,x.id LIMIT 1000""",
                (run_id,),
            ),
        }

    def dashboard(self) -> dict[str, Any]:
        runs = self.list_runs(10)
        latest = self.detail(runs[0]["id"]) if runs and runs[0]["status"] == "success" else None
        return {"runs": runs, "latest": latest}


backtest_service = BacktestService()
