from __future__ import annotations

import json
import math
import threading
import uuid
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .brokers import BrokerAdapter, BrokerFill, BrokerOrderResult, PaperBrokerAdapter
from .config import settings
from .database import Database, db, utcnow


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
ACCOUNT_ID = "paper-main"
ENTRY_ACTIONS = {"signal_buy", "signal_add"}
EXIT_ACTIONS = {"signal_reduce", "signal_exit", "signal_stop"}
TERMINAL_STATUSES = {"filled", "canceled", "rejected", "expired", "blocked", "completed_noop"}


class TradingService:
    def __init__(self, database: Database = db, broker: BrokerAdapter | None = None):
        self.db = database
        if broker is None and settings.broker_adapter != "paper":
            raise RuntimeError(f"尚未安装券商适配器：{settings.broker_adapter}")
        self.broker = broker or PaperBrokerAdapter(database)
        self._lock = threading.Lock()

    def initialize(self) -> None:
        now = utcnow()
        self.db.execute(
            """INSERT OR IGNORE INTO trading_accounts(id,name,broker,mode,status,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?)""",
            (ACCOUNT_ID, "本地模拟账户", self.broker.name, settings.trading_mode, "connected", now, now),
        )
        self.db.execute(
            "UPDATE trading_accounts SET broker=?,mode=?,updated_at=? WHERE id=?",
            (self.broker.name, settings.trading_mode, now, ACCOUNT_ID),
        )
        for currency, amount in (("CNY", settings.paper_cny_cash), ("HKD", settings.paper_hkd_cash)):
            self.db.execute(
                """INSERT OR IGNORE INTO account_balances(account_id,currency,cash,available,frozen,updated_at)
                   VALUES(?,?,?,?,0,?)""",
                (ACCOUNT_ID, currency, amount, amount, now),
            )
        self._import_existing_paper_positions()
        self.snapshot_equity()

    def _import_existing_paper_positions(self) -> None:
        if self.db.one("SELECT 1 FROM broker_positions WHERE account_id=? LIMIT 1", (ACCOUNT_ID,)):
            return
        rows = self.db.all(
            """SELECT p.security_id,SUM(p.quantity) AS quantity,
                      SUM(p.quantity*p.avg_cost)/SUM(p.quantity) AS avg_cost,s.market
               FROM positions p JOIN securities s ON s.id=p.security_id
               GROUP BY p.security_id,s.market"""
        )
        now = utcnow()
        for row in rows:
            self.db.execute(
                """INSERT OR IGNORE INTO broker_positions(
                       account_id,security_id,quantity,available_quantity,avg_cost,acquired_date,updated_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (ACCOUNT_ID, row["security_id"], row["quantity"], row["quantity"], row["avg_cost"], date.today().isoformat(), now),
            )

    def _audit(self, event_type: str, entity_type: str | None = None, entity_id: str | None = None, detail: dict | None = None) -> None:
        self.db.execute(
            """INSERT INTO trading_audit(id,account_id,event_type,entity_type,entity_id,detail_json,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), ACCOUNT_ID, event_type, entity_type, entity_id, self.db.dump(detail or {}), utcnow()),
        )

    def configure_allocation(self, strategy_id: str, currency: str, capital_limit: float, enabled: bool) -> dict:
        if currency not in {"CNY", "HKD"}:
            raise ValueError("币种仅支持CNY或HKD")
        if capital_limit <= 0:
            raise ValueError("策略资金配额必须大于0")
        if not self.db.one("SELECT 1 FROM strategies WHERE id=? AND deleted_at IS NULL", (strategy_id,)):
            raise LookupError("策略不存在")
        now = utcnow()
        self.db.execute(
            """INSERT INTO strategy_allocations(account_id,strategy_id,currency,capital_limit,enabled,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(account_id,strategy_id,currency) DO UPDATE SET
                 capital_limit=excluded.capital_limit,enabled=excluded.enabled,updated_at=excluded.updated_at""",
            (ACCOUNT_ID, strategy_id, currency, capital_limit, int(enabled), now, now),
        )
        self._audit("allocation_updated", "strategy", strategy_id, {"currency": currency, "capital_limit": capital_limit, "enabled": enabled})
        return self.db.one(
            "SELECT * FROM strategy_allocations WHERE account_id=? AND strategy_id=? AND currency=?",
            (ACCOUNT_ID, strategy_id, currency),
        ) or {}

    def _next_session(self, market: str, signal_date: str) -> date:
        current = date.fromisoformat(signal_date) + timedelta(days=1)
        while current.weekday() >= 5 or self.db.one(
            "SELECT 1 FROM market_holidays WHERE market=? AND trade_date=?", (market, current.isoformat())
        ):
            current += timedelta(days=1)
        return current

    def create_intent(self, strategy_id: str, rule: Any, security_id: str, reason: str) -> dict | None:
        if rule.action not in ENTRY_ACTIONS | EXIT_ACTIONS:
            return None
        security = self.db.one("SELECT * FROM securities WHERE id=?", (security_id,))
        if not security:
            return None
        signal_bar = self.db.one(
            """SELECT trade_date,close FROM bars WHERE security_id=? AND is_provisional=0
               ORDER BY trade_date DESC LIMIT 1""",
            (security_id,),
        )
        if signal_bar:
            signal_date = signal_bar["trade_date"]
            signal_price = float(signal_bar["close"])
        else:
            quote_time = str(security.get("quote_time") or "")
            if security["latest_price"] is None or len(quote_time) < 10:
                return None
            signal_date = quote_time[:10]
            signal_price = float(security["latest_price"])
        allocation = self.db.one(
            """SELECT * FROM strategy_allocations
               WHERE account_id=? AND strategy_id=? AND currency=? AND enabled=1""",
            (ACCOUNT_ID, strategy_id, security["currency"]),
        )
        if not allocation:
            return None
        target = rule.target_position_pct
        if rule.action in {"signal_exit", "signal_stop"}:
            target = 0.0
        if target is None:
            self._audit("intent_rejected_missing_target", "strategy", strategy_id, {"rule_id": rule.id, "security_id": security_id})
            return None
        side = "buy" if rule.action in ENTRY_ACTIONS else "sell"
        session = self._next_session(security["market"], signal_date)
        eligible = datetime.combine(session, time(9, 30), SHANGHAI_TZ)
        approval_deadline = datetime.combine(session, time(9, 20), SHANGHAI_TZ)
        expires = datetime.combine(session, time(10, 0), SHANGHAI_TZ) if side == "buy" else datetime.combine(session, time(15, 59), SHANGHAI_TZ)
        lifecycle = self.db.one(
            "SELECT id,opened_at FROM positions WHERE strategy_id=? AND security_id=?",
            (strategy_id, security_id),
        )
        lifecycle_key = lifecycle["id"] if lifecycle else "flat"
        idempotency_key = f"{strategy_id}:{rule.id}:{security_id}:{rule.action}:{signal_date}:{lifecycle_key}"
        intent_id = str(uuid.uuid4())
        requires_approval = side == "buy"
        status = "pending_approval" if requires_approval else "approved"
        now = utcnow()
        self.db.execute(
            """INSERT OR IGNORE INTO order_intents(
                   id,idempotency_key,account_id,strategy_id,rule_id,security_id,action,side,status,
                   target_position_pct,signal_price,signal_date,requires_approval,is_stop,approved_at,
                   approval_deadline,eligible_at,expires_at,reason,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                intent_id,
                idempotency_key,
                ACCOUNT_ID,
                strategy_id,
                rule.id,
                security_id,
                rule.action,
                side,
                status,
                float(target),
                signal_price,
                signal_date,
                int(requires_approval),
                int(rule.action == "signal_stop"),
                None if requires_approval else now,
                approval_deadline.isoformat(),
                eligible.isoformat(),
                expires.isoformat(),
                reason,
                now,
                now,
            ),
        )
        row = self.db.one("SELECT * FROM order_intents WHERE idempotency_key=?", (idempotency_key,))
        if row and row["id"] == intent_id:
            self._audit("intent_created", "order_intent", intent_id, {"action": rule.action, "requires_approval": requires_approval})
        return row

    def approve(self, intent_id: str) -> dict:
        intent = self.db.one("SELECT * FROM order_intents WHERE id=?", (intent_id,))
        if not intent:
            raise LookupError("订单意图不存在")
        if intent["status"] != "pending_approval":
            raise ValueError("当前订单不处于待批准状态")
        now = datetime.now(timezone.utc)
        if intent.get("approval_deadline") and now > datetime.fromisoformat(intent["approval_deadline"]).astimezone(timezone.utc):
            self.db.execute("UPDATE order_intents SET status='expired',updated_at=? WHERE id=?", (utcnow(), intent_id))
            raise ValueError("买入批准时间已截止")
        self.db.execute(
            "UPDATE order_intents SET status='approved',approved_at=?,updated_at=? WHERE id=?",
            (utcnow(), utcnow(), intent_id),
        )
        self._audit("intent_approved", "order_intent", intent_id)
        return self.intent(intent_id)

    def reject(self, intent_id: str) -> dict:
        intent = self.db.one("SELECT * FROM order_intents WHERE id=?", (intent_id,))
        if not intent:
            raise LookupError("订单意图不存在")
        if intent["status"] not in {"pending_approval", "approved"}:
            raise ValueError("当前订单不能拒绝")
        self.db.execute("UPDATE order_intents SET status='canceled',updated_at=? WHERE id=?", (utcnow(), intent_id))
        self._audit("intent_rejected", "order_intent", intent_id)
        return self.intent(intent_id)

    def intent(self, intent_id: str) -> dict:
        row = self.db.one(
            """SELECT i.*,s.name AS strategy_name,q.code,q.name AS security_name,q.market,q.currency
               FROM order_intents i JOIN strategies s ON s.id=i.strategy_id
               JOIN securities q ON q.id=i.security_id WHERE i.id=?""",
            (intent_id,),
        )
        if not row:
            raise LookupError("订单意图不存在")
        return row

    def list_intents(self, limit: int = 100) -> list[dict]:
        return self.db.all(
            """SELECT i.*,s.name AS strategy_name,q.code,q.name AS security_name,q.market,q.currency
               FROM order_intents i JOIN strategies s ON s.id=i.strategy_id
               JOIN securities q ON q.id=i.security_id ORDER BY i.created_at DESC LIMIT ?""",
            (limit,),
        )

    def _currency_equity(self, currency: str) -> float:
        balance = self.db.one(
            "SELECT cash FROM account_balances WHERE account_id=? AND currency=?", (ACCOUNT_ID, currency)
        )
        positions = self.db.one(
            """SELECT COALESCE(SUM(p.quantity*s.latest_price),0) AS value
               FROM broker_positions p JOIN securities s ON s.id=p.security_id
               WHERE p.account_id=? AND s.currency=?""",
            (ACCOUNT_ID, currency),
        )
        return float((balance or {}).get("cash") or 0) + float((positions or {}).get("value") or 0)

    def snapshot_equity(self) -> None:
        today = date.today().isoformat()
        for currency in ("CNY", "HKD"):
            self.db.execute(
                """INSERT OR IGNORE INTO daily_equity_snapshots(account_id,trade_date,currency,equity,created_at)
                   VALUES(?,?,?,?,?)""",
                (ACCOUNT_ID, today, currency, self._currency_equity(currency), utcnow()),
            )

    def _risk_block(self, intent: dict, security: dict, allocation: dict, quote: float) -> str | None:
        account = self.db.one("SELECT * FROM trading_accounts WHERE id=?", (ACCOUNT_ID,)) or {}
        if account.get("emergency_stop"):
            return "账户处于紧急停止状态"
        if intent["side"] == "buy" and account.get("entry_paused"):
            return "账户已暂停买入"
        if account.get("mode") == "live":
            if self.broker.name == "paper":
                return "实盘模式不能使用模拟券商适配器"
            if not settings.live_trading_enabled:
                return "实盘环境开关未开启"
            armed_until = account.get("armed_until")
            if not armed_until or datetime.fromisoformat(armed_until) <= datetime.now(timezone.utc):
                return "实盘账户尚未完成当日授权"
        health = self.broker.health(ACCOUNT_ID)
        if not health.get("connected"):
            return "券商连接不可用"
        equity = self._currency_equity(security["currency"])
        if equity <= 0:
            return "账户权益无效"
        if float(allocation["capital_limit"]) > equity * settings.max_strategy_exposure_pct / 100 + 1e-9:
            return f"策略配额超过账户权益{settings.max_strategy_exposure_pct:g}%"
        snapshot = self.db.one(
            """SELECT equity FROM daily_equity_snapshots
               WHERE account_id=? AND trade_date=? AND currency=?""",
            (ACCOUNT_ID, date.today().isoformat(), security["currency"]),
        )
        if intent["side"] == "buy" and snapshot and float(snapshot["equity"]) > 0:
            loss_pct = (equity / float(snapshot["equity"]) - 1) * 100
            if loss_pct <= -settings.daily_loss_pause_pct:
                return f"当日权益亏损达到{settings.daily_loss_pause_pct:g}%，已冻结买入"
        broker_position = self.db.one(
            "SELECT quantity FROM broker_positions WHERE account_id=? AND security_id=?", (ACCOUNT_ID, security["id"])
        )
        current_value = float((broker_position or {}).get("quantity") or 0) * quote
        if intent["side"] == "buy" and current_value >= equity * settings.max_security_exposure_pct / 100:
            return f"单股敞口已达到{settings.max_security_exposure_pct:g}%上限"
        return None

    @staticmethod
    def _round_price(value: float, tick: float, direction: str) -> float:
        units = value / tick
        rounded = math.floor(units + 1e-9) if direction == "down" else math.ceil(units - 1e-9)
        return round(rounded * tick, 6)

    def _size_and_price(self, intent: dict, security: dict, allocation: dict) -> tuple[float, float]:
        quote = self.broker.quote(security["id"]).price
        target_value = float(allocation["capital_limit"]) * float(intent["target_position_pct"]) / 100
        if intent["side"] == "buy":
            equity = self._currency_equity(security["currency"])
            target_value = min(target_value, equity * settings.max_security_exposure_pct / 100)
        local = self.db.one(
            "SELECT quantity FROM positions WHERE strategy_id=? AND security_id=?",
            (intent["strategy_id"], security["id"]),
        )
        local_quantity = float((local or {}).get("quantity") or 0)
        target_quantity = target_value / quote if quote > 0 else 0
        tick = float(security.get("price_tick") or 0.01)
        lot = max(1, int(security.get("lot_size") or 100))
        if intent["side"] == "buy":
            raw = max(0.0, target_quantity - local_quantity)
            quantity = math.floor(raw / lot) * lot
            limit_price = self._round_price(float(intent["signal_price"]) * 1.01, tick, "down")
        else:
            raw = max(0.0, local_quantity - target_quantity)
            quantity = min(local_quantity, math.ceil(raw))
            if intent["is_stop"]:
                value = quote if security["market"] in {"SH", "SZ"} else quote * 0.995
                limit_price = self._round_price(value, tick, "down")
            else:
                limit_price = self._round_price(float(intent["signal_price"]) * 0.98, tick, "up")
        return float(quantity), limit_price

    def _record_fill(self, intent: dict, broker_order_id: str, fill: BrokerFill) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO trade_fills(
                   id,broker_fill_id,broker_order_id,intent_id,security_id,side,quantity,price,fee,filled_at,raw_json
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()), fill.broker_fill_id, broker_order_id, intent["id"], intent["security_id"], intent["side"],
                fill.quantity, fill.price, fill.fee, fill.filled_at, "{}",
            ),
        )
        self._apply_fill_to_strategy(intent, fill)
        self._audit("fill_recorded", "order_intent", intent["id"], {"quantity": fill.quantity, "price": fill.price})

    def _apply_fill_to_strategy(self, intent: dict, fill: BrokerFill) -> None:
        now = fill.filled_at
        with self.db.transaction() as connection:
            position = connection.execute(
                "SELECT * FROM positions WHERE strategy_id=? AND security_id=?",
                (intent["strategy_id"], intent["security_id"]),
            ).fetchone()
            if intent["side"] == "buy":
                if position:
                    old_quantity = float(position["quantity"])
                    new_quantity = old_quantity + fill.quantity
                    new_cost = (old_quantity * float(position["avg_cost"]) + fill.quantity * fill.price + fill.fee) / new_quantity
                    connection.execute(
                        "UPDATE positions SET quantity=?,avg_cost=? WHERE id=?", (new_quantity, new_cost, position["id"])
                    )
                else:
                    connection.execute(
                        """INSERT INTO positions(id,strategy_id,security_id,quantity,avg_cost,opened_at,created_at,entry_trade_date)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (str(uuid.uuid4()), intent["strategy_id"], intent["security_id"], fill.quantity, fill.price, now, now, now[:10]),
                    )
                    connection.execute(
                        "DELETE FROM candidates WHERE strategy_id=? AND security_id=?",
                        (intent["strategy_id"], intent["security_id"]),
                    )
            else:
                if not position:
                    raise ValueError("策略持仓不存在，无法登记卖出成交")
                remaining = float(position["quantity"]) - fill.quantity
                if remaining <= 1e-9:
                    connection.execute("DELETE FROM positions WHERE id=?", (position["id"],))
                else:
                    connection.execute("UPDATE positions SET quantity=? WHERE id=?", (remaining, position["id"]))
            connection.execute(
                """INSERT INTO position_events(id,strategy_id,security_id,event_type,quantity,price,occurred_at,detail_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()), intent["strategy_id"], intent["security_id"],
                    "trade_buy" if intent["side"] == "buy" else "trade_sell",
                    fill.quantity, fill.price, now, json.dumps({"intent_id": intent["id"], "fee": fill.fee}),
                ),
            )

    def _store_broker_order(self, intent: dict, result: BrokerOrderResult) -> None:
        order_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO broker_orders(
                   id,intent_id,broker_order_id,status,side,quantity,filled_quantity,limit_price,submitted_at,updated_at,raw_json
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                order_id, intent["id"], result.broker_order_id, result.status, intent["side"], result.quantity,
                result.filled_quantity, result.limit_price, result.submitted_at, utcnow(), "{}",
            ),
        )
        if result.fill:
            self._record_fill(intent, result.broker_order_id, result.fill)
        self.db.execute(
            """UPDATE order_intents SET status=?,filled_quantity=filled_quantity+?,updated_at=? WHERE id=?""",
            (result.status, result.filled_quantity, utcnow(), intent["id"]),
        )
        self._audit("order_submitted", "order_intent", intent["id"], {"broker_order_id": result.broker_order_id, "status": result.status})

    def _submit(self, intent: dict) -> None:
        security = self.db.one("SELECT * FROM securities WHERE id=?", (intent["security_id"],))
        allocation = self.db.one(
            """SELECT * FROM strategy_allocations WHERE account_id=? AND strategy_id=? AND currency=? AND enabled=1""",
            (ACCOUNT_ID, intent["strategy_id"], security["currency"] if security else ""),
        )
        if not security or not allocation:
            self._block(intent["id"], "策略资金配额不存在或已停用")
            return
        quote = self.broker.quote(security["id"]).price
        risk = self._risk_block(intent, security, allocation, quote)
        if risk:
            self._block(intent["id"], risk)
            return
        quantity, limit_price = self._size_and_price(intent, security, allocation)
        if quantity <= 0:
            self.db.execute(
                "UPDATE order_intents SET status='completed_noop',blocked_reason='已达到目标仓位',updated_at=? WHERE id=?",
                (utcnow(), intent["id"]),
            )
            return
        self.db.execute(
            """UPDATE order_intents SET status='submitting',desired_quantity=?,limit_price=?,updated_at=? WHERE id=?""",
            (quantity, limit_price, utcnow(), intent["id"]),
        )
        try:
            result = self.broker.submit_limit_order(ACCOUNT_ID, security["id"], intent["side"], quantity, limit_price)
            self._store_broker_order(intent, result)
        except Exception as error:
            if intent["is_stop"]:
                self.db.execute(
                    "UPDATE order_intents SET status='approved',blocked_reason=?,updated_at=? WHERE id=?",
                    (f"止损待重试：{str(error)[:260]}", utcnow(), intent["id"]),
                )
                self._audit("stop_retry_scheduled", "order_intent", intent["id"], {"reason": str(error)[:300]})
            else:
                self._block(intent["id"], str(error))

    def _block(self, intent_id: str, reason: str) -> None:
        self.db.execute(
            "UPDATE order_intents SET status='blocked',blocked_reason=?,updated_at=? WHERE id=?",
            (reason[:300], utcnow(), intent_id),
        )
        self._audit("intent_blocked", "order_intent", intent_id, {"reason": reason[:300]})

    def _process_open_orders(self, now: datetime) -> None:
        orders = self.db.all(
            """SELECT o.*,i.account_id,i.security_id,i.side AS intent_side,i.expires_at,i.is_stop,i.status AS intent_status
               FROM broker_orders o JOIN order_intents i ON i.id=o.intent_id
               WHERE o.status IN ('submitted','partially_filled')"""
        )
        for order in orders:
            if order["expires_at"] and now > datetime.fromisoformat(order["expires_at"]).astimezone(timezone.utc) and not order["is_stop"]:
                if self.broker.cancel_order(ACCOUNT_ID, order["broker_order_id"]):
                    self.db.execute("UPDATE broker_orders SET status='canceled',updated_at=? WHERE id=?", (utcnow(), order["id"]))
                    self.db.execute("UPDATE order_intents SET status='expired',updated_at=? WHERE id=?", (utcnow(), order["intent_id"]))
                continue
            remaining = float(order["quantity"]) - float(order["filled_quantity"])
            if remaining <= 0:
                continue
            fill = self.broker.try_fill_order(
                ACCOUNT_ID, order["broker_order_id"], order["security_id"], order["intent_side"], remaining, float(order["limit_price"])
            )
            if fill:
                intent = self.db.one("SELECT * FROM order_intents WHERE id=?", (order["intent_id"],)) or {}
                self._record_fill(intent, order["broker_order_id"], fill)
                new_filled = float(order["filled_quantity"]) + fill.quantity
                status = "filled" if new_filled + 1e-9 >= float(order["quantity"]) else "partially_filled"
                self.db.execute(
                    "UPDATE broker_orders SET status=?,filled_quantity=?,updated_at=? WHERE id=?",
                    (status, new_filled, utcnow(), order["id"]),
                )
                self.db.execute(
                    "UPDATE order_intents SET status=?,filled_quantity=filled_quantity+?,updated_at=? WHERE id=?",
                    (status, fill.quantity, utcnow(), order["intent_id"]),
                )

    def process(self, now: datetime | None = None) -> dict:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if not self._lock.acquire(blocking=False):
            return {"status": "busy"}
        try:
            self.snapshot_equity()
            for pending in self.db.all(
                "SELECT id,approval_deadline FROM order_intents WHERE status='pending_approval'"
            ):
                deadline = pending.get("approval_deadline")
                if deadline and datetime.fromisoformat(deadline).astimezone(timezone.utc) < current:
                    self.db.execute(
                        "UPDATE order_intents SET status='expired',updated_at=? WHERE id=?",
                        (utcnow(), pending["id"]),
                    )
            self._process_open_orders(current)
            ready = [
                intent
                for intent in self.db.all(
                    "SELECT * FROM order_intents WHERE status='approved' ORDER BY is_stop DESC,created_at"
                )
                if datetime.fromisoformat(intent["eligible_at"]).astimezone(timezone.utc) <= current
            ]
            for intent in ready:
                self._submit(intent)
            return {"status": "success", "processed": len(ready)}
        finally:
            self._lock.release()

    def cancel(self, intent_id: str) -> dict:
        intent = self.db.one("SELECT * FROM order_intents WHERE id=?", (intent_id,))
        if not intent:
            raise LookupError("订单意图不存在")
        orders = self.db.all(
            "SELECT * FROM broker_orders WHERE intent_id=? AND status IN ('submitted','partially_filled')", (intent_id,)
        )
        for order in orders:
            if self.broker.cancel_order(ACCOUNT_ID, order["broker_order_id"]):
                self.db.execute("UPDATE broker_orders SET status='canceled',updated_at=? WHERE id=?", (utcnow(), order["id"]))
        self.db.execute("UPDATE order_intents SET status='canceled',updated_at=? WHERE id=?", (utcnow(), intent_id))
        self._audit("intent_canceled", "order_intent", intent_id)
        return self.intent(intent_id)

    def set_controls(self, entry_paused: bool | None = None, emergency_stop: bool | None = None) -> dict:
        account = self.db.one("SELECT * FROM trading_accounts WHERE id=?", (ACCOUNT_ID,))
        if not account:
            raise LookupError("交易账户不存在")
        paused = int(account["entry_paused"] if entry_paused is None else entry_paused)
        stopped = int(account["emergency_stop"] if emergency_stop is None else emergency_stop)
        self.db.execute(
            "UPDATE trading_accounts SET entry_paused=?,emergency_stop=?,updated_at=? WHERE id=?",
            (paused, stopped, utcnow(), ACCOUNT_ID),
        )
        if stopped:
            for intent in self.db.all("SELECT id FROM order_intents WHERE status IN ('approved','submitted','partially_filled')"):
                self.cancel(intent["id"])
        self._audit("controls_updated", "account", ACCOUNT_ID, {"entry_paused": bool(paused), "emergency_stop": bool(stopped)})
        return self.account_summary()

    def arm(self, hours: int = 16) -> dict:
        if settings.trading_mode == "live" and self.broker.name == "paper":
            raise ValueError("实盘模式不能使用模拟券商适配器")
        if settings.trading_mode == "live" and not settings.live_trading_enabled:
            raise ValueError("LIVE_TRADING_ENABLED未开启")
        armed_until = datetime.now(timezone.utc) + timedelta(hours=max(1, min(hours, 24)))
        self.db.execute(
            "UPDATE trading_accounts SET armed_until=?,updated_at=? WHERE id=?",
            (armed_until.isoformat(), utcnow(), ACCOUNT_ID),
        )
        self._audit("account_armed", "account", ACCOUNT_ID, {"armed_until": armed_until.isoformat()})
        return self.account_summary()

    def reconcile(self) -> dict:
        broker_positions = {row["security_id"]: float(row["quantity"]) for row in self.broker.positions(ACCOUNT_ID)}
        local_rows = self.db.all("SELECT security_id,SUM(quantity) AS quantity FROM positions GROUP BY security_id")
        local_positions = {row["security_id"]: float(row["quantity"]) for row in local_rows}
        differences = []
        for security_id in sorted(set(broker_positions) | set(local_positions)):
            broker_quantity = broker_positions.get(security_id, 0.0)
            local_quantity = local_positions.get(security_id, 0.0)
            if abs(broker_quantity - local_quantity) > 1e-9:
                differences.append({"security_id": security_id, "broker_quantity": broker_quantity, "local_quantity": local_quantity})
        if differences:
            self.set_controls(entry_paused=True)
            self._audit("reconciliation_mismatch", "account", ACCOUNT_ID, {"differences": differences})
        else:
            self._audit("reconciliation_ok", "account", ACCOUNT_ID)
        return {"ok": not differences, "differences": differences, "checked_at": utcnow()}

    def account_summary(self) -> dict:
        account = self.db.one("SELECT * FROM trading_accounts WHERE id=?", (ACCOUNT_ID,)) or {}
        return {
            **account,
            "balances": self.broker.balances(ACCOUNT_ID),
            "positions": self.broker.positions(ACCOUNT_ID),
            "allocations": self.db.all(
                """SELECT a.*,s.name AS strategy_name FROM strategy_allocations a
                   JOIN strategies s ON s.id=a.strategy_id WHERE a.account_id=? ORDER BY s.name,a.currency""",
                (ACCOUNT_ID,),
            ),
            "pending_approvals": int((self.db.one(
                "SELECT COUNT(*) AS count FROM order_intents WHERE status='pending_approval'"
            ) or {}).get("count") or 0),
            "open_orders": int((self.db.one(
                "SELECT COUNT(*) AS count FROM order_intents WHERE status IN ('approved','submitting','submitted','partially_filled')"
            ) or {}).get("count") or 0),
            "health": self.broker.health(ACCOUNT_ID),
        }

    def dashboard(self) -> dict:
        return {
            "account": self.account_summary(),
            "intents": self.list_intents(50),
            "fills": self.db.all(
                """SELECT f.*,q.code,q.name AS security_name,q.market,q.currency,s.name AS strategy_name
                   FROM trade_fills f JOIN order_intents i ON i.id=f.intent_id
                   JOIN strategies s ON s.id=i.strategy_id JOIN securities q ON q.id=f.security_id
                   ORDER BY f.filled_at DESC LIMIT 50"""
            ),
            "audit": self.db.all("SELECT * FROM trading_audit ORDER BY created_at DESC LIMIT 50"),
        }


trading_service = TradingService()
