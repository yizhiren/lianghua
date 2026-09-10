from __future__ import annotations

import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .database import Database, db, utcnow
from .dsl import evaluate_rule, validate_dsl
from .repository import Repository, repo
from .trading import TradingService


class StrategyService:
    ACTION_PRIORITY = {
        "signal_stop": 100,
        "signal_exit": 90,
        "signal_reduce": 80,
        "signal_add": 70,
        "signal_buy": 60,
        "highlight_position": 50,
        "highlight_candidate": 40,
        "signal_hold": 20,
    }

    def __init__(self, database: Database = db, repository: Repository = repo, trading: TradingService | None = None):
        self.db = database
        self.repo = repository
        self.trading = trading or TradingService(database)
        self._lock = threading.Lock()

    def _bars(self, security_id: str) -> list[dict[str, Any]]:
        return self.db.all(
            "SELECT trade_date,open,high,low,close,volume,is_provisional,trade_status,is_st FROM bars WHERE security_id=? ORDER BY trade_date",
            (security_id,),
        )

    def _previous_market_session(self, market: str, value: date) -> date:
        current = value - timedelta(days=1)
        while current.weekday() >= 5 or self.db.one(
            "SELECT 1 FROM market_holidays WHERE market=? AND trade_date=?",
            (market, current.isoformat()),
        ):
            current -= timedelta(days=1)
        return current

    def _market_session_on_or_before(self, market: str, value: date) -> date:
        current = value
        while current.weekday() >= 5 or self.db.one(
            "SELECT 1 FROM market_holidays WHERE market=? AND trade_date=?",
            (market, current.isoformat()),
        ):
            current -= timedelta(days=1)
        return current

    def _latest_completed_dates(self) -> dict[str, str]:
        """Return the completed daily-bar date each market must have for a live signal.

        During a session, today's provisional bar is intentionally ignored and the
        previous market session is required.  After close and on non-trading days,
        the latest quote snapshot date is already the latest completed session.
        """
        local_today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        result: dict[str, str] = {}
        for row in self.db.all("SELECT market,state,quote_time FROM market_status"):
            raw_date = str(row.get("quote_time") or "")[:10]
            try:
                quote_date = date.fromisoformat(raw_date)
            except ValueError:
                continue
            if quote_date == local_today and row.get("state") in {"open", "lunch"}:
                quote_date = self._previous_market_session(row["market"], quote_date)
            else:
                quote_date = self._market_session_on_or_before(row["market"], quote_date)
            result[row["market"]] = quote_date.isoformat()
        missing = sorted({"SH", "SZ", "HK"} - set(result))
        if missing:
            placeholders = ",".join("?" for _item in missing)
            for row in self.db.all(
                f"""SELECT s.market,MAX(b.trade_date) AS trade_date
                    FROM bars b JOIN securities s ON s.id=b.security_id
                    WHERE b.is_provisional=0 AND s.market IN ({placeholders})
                    GROUP BY s.market""",
                tuple(missing),
            ):
                if row.get("trade_date"):
                    result[row["market"]] = row["trade_date"]
        return result

    @staticmethod
    def _has_current_completed_bar(
        item: dict[str, Any],
        bars: list[dict[str, Any]],
        latest_completed_dates: dict[str, str],
    ) -> bool:
        expected = latest_completed_dates.get(str(item.get("market") or ""))
        latest = next(
            (str(bar["trade_date"]) for bar in reversed(bars) if not bool(bar.get("is_provisional", 0))),
            None,
        )
        return bool(expected and latest == expected)

    def _position_items(self, strategy_id: str) -> list[dict[str, Any]]:
        return self.db.all(
            """SELECT p.security_id,p.avg_cost,p.entry_trade_date,COALESCE(entry_bar.low,p.entry_day_low) AS entry_day_low,
               p.add_count,p.last_add_price,
               CASE WHEN p.last_add_at IS NULL THEN NULL
                    ELSE MAX(0,julianday('now')-julianday(p.last_add_at)) END AS days_since_last_add,
               MAX(0,julianday('now')-julianday(COALESCE(p.last_add_at,p.opened_at))) AS days_since_last_buy,
               p.first_add_20d_low,p.first_add_post_low_high,p.first_add_rebound_pct,
               p.first_add_rebound_confirmed,s.market,s.code,s.name
               FROM positions p JOIN securities s ON s.id=p.security_id
               LEFT JOIN bars entry_bar ON entry_bar.security_id=p.security_id AND entry_bar.trade_date=p.entry_trade_date
               WHERE p.strategy_id=?""",
            (strategy_id,),
        )

    def _set_signal(
        self,
        strategy_id: str,
        security_id: str,
        scope: str,
        active: bool,
        reason: str,
        run_id: str,
    ) -> None:
        table = "candidates" if scope == "candidate" else "positions"
        existing = self.db.one(
            f"SELECT signal_active,signal_reason FROM {table} WHERE strategy_id=? AND security_id=?",
            (strategy_id, security_id),
        )
        if not existing:
            return
        old_active = bool(existing["signal_active"])
        old_reason = existing.get("signal_reason")
        self.db.execute(
            f"UPDATE {table} SET signal_active=?,signal_reason=?,signal_updated_at=?,signal_stale=0 WHERE strategy_id=? AND security_id=?",
            (int(active), reason if active else None, utcnow(), strategy_id, security_id),
        )
        if old_active != active or (active and old_reason != reason):
            self.db.execute(
                "INSERT INTO signals(id,strategy_id,security_id,list_type,active,reason,created_at,run_id) VALUES(?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), strategy_id, security_id, scope, int(active), reason, utcnow(), run_id),
            )

    @staticmethod
    def _signal_reason(rule) -> str:
        action_text = {
            "highlight_candidate": "优选关注",
            "highlight_position": "持仓提醒",
            "signal_buy": "建议建仓",
            "signal_add": "建议加仓",
            "signal_hold": "继续持有",
            "signal_reduce": "建议减仓",
            "signal_exit": "建议清仓",
            "signal_stop": "止损清仓",
        }.get(rule.action, "策略提示")
        target = "" if rule.target_position_pct is None else f"至{rule.target_position_pct:g}%仓位"
        return f"{action_text}{target} · {rule.label} · 每15分钟跟踪"

    def execute(self, strategy_id: str, slot: str | None = None) -> dict[str, Any]:
        strategy = self.repo.get_strategy(strategy_id)
        if not strategy:
            raise LookupError("策略不存在")
        if not strategy["active"]:
            raise ValueError("策略已暂停")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("已有策略任务正在执行，请稍后重试")
        run_id = str(uuid.uuid4())
        slot = slot or datetime.now(timezone.utc).strftime("manual-%Y%m%d%H%M%S%f")
        key = f"{strategy_id}:{slot}"
        started = utcnow()
        try:
            try:
                self.db.execute(
                    "INSERT INTO strategy_runs(id,strategy_id,idempotency_key,status,started_at) VALUES(?,?,?,?,?)",
                    (run_id, strategy_id, key, "running", started),
                )
            except Exception:
                existing = self.db.one("SELECT * FROM strategy_runs WHERE idempotency_key=?", (key,))
                if existing:
                    return existing
                raise
            dsl = validate_dsl(strategy["dsl"])
            added = 0
            active_signals = 0
            order_intents_created = 0
            scanned_ids: set[str] = set()
            signal_seen: set[tuple[str, str]] = set()
            signal_matches: dict[tuple[str, str], list[Any]] = {}
            latest_completed_dates = self._latest_completed_dates()
            universe_rules = [rule for rule in dsl.rules if rule.scope == "universe"]
            other_rules = [rule for rule in dsl.rules if rule.scope != "universe"]

            # 自动待定池是本轮扫描快照，不是只增不减的历史合集。
            # 先完整评估所有全市场规则，再一次性同步，避免多条入池规则之间互相删除。
            if universe_rules:
                universe_items = self.db.all(
                    """SELECT id AS security_id,market,code,name,1 AS live_snapshot
                       FROM securities
                       WHERE is_active=1
                         AND (delisting_date IS NULL OR delisting_date>date('now','+8 hours'))"""
                )
                universe_matches: set[str] = set()
                evaluation_counts: dict[str, int] = {}
                for rule in universe_rules:
                    for item in universe_items:
                        security_id = item["security_id"]
                        bars = self._bars(security_id)
                        if len(bars) < 30:
                            continue
                        scanned_ids.add(security_id)
                        evaluation_counts[security_id] = evaluation_counts.get(security_id, 0) + 1
                        if not self._has_current_completed_bar(item, bars, latest_completed_dates):
                            continue
                        if evaluate_rule(rule, bars, context=item, market=item["market"]):
                            universe_matches.add(security_id)
                fully_evaluated = {
                    security_id
                    for security_id, count in evaluation_counts.items()
                    if count == len(universe_rules)
                }
                added, _removed = self.repo.sync_auto_candidates(
                    strategy_id,
                    universe_matches,
                    fully_evaluated,
                )

            # 候选规则必须在快照同步完成后评估，确保新命中和已失效标的
            # 都不会沿用上一个交易日的待定状态。
            for rule in other_rules:
                if rule.scope == "candidate":
                    items = self.db.all(
                        """SELECT c.security_id,s.market,s.code,s.name
                           FROM candidates c JOIN securities s ON s.id=c.security_id WHERE c.strategy_id=?""",
                        (strategy_id,),
                    )
                else:
                    items = self._position_items(strategy_id)
                for item in items:
                    security_id = item["security_id"]
                    bars = self._bars(security_id)
                    if len(bars) < 30:
                        continue
                    scanned_ids.add(security_id)
                    key_scope = "candidate" if rule.scope == "candidate" else "position"
                    signal_key = (key_scope, security_id)
                    signal_seen.add(signal_key)
                    if not self._has_current_completed_bar(item, bars, latest_completed_dates):
                        continue
                    matched = evaluate_rule(rule, bars, context=item, market=item["market"])
                    if matched:
                        signal_matches.setdefault(signal_key, []).append(rule)
            for (scope, security_id) in signal_seen:
                matches = signal_matches.get((scope, security_id), [])
                winner = max(matches, key=lambda item: self.ACTION_PRIORITY.get(item.action, 0)) if matches else None
                reason = self._signal_reason(winner) if winner else "本轮条件未命中"
                self._set_signal(strategy_id, security_id, scope, winner is not None, reason, run_id)
                active_signals += int(winner is not None)
                if winner is not None:
                    before = self.db.one(
                        """SELECT COUNT(*) AS count FROM order_intents
                           WHERE strategy_id=? AND security_id=?""",
                        (strategy_id, security_id),
                    )
                    self.trading.create_intent(strategy_id, winner, security_id, reason)
                    after = self.db.one(
                        """SELECT COUNT(*) AS count FROM order_intents
                           WHERE strategy_id=? AND security_id=?""",
                        (strategy_id, security_id),
                    )
                    order_intents_created += int((after or {}).get("count", 0)) - int((before or {}).get("count", 0))
            self.db.execute(
                """UPDATE strategy_runs SET status='success',finished_at=?,candidates_added=?,signals_active=?,
                   order_intents_created=?,securities_scanned=? WHERE id=?""",
                (utcnow(), added, active_signals, order_intents_created, len(scanned_ids), run_id),
            )
            return self.db.one("SELECT * FROM strategy_runs WHERE id=?", (run_id,)) or {}
        except Exception as error:
            self.db.execute(
                "UPDATE strategy_runs SET status='failed',finished_at=?,error=? WHERE id=?",
                (utcnow(), str(error)[:500], run_id),
            )
            self.db.execute("UPDATE candidates SET signal_stale=1 WHERE strategy_id=?", (strategy_id,))
            self.db.execute("UPDATE positions SET signal_stale=1 WHERE strategy_id=?", (strategy_id,))
            raise
        finally:
            self._lock.release()

    def execute_all(self, slot: str | None = None) -> list[dict[str, Any]]:
        results = []
        for strategy in self.repo.list_strategies():
            if not strategy["active"]:
                continue
            try:
                results.append(self.execute(strategy["id"], slot))
            except Exception as error:
                results.append({"strategy_id": strategy["id"], "status": "failed", "error": str(error)})
        return results


strategy_service = StrategyService()
