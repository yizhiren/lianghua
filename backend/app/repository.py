from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .database import Database, db, utcnow
from .dsl import DEFAULT_DSL, DEFAULT_EXPLANATION
from .logging_config import logger
from .market import market_session_state


def _strategy_detail(database: Database, row: dict[str, Any]) -> dict[str, Any]:
    version = database.one(
        "SELECT * FROM strategy_versions WHERE strategy_id=? AND version=?",
        (row["id"], row["current_version"]),
    )
    candidates = database.all(
        """SELECT c.*,s.code,s.name,s.market,s.currency,s.latest_price,s.quote_time
           FROM candidates c JOIN securities s ON s.id=c.security_id
           WHERE c.strategy_id=? ORDER BY c.signal_active DESC,c.created_at DESC""",
        (row["id"],),
    )
    positions = database.all(
        """SELECT p.*,s.code,s.name,s.market,s.currency,s.latest_price,s.quote_time,
           CASE WHEN s.latest_price IS NULL THEN NULL ELSE (s.latest_price-p.avg_cost)*p.quantity END AS unrealized_pnl,
           CASE WHEN s.latest_price IS NULL THEN NULL ELSE (s.latest_price/p.avg_cost-1)*100 END AS unrealized_pnl_pct
           FROM positions p JOIN securities s ON s.id=p.security_id
           WHERE p.strategy_id=? ORDER BY p.signal_active DESC,p.created_at DESC""",
        (row["id"],),
    )
    add_events = database.all(
        """SELECT id,security_id,quantity,price,occurred_at,detail_json
           FROM position_events WHERE strategy_id=? AND event_type='added'
           ORDER BY occurred_at,id""",
        (row["id"],),
    )
    events_by_security: dict[str, list[dict[str, Any]]] = {}
    for event in add_events:
        events_by_security.setdefault(event["security_id"], []).append(
            {**event, "detail": database.load(event.get("detail_json"), {})}
        )
    position_items = []
    for item in positions:
        rebound_value = item.get("first_add_rebound_confirmed")
        position_items.append(
            {
                **item,
                "signal_active": bool(item["signal_active"]),
                "signal_stale": bool(item["signal_stale"]),
                "first_add_rebound_confirmed": None if rebound_value is None else bool(rebound_value),
                "add_events": events_by_security.get(item["security_id"], []),
            }
        )
    latest_run = database.one(
        "SELECT * FROM strategy_runs WHERE strategy_id=? ORDER BY started_at DESC LIMIT 1", (row["id"],)
    )
    return {
        **row,
        "active": bool(row["active"]),
        "dsl": database.load(version["dsl_json"], DEFAULT_DSL) if version else DEFAULT_DSL,
        "explanation": database.load(version["explanation_json"], DEFAULT_EXPLANATION) if version else DEFAULT_EXPLANATION,
        "candidates": [{**item, "signal_active": bool(item["signal_active"]), "signal_stale": bool(item["signal_stale"])} for item in candidates],
        "positions": position_items,
        "latest_run": latest_run,
    }


class Repository:
    def __init__(self, database: Database = db):
        self.db = database

    def dashboard(self) -> dict[str, Any]:
        strategies = [
            _strategy_detail(self.db, row)
            for row in self.db.all("SELECT * FROM strategies WHERE deleted_at IS NULL ORDER BY created_at")
        ]
        return {
            "strategies": strategies,
            "market_status": [
                {**item, "state": market_session_state(item["market"])}
                for item in self.db.all("SELECT * FROM market_status ORDER BY market")
            ],
            "recent_runs": self.db.all(
                """SELECT r.*,s.name AS strategy_name FROM strategy_runs r JOIN strategies s ON s.id=r.strategy_id
                   ORDER BY r.started_at DESC LIMIT 20"""
            ),
            "recent_signals": self.db.all(
                """SELECT x.*,s.name AS strategy_name,q.code,q.name AS security_name,q.market
                   FROM signals x JOIN strategies s ON s.id=x.strategy_id JOIN securities q ON q.id=x.security_id
                   ORDER BY x.created_at DESC LIMIT 30"""
            ),
        }

    def list_strategies(self) -> list[dict[str, Any]]:
        return [_strategy_detail(self.db, row) for row in self.db.all("SELECT * FROM strategies WHERE deleted_at IS NULL ORDER BY created_at")]

    def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM strategies WHERE id=? AND deleted_at IS NULL", (strategy_id,))
        return _strategy_detail(self.db, row) if row else None

    def create_strategy(self, name: str, description: str, dsl: dict | None = None, explanation: list[str] | None = None) -> dict[str, Any]:
        strategy_id = str(uuid.uuid4())
        now = utcnow()
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO strategies(id,name,description,active,current_version,created_at,updated_at) VALUES(?,?,?,1,1,?,?)",
                (strategy_id, name, description, now, now),
            )
            connection.execute(
                "INSERT INTO strategy_versions(id,strategy_id,version,description,dsl_json,explanation_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), strategy_id, 1, description, self.db.dump(dsl or DEFAULT_DSL), self.db.dump(explanation or DEFAULT_EXPLANATION), now),
            )
        return self.get_strategy(strategy_id)  # type: ignore[return-value]

    def update_strategy(self, strategy_id: str, name: str | None, active: bool | None) -> dict[str, Any] | None:
        existing = self.get_strategy(strategy_id)
        if not existing:
            return None
        self.db.execute(
            "UPDATE strategies SET name=?,active=?,updated_at=? WHERE id=?",
            (name or existing["name"], int(existing["active"] if active is None else active), utcnow(), strategy_id),
        )
        return self.get_strategy(strategy_id)

    def archive_strategy(self, strategy_id: str) -> bool:
        return self.db.execute("UPDATE strategies SET active=0,deleted_at=?,updated_at=? WHERE id=? AND deleted_at IS NULL", (utcnow(), utcnow(), strategy_id)).rowcount > 0

    def activate_version(self, strategy_id: str, description: str, dsl: dict, explanation: list[str]) -> dict[str, Any] | None:
        existing = self.get_strategy(strategy_id)
        if not existing:
            return None
        version = int(existing["current_version"]) + 1
        now = utcnow()
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO strategy_versions(id,strategy_id,version,description,dsl_json,explanation_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), strategy_id, version, description, self.db.dump(dsl), self.db.dump(explanation), now),
            )
            connection.execute(
                "UPDATE strategies SET description=?,current_version=?,updated_at=? WHERE id=?",
                (description, version, now, strategy_id),
            )
        return self.get_strategy(strategy_id)

    def search_securities(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        term = f"%{query.strip()}%"
        return self.db.all(
            "SELECT * FROM securities WHERE is_active=1 AND (id LIKE ? OR code LIKE ? OR name LIKE ?) ORDER BY market,code LIMIT ?",
            (term, term, term, limit),
        )

    def add_candidate(self, strategy_id: str, security_id: str, added_by: str = "manual") -> dict[str, Any]:
        if self.db.one("SELECT 1 FROM positions WHERE strategy_id=? AND security_id=?", (strategy_id, security_id)):
            raise ValueError("股票已在持仓列表")
        if not self.db.one("SELECT 1 FROM securities WHERE id=?", (security_id,)):
            raise LookupError("股票不存在")
        self.db.execute(
            "INSERT OR IGNORE INTO candidates(id,strategy_id,security_id,added_by,created_at) VALUES(?,?,?,?,?)",
            (str(uuid.uuid4()), strategy_id, security_id, added_by, utcnow()),
        )
        return self.get_strategy(strategy_id)  # type: ignore[return-value]

    def sync_auto_candidates(
        self,
        strategy_id: str,
        matched_security_ids: set[str],
        evaluated_security_ids: set[str],
    ) -> tuple[int, int]:
        """将策略自动待定池同步为本轮扫描快照。

        手动加入和从持仓放回的标的由用户维护，不参与自动清理。
        数据不足而未能完成本轮评估的标的也会保留，避免把“未知”误判为“未命中”。
        """
        matched = set(matched_security_ids)
        evaluated = set(evaluated_security_ids)
        now = utcnow()
        with self.db.transaction() as connection:
            existing_rows = connection.execute(
                "SELECT security_id,added_by FROM candidates WHERE strategy_id=?",
                (strategy_id,),
            ).fetchall()
            existing = {str(row["security_id"]): str(row["added_by"]) for row in existing_rows}
            positions = {
                str(row["security_id"])
                for row in connection.execute(
                    "SELECT security_id FROM positions WHERE strategy_id=?",
                    (strategy_id,),
                ).fetchall()
            }
            inactive_auto = {
                str(row["security_id"])
                for row in connection.execute(
                    """SELECT c.security_id
                       FROM candidates c JOIN securities s ON s.id=c.security_id
                       WHERE c.strategy_id=? AND c.added_by='auto' AND s.is_active=0""",
                    (strategy_id,),
                ).fetchall()
            }
            to_add = sorted(matched - set(existing) - positions)
            to_remove = sorted(
                security_id
                for security_id, added_by in existing.items()
                if added_by == "auto"
                and security_id not in matched
                and (security_id in evaluated or security_id in inactive_auto)
            )
            connection.executemany(
                "INSERT INTO candidates(id,strategy_id,security_id,added_by,created_at) VALUES(?,?,?,?,?)",
                [
                    (str(uuid.uuid4()), strategy_id, security_id, "auto", now)
                    for security_id in to_add
                ],
            )
            connection.executemany(
                "DELETE FROM candidates WHERE strategy_id=? AND security_id=? AND added_by='auto'",
                [(strategy_id, security_id) for security_id in to_remove],
            )
        return len(to_add), len(to_remove)

    def delete_candidate(self, strategy_id: str, security_id: str) -> bool:
        return self.db.execute("DELETE FROM candidates WHERE strategy_id=? AND security_id=?", (strategy_id, security_id)).rowcount > 0

    @staticmethod
    def _entry_day_snapshot(connection, security_id: str, occurred_at: str) -> tuple[str, float | None]:
        try:
            occurred = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("建仓时间格式无效") from error
        shanghai = ZoneInfo("Asia/Shanghai")
        if occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=shanghai)
        trade_date = occurred.astimezone(shanghai).date().isoformat()
        bar = connection.execute(
            "SELECT low FROM bars WHERE security_id=? AND trade_date=?",
            (security_id, trade_date),
        ).fetchone()
        return trade_date, None if bar is None else float(bar["low"])

    def add_position(self, strategy_id: str, security_id: str, quantity: float, avg_cost: float, opened_at: str | None = None) -> dict[str, Any]:
        if not self.db.one("SELECT 1 FROM securities WHERE id=?", (security_id,)):
            raise LookupError("股票不存在")
        position_id = str(uuid.uuid4())
        occurred = opened_at or utcnow()
        with self.db.transaction() as connection:
            entry_trade_date, entry_day_low = self._entry_day_snapshot(connection, security_id, occurred)
            connection.execute("DELETE FROM candidates WHERE strategy_id=? AND security_id=?", (strategy_id, security_id))
            # 兼容旧版本可能遗留的孤立补仓事件；新建一次持仓必须从干净状态开始。
            connection.execute(
                "DELETE FROM position_events WHERE strategy_id=? AND security_id=? AND event_type='added'",
                (strategy_id, security_id),
            )
            connection.execute(
                """INSERT INTO positions(
                       id,strategy_id,security_id,quantity,avg_cost,opened_at,created_at,entry_trade_date,entry_day_low
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (position_id, strategy_id, security_id, quantity, avg_cost, occurred, utcnow(), entry_trade_date, entry_day_low),
            )
            connection.execute(
                "INSERT INTO position_events(id,strategy_id,security_id,event_type,quantity,price,occurred_at) VALUES(?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), strategy_id, security_id, "opened", quantity, avg_cost, occurred),
            )
        return self.get_strategy(strategy_id)  # type: ignore[return-value]

    @staticmethod
    def _first_add_snapshot(connection, security_id: str, occurred_at: str) -> dict[str, Any]:
        security = connection.execute("SELECT market FROM securities WHERE id=?", (security_id,)).fetchone()
        if not security:
            raise LookupError("股票不存在")
        try:
            occurred = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("补仓时间格式无效") from error
        shanghai = ZoneInfo("Asia/Shanghai")
        if occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=shanghai)
        local_time = occurred.astimezone(shanghai)
        market_close = time(16, 0) if security["market"] == "HK" else time(15, 0)
        cutoff = local_time.date() if local_time.time() >= market_close else local_time.date() - timedelta(days=1)
        newest_first = connection.execute(
            """SELECT trade_date,low,high FROM bars
               WHERE security_id=? AND is_provisional=0 AND trade_date<=?
               ORDER BY trade_date DESC LIMIT 20""",
            (security_id, cutoff.isoformat()),
        ).fetchall()
        if len(newest_first) < 20:
            raise ValueError(
                f"首次补仓需要20个已完成交易日，当前截至{cutoff.isoformat()}只有{len(newest_first)}个；请先补齐真实历史行情"
            )
        bars = list(reversed(newest_first))
        low_value = min(float(bar["low"]) for bar in bars)
        # 相同最低价重复出现时使用最后一次，防止把两次探底之间的反弹误算成第二次探底后的反弹。
        low_index = max(index for index, bar in enumerate(bars) if float(bar["low"]) == low_value)
        bars_after_low = bars[low_index + 1 :]
        post_low_high = max((float(bar["high"]) for bar in bars_after_low), default=None)
        rebound_pct = None if post_low_high is None else (post_low_high / low_value - 1) * 100
        rebound_confirmed = post_low_high is not None and post_low_high >= low_value * 1.05
        return {
            "window_start": bars[0]["trade_date"],
            "window_end": bars[-1]["trade_date"],
            "low": low_value,
            "low_date": bars[low_index]["trade_date"],
            "post_low_high": post_low_high,
            "rebound_pct": rebound_pct,
            "rebound_confirmed": rebound_confirmed,
            "formula": "最低点之后的日K最高价 >= 20日最低价 * 1.05",
        }

    def add_to_position(
        self,
        strategy_id: str,
        security_id: str,
        quantity: float,
        price: float,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        if quantity <= 0 or price <= 0:
            raise ValueError("补仓数量和成交价必须大于0")
        occurred = occurred_at or utcnow()
        with self.db.transaction() as connection:
            position = connection.execute(
                "SELECT * FROM positions WHERE strategy_id=? AND security_id=?",
                (strategy_id, security_id),
            ).fetchone()
            if not position:
                raise LookupError("持仓股票不存在")
            first_add = int(position["add_count"] or 0) == 0
            snapshot = self._first_add_snapshot(connection, security_id, occurred) if first_add else None
            old_quantity = float(position["quantity"])
            old_cost = float(position["avg_cost"])
            new_quantity = old_quantity + quantity
            new_cost = (old_quantity * old_cost + quantity * price) / new_quantity
            if first_add and snapshot:
                connection.execute(
                    """UPDATE positions SET quantity=?,avg_cost=?,add_count=1,last_add_price=?,last_add_at=?,
                       first_add_at=?,first_add_20d_low=?,first_add_20d_low_date=?,first_add_post_low_high=?,
                       first_add_rebound_pct=?,first_add_rebound_confirmed=?
                       WHERE id=?""",
                    (
                        new_quantity,
                        new_cost,
                        price,
                        occurred,
                        occurred,
                        snapshot["low"],
                        snapshot["low_date"],
                        snapshot["post_low_high"],
                        snapshot["rebound_pct"],
                        int(snapshot["rebound_confirmed"]),
                        position["id"],
                    ),
                )
            else:
                connection.execute(
                    """UPDATE positions SET quantity=?,avg_cost=?,add_count=add_count+1,
                       last_add_price=?,last_add_at=? WHERE id=?""",
                    (new_quantity, new_cost, price, occurred, position["id"]),
                )
            detail = {
                "previous_quantity": old_quantity,
                "previous_avg_cost": old_cost,
                "new_quantity": new_quantity,
                "new_avg_cost": new_cost,
                "first_add_snapshot": snapshot,
            }
            connection.execute(
                """INSERT INTO position_events(id,strategy_id,security_id,event_type,quantity,price,occurred_at,detail_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    strategy_id,
                    security_id,
                    "added",
                    quantity,
                    price,
                    occurred,
                    self.db.dump(detail),
                ),
            )
        logger.info(
            "Position add recorded",
            extra={
                "event": "position_add_recorded",
                "strategy_id": strategy_id,
                "security_id": security_id,
                "quantity": quantity,
                "price": price,
                "occurred_at": occurred,
                "first_add": first_add,
                "rebound_confirmed": snapshot["rebound_confirmed"] if snapshot else None,
            },
        )
        return self.get_strategy(strategy_id)  # type: ignore[return-value]

    def delete_position(self, strategy_id: str, security_id: str, return_to_candidate: bool = False) -> bool:
        position = self.db.one("SELECT * FROM positions WHERE strategy_id=? AND security_id=?", (strategy_id, security_id))
        if not position:
            return False
        with self.db.transaction() as connection:
            connection.execute("DELETE FROM positions WHERE id=?", (position["id"],))
            # 补仓成交记录属于当前持仓生命周期。删除或放回待定后必须清除，
            # 避免同一策略以后重新建仓时继承旧的补仓次数、价格或界面历史。
            connection.execute(
                "DELETE FROM position_events WHERE strategy_id=? AND security_id=? AND event_type='added'",
                (strategy_id, security_id),
            )
            connection.execute(
                "INSERT INTO position_events(id,strategy_id,security_id,event_type,quantity,price,occurred_at) VALUES(?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), strategy_id, security_id, "returned" if return_to_candidate else "removed", position["quantity"], position["avg_cost"], utcnow()),
            )
            if return_to_candidate:
                connection.execute(
                    "INSERT OR IGNORE INTO candidates(id,strategy_id,security_id,added_by,created_at) VALUES(?,?,?,?,?)",
                    (str(uuid.uuid4()), strategy_id, security_id, "returned", utcnow()),
                )
        return True


repo = Repository()
