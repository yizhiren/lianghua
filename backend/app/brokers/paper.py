from __future__ import annotations

import uuid
from datetime import date

from ..database import Database, utcnow
from .base import BrokerAdapter, BrokerFill, BrokerOrderResult, BrokerQuote


class PaperBrokerAdapter(BrokerAdapter):
    name = "paper"

    def __init__(self, database: Database):
        self.db = database

    def health(self, account_id: str) -> dict:
        account = self.db.one("SELECT status,mode FROM trading_accounts WHERE id=?", (account_id,))
        return {"connected": bool(account and account["status"] == "connected"), "adapter": self.name}

    def quote(self, security_id: str) -> BrokerQuote:
        row = self.db.one("SELECT latest_price,quote_time FROM securities WHERE id=?", (security_id,))
        if not row or row["latest_price"] is None:
            raise ValueError("缺少可执行行情")
        return BrokerQuote(security_id, float(row["latest_price"]), row.get("quote_time"))

    def balances(self, account_id: str) -> list[dict]:
        return self.db.all("SELECT * FROM account_balances WHERE account_id=? ORDER BY currency", (account_id,))

    def positions(self, account_id: str) -> list[dict]:
        self._roll_available(account_id)
        return self.db.all(
            """SELECT p.*,s.market,s.code,s.name,s.currency,s.latest_price
               FROM broker_positions p JOIN securities s ON s.id=p.security_id
               WHERE p.account_id=? AND p.quantity>0 ORDER BY s.market,s.code""",
            (account_id,),
        )

    def _roll_available(self, account_id: str) -> None:
        today = date.today().isoformat()
        self.db.execute(
            """UPDATE broker_positions SET available_quantity=quantity,updated_at=?
               WHERE account_id=? AND acquired_date<?
                 AND (security_id LIKE 'SH.%' OR security_id LIKE 'SZ.%')""",
            (utcnow(), account_id, today),
        )

    def _can_fill(self, security_id: str, side: str, limit_price: float) -> tuple[bool, float]:
        price = self.quote(security_id).price
        return ((price <= limit_price) if side == "buy" else (price >= limit_price)), price

    def _apply_fill(self, account_id: str, security_id: str, side: str, quantity: float, price: float) -> None:
        security = self.db.one("SELECT market,currency FROM securities WHERE id=?", (security_id,))
        if not security:
            raise LookupError("股票不存在")
        if side == "sell":
            self._roll_available(account_id)
        currency = security["currency"]
        with self.db.transaction() as connection:
            balance = connection.execute(
                "SELECT * FROM account_balances WHERE account_id=? AND currency=?", (account_id, currency)
            ).fetchone()
            if not balance:
                raise ValueError(f"账户缺少{currency}资金")
            position = connection.execute(
                "SELECT * FROM broker_positions WHERE account_id=? AND security_id=?", (account_id, security_id)
            ).fetchone()
            amount = quantity * price
            if side == "buy":
                if float(balance["available"]) + 1e-9 < amount:
                    raise ValueError("可用资金不足")
                old_quantity = float(position["quantity"]) if position else 0.0
                old_cost = float(position["avg_cost"]) if position else 0.0
                new_quantity = old_quantity + quantity
                avg_cost = (old_quantity * old_cost + amount) / new_quantity
                available_add = quantity if security["market"] == "HK" else 0.0
                if position:
                    connection.execute(
                        """UPDATE broker_positions SET quantity=?,available_quantity=available_quantity+?,avg_cost=?,
                           acquired_date=?,updated_at=? WHERE account_id=? AND security_id=?""",
                        (new_quantity, available_add, avg_cost, date.today().isoformat(), utcnow(), account_id, security_id),
                    )
                else:
                    connection.execute(
                        """INSERT INTO broker_positions(account_id,security_id,quantity,available_quantity,avg_cost,acquired_date,updated_at)
                           VALUES(?,?,?,?,?,?,?)""",
                        (account_id, security_id, quantity, available_add, price, date.today().isoformat(), utcnow()),
                    )
                connection.execute(
                    """UPDATE account_balances SET cash=cash-?,available=available-?,updated_at=?
                       WHERE account_id=? AND currency=?""",
                    (amount, amount, utcnow(), account_id, currency),
                )
            else:
                if not position or float(position["available_quantity"]) + 1e-9 < quantity:
                    raise ValueError("可卖数量不足（可能受T+1限制）")
                new_quantity = float(position["quantity"]) - quantity
                new_available = float(position["available_quantity"]) - quantity
                connection.execute(
                    """UPDATE broker_positions SET quantity=?,available_quantity=?,updated_at=?
                       WHERE account_id=? AND security_id=?""",
                    (new_quantity, new_available, utcnow(), account_id, security_id),
                )
                connection.execute(
                    """UPDATE account_balances SET cash=cash+?,available=available+?,updated_at=?
                       WHERE account_id=? AND currency=?""",
                    (amount, amount, utcnow(), account_id, currency),
                )

    def _fill(self, account_id: str, security_id: str, side: str, quantity: float, limit_price: float) -> BrokerFill | None:
        can_fill, price = self._can_fill(security_id, side, limit_price)
        if not can_fill:
            return None
        self._apply_fill(account_id, security_id, side, quantity, price)
        return BrokerFill(str(uuid.uuid4()), quantity, price, 0.0, utcnow())

    def submit_limit_order(
        self,
        account_id: str,
        security_id: str,
        side: str,
        quantity: float,
        limit_price: float,
    ) -> BrokerOrderResult:
        submitted_at = utcnow()
        broker_order_id = f"paper-{uuid.uuid4()}"
        fill = self._fill(account_id, security_id, side, quantity, limit_price)
        return BrokerOrderResult(
            broker_order_id=broker_order_id,
            status="filled" if fill else "submitted",
            quantity=quantity,
            filled_quantity=fill.quantity if fill else 0.0,
            limit_price=limit_price,
            submitted_at=submitted_at,
            fill=fill,
        )

    def try_fill_order(
        self,
        account_id: str,
        broker_order_id: str,
        security_id: str,
        side: str,
        remaining_quantity: float,
        limit_price: float,
    ) -> BrokerFill | None:
        return self._fill(account_id, security_id, side, remaining_quantity, limit_price)

    def cancel_order(self, account_id: str, broker_order_id: str) -> bool:
        return True
