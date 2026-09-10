from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BrokerQuote:
    security_id: str
    price: float
    quote_time: str | None


@dataclass(frozen=True)
class BrokerFill:
    broker_fill_id: str
    quantity: float
    price: float
    fee: float
    filled_at: str


@dataclass(frozen=True)
class BrokerOrderResult:
    broker_order_id: str
    status: str
    quantity: float
    filled_quantity: float
    limit_price: float
    submitted_at: str
    fill: BrokerFill | None = None
    error: str | None = None


class BrokerAdapter(ABC):
    name: str

    @abstractmethod
    def health(self, account_id: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def quote(self, security_id: str) -> BrokerQuote:
        raise NotImplementedError

    @abstractmethod
    def balances(self, account_id: str) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def positions(self, account_id: str) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def submit_limit_order(
        self,
        account_id: str,
        security_id: str,
        side: str,
        quantity: float,
        limit_price: float,
    ) -> BrokerOrderResult:
        raise NotImplementedError

    @abstractmethod
    def try_fill_order(
        self,
        account_id: str,
        broker_order_id: str,
        security_id: str,
        side: str,
        remaining_quantity: float,
        limit_price: float,
    ) -> BrokerFill | None:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, account_id: str, broker_order_id: str) -> bool:
        raise NotImplementedError
