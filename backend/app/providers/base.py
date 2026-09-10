from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityInfo:
    id: str
    market: str
    code: str
    name: str
    currency: str


@dataclass(frozen=True)
class Quote:
    security_id: str
    market: str
    price: float
    open: float
    high: float
    low: float
    previous_close: float
    volume: float
    quote_time: str


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_universe_and_quotes(self) -> tuple[list[SecurityInfo], list[Quote]]:
        raise NotImplementedError

    @abstractmethod
    def fetch_daily_bars(self, security: SecurityInfo, start_date: str, end_date: str) -> list[dict]:
        raise NotImplementedError

