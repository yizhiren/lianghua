from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path(os.getenv("DATABASE_PATH", "./data/lianghua.db"))
    market_data_mode: str = os.getenv("MARKET_DATA_MODE", "akshare").lower()
    scan_interval_minutes: int = _int("SCAN_INTERVAL_MINUTES", 15)
    backfill_batch_size: int = _int("BACKFILL_BATCH_SIZE", 100)
    market_history_years: int = max(10, min(30, _int("MARKET_HISTORY_YEARS", 15)))
    market_backfill_retries: int = max(1, min(6, _int("MARKET_BACKFILL_RETRIES", 3)))
    backfill_max_concurrency: int = max(1, min(8, _int("BACKFILL_MAX_CONCURRENCY", 3)))
    ai_base_url: str = os.getenv("AI_BASE_URL", "").rstrip("/")
    ai_model: str = os.getenv("AI_MODEL", "")
    ai_api_key: str = os.getenv("AI_API_KEY", "")
    ai_timeout_seconds: int = _int("AI_TIMEOUT_SECONDS", 60)
    ai_retry_attempts: int = max(1, min(5, _int("AI_RETRY_ATTEMPTS", 3)))
    ai_retry_backoff_seconds: int = max(0, min(30, _int("AI_RETRY_BACKOFF_SECONDS", 1)))
    backend_log_path: Path = Path(os.getenv("BACKEND_LOG_PATH", ".runtime/backend.log"))
    trading_mode: str = os.getenv("TRADING_MODE", "paper").lower()
    broker_adapter: str = os.getenv("BROKER_ADAPTER", "paper").lower()
    live_trading_enabled: bool = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
    paper_cny_cash: float = _float("PAPER_CNY_CASH", 1_000_000)
    paper_hkd_cash: float = _float("PAPER_HKD_CASH", 1_000_000)
    max_security_exposure_pct: float = _float("MAX_SECURITY_EXPOSURE_PCT", 10)
    max_strategy_exposure_pct: float = _float("MAX_STRATEGY_EXPOSURE_PCT", 30)
    daily_loss_pause_pct: float = _float("DAILY_LOSS_PAUSE_PCT", 2)
    training_snapshot_dir: Path = Path(os.getenv("TRAINING_SNAPSHOT_DIR", "./data/training_snapshots"))
    training_weekly_budget: int = max(8, min(1000, _int("TRAINING_WEEKLY_BUDGET", 200)))
    training_smoke_budget: int = max(4, min(100, _int("TRAINING_SMOKE_BUDGET", 12)))
    training_min_years: float = _float("TRAINING_MIN_YEARS", 10)
    training_max_drawdown_pct: float = _float("TRAINING_MAX_DRAWDOWN_PCT", 15)

    @property
    def ai_enabled(self) -> bool:
        return bool(self.ai_base_url and self.ai_model and self.ai_api_key)


settings = Settings()
