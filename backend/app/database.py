from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import settings


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS securities (
  id TEXT PRIMARY KEY,
  market TEXT NOT NULL,
  code TEXT NOT NULL,
  name TEXT NOT NULL,
  currency TEXT NOT NULL,
  security_type TEXT NOT NULL DEFAULT 'stock',
  is_active INTEGER NOT NULL DEFAULT 1,
  latest_price REAL,
  quote_time TEXT,
  lot_size INTEGER NOT NULL DEFAULT 100,
  price_tick REAL NOT NULL DEFAULT 0.01,
  UNIQUE(market, code)
);

CREATE TABLE IF NOT EXISTS bars (
  security_id TEXT NOT NULL REFERENCES securities(id),
  trade_date TEXT NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL NOT NULL DEFAULT 0,
  is_provisional INTEGER NOT NULL DEFAULT 0,
  trade_status INTEGER,
  is_st INTEGER,
  data_source TEXT,
  PRIMARY KEY(security_id, trade_date)
);

CREATE TABLE IF NOT EXISTS market_backfill_state (
  security_id TEXT PRIMARY KEY REFERENCES securities(id) ON DELETE CASCADE,
  desired_start TEXT NOT NULL,
  actual_start TEXT,
  actual_end TEXT,
  bar_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  source TEXT,
  last_error TEXT,
  next_retry_at TEXT,
  freshness_target_date TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategies (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  current_version INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS strategy_versions (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL REFERENCES strategies(id),
  version INTEGER NOT NULL,
  description TEXT NOT NULL,
  dsl_json TEXT NOT NULL,
  explanation_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(strategy_id, version)
);

CREATE TABLE IF NOT EXISTS candidates (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL REFERENCES strategies(id),
  security_id TEXT NOT NULL REFERENCES securities(id),
  added_by TEXT NOT NULL DEFAULT 'manual',
  created_at TEXT NOT NULL,
  signal_active INTEGER NOT NULL DEFAULT 0,
  signal_reason TEXT,
  signal_updated_at TEXT,
  signal_stale INTEGER NOT NULL DEFAULT 0,
  UNIQUE(strategy_id, security_id)
);

CREATE TABLE IF NOT EXISTS positions (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL REFERENCES strategies(id),
  security_id TEXT NOT NULL REFERENCES securities(id),
  quantity REAL NOT NULL CHECK(quantity > 0),
  avg_cost REAL NOT NULL CHECK(avg_cost > 0),
  opened_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  signal_active INTEGER NOT NULL DEFAULT 0,
  signal_reason TEXT,
  signal_updated_at TEXT,
  signal_stale INTEGER NOT NULL DEFAULT 0,
  add_count INTEGER NOT NULL DEFAULT 0,
  last_add_price REAL,
  last_add_at TEXT,
  first_add_at TEXT,
  first_add_20d_low REAL,
  first_add_20d_low_date TEXT,
  first_add_post_low_high REAL,
  first_add_rebound_pct REAL,
  first_add_rebound_confirmed INTEGER,
  entry_trade_date TEXT,
  entry_day_low REAL,
  UNIQUE(strategy_id, security_id)
);

CREATE TABLE IF NOT EXISTS position_events (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL,
  security_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  quantity REAL,
  price REAL,
  occurred_at TEXT NOT NULL,
  detail_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS strategy_runs (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL REFERENCES strategies(id),
  idempotency_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  candidates_added INTEGER NOT NULL DEFAULT 0,
  signals_active INTEGER NOT NULL DEFAULT 0,
  order_intents_created INTEGER NOT NULL DEFAULT 0,
  securities_scanned INTEGER NOT NULL DEFAULT 0,
  error TEXT
);

CREATE TABLE IF NOT EXISTS signals (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL,
  security_id TEXT NOT NULL,
  list_type TEXT NOT NULL,
  active INTEGER NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL,
  run_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_dialogues (
  id TEXT PRIMARY KEY,
  strategy_id TEXT REFERENCES strategies(id),
  strategy_name TEXT NOT NULL,
  original_description TEXT NOT NULL,
  normalized_text TEXT NOT NULL DEFAULT '',
  compiled_dsl_json TEXT,
  explanation_json TEXT NOT NULL DEFAULT '[]',
  validation_errors_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'clarifying',
  ai_status TEXT NOT NULL DEFAULT 'not_requested',
  ai_error TEXT,
  activated_version INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_dialogue_messages (
  id TEXT PRIMARY KEY,
  dialogue_id TEXT NOT NULL REFERENCES strategy_dialogues(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_status (
  market TEXT PRIMARY KEY,
  state TEXT NOT NULL,
  quote_time TEXT,
  last_sync_at TEXT,
  data_status TEXT NOT NULL DEFAULT 'demo',
  message TEXT,
  initialized INTEGER NOT NULL DEFAULT 0,
  backfilled INTEGER NOT NULL DEFAULT 0,
  total INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trading_accounts (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  broker TEXT NOT NULL,
  mode TEXT NOT NULL CHECK(mode IN ('paper','shadow','live')),
  status TEXT NOT NULL DEFAULT 'connected',
  armed_until TEXT,
  entry_paused INTEGER NOT NULL DEFAULT 0,
  emergency_stop INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS account_balances (
  account_id TEXT NOT NULL REFERENCES trading_accounts(id),
  currency TEXT NOT NULL,
  cash REAL NOT NULL DEFAULT 0,
  available REAL NOT NULL DEFAULT 0,
  frozen REAL NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(account_id,currency)
);

CREATE TABLE IF NOT EXISTS strategy_allocations (
  account_id TEXT NOT NULL REFERENCES trading_accounts(id),
  strategy_id TEXT NOT NULL REFERENCES strategies(id),
  currency TEXT NOT NULL,
  capital_limit REAL NOT NULL CHECK(capital_limit > 0),
  enabled INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(account_id,strategy_id,currency)
);

CREATE TABLE IF NOT EXISTS broker_positions (
  account_id TEXT NOT NULL REFERENCES trading_accounts(id),
  security_id TEXT NOT NULL REFERENCES securities(id),
  quantity REAL NOT NULL DEFAULT 0,
  available_quantity REAL NOT NULL DEFAULT 0,
  avg_cost REAL NOT NULL DEFAULT 0,
  acquired_date TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(account_id,security_id)
);

CREATE TABLE IF NOT EXISTS order_intents (
  id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  account_id TEXT NOT NULL REFERENCES trading_accounts(id),
  strategy_id TEXT NOT NULL REFERENCES strategies(id),
  rule_id TEXT NOT NULL,
  security_id TEXT NOT NULL REFERENCES securities(id),
  action TEXT NOT NULL,
  side TEXT NOT NULL CHECK(side IN ('buy','sell')),
  status TEXT NOT NULL,
  target_position_pct REAL,
  signal_price REAL NOT NULL,
  signal_date TEXT NOT NULL,
  desired_quantity REAL,
  filled_quantity REAL NOT NULL DEFAULT 0,
  limit_price REAL,
  requires_approval INTEGER NOT NULL DEFAULT 0,
  is_stop INTEGER NOT NULL DEFAULT 0,
  approved_at TEXT,
  approval_deadline TEXT,
  eligible_at TEXT NOT NULL,
  expires_at TEXT,
  blocked_reason TEXT,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS broker_orders (
  id TEXT PRIMARY KEY,
  intent_id TEXT NOT NULL REFERENCES order_intents(id),
  broker_order_id TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity REAL NOT NULL,
  filled_quantity REAL NOT NULL DEFAULT 0,
  limit_price REAL NOT NULL,
  submitted_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  raw_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS trade_fills (
  id TEXT PRIMARY KEY,
  broker_fill_id TEXT NOT NULL UNIQUE,
  broker_order_id TEXT NOT NULL,
  intent_id TEXT NOT NULL REFERENCES order_intents(id),
  security_id TEXT NOT NULL REFERENCES securities(id),
  side TEXT NOT NULL,
  quantity REAL NOT NULL,
  price REAL NOT NULL,
  fee REAL NOT NULL DEFAULT 0,
  filled_at TEXT NOT NULL,
  raw_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS daily_equity_snapshots (
  account_id TEXT NOT NULL REFERENCES trading_accounts(id),
  trade_date TEXT NOT NULL,
  currency TEXT NOT NULL,
  equity REAL NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(account_id,trade_date,currency)
);

CREATE TABLE IF NOT EXISTS trading_audit (
  id TEXT PRIMARY KEY,
  account_id TEXT,
  event_type TEXT NOT NULL,
  entity_type TEXT,
  entity_id TEXT,
  detail_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_holidays (
  market TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  reason TEXT,
  PRIMARY KEY(market,trade_date)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL REFERENCES strategies(id),
  status TEXT NOT NULL,
  start_date TEXT NOT NULL,
  end_date TEXT NOT NULL,
  config_json TEXT NOT NULL,
  summary_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  finished_at TEXT,
  error TEXT
);

CREATE TABLE IF NOT EXISTS backtest_signals (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
  security_id TEXT NOT NULL,
  signal_date TEXT NOT NULL,
  rule_id TEXT NOT NULL,
  action TEXT NOT NULL,
  signal_price REAL NOT NULL,
  reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_trades (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
  security_id TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity REAL NOT NULL,
  price REAL NOT NULL,
  fee REAL NOT NULL DEFAULT 0,
  signal_date TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  action TEXT NOT NULL,
  reason TEXT NOT NULL,
  realized_pnl REAL
);

CREATE TABLE IF NOT EXISTS backtest_equity_curve (
  run_id TEXT NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
  trade_date TEXT NOT NULL,
  cny_equity REAL NOT NULL,
  hkd_equity REAL NOT NULL,
  normalized_equity REAL NOT NULL,
  drawdown_pct REAL NOT NULL,
  PRIMARY KEY(run_id,trade_date)
);

CREATE TABLE IF NOT EXISTS training_data_snapshots (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  start_date TEXT,
  end_date TEXT,
  markets_json TEXT NOT NULL,
  security_count INTEGER NOT NULL DEFAULT 0,
  bar_count INTEGER NOT NULL DEFAULT 0,
  coverage_pct REAL NOT NULL DEFAULT 0,
  content_hash TEXT,
  storage_path TEXT,
  quality_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_campaigns (
  id TEXT PRIMARY KEY,
  track TEXT NOT NULL,
  market TEXT NOT NULL CHECK(market IN ('A','HK')),
  style TEXT NOT NULL CHECK(style IN ('left','right')),
  status TEXT NOT NULL,
  trigger_type TEXT NOT NULL DEFAULT 'manual',
  budget INTEGER NOT NULL,
  completed_trials INTEGER NOT NULL DEFAULT 0,
  snapshot_id TEXT REFERENCES training_data_snapshots(id),
  best_trial_id TEXT,
  progress REAL NOT NULL DEFAULT 0,
  config_json TEXT NOT NULL DEFAULT '{}',
  summary_json TEXT NOT NULL DEFAULT '{}',
  error TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS training_trials (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES training_campaigns(id) ON DELETE CASCADE,
  trial_number INTEGER NOT NULL,
  status TEXT NOT NULL,
  params_json TEXT NOT NULL,
  score REAL,
  summary_json TEXT NOT NULL DEFAULT '{}',
  folds_json TEXT NOT NULL DEFAULT '[]',
  gates_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  finished_at TEXT,
  UNIQUE(campaign_id,trial_number)
);

CREATE TABLE IF NOT EXISTS training_events (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES training_campaigns(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  level TEXT NOT NULL DEFAULT 'info',
  message TEXT NOT NULL,
  detail_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_champions (
  track TEXT PRIMARY KEY,
  strategy_id TEXT REFERENCES strategies(id),
  campaign_id TEXT NOT NULL REFERENCES training_campaigns(id),
  trial_id TEXT NOT NULL REFERENCES training_trials(id),
  status TEXT NOT NULL,
  params_json TEXT NOT NULL,
  gates_json TEXT NOT NULL,
  approved_at TEXT,
  paper_started_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_daily_checks (
  id TEXT PRIMARY KEY,
  track TEXT NOT NULL,
  check_date TEXT NOT NULL,
  status TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(track,check_date)
);

CREATE INDEX IF NOT EXISTS bars_security_date_idx ON bars(security_id, trade_date);
CREATE INDEX IF NOT EXISTS market_backfill_state_status_retry_idx ON market_backfill_state(status,next_retry_at);
CREATE INDEX IF NOT EXISTS runs_strategy_started_idx ON strategy_runs(strategy_id, started_at DESC);
CREATE INDEX IF NOT EXISTS signals_strategy_created_idx ON signals(strategy_id, created_at DESC);
CREATE INDEX IF NOT EXISTS dialogue_messages_dialogue_created_idx ON strategy_dialogue_messages(dialogue_id, created_at);
CREATE INDEX IF NOT EXISTS position_events_strategy_security_time_idx ON position_events(strategy_id, security_id, occurred_at);
CREATE INDEX IF NOT EXISTS order_intents_status_eligible_idx ON order_intents(status, eligible_at);
CREATE INDEX IF NOT EXISTS order_intents_strategy_created_idx ON order_intents(strategy_id, created_at DESC);
CREATE INDEX IF NOT EXISTS broker_orders_intent_idx ON broker_orders(intent_id, submitted_at DESC);
CREATE INDEX IF NOT EXISTS trade_fills_intent_idx ON trade_fills(intent_id, filled_at DESC);
CREATE INDEX IF NOT EXISTS trading_audit_created_idx ON trading_audit(created_at DESC);
CREATE INDEX IF NOT EXISTS backtest_runs_created_idx ON backtest_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS backtest_signals_run_date_idx ON backtest_signals(run_id,signal_date);
CREATE INDEX IF NOT EXISTS backtest_trades_run_date_idx ON backtest_trades(run_id,trade_date);
CREATE INDEX IF NOT EXISTS training_campaigns_created_idx ON training_campaigns(created_at DESC);
CREATE INDEX IF NOT EXISTS training_trials_campaign_score_idx ON training_trials(campaign_id,score DESC);
CREATE INDEX IF NOT EXISTS training_events_campaign_created_idx ON training_events(campaign_id,created_at);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | None = None):
        self.path = path or settings.database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()

    def connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection
        return connection

    def initialize(self) -> None:
        with self._write_lock:
            # WAL is persistent database state. Setting it once during startup avoids
            # every request thread trying to acquire a write lock for the PRAGMA.
            self.connection().execute("PRAGMA journal_mode=WAL")
            self.connection().executescript(SCHEMA)
            dialogue_columns = {
                row["name"] for row in self.connection().execute("PRAGMA table_info(strategy_dialogues)").fetchall()
            }
            if "ai_status" not in dialogue_columns:
                self.connection().execute(
                    "ALTER TABLE strategy_dialogues ADD COLUMN ai_status TEXT NOT NULL DEFAULT 'not_requested'"
                )
            if "ai_error" not in dialogue_columns:
                self.connection().execute("ALTER TABLE strategy_dialogues ADD COLUMN ai_error TEXT")
            security_columns = {
                row["name"] for row in self.connection().execute("PRAGMA table_info(securities)").fetchall()
            }
            for column, definition in {
                "lot_size": "INTEGER NOT NULL DEFAULT 100",
                "price_tick": "REAL NOT NULL DEFAULT 0.01",
                "listing_date": "TEXT",
                "delisting_date": "TEXT",
                "reference_source": "TEXT",
            }.items():
                if column not in security_columns:
                    self.connection().execute(f"ALTER TABLE securities ADD COLUMN {column} {definition}")
            # 旧库通过ALTER TABLE加入带默认值的列时，SQLite读取可返回默认值，但底层记录仍可能被
            # integrity_check视为NULL；显式重写一次可将默认值物化到历史行。
            self.connection().execute(
                "UPDATE securities SET lot_size=COALESCE(lot_size,100),price_tick=COALESCE(price_tick,0.01)"
            )
            bar_columns = {row["name"] for row in self.connection().execute("PRAGMA table_info(bars)").fetchall()}
            for column, definition in {
                "trade_status": "INTEGER",
                "is_st": "INTEGER",
                "data_source": "TEXT",
            }.items():
                if column not in bar_columns:
                    self.connection().execute(f"ALTER TABLE bars ADD COLUMN {column} {definition}")
            backfill_columns = {
                row["name"] for row in self.connection().execute(
                    "PRAGMA table_info(market_backfill_state)"
                ).fetchall()
            }
            if "freshness_target_date" not in backfill_columns:
                self.connection().execute(
                    "ALTER TABLE market_backfill_state ADD COLUMN freshness_target_date TEXT"
                )
            position_columns = {
                row["name"] for row in self.connection().execute("PRAGMA table_info(positions)").fetchall()
            }
            position_migrations = {
                "add_count": "INTEGER NOT NULL DEFAULT 0",
                "last_add_price": "REAL",
                "last_add_at": "TEXT",
                "first_add_at": "TEXT",
                "first_add_20d_low": "REAL",
                "first_add_20d_low_date": "TEXT",
                "first_add_post_low_high": "REAL",
                "first_add_rebound_pct": "REAL",
                "first_add_rebound_confirmed": "INTEGER",
                "entry_trade_date": "TEXT",
                "entry_day_low": "REAL",
            }
            for column, definition in position_migrations.items():
                if column not in position_columns:
                    self.connection().execute(f"ALTER TABLE positions ADD COLUMN {column} {definition}")
            run_columns = {row["name"] for row in self.connection().execute("PRAGMA table_info(strategy_runs)").fetchall()}
            if "order_intents_created" not in run_columns:
                self.connection().execute(
                    "ALTER TABLE strategy_runs ADD COLUMN order_intents_created INTEGER NOT NULL DEFAULT 0"
                )
            self.connection().execute(
                """UPDATE positions SET entry_trade_date=date(opened_at,'+8 hours')
                   WHERE entry_trade_date IS NULL"""
            )
            self.connection().execute(
                """UPDATE positions SET entry_day_low=(
                       SELECT b.low FROM bars b
                       WHERE b.security_id=positions.security_id AND b.trade_date=positions.entry_trade_date
                   )
                   WHERE entry_day_low IS NULL AND entry_trade_date IS NOT NULL"""
            )
            self.connection().commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            connection = self.connection()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            connection = self.connection()
            try:
                cursor = connection.execute(sql, params)
                connection.commit()
                return cursor
            except Exception:
                # SQLite 在约束冲突后仍可能保持事务开启。如果不回滚，
                # 幂等任务的重复键冲突会遗留写锁，阻塞仪表盘读取。
                connection.rollback()
                raise

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        with self._write_lock:
            connection = self.connection()
            try:
                connection.executemany(sql, rows)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        row = self.connection().execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection().execute(sql, params).fetchall()]

    @staticmethod
    def dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def load(value: str | None, default: Any = None) -> Any:
        if value is None:
            return default
        return json.loads(value)


db = Database()
