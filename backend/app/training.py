from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import optuna
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .config import settings
from .database import Database, db, utcnow
from .indicators import adx, atr, bollinger, ema, macd, rsi
from .repository import Repository


TRACKS = {
    "left-A": ("left", "A"),
    "right-A": ("right", "A"),
    "left-HK": ("left", "HK"),
    "right-HK": ("right", "HK"),
}

DATA_GATE_NAMES = ("data_history", "data_coverage", "data_freshness", "point_in_time_status")
GATE_LABELS = {
    "data_history": "历史长度",
    "data_coverage": "K线覆盖",
    "data_freshness": "数据新鲜度",
    "point_in_time_status": "历史状态完整性",
    "max_drawdown": "最大回撤",
    "positive_return": "留出集正收益",
    "sharpe": "留出集夏普",
    "profitable_folds": "开发区间一致盈利",
    "trade_count": "成交样本数",
    "cost_stress": "成本压力测试",
    "excess_return": "超额收益",
    "deflated_sharpe": "去偏夏普可信度",
    "parameter_stability": "参数稳定性",
}
HK_TEMPORARY_NAME_SUFFIXES = ("（新）", "(新)", "－新", "-新", "（旧）", "(旧)", "－旧", "-旧")
TRAINING_ALGORITHM_VERSION = 16
LEFT_LOW_WINDOW = 120
LEFT_TREND_SLOPE_DAYS = 40
MIN_DEVELOPMENT_TRADES = 180
MIN_FOLD_TRADES = 25
TARGET_DEVELOPMENT_TRADES = 600
LEFT_RSI_PERIOD = 6


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return str(value)[:10]
    raise TypeError(f"unsupported JSON value {type(value)!r}")


class TrainingService:
    def __init__(self, database: Database = db):
        self.db = database
        self.repo = Repository(database)
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._feature_cache_snapshot_id: str | None = None
        self._feature_cache_frames: list[pd.DataFrame] | None = None
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    @staticmethod
    def _validation_tier(gates: dict[str, Any]) -> str:
        """Separate strict statistical promotion from safe observation use."""
        if gates.get("passed"):
            return "strict_champion"
        reasons = set(gates.get("reasons") or [])
        if reasons and reasons <= {"deflated_sharpe"}:
            return "paper_observation"
        return "research_only"

    def initialize(self) -> None:
        """Resume campaigns whose worker disappeared during a process restart."""
        interrupted = self.db.all(
            """SELECT id,status FROM training_campaigns
               WHERE status IN ('queued','running') ORDER BY created_at,id"""
        )
        if not interrupted:
            return
        for row in interrupted:
            self.db.execute(
                """UPDATE training_campaigns
                   SET status='queued',error=NULL,finished_at=NULL,cancel_requested=0
                   WHERE id=?""",
                (row["id"],),
            )
            self.db.execute(
                """UPDATE training_trials SET status='interrupted',finished_at=?
                   WHERE campaign_id=? AND status='running'""",
                (utcnow(), row["id"]),
            )
            self._event(
                row["id"], "resuming_after_restart",
                "检测到服务重启，将从已有快照和成功试验继续训练",
                level="warning",
            )
        campaign_ids = [row["id"] for row in interrupted]
        thread = threading.Thread(target=self._run_queue, args=(campaign_ids,), daemon=True)
        thread.start()
        for campaign_id in campaign_ids:
            self._threads[campaign_id] = thread

    def _event(self, campaign_id: str, event_type: str, message: str, detail: dict | None = None, level: str = "info") -> None:
        self.db.execute(
            """INSERT INTO training_events(id,campaign_id,event_type,level,message,detail_json,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), campaign_id, event_type, level, message, self.db.dump(detail or {}), utcnow()),
        )

    def data_quality(self) -> dict[str, Any]:
        result: dict[str, Any] = {"markets": {}, "minimum_years": settings.training_min_years}
        has_backfill_summary = bool((self.db.one(
            "SELECT 1 AS available FROM market_backfill_state LIMIT 1"
        ) or {}).get("available"))
        for market_key, sql_markets in (("A", ("SH", "SZ")), ("HK", ("HK",))):
            placeholders = ",".join("?" for _ in sql_markets)
            if has_backfill_summary:
                # Historical backfill already maintains one compact row per security.
                # Reading it keeps the dashboard responsive while the multi-gigabyte
                # bars table is being extended in the background.
                row = self.db.one(
                    f"""SELECT
                               MIN(CASE WHEN COALESCE(m.bar_count,0)>0 THEN m.actual_start END) AS start_date,
                               MAX(CASE WHEN COALESCE(m.bar_count,0)>0 THEN m.actual_end END) AS end_date,
                               COALESCE(SUM(m.bar_count),0) AS bars,
                               SUM(CASE WHEN s.is_active=1 AND COALESCE(m.bar_count,0)>0 THEN 1 ELSE 0 END) AS active_securities,
                               SUM(CASE WHEN s.is_active=1 AND COALESCE(m.bar_count,0)>0
                                             AND (ms.quote_time IS NULL OR m.freshness_target_date>=substr(ms.quote_time,1,10))
                                             AND (ms.quote_time IS NULL OR m.actual_end>=date(substr(ms.quote_time,1,10),'-30 days'))
                                        THEN 1 ELSE 0 END) AS fresh_active_securities,
                               SUM(CASE WHEN s.is_active=1
                                             AND (ms.quote_time IS NULL OR m.actual_end>=date(substr(ms.quote_time,1,10),'-30 days'))
                                        THEN 1 ELSE 0 END) AS freshness_universe,
                               MAX(COALESCE(substr(ms.quote_time,1,10),m.actual_end)) AS freshness_target_date,
                               SUM(CASE WHEN s.is_active=1 THEN 1 ELSE 0 END) AS universe,
                               SUM(CASE WHEN s.is_active=0 THEN 1 ELSE 0 END) AS inactive_total,
                               SUM(CASE WHEN s.is_active=0 AND COALESCE(m.bar_count,0)>0 THEN 1 ELSE 0 END) AS inactive_with_bars,
                               COALESCE(SUM(CASE WHEN m.source LIKE 'baostock%' THEN m.bar_count ELSE 0 END),0) AS status_bars
                        FROM securities s
                        LEFT JOIN market_backfill_state m ON m.security_id=s.id
                        LEFT JOIN market_status ms ON ms.market=s.market
                        WHERE s.market IN ({placeholders})""",
                    sql_markets,
                ) or {}
                universe = int(row.get("universe") or 0)
                status_row = {"bars": row.get("bars"), "status_bars": row.get("status_bars")}
                inactive_row = {"total": row.get("inactive_total"), "with_bars": row.get("inactive_with_bars")}
            else:
                universe = int((self.db.one(
                    f"SELECT COUNT(*) AS count FROM securities WHERE is_active=1 AND market IN ({placeholders})", sql_markets
                ) or {}).get("count") or 0)
                row = self.db.one(
                    f"""SELECT MIN(b.trade_date) AS start_date,MAX(b.trade_date) AS end_date,
                               COUNT(*) AS bars,
                               COUNT(DISTINCT CASE WHEN s.is_active=1 THEN b.security_id END) AS active_securities
                        FROM bars b JOIN securities s ON s.id=b.security_id
                        WHERE b.is_provisional=0 AND s.market IN ({placeholders})""",
                    sql_markets,
                ) or {}
                status_row = self.db.one(
                    f"""SELECT COUNT(*) AS bars,
                               SUM(CASE WHEN b.is_st IS NOT NULL AND b.trade_status IS NOT NULL THEN 1 ELSE 0 END) AS status_bars
                        FROM bars b JOIN securities s ON s.id=b.security_id
                        WHERE b.is_provisional=0 AND s.market IN ({placeholders})""",
                    sql_markets,
                ) or {}
                inactive_row = self.db.one(
                    f"""SELECT COUNT(*) AS total,
                               SUM(CASE WHEN EXISTS(SELECT 1 FROM bars b WHERE b.security_id=s.id AND b.is_provisional=0) THEN 1 ELSE 0 END) AS with_bars
                        FROM securities s WHERE s.is_active=0 AND s.market IN ({placeholders})""",
                    sql_markets,
                ) or {}
                row["fresh_active_securities"] = row.get("active_securities")
                row["freshness_universe"] = universe
                row["freshness_target_date"] = row.get("end_date")
            start, end = row.get("start_date"), row.get("end_date")
            # Delisted securities that left the market before the training
            # window cannot create survivorship bias inside that window.  BaoStock
            # correctly returns no rows for them, so counting them in the
            # denominator permanently blocked A-share training at 78.9% even
            # though every overlapping delisted security was present.
            if start and has_backfill_summary:
                inactive_row = self.db.one(
                    f"""SELECT COUNT(*) AS total,
                               SUM(CASE WHEN COALESCE(m.bar_count,0)>0 AND m.actual_end>=? THEN 1 ELSE 0 END) AS with_bars
                        FROM securities s
                        LEFT JOIN market_backfill_state m ON m.security_id=s.id
                        WHERE s.is_active=0 AND s.market IN ({placeholders})
                          AND (s.delisting_date IS NULL OR s.delisting_date>=?)""",
                    (start, *sql_markets, start),
                ) or {}
            elif start:
                inactive_row = self.db.one(
                    f"""SELECT COUNT(*) AS total,
                               SUM(CASE WHEN EXISTS(
                                   SELECT 1 FROM bars b
                                   WHERE b.security_id=s.id AND b.is_provisional=0 AND b.trade_date>=?
                               ) THEN 1 ELSE 0 END) AS with_bars
                        FROM securities s
                        WHERE s.is_active=0 AND s.market IN ({placeholders})
                          AND (s.delisting_date IS NULL OR s.delisting_date>=?)""",
                    (start, *sql_markets, start),
                ) or {}
            years = 0.0 if not start or not end else (date.fromisoformat(end) - date.fromisoformat(start)).days / 365.25
            securities = int(row.get("active_securities") or 0)
            coverage = securities / universe * 100 if universe else 0
            fresh_securities = int(row.get("fresh_active_securities") or 0)
            freshness_universe = int(row.get("freshness_universe") or 0)
            freshness_coverage = fresh_securities / freshness_universe * 100 if freshness_universe else 0
            history_gate = years >= settings.training_min_years
            freshness_gate = freshness_coverage >= 95
            status_coverage = (
                int(status_row.get("status_bars") or 0) / int(status_row.get("bars") or 1) * 100
            )
            inactive_total = int(inactive_row.get("total") or 0)
            inactive_with_bars = int(inactive_row.get("with_bars") or 0)
            inactive_coverage = inactive_with_bars / inactive_total * 100 if inactive_total else 0
            point_in_time = (
                inactive_total > 0 and inactive_coverage >= 80
                and (market_key == "HK" or status_coverage >= 95)
            )
            blocking_reasons = [
                key for key, passed in (
                    ("data_history", history_gate),
                    ("data_coverage", coverage >= 95),
                    ("data_freshness", freshness_gate),
                    ("point_in_time_status", point_in_time),
                ) if not passed
            ]
            result["markets"][market_key] = {
                "start_date": start, "end_date": end, "years": years, "bars": int(row.get("bars") or 0),
                "universe": universe, "securities_with_bars": securities, "coverage_pct": coverage,
                "history_gate": history_gate,
                "coverage_gate": coverage >= 95,
                "freshness_target_date": row.get("freshness_target_date"),
                "freshness_universe": freshness_universe,
                "fresh_securities": fresh_securities,
                "freshness_coverage_pct": freshness_coverage,
                "freshness_gate": freshness_gate,
                "status_coverage_pct": status_coverage,
                "inactive_universe": inactive_total,
                "inactive_with_bars": inactive_with_bars,
                "inactive_coverage_pct": inactive_coverage,
                "point_in_time_status": point_in_time,
                "ready_for_training": not blocking_reasons,
                "blocking_reasons": blocking_reasons,
                "warnings": [
                    warning for warning, applies in (
                        (f"历史长度不足{settings.training_min_years:g}年", not history_gate),
                        ("历史K线覆盖不足95%", coverage < 95),
                        ("最新复权日K覆盖不足95%", not freshness_gate),
                        ("历史ST/停牌日状态覆盖不足95%", market_key == "A" and status_coverage < 95),
                        ("历史退市股票覆盖不足80%", inactive_coverage < 80),
                        ("港股历史退市股票主数据尚未完成", market_key == "HK" and inactive_coverage < 80),
                    ) if applies
                ],
            }
        result["ready_for_promotion"] = all(
            item["ready_for_training"]
            for item in result["markets"].values()
        )
        return result

    def ready_tracks(self, tracks: list[str] | None = None) -> list[str]:
        """Return tracks whose market data is fit for formal optimization."""
        selected = list(TRACKS) if tracks is None else tracks
        quality = self.data_quality()["markets"]
        return [track for track in selected if quality[TRACKS[track][1]]["ready_for_training"]]

    def formal_training_due_tracks(self, minimum_interval_days: int = 6) -> list[str]:
        """Start promptly after data becomes ready, then leave iterations to the weekly cadence."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=minimum_interval_days)).isoformat()
        due = []
        for track in self.ready_tracks():
            recent = self.db.one(
                """SELECT id FROM training_campaigns
                   WHERE track=? AND trigger_type IN ('manual','scheduled')
                     AND status IN ('queued','running','research_only','candidate_ready')
                     AND created_at>=?
                   ORDER BY created_at DESC LIMIT 1""",
                (track, cutoff),
            )
            if not recent:
                due.append(track)
        return due

    @staticmethod
    def _prepare_snapshot_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Remove rows that cannot represent an ordinary tradable equity.

        The HK spot feed also exposes temporary consolidation counters, ETFs,
        debt, preference shares and foreign trading counters.  Their price
        series are not continuous ordinary-share histories and previously
        produced single-day returns of hundreds of times inside the benchmark.
        """
        original_rows = len(frame)
        original_securities = int(frame["security_id"].nunique())
        prepared = frame.copy()
        prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
        valid = np.ones(len(prepared), dtype=bool)
        for field in ("open", "high", "low", "close"):
            values = pd.to_numeric(prepared[field], errors="coerce").to_numpy(dtype=float, copy=False)
            valid &= np.isfinite(values) & (values > 0)

        listing = pd.to_datetime(prepared["listing_date"], errors="coerce")
        delisting = pd.to_datetime(prepared["delisting_date"], errors="coerce")
        valid &= listing.isna().to_numpy() | (prepared["trade_date"].to_numpy() >= listing.to_numpy())
        valid &= delisting.isna().to_numpy() | (prepared["trade_date"].to_numpy() <= delisting.to_numpy())

        hk = prepared["market"].eq("HK").to_numpy()
        codes = pd.to_numeric(prepared["code"], errors="coerce").fillna(-1).to_numpy(dtype=int, copy=False)
        hk_equity_code = (
            (codes >= 1) & (codes <= 9999)
            & ~((codes >= 2800) & (codes <= 2849))
            & ~((codes >= 2900) & (codes <= 3199))
            & ~((codes >= 4000) & (codes <= 4199))
            & ~((codes >= 4200) & (codes <= 4299))
            & ~((codes >= 4300) & (codes <= 4699))
            & ~((codes >= 5200) & (codes <= 6029))
            & ~((codes >= 6200) & (codes <= 6499))
        )
        valid &= ~hk | hk_equity_code
        temporary_name = prepared["name"].fillna("").astype(str).str.endswith(HK_TEMPORARY_NAME_SUFFIXES).to_numpy()
        valid &= ~hk | ~temporary_name

        prepared = prepared.loc[valid].sort_values(["security_id", "trade_date"]).reset_index(drop=True)
        stats = {
            "excluded_rows": original_rows - len(prepared),
            "excluded_securities": original_securities - int(prepared["security_id"].nunique()),
        }
        return prepared, stats

    def _create_snapshot(self, markets: tuple[str, ...]) -> tuple[dict[str, Any], pd.DataFrame]:
        placeholders = ",".join("?" for _ in markets)
        snapshot_id = str(uuid.uuid4())
        target = settings.training_snapshot_dir / snapshot_id
        target.mkdir(parents=True, exist_ok=True)
        parquet_path = target / "bars.parquet"
        sql = f"""SELECT b.security_id,s.market,s.code,s.name,s.currency,s.is_active,
                         s.listing_date,s.delisting_date,b.trade_date,b.open,b.high,b.low,b.close,b.volume,
                         b.trade_status,b.is_st,b.data_source
                  FROM bars b JOIN securities s ON s.id=b.security_id
                  WHERE b.is_provisional=0 AND s.market IN ({placeholders})"""
        cursor = self.db.connection().execute(sql, markets)
        columns = [item[0] for item in cursor.description]
        parquet_schema = pa.schema([
            ("security_id", pa.string()), ("market", pa.string()), ("code", pa.string()),
            ("name", pa.string()), ("currency", pa.string()), ("is_active", pa.int64()),
            ("listing_date", pa.string()), ("delisting_date", pa.string()),
            ("trade_date", pa.timestamp("ns")), ("open", pa.float64()),
            ("high", pa.float64()), ("low", pa.float64()), ("close", pa.float64()),
            ("volume", pa.float64()), ("trade_status", pa.int64()),
            ("is_st", pa.int64()), ("data_source", pa.string()),
        ])
        writer: pq.ParquetWriter | None = None
        digest = hashlib.sha256()
        original_rows = 0
        kept_rows = 0
        original_securities: set[str] = set()
        kept_securities: set[str] = set()
        try:
            while True:
                rows = cursor.fetchmany(100_000)
                if not rows:
                    break
                chunk = pd.DataFrame.from_records(
                    [tuple(row) for row in rows], columns=columns,
                )
                original_rows += len(chunk)
                original_securities.update(chunk["security_id"].astype(str).unique())
                prepared, _stats = self._prepare_snapshot_frame(chunk)
                if prepared.empty:
                    continue
                kept_rows += len(prepared)
                kept_securities.update(prepared["security_id"].astype(str).unique())
                digest.update(
                    pd.util.hash_pandas_object(prepared, index=False).values.tobytes()
                )
                table = pa.Table.from_pandas(
                    prepared, schema=parquet_schema, preserve_index=False, safe=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, parquet_schema, compression="snappy")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if original_rows == 0:
            raise ValueError("所选训练轨道没有历史日K")
        if kept_rows == 0 or not parquet_path.is_file():
            raise ValueError("过滤非普通股票和无效行情后没有可训练日K")
        preparation = {
            "excluded_rows": original_rows - kept_rows,
            "excluded_securities": len(original_securities - kept_securities),
        }
        # Training still needs the complete immutable frame, but reading Arrow
        # columns is much cheaper than first materializing millions of sqlite3.Row
        # dictionaries and then converting them to pandas objects.
        frame = pd.read_parquet(parquet_path)
        with duckdb.connect() as connection:
            parquet_stats = connection.execute(
                """SELECT COUNT(*) AS bars, COUNT(DISTINCT security_id) AS securities,
                          COUNT(*) - COUNT(DISTINCT security_id || ':' || CAST(trade_date AS VARCHAR)) AS duplicates
                   FROM read_parquet(?)""",
                [str(parquet_path)],
            ).fetchone()
        content_hash = digest.hexdigest()
        start, end = str(frame["trade_date"].min())[:10], str(frame["trade_date"].max())[:10]
        active = int((self.db.one(
            f"SELECT COUNT(*) AS count FROM securities WHERE is_active=1 AND market IN ({placeholders})", markets
        ) or {}).get("count") or 0)
        security_count = int(frame["security_id"].nunique())
        active_with_bars = int(frame.loc[frame["is_active"] == 1, "security_id"].nunique())
        coverage = active_with_bars / active * 100 if active else 0
        years = (date.fromisoformat(end) - date.fromisoformat(start)).days / 365.25
        market_key = "HK" if markets == ("HK",) else "A"
        market_quality = self.data_quality()["markets"][market_key]
        quality = {
            "years": years, "history_gate": bool(market_quality["history_gate"]),
            "coverage_gate": coverage >= 95,
            "freshness_gate": bool(market_quality["freshness_gate"]),
            "point_in_time_status": bool(market_quality["point_in_time_status"]),
            "survivorship_bias": not bool(market_quality["point_in_time_status"]),
            "status_coverage_pct": market_quality["status_coverage_pct"],
            "inactive_coverage_pct": market_quality["inactive_coverage_pct"],
            "duplicate_rows": int(frame.duplicated(["security_id", "trade_date"]).sum()),
            "preparation": preparation,
            "parquet_validation": {
                "bars": int(parquet_stats[0]), "securities": int(parquet_stats[1]),
                "duplicates": int(parquet_stats[2]), "engine": "duckdb",
            },
        }
        self.db.execute(
            """INSERT INTO training_data_snapshots(
                   id,status,start_date,end_date,markets_json,security_count,bar_count,coverage_pct,
                   content_hash,storage_path,quality_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (snapshot_id, "ready", start, end, self.db.dump(list(markets)), security_count, len(frame), coverage,
             content_hash, str(parquet_path), self.db.dump(quality), utcnow()),
        )
        return self.db.one("SELECT * FROM training_data_snapshots WHERE id=?", (snapshot_id,)) or {}, frame

    def _load_snapshot(self, snapshot_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
        snapshot = self.db.one(
            "SELECT * FROM training_data_snapshots WHERE id=? AND status='ready'",
            (snapshot_id,),
        )
        if not snapshot:
            raise ValueError("训练快照不存在或尚未就绪")
        storage_path = Path(str(snapshot.get("storage_path") or ""))
        if not storage_path.is_file():
            raise ValueError("训练快照文件不存在")
        frame = pd.read_parquet(storage_path)
        expected_rows = int(snapshot.get("bar_count") or 0)
        if expected_rows and len(frame) != expected_rows:
            raise ValueError(f"训练快照行数校验失败：预期{expected_rows}，实际{len(frame)}")
        return snapshot, frame

    def _compatible_batch_snapshot(self, campaign: dict[str, Any]) -> dict[str, Any] | None:
        """Find a ready snapshot made for another track in the same launch batch.

        The two styles for one market consume identical source data.  A durable
        batch id is used instead of a creation-time heuristic so that a fast
        subsequent generation cannot be mislabeled as part of the first launch.
        """
        config = self.db.load(campaign.get("config_json"), {})
        batch_id = str(config.get("batch_id") or "")
        if not batch_id:
            return None
        return self.db.one(
            """SELECT s.* FROM training_campaigns peer
               JOIN training_data_snapshots s ON s.id=peer.snapshot_id
               WHERE peer.id<>? AND peer.market=?
                 AND json_extract(peer.config_json,'$.batch_id')=?
                 AND s.status='ready'
               ORDER BY peer.created_at,peer.id LIMIT 1""",
            (campaign["id"], campaign["market"], batch_id),
        )

    def _latest_unchanged_snapshot(self, campaign: dict[str, Any]) -> dict[str, Any] | None:
        """Reuse a prior immutable snapshot only when source data is unchanged.

        Repeated research generations often run minutes apart. Exporting the
        same 13-million-row market again wastes time and disk, while reusing a
        snapshot after a backfill would silently ignore corrected data.  The
        backfill watermark and latest finalized bar date make that distinction
        explicit.
        """
        markets = ("HK",) if campaign["market"] == "HK" else ("SH", "SZ")
        candidate = self.db.one(
            """SELECT * FROM training_data_snapshots
               WHERE status='ready' AND markets_json=?
               ORDER BY created_at DESC LIMIT 1""",
            (self.db.dump(list(markets)),),
        )
        if not candidate or not Path(str(candidate.get("storage_path") or "")).is_file():
            return None
        placeholders = ",".join("?" for _ in markets)
        source_state = self.db.one(
            f"""SELECT MAX(m.actual_end) AS latest_bar,
                       MAX(m.updated_at) AS watermark
                  FROM market_backfill_state m
                  JOIN securities s ON s.id=m.security_id
                 WHERE s.market IN ({placeholders})""",
            markets,
        ) or {}
        latest_bar = source_state.get("latest_bar")
        if not latest_bar:
            # Small/demo databases may not have a backfill ledger yet.
            latest_bar = (self.db.one(
                f"""SELECT MAX(b.trade_date) AS value FROM bars b
                     JOIN securities s ON s.id=b.security_id
                     WHERE b.is_provisional=0 AND s.market IN ({placeholders})""",
                markets,
            ) or {}).get("value")
        if str(candidate.get("end_date") or "") != str(latest_bar or ""):
            return None
        if source_state.get("watermark") and str(candidate.get("created_at") or "") < str(source_state["watermark"]):
            return None
        return candidate

    @staticmethod
    def _feature_frame(group: pd.DataFrame) -> pd.DataFrame:
        frame = group.sort_values("trade_date").reset_index(drop=True).copy()
        closes = frame["close"].astype(float).tolist()
        bars = frame[["trade_date", "open", "high", "low", "close", "volume"]].to_dict("records")
        macd_values = macd(closes)
        boll = bollinger(closes)
        adx_values = adx(bars)
        # RSI(14) remains the slower entry-quality measure for both strategy
        # styles. RSI(6) is calculated separately for the left-side exit so a
        # recovered holding can be released more responsively, as the deployed
        # DSL does.
        frame["rsi"] = rsi(closes)
        frame["rsi6"] = rsi(closes, LEFT_RSI_PERIOD)
        frame["dif"] = macd_values["dif"]
        frame["dea"] = macd_values["dea"]
        frame["hist"] = macd_values["histogram"]
        frame["boll_mid"] = boll["middle"]
        frame["boll_b"] = boll["percent_b"]
        frame["boll_width"] = boll["bandwidth"]
        frame["atr"] = atr(bars)
        frame["atr_pct"] = frame["atr"] / frame["close"].replace(0, np.nan) * 100
        frame["adx"] = adx_values["value"]
        frame["ema20"] = ema(closes, 20)
        frame["ma60"] = frame["close"].rolling(60).mean()
        frame["ma120"] = frame["close"].rolling(120).mean()
        frame["ma200"] = frame["close"].rolling(200).mean()
        frame["low20"] = frame["close"].rolling(20).min()
        frame["low60"] = frame["close"].rolling(60).min()
        frame["low120"] = frame["close"].rolling(120).min()
        frame["low20_prev"] = frame["close"].shift(1).rolling(20).min()
        frame["low40_prev"] = frame["close"].shift(1).rolling(40).min()
        frame["low60_prev"] = frame["close"].shift(1).rolling(60).min()
        frame["high20"] = frame["close"].shift(1).rolling(20).max()
        frame["high55"] = frame["close"].shift(1).rolling(55).max()
        frame["high120"] = frame["close"].shift(1).rolling(120).max()
        frame["high250"] = frame["close"].shift(1).rolling(250).max()
        frame["vol_ma5"] = frame["volume"].rolling(5).mean()
        frame["vol_ma20"] = frame["volume"].rolling(20).mean()
        frame["vol_ratio"] = frame["vol_ma5"] / frame["vol_ma20"].replace(0, np.nan)
        frame["day_vol_ratio"] = frame["volume"] / frame["vol_ma20"].replace(0, np.nan)
        frame["return1"] = frame["close"].pct_change() * 100
        frame["return20"] = frame["close"].pct_change(20) * 100
        frame["return60"] = frame["close"].pct_change(60) * 100
        frame["recent_abs_return_max"] = frame["return1"].abs().rolling(5).max()
        frame["recent_calendar_gap_max"] = frame["trade_date"].diff().dt.days.rolling(20).max()

        week_key = frame["trade_date"].dt.to_period("W-FRI")
        weekly = frame.assign(week_key=week_key).groupby("week_key", as_index=False).agg(
            trade_date=("trade_date", "max"), open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
        )
        weekly_closes = weekly["close"].astype(float).tolist()
        weekly_macd = macd(weekly_closes)
        weekly_boll = bollinger(weekly_closes, 20, 2)
        weekly["week_dif"] = weekly_macd["dif"]
        weekly["week_dea"] = weekly_macd["dea"]
        weekly["week_boll_mid"] = weekly_boll["middle"]
        weekly["week_trend"] = (weekly["week_dif"] > weekly["week_dea"]) & (weekly["close"] > weekly["week_boll_mid"])
        week_records = weekly.set_index("week_key").to_dict("index")
        week_last = weekly.set_index("week_key")["trade_date"].to_dict()
        week_values = []
        for current_date, period in zip(frame["trade_date"], week_key):
            usable = period if week_last.get(period) == current_date else period - 1
            week_values.append(week_records.get(usable, {}))
        for field in ("week_dif", "week_dea", "week_boll_mid", "week_trend"):
            frame[field] = [item.get(field, np.nan) for item in week_values]
        return frame

    @staticmethod
    def _suggest(trial: optuna.Trial, style: str, market: str = "HK") -> dict[str, Any]:
        if style == "left":
            if market == "A":
                return {
                    "rsi_max": trial.suggest_float("rsi_max", 30, 40),
                    "boll_b_max": trial.suggest_float("boll_b_max", 0.05, 0.20),
                    "low_window": trial.suggest_categorical("low_window", [20, 60]),
                    "low_proximity_pct": trial.suggest_float("low_proximity_pct", 3, 5),
                    "divergence_window": trial.suggest_categorical("divergence_window", [20, 40]),
                    "volume_ratio_max": trial.suggest_float("volume_ratio_max", 1.2, 2.2),
                    "confirm_return_max": trial.suggest_float("confirm_return_max", 3, 8),
                    "trend_floor_ratio": trial.suggest_float("trend_floor_ratio", 0.82, 0.98),
                    "trend_slope_floor": trial.suggest_float("trend_slope_floor", 0.94, 1.02),
                    "adx_max": trial.suggest_float("adx_max", 40, 60),
                    "stop_atr": trial.suggest_float("stop_atr", 2, 6),
                    "exit_rsi": trial.suggest_float("exit_rsi", 55, 75),
                    "max_hold": trial.suggest_categorical("max_hold", [5, 10, 15, 20, 30]),
                }
            return {
                "rsi_max": trial.suggest_float("rsi_max", 32, 44),
                "low_window": trial.suggest_categorical("low_window", [20, 60]),
                "low_proximity_pct": trial.suggest_float("low_proximity_pct", 3, 5),
                "volume_ratio_max": trial.suggest_float("volume_ratio_max", 0.6, 1.2),
                "confirm_return_max": trial.suggest_float("confirm_return_max", 3, 9),
                "trend_floor_ratio": trial.suggest_float("trend_floor_ratio", 0.78, 0.98),
                "trend_slope_floor": trial.suggest_float("trend_slope_floor", 0.92, 1.02),
                "adx_max": trial.suggest_float("adx_max", 40, 60),
                "stop_atr": trial.suggest_float("stop_atr", 2, 6),
                "exit_rsi": trial.suggest_float("exit_rsi", 55, 75),
                "max_hold": trial.suggest_categorical("max_hold", [5, 10, 15, 20, 30, 45]),
            }
        if market == "A":
            return {
                "rsi_min": trial.suggest_float("rsi_min", 40, 65),
                "rsi_max": trial.suggest_float("rsi_max", 70, 90),
                "adx_min": trial.suggest_float("adx_min", 8, 35),
                "volume_ratio_min": trial.suggest_float("volume_ratio_min", 0.8, 2.5),
                "signal_return_max": trial.suggest_float("signal_return_max", 2, 12),
                "stop_atr": trial.suggest_float("stop_atr", 1, 4),
                "trail_atr": trial.suggest_float("trail_atr", 1.5, 5),
                "max_hold": trial.suggest_categorical("max_hold", [30, 60, 90, 120, 150, 180]),
                "breakout_window": trial.suggest_categorical("breakout_window", [20, 55, 120, 250]),
                "channel_exit_window": trial.suggest_categorical("channel_exit_window", [20, 40, 60]),
                "trend_slope_days": trial.suggest_categorical("trend_slope_days", [20, 40, 60, 120]),
                "atr_pct_max": trial.suggest_float("atr_pct_max", 2.5, 8),
                "require_weekly": trial.suggest_categorical("require_weekly", [True, False]),
            }
        return {
            "rsi_min": trial.suggest_float("rsi_min", 45, 58),
            "rsi_max": trial.suggest_float("rsi_max", 68, 82),
            "adx_min": trial.suggest_float("adx_min", 18, 30),
            "volume_ratio_min": trial.suggest_float("volume_ratio_min", 1.0, 1.6),
            "signal_return_max": trial.suggest_float("signal_return_max", 3, 9),
            "stop_atr": trial.suggest_float("stop_atr", 1, 3),
            "trail_atr": trial.suggest_float("trail_atr", 2, 4),
            "max_hold": trial.suggest_int("max_hold", 45, 120, step=15),
            "breakout_window": trial.suggest_categorical("breakout_window", [20, 55, 120]),
            "trend_slope_days": trial.suggest_categorical("trend_slope_days", [40, 60]),
            "atr_pct_max": trial.suggest_float("atr_pct_max", 5, 10),
            "require_weekly": trial.suggest_categorical("require_weekly", [True, False]),
        }

    @staticmethod
    def _distributions(style: str, market: str = "HK") -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the distributions needed to rebuild Optuna after a restart."""
        if style == "left":
            if market == "A":
                return {
                    "rsi_max": optuna.distributions.FloatDistribution(30, 40),
                    "boll_b_max": optuna.distributions.FloatDistribution(0.05, 0.20),
                    "low_window": optuna.distributions.CategoricalDistribution([20, 60]),
                    "low_proximity_pct": optuna.distributions.FloatDistribution(3, 5),
                    "divergence_window": optuna.distributions.CategoricalDistribution([20, 40]),
                    "volume_ratio_max": optuna.distributions.FloatDistribution(1.2, 2.2),
                    "confirm_return_max": optuna.distributions.FloatDistribution(3, 8),
                    "trend_floor_ratio": optuna.distributions.FloatDistribution(0.82, 0.98),
                    "trend_slope_floor": optuna.distributions.FloatDistribution(0.94, 1.02),
                    "adx_max": optuna.distributions.FloatDistribution(40, 60),
                    "stop_atr": optuna.distributions.FloatDistribution(2, 6),
                    "exit_rsi": optuna.distributions.FloatDistribution(55, 75),
                    "max_hold": optuna.distributions.CategoricalDistribution([5, 10, 15, 20, 30]),
                }
            return {
                "rsi_max": optuna.distributions.FloatDistribution(32, 44),
                "low_window": optuna.distributions.CategoricalDistribution([20, 60]),
                "low_proximity_pct": optuna.distributions.FloatDistribution(3, 5),
                "volume_ratio_max": optuna.distributions.FloatDistribution(0.6, 1.2),
                "confirm_return_max": optuna.distributions.FloatDistribution(3, 9),
                "trend_floor_ratio": optuna.distributions.FloatDistribution(0.78, 0.98),
                "trend_slope_floor": optuna.distributions.FloatDistribution(0.92, 1.02),
                "adx_max": optuna.distributions.FloatDistribution(40, 60),
                "stop_atr": optuna.distributions.FloatDistribution(2, 6),
                "exit_rsi": optuna.distributions.FloatDistribution(55, 75),
                "max_hold": optuna.distributions.CategoricalDistribution([5, 10, 15, 20, 30, 45]),
            }
        if market == "A":
            return {
                "rsi_min": optuna.distributions.FloatDistribution(40, 65),
                "rsi_max": optuna.distributions.FloatDistribution(70, 90),
                "adx_min": optuna.distributions.FloatDistribution(8, 35),
                "volume_ratio_min": optuna.distributions.FloatDistribution(0.8, 2.5),
                "signal_return_max": optuna.distributions.FloatDistribution(2, 12),
                "stop_atr": optuna.distributions.FloatDistribution(1, 4),
                "trail_atr": optuna.distributions.FloatDistribution(1.5, 5),
                "max_hold": optuna.distributions.CategoricalDistribution([30, 60, 90, 120, 150, 180]),
                "breakout_window": optuna.distributions.CategoricalDistribution([20, 55, 120, 250]),
                "channel_exit_window": optuna.distributions.CategoricalDistribution([20, 40, 60]),
                "trend_slope_days": optuna.distributions.CategoricalDistribution([20, 40, 60, 120]),
                "atr_pct_max": optuna.distributions.FloatDistribution(2.5, 8),
                "require_weekly": optuna.distributions.CategoricalDistribution([True, False]),
            }
        return {
            "rsi_min": optuna.distributions.FloatDistribution(45, 58),
            "rsi_max": optuna.distributions.FloatDistribution(68, 82),
            "adx_min": optuna.distributions.FloatDistribution(18, 30),
            "volume_ratio_min": optuna.distributions.FloatDistribution(1.0, 1.6),
            "signal_return_max": optuna.distributions.FloatDistribution(3, 9),
            "stop_atr": optuna.distributions.FloatDistribution(1, 3),
            "trail_atr": optuna.distributions.FloatDistribution(2, 4),
            "max_hold": optuna.distributions.IntDistribution(45, 120, step=15),
            "breakout_window": optuna.distributions.CategoricalDistribution([20, 55, 120]),
            "trend_slope_days": optuna.distributions.CategoricalDistribution([40, 60]),
            "atr_pct_max": optuna.distributions.FloatDistribution(5, 10),
            "require_weekly": optuna.distributions.CategoricalDistribution([True, False]),
        }

    @staticmethod
    def _params_match_distributions(
        params: dict[str, Any],
        distributions: dict[str, optuna.distributions.BaseDistribution],
    ) -> bool:
        """Reject legacy/restart parameters that sit outside the active search space.

        Matching keys is not sufficient after a range is tightened: Optuna will
        otherwise accept an enqueued float outside the new bounds and let the
        legacy candidate win the new generation unchanged.
        """
        if set(params) != set(distributions):
            return False
        try:
            return all(
                distribution._contains(distribution.to_internal_repr(params[name]))
                for name, distribution in distributions.items()
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _trades(frame: pd.DataFrame, style: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Replay one security with vectorized entry filtering.

        Parameter optimization calls this function once per security for every
        trial.  Scanning every daily row through ``DataFrame.iloc`` made a
        200-trial market run take several days.  Entry conditions are independent
        until a position opens, so calculate them as NumPy masks first and only
        iterate over actual entry candidates and their bounded exit windows.
        """
        trades: list[dict[str, Any]] = []
        # Both styles now use the 200-day trend as a guardrail.  Keeping the
        # warm-up identical also makes left/right comparisons use the same
        # eligible history.
        start_index = 220
        row_count = len(frame)
        if row_count < start_index + 3:
            return trades

        close = frame["close"].to_numpy(dtype=float, copy=False)
        open_price = frame["open"].to_numpy(dtype=float, copy=False)
        high = frame["high"].to_numpy(dtype=float, copy=False)
        low = frame["low"].to_numpy(dtype=float, copy=False)
        rsi_value = frame["rsi"].to_numpy(dtype=float, copy=False)
        # Old/test frames may not yet carry RSI(6), so keep a compatibility
        # fallback. Production feature snapshots always have both columns.
        exit_rsi_value = (
            frame["rsi6"].to_numpy(dtype=float, copy=False)
            if style == "left" and "rsi6" in frame.columns
            else rsi_value
        )
        atr_value = frame["atr"].to_numpy(dtype=float, copy=False)
        dif = frame["dif"].to_numpy(dtype=float, copy=False)
        histogram = frame["hist"].to_numpy(dtype=float, copy=False)
        adx_value = frame["adx"].to_numpy(dtype=float, copy=False)
        atr_pct = frame["atr_pct"].to_numpy(dtype=float, copy=False)
        raw_market = (
            str(frame["market"].iloc[0])
            if "market" in frame.columns and not frame.empty
            else ("HK" if str(frame["security_id"].iloc[0]).startswith("HK.") else "A")
        )
        # Stored A-share rows carry their exchange (SH/SZ), while campaign
        # configuration and strategy branches use the logical market A.
        # Without normalization, A-share replays silently used HK volume and
        # trend rules even though the generated DSL correctly said A-share.
        market = "HK" if raw_market == "HK" else "A"
        raw_trade_status = pd.to_numeric(frame["trade_status"], errors="coerce").to_numpy(dtype=float, copy=False)
        raw_is_st = pd.to_numeric(frame["is_st"], errors="coerce").to_numpy(dtype=float, copy=False)
        trade_status = np.where(np.isnan(raw_trade_status), 1, raw_trade_status).astype(int)
        is_st = np.where(np.isnan(raw_is_st), 0, raw_is_st).astype(int)
        signal_mask = (
            (trade_status == 1)
            & (np.roll(trade_status, 1) == 1)
            & (is_st == 0)
            & (np.roll(is_st, 1) == 0)
            & ~np.isnan(rsi_value)
            & ~np.isnan(atr_value)
        )
        signal_mask &= (
            (frame["recent_abs_return_max"].to_numpy(dtype=float, copy=False) <= (25 if market == "A" else 80))
            & (frame["recent_calendar_gap_max"].to_numpy(dtype=float, copy=False) <= 14)
        )
        signal_mask[:start_index] = False
        signal_mask[row_count - 2:] = False

        previous_close = np.roll(close, 1)
        volume_ratio = frame["vol_ratio"].to_numpy(dtype=float, copy=False)
        day_volume_ratio = frame["day_vol_ratio"].to_numpy(dtype=float, copy=False)
        entry_volume_ratio = day_volume_ratio if market == "A" else volume_ratio
        return1 = frame["return1"].to_numpy(dtype=float, copy=False)
        return20 = frame["return20"].to_numpy(dtype=float, copy=False)
        return60 = frame["return60"].to_numpy(dtype=float, copy=False)
        ma120 = frame["ma120"].to_numpy(dtype=float, copy=False)
        ma200 = frame["ma200"].to_numpy(dtype=float, copy=False)
        if style == "left":
            low_window = int(params.get("low_window", LEFT_LOW_WINDOW))
            rolling_low = frame[f"low{low_window}"].to_numpy(dtype=float, copy=False)
            previous_low = np.roll(rolling_low, 1)
            previous_rsi = np.roll(rsi_value, 1)
            previous_histogram = np.roll(histogram, 1)
            prior_ma120 = np.roll(ma120, LEFT_TREND_SLOPE_DAYS)
            quality_filter = entry_volume_ratio <= params["volume_ratio_max"]
            if market == "A":
                divergence_window = int(params["divergence_window"])
                prior_histogram_min = (
                    frame["hist"].shift(2).rolling(
                        divergence_window,
                        min_periods=max(5, divergence_window // 2),
                    ).min()
                    .to_numpy(dtype=float, copy=False)
                )
                previous_boll_b = np.roll(
                    frame["boll_b"].to_numpy(dtype=float, copy=False), 1,
                )
                # A-share reversals need one independent low-position quality
                # confirmation. Requiring both made live intersections vanish;
                # requiring neither admitted weak one-day bounces.
                quality_filter &= (
                    (previous_boll_b <= params["boll_b_max"])
                    | (previous_histogram > prior_histogram_min)
                )
            left_entry_mask = (
                (previous_close <= previous_low * (1 + params["low_proximity_pct"] / 100))
                & (previous_rsi < params["rsi_max"])
                & (previous_histogram < 0)
                & (close > previous_close)
                & (histogram > previous_histogram)
                & quality_filter
                & (return1 > 0)
                & (return1 <= params["confirm_return_max"])
                & (close >= ma200 * params["trend_floor_ratio"])
                & (ma120 >= prior_ma120 * params["trend_slope_floor"])
                & (adx_value <= params["adx_max"])
            )
            signal_mask &= left_entry_mask
            histogram_turn_pct = np.divide(
                histogram - previous_histogram,
                close,
                out=np.zeros(row_count, dtype=float),
                where=close != 0,
            ) * 100
            signal_scores = (
                (rsi_value - previous_rsi)
                + histogram_turn_pct * 5
                + return1
                - adx_value * 0.05
            )
        else:
            breakout_window = int(params.get("breakout_window", 55))
            breakout_high = frame[f"high{breakout_window}"].to_numpy(dtype=float, copy=False)
            breakout = (close > breakout_high) & (entry_volume_ratio >= params["volume_ratio_min"])
            prior_ma120 = np.roll(ma120, int(params["trend_slope_days"]))
            signal_mask &= (
                breakout
                & (return1 <= params["signal_return_max"])
                & (rsi_value >= params["rsi_min"])
                & (rsi_value <= params["rsi_max"])
                & (adx_value >= params["adx_min"])
                & (dif > frame["dea"].to_numpy(dtype=float, copy=False))
                & (frame["ma60"].to_numpy(dtype=float, copy=False) > ma120)
                & (close > ma200)
                & (ma120 > prior_ma120)
                & (atr_pct <= params["atr_pct_max"])
            )
            if market == "A":
                signal_mask &= ma200 > np.roll(ma200, 60)
            week_raw = frame["week_trend"].to_numpy(copy=False)
            # An unavailable completed-week value is not trend confirmation.
            week_trend = np.zeros(row_count, dtype=bool)
            known_week = pd.notna(week_raw)
            week_trend[known_week] = week_raw[known_week].astype(bool)
            if params["require_weekly"]:
                signal_mask &= week_trend
            momentum = return20 if market == "A" else return60
            signal_scores = momentum + adx_value + entry_volume_ratio * 10 - atr_pct * 2

        signal_scores = np.nan_to_num(signal_scores, nan=-1e9, posinf=-1e9, neginf=-1e9)

        candidate_indices = np.flatnonzero(signal_mask)
        dates = frame["trade_date"].to_numpy(copy=False)
        security_ids = frame["security_id"].to_numpy(copy=False)
        boll_mid = frame["boll_mid"].to_numpy(dtype=float, copy=False) if style == "left" else None
        channel_floor = None
        if style == "right" and market == "A":
            channel_exit_window = int(params.get("channel_exit_window", 20))
            channel_floor = frame[f"low{channel_exit_window}_prev"].to_numpy(dtype=float, copy=False)
        index = start_index
        while index < row_count - 2:
            candidate_offset = int(np.searchsorted(candidate_indices, index))
            if candidate_offset >= len(candidate_indices):
                break
            signal_index = int(candidate_indices[candidate_offset])
            entry_index = signal_index + 1
            entry_price = float(open_price[entry_index]) * 1.001
            if entry_price > float(close[signal_index]) * 1.01:
                index = signal_index + 1
                continue
            entry_atr = float(atr_value[signal_index])
            highest = entry_price
            exit_index = min(row_count - 1, entry_index + int(params["max_hold"]))
            reason = "max_hold"
            mfe, mae = 0.0, 0.0
            for cursor in range(entry_index, min(row_count - 1, entry_index + int(params["max_hold"])) + 1):
                highest = max(highest, float(high[cursor]))
                mfe = max(mfe, float(high[cursor]) / entry_price - 1)
                mae = min(mae, float(low[cursor]) / entry_price - 1)
                if style == "left":
                    stopped = float(close[cursor]) < entry_price - params["stop_atr"] * entry_atr
                    target = float(exit_rsi_value[cursor]) >= params["exit_rsi"] or float(close[cursor]) >= float(boll_mid[cursor])
                    should_exit = stopped or target
                    exit_reason = "stop" if stopped else "mean_reversion"
                else:
                    trailing = float(close[cursor]) < highest - params["trail_atr"] * float(atr_value[cursor])
                    # The initial stop is fixed from entry ATR.  Recomputing it
                    # with a later ATR made the backtest silently widen or
                    # tighten a risk limit that the deployed DSL keeps fixed.
                    initial = float(close[cursor]) < entry_price - params["stop_atr"] * entry_atr
                    channel_break = (
                        channel_floor is not None
                        and cursor > entry_index + 5
                        and float(close[cursor]) < float(channel_floor[cursor])
                    )
                    weekly_break = (
                        bool(params["require_weekly"])
                        and not bool(week_trend[cursor])
                        and cursor > entry_index + 5
                    )
                    should_exit = trailing or initial or channel_break or weekly_break
                    exit_reason = (
                        "stop" if trailing or initial
                        else "channel_break" if channel_break
                        else "weekly_break"
                    )
                if should_exit:
                    exit_index = min(row_count - 1, cursor + 1)
                    reason = exit_reason
                    break
            exit_price = float(open_price[exit_index]) * 0.999
            net_return = exit_price / entry_price - 1 - 0.0016
            trades.append({
                "security_id": str(security_ids[signal_index]), "entry_date": str(dates[entry_index])[:10],
                "exit_date": str(dates[exit_index])[:10], "return": net_return,
                "duration": exit_index - entry_index, "mfe": mfe, "mae": mae, "reason": reason,
                "signal_score": float(signal_scores[signal_index]),
                "_dates": dates, "_close": close, "_entry_index": entry_index, "_exit_index": exit_index,
                "_entry_price": entry_price,
            })
            index = exit_index + 1
        return trades

    @staticmethod
    def _portfolio_metrics(
        trades: list[dict[str, Any]],
        start_date: str,
        end_date: str,
        stress: float = 0.0,
        include_monthly_returns: bool = False,
    ) -> dict[str, Any]:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        position_fraction = min(0.10, max(0.001, settings.max_security_exposure_pct / 100))
        strategy_fraction = min(1.0, max(position_fraction, settings.max_strategy_exposure_pct / 100))
        max_positions = max(1, int(math.floor(strategy_fraction / position_fraction + 1e-9)))
        selected = []
        active: list[str] = []
        # When more signals arrive than the exposure cap can hold, choose the
        # strongest point-in-time signal.  Sorting by security code made the
        # old backtest's result depend on ticker naming instead of the strategy.
        for trade in sorted(
            trades,
            key=lambda item: (
                item["entry_date"],
                -float(item.get("signal_score") or 0),
                item["security_id"],
            ),
        ):
            if not (start_date <= trade["entry_date"] <= end_date):
                continue
            active = [exit_date for exit_date in active if exit_date >= trade["entry_date"]]
            if len(active) >= max_positions:
                continue
            selected.append(trade); active.append(trade["exit_date"])
        calendar = pd.date_range(start, end, freq="B")
        daily = pd.Series(0.0, index=calendar)
        daily_exposure = pd.Series(0.0, index=calendar)
        has_paths = selected and all(
            all(key in trade for key in ("_dates", "_close", "_entry_index", "_exit_index", "_entry_price"))
            for trade in selected
        )
        if has_paths:
            entries: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            exits: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for trade_number, trade in enumerate(selected):
                entries.setdefault(trade["entry_date"], []).append((trade_number, trade))
                exits.setdefault(trade["exit_date"], []).append((trade_number, trade))
            cash = 1.0
            previous_equity = 1.0
            open_positions: dict[int, dict[str, Any]] = {}
            for mark_day in calendar:
                day_key = mark_day.date().isoformat()
                for trade_number, trade in exits.get(day_key, []):
                    position = open_positions.pop(trade_number, None)
                    if position:
                        terminal_factor = float(np.clip(1 + float(trade["return"]) - stress, 0.05, 2.0))
                        cash += position["notional"] * terminal_factor
                equity_at_open = cash + sum(position["mark"] for position in open_positions.values())
                for trade_number, trade in entries.get(day_key, []):
                    notional = min(cash, max(0.0, equity_at_open * position_fraction))
                    if notional <= 0:
                        continue
                    cash -= notional
                    open_positions[trade_number] = {
                        "trade": trade,
                        "notional": notional,
                        "shares": notional / float(trade["_entry_price"]),
                        "mark": notional,
                        "cursor": int(trade["_entry_index"]),
                    }
                for position in open_positions.values():
                    trade = position["trade"]
                    cursor = int(position["cursor"])
                    dates = trade["_dates"]
                    exit_index = int(trade["_exit_index"])
                    while cursor < exit_index and pd.Timestamp(dates[cursor]).date() <= mark_day.date():
                        if pd.Timestamp(dates[cursor]).date() == mark_day.date():
                            close_value = float(trade["_close"][cursor])
                            raw_factor = close_value / float(trade["_entry_price"])
                            robust_factor = float(np.clip(raw_factor, 0.05, 2.0))
                            position["mark"] = position["notional"] * robust_factor
                        cursor += 1
                    position["cursor"] = cursor
                equity = cash + sum(position["mark"] for position in open_positions.values())
                invested = sum(position["mark"] for position in open_positions.values())
                daily_exposure.loc[mark_day] = invested / equity if equity > 0 else 0
                daily.loc[mark_day] = equity / previous_equity - 1 if previous_equity > 0 else 0
                previous_equity = equity
        else:
            for trade in selected:
                exit_day = pd.Timestamp(trade["exit_date"])
                if exit_day in daily.index:
                    daily.loc[exit_day] += float(np.clip(float(trade["return"]) - stress, -0.95, 1.0)) * position_fraction
                active_days = daily_exposure.loc[trade["entry_date"]:trade["exit_date"]].index
                if len(active_days):
                    daily_exposure.loc[active_days] += position_fraction
            daily_exposure = daily_exposure.clip(upper=strategy_fraction)
        equity = (1 + daily).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1
        years = max((end - start).days / 365.25, 1 / 365.25)
        total_return = float(equity.iloc[-1] - 1) if len(equity) else 0
        std = float(daily.std())
        sharpe = float(daily.mean() / std * math.sqrt(252)) if std > 0 else 0
        downside = float(daily[daily < 0].std())
        sortino = float(daily.mean() / downside * math.sqrt(252)) if downside > 0 else 0
        annual = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
        max_dd = float(drawdown.min()) if len(drawdown) else 0
        calmar = annual / abs(max_dd) if max_dd < 0 else 0
        returns = [float(np.clip(float(item["return"]) - stress, -0.95, 1.0)) for item in selected]
        gross_profit = sum(value for value in returns if value > 0)
        gross_loss = abs(sum(value for value in returns if value < 0))
        skewness = float(daily.skew()) if len(daily) >= 3 else 0.0
        kurtosis = float(daily.kurt()) + 3 if len(daily) >= 4 else 3.0
        if not math.isfinite(skewness):
            skewness = 0.0
        if not math.isfinite(kurtosis):
            kurtosis = 3.0
        result = {
            "trades": len(selected), "total_return_pct": total_return * 100, "annual_return_pct": annual * 100,
            "max_drawdown_pct": max_dd * 100, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "win_rate_pct": sum(value > 0 for value in returns) / len(returns) * 100 if returns else None,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (99 if gross_profit > 0 else 0),
            "turnover": len(selected) * position_fraction * 2 / years,
            "position_exposure_pct": position_fraction * 100,
            "strategy_exposure_cap_pct": strategy_fraction * 100,
            "average_gross_exposure_pct": float(daily_exposure.mean()) * 100 if len(daily_exposure) else 0,
            "observations": len(daily), "return_skewness": skewness, "return_kurtosis": kurtosis,
            "average_mfe_pct": np.mean([item["mfe"] for item in selected]) * 100 if selected else None,
            "average_mae_pct": np.mean([item["mae"] for item in selected]) * 100 if selected else None,
        }
        if include_monthly_returns:
            monthly = (1 + daily).resample("ME").prod() - 1
            result["monthly_returns"] = [float(value) for value in monthly]
        return result

    @staticmethod
    def _benchmark_metrics(feature_frames: list[pd.DataFrame], start_date: str, end_date: str) -> dict[str, Any]:
        """Winsorized equal-weight point-in-time proxy when index history is unavailable."""
        samples = []
        for frame in feature_frames:
            selected = frame.loc[
                (frame["trade_date"] >= pd.Timestamp(start_date))
                & (frame["trade_date"] <= pd.Timestamp(end_date)),
                ["trade_date", "return1"],
            ].dropna()
            if not selected.empty:
                samples.append(selected)
        if not samples:
            return {"total_return_pct": 0.0, "annual_return_pct": 0.0}
        combined = pd.concat(samples, ignore_index=True)
        combined = combined.loc[np.isfinite(combined["return1"])].copy()
        grouped = combined.groupby("trade_date")["return1"]
        lower = grouped.transform(lambda values: values.quantile(0.01))
        upper = grouped.transform(lambda values: values.quantile(0.99))
        combined["robust_return"] = combined["return1"].clip(lower=lower, upper=upper).clip(-95, 100)
        exposure_fraction = min(1.0, max(0.0, settings.max_strategy_exposure_pct / 100))
        market_daily = combined.groupby("trade_date")["robust_return"].mean() / 100
        daily = market_daily * exposure_fraction
        total_return = float((1 + daily).prod() - 1)
        years = max((date.fromisoformat(end_date) - date.fromisoformat(start_date)).days / 365.25, 1 / 365.25)
        annual = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
        return {
            "total_return_pct": total_return * 100,
            "annual_return_pct": annual * 100,
            "winsorization": "cross_section_1_99_pct",
            "exposure_pct": exposure_fraction * 100,
            # Kept only in memory so each candidate can be compared with the
            # same market path at its own realized average gross exposure.
            "_daily_market_returns": [float(value) for value in market_daily],
            "_years": years,
        }

    @staticmethod
    def _benchmark_at_exposure(benchmark: dict[str, Any], exposure_pct: float) -> dict[str, Any]:
        """Scale the market proxy to a candidate's realized average exposure.

        A sparse 30%-capped strategy must not be compared with a market proxy
        that stays 30% invested every day.  The hidden-period benchmark is
        therefore replayed at the strategy's average gross exposure.
        """
        exposure_fraction = min(1.0, max(0.0, float(exposure_pct) / 100))
        market_returns = benchmark.get("_daily_market_returns")
        if market_returns is None:
            # Compatibility path for tests and imported historical records
            # that already contain a public benchmark result only.
            return {
                key: value for key, value in benchmark.items()
                if not str(key).startswith("_")
            }
        daily = np.asarray(market_returns, dtype=float) * exposure_fraction
        total_return = float(np.prod(1 + daily) - 1) if len(daily) else 0.0
        years = max(float(benchmark.get("_years") or len(daily) / 252), 1 / 252)
        annual = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
        return {
            "total_return_pct": total_return * 100,
            "annual_return_pct": annual * 100,
            "winsorization": benchmark.get("winsorization", "cross_section_1_99_pct"),
            "exposure_pct": exposure_fraction * 100,
            "exposure_basis": "strategy_average_gross_exposure",
        }

    @staticmethod
    def _fold_ranges(start: str, end: str) -> tuple[list[tuple[str, str]], tuple[str, str]]:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        days = (last - first).days
        dev_start = first + timedelta(days=int(days * 0.4))
        holdout_start = first + timedelta(days=int(days * 0.8))
        development_days = (holdout_start - dev_start).days
        if development_days < 3:
            raise ValueError("训练开发区间不足3天")
        # Exactly three folds.  The previous floor-division/while loop emitted a
        # fourth 1-2 day remainder fold, which made the sample and regime
        # consistency gates fail for otherwise well-sampled candidates.
        boundaries = [
            dev_start + timedelta(days=(development_days * index) // 3)
            for index in range(4)
        ]
        folds = [
            (boundaries[index].isoformat(), (boundaries[index + 1] - timedelta(days=1)).isoformat())
            for index in range(3)
        ]
        return folds, (holdout_start.isoformat(), last.isoformat())

    @staticmethod
    def _deflated_sharpe_probability(
        metrics: dict[str, Any],
        trial_sharpes: Sequence[float],
        independent_trials: int | None = None,
    ) -> float:
        """Deflated Sharpe probability using the campaign's empirical SR spread.

        DSR requires the variance of the Sharpe ratios actually tried and the
        number of independent trials.  The previous implementation substituted
        the sampling error of one hidden series and treated every TPE proposal
        as independent, which made a 95% pass effectively impossible.  This
        implementation is applied to the development period after all trials
        finish; the hidden holdout remains a one-shot confirmation sample.
        """
        observations = max(2, int(metrics.get("observations") or 0))
        annual_sharpe = float(metrics.get("sharpe") or 0)
        period_sharpe = annual_sharpe / math.sqrt(252)
        skewness = float(metrics.get("return_skewness") or 0)
        kurtosis = max(1.0, float(metrics.get("return_kurtosis") or 3))
        finite_sharpes = np.asarray(
            [float(value) for value in trial_sharpes if math.isfinite(float(value))],
            dtype=float,
        )
        if len(finite_sharpes) < 2:
            return 0.0
        trials = max(2, min(len(finite_sharpes), int(independent_trials or len(finite_sharpes))))
        normal = NormalDist()
        euler_gamma = 0.5772156649015329
        expected_max_z = (
            (1 - euler_gamma) * normal.inv_cdf(1 - 1 / trials)
            + euler_gamma * normal.inv_cdf(1 - 1 / (trials * math.e))
        )
        trial_sharpe_std = float(np.std(finite_sharpes, ddof=1))
        expected_max_period_sharpe = trial_sharpe_std * expected_max_z / math.sqrt(252)
        variance_adjustment = max(
            1e-12,
            1 - skewness * period_sharpe + ((kurtosis - 1) / 4) * period_sharpe ** 2,
        )
        z_score = (
            (period_sharpe - expected_max_period_sharpe)
            * math.sqrt(observations - 1)
            / math.sqrt(variance_adjustment)
        )
        return float(normal.cdf(z_score))

    @staticmethod
    def _effective_trial_count(monthly_return_paths: Sequence[Sequence[float]]) -> tuple[int, float | None]:
        """Estimate independent trials from average strategy-return correlation.

        Bailey and Lopez de Prado's correlation interpolation is N = rho +
        (1-rho)M.  Monthly paths keep the persisted audit record compact while
        retaining the common time axis needed to measure trial dependence.
        """
        paths = [np.asarray(path, dtype=float) for path in monthly_return_paths if len(path) >= 3]
        if len(paths) < 2:
            return max(2, len(monthly_return_paths)), None
        width = min(len(path) for path in paths)
        matrix = np.asarray([path[-width:] for path in paths], dtype=float)
        varying = np.std(matrix, axis=1) > 1e-12
        matrix = matrix[varying]
        if len(matrix) < 2:
            return max(2, len(monthly_return_paths)), None
        correlations = np.corrcoef(matrix)
        upper = correlations[np.triu_indices(len(correlations), k=1)]
        finite = upper[np.isfinite(upper)]
        if not len(finite):
            return max(2, len(monthly_return_paths)), None
        average_correlation = float(np.clip(np.mean(finite), 0.0, 0.99))
        attempted = max(2, len(monthly_return_paths))
        effective = int(round(average_correlation + (1 - average_correlation) * attempted))
        return max(2, min(attempted, effective)), average_correlation

    @staticmethod
    def _refresh_gate_outcome(gates: dict[str, Any]) -> dict[str, Any]:
        gate_names = [
            key for key, value in gates.items()
            if isinstance(value, bool) and key != "passed"
        ]
        gates["reasons"] = [key for key in gate_names if not gates[key]]
        gates["passed"] = not gates["reasons"]
        return gates

    def _finalize_campaign_statistical_gates(self, campaign_id: str) -> list[dict[str, Any]]:
        """Apply DSR only after the full development-trial distribution exists."""
        rows = self.db.all(
            """SELECT * FROM training_trials
               WHERE campaign_id=? AND status='success' AND score IS NOT NULL
               ORDER BY score DESC""",
            (campaign_id,),
        )
        summaries = [self.db.load(row.get("summary_json"), {}) for row in rows]
        trial_sharpes = [
            float((summary.get("development") or {}).get("sharpe") or 0)
            for summary in summaries
        ]
        monthly_paths = [
            list((summary.get("development") or {}).get("monthly_returns") or [])
            for summary in summaries
        ]
        effective_trials, average_correlation = self._effective_trial_count(monthly_paths)
        trial_sharpe_std = float(np.std(trial_sharpes, ddof=1)) if len(trial_sharpes) >= 2 else 0.0
        for row, summary in zip(rows, summaries):
            development = summary.get("development") or {}
            probability = self._deflated_sharpe_probability(
                development, trial_sharpes, effective_trials,
            )
            gates = self.db.load(row.get("gates_json"), {})
            gates["deflated_sharpe"] = probability >= 0.95
            gates["dsr_probability"] = probability
            self._refresh_gate_outcome(gates)
            summary["dsr_probability"] = probability
            summary["dsr_diagnostics"] = {
                "scope": "development",
                "method": "empirical_trial_sharpe_variance_with_correlation_adjusted_trials",
                "attempted_trials": len(trial_sharpes),
                "effective_trials": effective_trials,
                "average_monthly_return_correlation": average_correlation,
                "trial_sharpe_std": trial_sharpe_std,
            }
            self.db.execute(
                "UPDATE training_trials SET summary_json=?,gates_json=? WHERE id=?",
                (self.db.dump(summary), self.db.dump(gates), row["id"]),
            )
            row["summary_json"] = self.db.dump(summary)
            row["gates_json"] = self.db.dump(gates)
        return rows

    @staticmethod
    def _development_score(folds: list[dict[str, Any]], params: dict[str, Any]) -> float:
        """Rank candidates on robust development-fold behavior only.

        Using medians alone allowed one of three folds to lose heavily while the
        other two dominated the objective. The promotion gates require all three
        folds to be profitable, so the optimizer should receive the same signal
        without ever looking at the hidden holdout period.
        """
        values = lambda key: np.asarray([float(item[key]) for item in folds], dtype=float)
        annual = values("annual_return_pct")
        sharpe = values("sharpe")
        sortino = values("sortino")
        calmar = values("calmar")
        drawdown = values("max_drawdown_pct")
        turnover = values("turnover")
        profitable_ratio = float(np.mean(annual > 0))
        total_trades = sum(int(item.get("trades") or 0) for item in folds)
        complexity = sum(
            bool(value) for key, value in params.items()
            if key.startswith("use_") or key.startswith("require_")
        )
        score = (
            float(np.quantile(annual, 0.25)) * 0.25
            + float(np.quantile(sharpe, 0.25)) * 3.0
            + float(np.quantile(sortino, 0.25)) * 1.5
            + float(np.quantile(calmar, 0.25)) * 0.75
            + profitable_ratio * 4.0
            - float(np.median(turnover)) * 0.05
            - float(np.std(annual)) * 0.05
            - max(0.0, -float(np.min(drawdown)) - settings.training_max_drawdown_pct) * 0.5
            - complexity * 0.02
        )
        # Once a candidate clears the formal sample gate, prefer the strategy
        # that proved itself on more independent trades, but cap the bonus so
        # raw signal frequency can never dominate risk-adjusted performance.
        # This prevents the optimizer from camping just above the minimum trade
        # count with a boundary ATR/volume/trend threshold that rarely produces
        # a usable live signal.
        score += min(total_trades, TARGET_DEVELOPMENT_TRADES) / TARGET_DEVELOPMENT_TRADES * 2.0
        if profitable_ratio < 1.0:
            score -= 8.0
        if total_trades < MIN_DEVELOPMENT_TRADES:
            # A candidate below the formal sample gate cannot be promoted.
            # Keep a continuous shortfall term so TPE can still distinguish a
            # near-miss from a rule that almost never trades.
            score -= 10.0 + (MIN_DEVELOPMENT_TRADES - total_trades) * 0.1
        if any(int(item.get("trades") or 0) < MIN_FOLD_TRADES for item in folds):
            score -= 10.0
        return float(score)

    def _evaluate(
        self,
        feature_frames: list[pd.DataFrame],
        style: str,
        params: dict[str, Any],
        start: str,
        end: str,
        quality: dict[str, Any],
        benchmark: dict[str, Any] | None = None,
    ) -> tuple[float, dict, list, dict]:
        trades = [trade for frame in feature_frames for trade in self._trades(frame, style, params)]
        folds, holdout_range = self._fold_ranges(start, end)
        fold_metrics = [self._portfolio_metrics(trades, fold_start, fold_end) for fold_start, fold_end in folds]
        development = self._portfolio_metrics(
            trades, folds[0][0], folds[-1][1], include_monthly_returns=True,
        )
        holdout = self._portfolio_metrics(trades, *holdout_range)
        benchmark_base = benchmark or self._benchmark_metrics(feature_frames, *holdout_range)
        matched_benchmark = self._benchmark_at_exposure(
            benchmark_base,
            float(holdout.get("average_gross_exposure_pct") or 0),
        )
        if not any(item["trades"] > 0 for item in fold_metrics):
            gates = {
                "data_history": bool(quality.get("history_gate")),
                "data_coverage": bool(quality.get("coverage_gate")),
                "data_freshness": bool(quality.get("freshness_gate")),
                "point_in_time_status": bool(quality.get("point_in_time_status")),
                "max_drawdown": False, "positive_return": False, "sharpe": False,
                "profitable_folds": False, "trade_count": False, "cost_stress": False,
                "excess_return": False,
            }
            gates["reasons"] = [key for key, passed in gates.items() if not passed]
            gates["passed"] = False
            return -999.0, {
                "development": development, "holdout": holdout,
                "total_generated_trades": len(trades),
                "benchmark": {**matched_benchmark, "kind": "equal_weight_market_proxy_exposure_matched"},
                "annual_excess_return_pct": holdout["annual_return_pct"] - matched_benchmark["annual_return_pct"],
                "dsr_probability": None,
            }, fold_metrics, gates
        # Every chronological development fold counts.  Dropping zero-trade
        # folds made a parameter set with one lucky trade in one regime look
        # like "100% profitable folds" and directed TPE toward vanishingly rare
        # signals.  A zero-trade fold is evidence that the rule did not work in
        # that regime, not missing data.
        profitable_ratio = sum(item["total_return_pct"] > 0 for item in fold_metrics) / len(fold_metrics)
        score = self._development_score(fold_metrics, params)
        stress = self._portfolio_metrics(trades, *holdout_range, stress=0.004)
        total_oos_trades = sum(item["trades"] for item in fold_metrics)
        gates = {
            "data_history": bool(quality.get("history_gate")), "data_coverage": bool(quality.get("coverage_gate")),
            "data_freshness": bool(quality.get("freshness_gate")),
            "point_in_time_status": bool(quality.get("point_in_time_status")),
            "max_drawdown": holdout["max_drawdown_pct"] >= -settings.training_max_drawdown_pct,
            "positive_return": holdout["total_return_pct"] > 0, "sharpe": holdout["sharpe"] >= 0.8,
            "profitable_folds": profitable_ratio >= 0.7,
            "trade_count": total_oos_trades >= MIN_DEVELOPMENT_TRADES
            and all(item["trades"] >= MIN_FOLD_TRADES for item in fold_metrics),
            "cost_stress": stress["total_return_pct"] > 0 and stress["max_drawdown_pct"] >= -18,
            "excess_return": holdout["annual_return_pct"] > matched_benchmark["annual_return_pct"],
        }
        reasons = [key for key, passed in gates.items() if not passed]
        gates.update({"passed": not reasons, "reasons": reasons})
        summary = {
            "development": development, "holdout": holdout, "stress": stress,
            "total_generated_trades": len(trades),
            "profitable_fold_ratio": profitable_ratio, "dsr_probability": None,
            "benchmark": {**matched_benchmark, "kind": "equal_weight_market_proxy_exposure_matched"},
            "annual_excess_return_pct": holdout["annual_return_pct"] - matched_benchmark["annual_return_pct"],
        }
        return score, summary, fold_metrics, gates

    def _stability_report(
        self,
        feature_frames: list[pd.DataFrame],
        style: str,
        params: dict[str, Any],
        start: str,
        end: str,
        quality: dict[str, Any],
        best_score: float,
        benchmark: dict[str, Any] | None = None,
        distributions: dict[str, optuna.distributions.BaseDistribution] | None = None,
    ) -> dict[str, Any]:
        """Test one local parameter move at a time around the winner.

        Moving every continuous threshold in the same direction produced five
        points on one arbitrary diagonal through a high-dimensional search
        space.  It could label a broad plateau unstable merely because several
        individually harmless changes were compounded.  One-factor neighbors
        measure the intended local robustness and use each search range to make
        perturbations comparable.
        """
        distributions = distributions or self._distributions(style)
        neighbors: list[dict[str, Any]] = [dict(params)]
        for key, distribution in distributions.items():
            value = params.get(key)
            candidates: list[Any] = []
            if isinstance(distribution, optuna.distributions.FloatDistribution) and isinstance(value, (int, float)):
                delta = (float(distribution.high) - float(distribution.low)) * 0.05
                candidates = [
                    max(float(distribution.low), float(value) - delta),
                    min(float(distribution.high), float(value) + delta),
                ]
            elif isinstance(distribution, optuna.distributions.IntDistribution) and isinstance(value, int):
                step = int(distribution.step or 1)
                candidates = [
                    max(int(distribution.low), value - step),
                    min(int(distribution.high), value + step),
                ]
            for candidate in candidates:
                if candidate == value:
                    continue
                neighbor = dict(params)
                neighbor[key] = candidate
                neighbors.append(neighbor)

        scores: list[float] = []
        for neighbor in neighbors:
            score, _summary, _folds, _gates = self._evaluate(
                feature_frames, style, neighbor, start, end, quality,
                benchmark,
            )
            scores.append(float(score))
        tolerance = max(0.5, abs(best_score) * 0.2)
        stable_count = sum(score >= best_score - tolerance for score in scores)
        required_count = max(3, math.ceil(len(scores) * 0.7))
        return {
            "passed": stable_count >= required_count,
            "stable_neighbors": stable_count,
            "total_neighbors": len(scores),
            "required_neighbors": required_count,
            "scores": scores,
            "tolerance": tolerance,
            "method": "one_parameter_at_a_time_5pct_range",
        }

    def start_campaigns(self, tracks: list[str], budget: int | None = None, trigger_type: str = "manual") -> list[dict[str, Any]]:
        unknown = [track for track in tracks if track not in TRACKS]
        if unknown:
            raise ValueError(f"未知训练轨道：{','.join(unknown)}")
        if not tracks:
            raise ValueError("至少选择一条训练轨道")
        if trigger_type != "smoke":
            quality = self.data_quality()["markets"]
            blocked = {
                track: quality[TRACKS[track][1]]["warnings"]
                for track in tracks
                if not quality[TRACKS[track][1]]["ready_for_training"]
            }
            if blocked:
                detail = "；".join(
                    f"{track}：{'、'.join(warnings) or '数据门槛未满足'}"
                    for track, warnings in blocked.items()
                )
                raise RuntimeError(f"正式训练暂缓，先完成数据建设（{detail}）。可使用快速训练验证流程。")
        if any(row["status"] in {"queued", "running"} for row in self.db.all("SELECT status FROM training_campaigns")):
            raise RuntimeError("已有训练任务正在运行")
        actual_budget = budget or (settings.training_smoke_budget if trigger_type == "smoke" else settings.training_weekly_budget)
        batch_id = str(uuid.uuid4())
        campaigns = []
        for track in tracks:
            style, market = TRACKS[track]
            campaign_id = str(uuid.uuid4())
            generation = int((self.db.one(
                "SELECT COUNT(*) AS count FROM training_campaigns WHERE track=?", (track,)
            ) or {}).get("count") or 0) + 1
            self.db.execute(
                """INSERT INTO training_campaigns(
                       id,track,market,style,status,trigger_type,budget,config_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (campaign_id, track, market, style, "queued", trigger_type, actual_budget,
                 self.db.dump({
                     "max_drawdown_pct": settings.training_max_drawdown_pct,
                     "generation": generation,
                     "algorithm_version": TRAINING_ALGORITHM_VERSION,
                     "batch_id": batch_id,
                 }), utcnow()),
            )
            campaigns.append(self.campaign(campaign_id))
        thread = threading.Thread(target=self._run_queue, args=([item["id"] for item in campaigns],), daemon=True)
        thread.start()
        for item in campaigns:
            self._threads[item["id"]] = thread
        return campaigns

    def _run_queue(self, campaign_ids: list[str]) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            for campaign_id in campaign_ids:
                campaign = self.db.one("SELECT * FROM training_campaigns WHERE id=?", (campaign_id,)) or {}
                if campaign.get("cancel_requested"):
                    self.db.execute("UPDATE training_campaigns SET status='canceled',finished_at=? WHERE id=?", (utcnow(), campaign_id)); continue
                try:
                    self._run_campaign(campaign_id)
                except Exception as error:
                    self.db.execute(
                        "UPDATE training_campaigns SET status='failed',error=?,finished_at=? WHERE id=?",
                        (str(error)[:500], utcnow(), campaign_id),
                    )
                    self._event(campaign_id, "failed", f"训练失败：{error}", level="error")
        finally:
            self._lock.release()

    def _run_campaign(self, campaign_id: str) -> None:
        campaign = self.db.one("SELECT * FROM training_campaigns WHERE id=?", (campaign_id,)) or {}
        markets = ("HK",) if campaign["market"] == "HK" else ("SH", "SZ")
        existing_snapshot_id = str(campaign.get("snapshot_id") or "")
        self.db.execute(
            """UPDATE training_campaigns
               SET status='running',started_at=COALESCE(started_at,?),error=NULL,finished_at=NULL
               WHERE id=?""",
            (utcnow(), campaign_id),
        )
        reused_snapshot = False
        if not existing_snapshot_id:
            reusable = self._compatible_batch_snapshot(campaign)
            reuse_message = f"复用同批次{campaign['market']}市场不可变数据快照"
            if not reusable:
                reusable = self._latest_unchanged_snapshot(campaign)
                reuse_message = f"{campaign['market']}市场数据未变化，复用最近不可变快照"
            if reusable:
                existing_snapshot_id = str(reusable["id"])
                reused_snapshot = True
                self.db.execute(
                    "UPDATE training_campaigns SET snapshot_id=? WHERE id=?",
                    (existing_snapshot_id, campaign_id),
                )
                self._event(
                    campaign_id, "snapshot_reused",
                    reuse_message,
                    {"snapshot_id": existing_snapshot_id},
                )
        if existing_snapshot_id:
            if not reused_snapshot:
                self._event(campaign_id, "resumed", f"继续训练{campaign['track']}")
            snapshot, frame = self._load_snapshot(existing_snapshot_id)
        else:
            self._event(campaign_id, "started", f"开始训练{campaign['track']}")
            snapshot, frame = self._create_snapshot(markets)
            self.db.execute("UPDATE training_campaigns SET snapshot_id=? WHERE id=?", (snapshot["id"], campaign_id))
        quality = self.db.load(snapshot.get("quality_json"), {})
        if self._feature_cache_snapshot_id == snapshot["id"] and self._feature_cache_frames:
            feature_frames = self._feature_cache_frames
            self._event(
                campaign_id, "feature_cache_reused",
                f"复用同一快照已构建的技术特征：{len(feature_frames):,}只股票",
                {"security_count": len(feature_frames), "snapshot_id": snapshot["id"]},
            )
        else:
            self._event(
                campaign_id, "feature_build_started",
                f"正在为{int(snapshot.get('security_count') or 0):,}只股票构建无未来数据的技术特征",
            )
            feature_frames = [self._feature_frame(group) for _, group in frame.groupby("security_id") if len(group) >= 150]
            if not feature_frames:
                raise ValueError("没有至少150根日K的股票")
            self._feature_cache_snapshot_id = str(snapshot["id"])
            self._feature_cache_frames = feature_frames
            self._event(
                campaign_id, "feature_build_completed",
                f"技术特征构建完成：{len(feature_frames):,}只股票",
                {"security_count": len(feature_frames)},
            )
        # The immutable raw frame is no longer needed after feature creation;
        # releasing it keeps enough memory available to cache features for the
        # paired left/right track.
        del frame
        start, end = snapshot["start_date"], snapshot["end_date"]
        style = campaign["style"]
        _folds, holdout_range = self._fold_ranges(start, end)
        holdout_benchmark = self._benchmark_metrics(feature_frames, *holdout_range)

        successful_rows = self.db.all(
            """SELECT * FROM training_trials
               WHERE campaign_id=? AND status='success' AND score IS NOT NULL
               ORDER BY trial_number""",
            (campaign_id,),
        )
        completed = [len(successful_rows)]
        max_trial_row = self.db.one(
            "SELECT MAX(trial_number) AS number FROM training_trials WHERE campaign_id=?",
            (campaign_id,),
        ) or {}
        next_trial_number = int(max_trial_row.get("number") if max_trial_row.get("number") is not None else -1) + 1

        sampler_seed = int(uuid.UUID(campaign_id)) % (2 ** 31 - 1)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=sampler_seed))
        distributions = self._distributions(style, campaign["market"])
        for previous in successful_rows:
            previous_params = self.db.load(previous.get("params_json"), {})
            if not self._params_match_distributions(previous_params, distributions):
                continue
            try:
                study.add_trial(optuna.trial.create_trial(
                    params=previous_params,
                    distributions=distributions,
                    value=float(previous["score"]),
                ))
            except (TypeError, ValueError):
                # Preserve audit history if a future release changes the search
                # space; new trials can still complete the campaign.
                continue
        existing_study_trials = len(study.trials)

        interrupted_rows = self.db.all(
            """SELECT params_json FROM training_trials
               WHERE campaign_id=? AND status='interrupted' ORDER BY trial_number""",
            (campaign_id,),
        )
        for interrupted in interrupted_rows:
            params = self.db.load(interrupted.get("params_json"), {})
            if self._params_match_distributions(params, distributions):
                study.enqueue_trial(params)
        incumbent_before_training = self.db.one(
            "SELECT params_json FROM strategy_champions WHERE track=?", (campaign["track"],)
        )
        if not successful_rows and not interrupted_rows and incumbent_before_training:
            incumbent_params = self.db.load(incumbent_before_training.get("params_json"), {})
            if self._params_match_distributions(incumbent_params, distributions):
                study.enqueue_trial(incumbent_params)

        def objective(trial: optuna.Trial) -> float:
            current = self.db.one("SELECT cancel_requested FROM training_campaigns WHERE id=?", (campaign_id,)) or {}
            if current.get("cancel_requested"):
                trial.study.stop()
                raise optuna.TrialPruned("cancel requested")
            params = self._suggest(trial, style, campaign["market"])
            trial_id = str(uuid.uuid4())
            trial_number = next_trial_number + max(0, trial.number - existing_study_trials)
            self.db.execute(
                "INSERT INTO training_trials(id,campaign_id,trial_number,status,params_json,created_at) VALUES(?,?,?,?,?,?)",
                (trial_id, campaign_id, trial_number, "running", self.db.dump(params), utcnow()),
            )
            try:
                score, summary, folds, gates = self._evaluate(
                    feature_frames, style, params, start, end, quality,
                    holdout_benchmark,
                )
                self.db.execute(
                    """UPDATE training_trials SET status='success',score=?,summary_json=?,folds_json=?,gates_json=?,finished_at=? WHERE id=?""",
                    (score, json.dumps(summary, ensure_ascii=False, default=_json_safe), json.dumps(folds, ensure_ascii=False, default=_json_safe),
                     json.dumps(gates, ensure_ascii=False, default=_json_safe), utcnow(), trial_id),
                )
            except Exception as error:
                self.db.execute("UPDATE training_trials SET status='failed',finished_at=? WHERE id=?", (utcnow(), trial_id))
                raise error
            completed[0] += 1
            self.db.execute(
                "UPDATE training_campaigns SET completed_trials=?,progress=? WHERE id=?",
                (completed[0], min(100, completed[0] / int(campaign["budget"]) * 100), campaign_id),
            )
            if completed[0] == 1 or completed[0] % 10 == 0:
                self._event(campaign_id, "progress", f"已完成{completed[0]}/{campaign['budget']}个候选", {"score": score})
            return score

        remaining = max(0, int(campaign["budget"]) - completed[0])
        if remaining:
            study.optimize(objective, n_trials=remaining, catch=(Exception,))
        canceled = self.db.one("SELECT cancel_requested FROM training_campaigns WHERE id=?", (campaign_id,)) or {}
        if canceled.get("cancel_requested"):
            self.db.execute(
                "UPDATE training_campaigns SET status='canceled',finished_at=? WHERE id=?",
                (utcnow(), campaign_id),
            )
            self._event(campaign_id, "canceled", "训练已安全取消")
            return
        retry_budget = max(0, int(campaign["budget"]) - completed[0])
        if retry_budget:
            study.optimize(objective, n_trials=retry_budget, catch=(Exception,))
        canceled = self.db.one("SELECT cancel_requested FROM training_campaigns WHERE id=?", (campaign_id,)) or {}
        if canceled.get("cancel_requested"):
            self.db.execute(
                "UPDATE training_campaigns SET status='canceled',finished_at=? WHERE id=?",
                (utcnow(), campaign_id),
            )
            self._event(campaign_id, "canceled", "训练已安全取消")
            return
        if completed[0] < int(campaign["budget"]):
            raise RuntimeError(
                f"成功候选不足：{completed[0]}/{campaign['budget']}，请检查失败试验记录"
            )
        ranked_trials = self._finalize_campaign_statistical_gates(campaign_id)
        if not ranked_trials:
            raise ValueError("所有候选试验均失败")
        development_sampled = [
            row for row in ranked_trials
            if (lambda candidate_gates: (
                bool(candidate_gates.get("profitable_folds"))
                and bool(candidate_gates.get("trade_count"))
            ))(self.db.load(row.get("gates_json"), {}))
        ]
        development_ready = [
            row for row in development_sampled
            if bool(self.db.load(row.get("gates_json"), {}).get("deflated_sharpe"))
        ]
        best = (development_ready or development_sampled or ranked_trials)[0]
        if best["id"] != ranked_trials[0]["id"]:
            self._event(
                campaign_id,
                "development_gate_selection",
                "最高原始得分候选未满足开发期硬门槛，已选择最高的合格研究候选",
                {
                    "raw_best_trial_id": ranked_trials[0]["id"],
                    "selected_trial_id": best["id"],
                    "development_sampled_candidates": len(development_sampled),
                    "development_dsr_candidates": len(development_ready),
                },
            )
        gates = self.db.load(best["gates_json"], {})
        params = self.db.load(best["params_json"], {})
        stability = self._stability_report(
            feature_frames, style, params, start, end, quality, float(best["score"]),
            holdout_benchmark, distributions=distributions,
        )
        gates["parameter_stability"] = stability["passed"]
        self._refresh_gate_outcome(gates)
        status = "candidate_ready" if gates.get("passed") else "research_only"
        summary = self.db.load(best["summary_json"], {})
        summary["parameter_stability"] = stability
        self.db.execute(
            "UPDATE training_trials SET summary_json=?,gates_json=? WHERE id=?",
            (self.db.dump(summary), self.db.dump(gates), best["id"]),
        )
        best["gates_json"] = self.db.dump(gates)
        incumbent = self.db.one(
            """SELECT ch.*,t.score AS score FROM strategy_champions ch
               JOIN training_trials t ON t.id=ch.trial_id WHERE ch.track=?""",
            (campaign["track"],),
        )
        incumbent_compatible = bool(
            incumbent
            and self._params_match_distributions(
                self.db.load(incumbent.get("params_json"), {}),
                distributions,
            )
        )
        incumbent_for_comparison = incumbent if incumbent_compatible else None
        replace_champion = incumbent_for_comparison is None
        decision_reason = (
            "首个冠军" if incumbent is None
            else "现任冠军参数超出当前稳健搜索边界，改用本代候选"
        )
        champion_trial = best
        champion_gates = gates
        champion_status = status
        incumbent_retest = None
        if incumbent_for_comparison:
            incumbent_retest = self.db.one(
                """SELECT * FROM training_trials
                   WHERE campaign_id=? AND status='success' AND params_json=?
                   ORDER BY trial_number LIMIT 1""",
                (campaign_id, incumbent["params_json"]),
            )
            incumbent_gates = self.db.load(
                (incumbent_retest or incumbent).get("gates_json"), {}
            )
            prior_incumbent_gates = self.db.load(incumbent.get("gates_json"), {})
            incumbent_is_best = bool(incumbent_retest and best["id"] == incumbent_retest["id"])
            incumbent_gates["parameter_stability"] = (
                bool(gates.get("parameter_stability")) if incumbent_is_best
                else bool(prior_incumbent_gates.get("parameter_stability"))
            )
            self._refresh_gate_outcome(incumbent_gates)
            incumbent_passed = bool(incumbent_gates.get("passed"))
            candidate_passed = bool(gates.get("passed"))
            incumbent_score = float((incumbent_retest or incumbent).get("score") or -999)
            improvement = float(best["score"]) - incumbent_score
            required_improvement = max(0.25, abs(incumbent_score) * 0.02)
            if incumbent_retest:
                champion_trial = incumbent_retest
                champion_gates = incumbent_gates
                champion_status = "candidate_ready" if incumbent_passed else "research_only"
            if incumbent_retest and best["id"] == incumbent_retest["id"]:
                replace_champion, decision_reason = False, "现任冠军在新快照复测中仍然胜出"
            elif candidate_passed and not incumbent_passed:
                replace_champion, decision_reason = True, "挑战者首次通过全部门槛"
            elif candidate_passed and incumbent_passed and improvement >= required_improvement:
                replace_champion, decision_reason = True, "挑战者风险调整得分显著提高"
            elif not candidate_passed and not incumbent_passed and improvement > 0:
                replace_champion, decision_reason = True, "挑战者开发期风险调整得分更高"
            else:
                decision_reason = "挑战者未显著优于现任冠军"
            if replace_champion:
                champion_trial, champion_gates, champion_status = best, gates, status
        summary["champion_challenge"] = {
            "replaced": replace_champion, "reason": decision_reason,
            "incumbent_trial_id": None if incumbent is None else incumbent["trial_id"],
            "incumbent_retest_trial_id": None if incumbent_retest is None else incumbent_retest["id"],
            "incumbent_retest_score": None if incumbent_retest is None else incumbent_retest["score"],
            "challenger_trial_id": best["id"],
        }
        self.db.execute(
            """UPDATE training_campaigns SET status=?,completed_trials=?,best_trial_id=?,progress=100,
               summary_json=?,finished_at=? WHERE id=?""",
            (status, completed[0], best["id"], self.db.dump({"best_score": best["score"], "gates": gates, **summary}), utcnow(), campaign_id),
        )
        now = utcnow()
        if replace_champion or incumbent_retest:
            self.db.execute(
                """INSERT INTO strategy_champions(track,campaign_id,trial_id,status,params_json,gates_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(track) DO UPDATE SET campaign_id=excluded.campaign_id,trial_id=excluded.trial_id,
                     status=excluded.status,params_json=excluded.params_json,gates_json=excluded.gates_json,updated_at=excluded.updated_at""",
                (campaign["track"], campaign_id, champion_trial["id"], champion_status,
                 champion_trial["params_json"], self.db.dump(champion_gates), now, now),
            )
            if (
                incumbent
                and incumbent.get("strategy_id")
                and (not champion_gates.get("passed") or not incumbent_compatible)
            ):
                self.db.execute(
                    "UPDATE strategies SET active=0,updated_at=? WHERE id=?",
                    (now, incumbent["strategy_id"]),
                )
                pause_reason = (
                    "旧冠军参数超出当前稳健边界，已暂停对应模拟盘策略"
                    if not incumbent_compatible
                    else "冠军复测退化，已暂停对应模拟盘策略"
                )
                self._event(campaign_id, "deployed_champion_paused", pause_reason, level="warning")
        self._event(
            campaign_id, "champion_replaced" if replace_champion else "champion_revalidated" if incumbent_retest else "challenger_rejected",
            decision_reason, {"best_score": best["score"]},
        )
        self._event(campaign_id, "completed", f"训练完成：{status}", {"best_score": best["score"], "failed_gates": gates.get("reasons", [])})

    def cancel(self, campaign_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM training_campaigns WHERE id=?", (campaign_id,))
        if not row:
            raise LookupError("训练任务不存在")
        if row["status"] not in {"queued", "running"}:
            raise ValueError("当前训练任务不能取消")
        self.db.execute("UPDATE training_campaigns SET cancel_requested=1 WHERE id=?", (campaign_id,))
        self._event(campaign_id, "cancel_requested", "已请求安全取消")
        return self.campaign(campaign_id)

    @staticmethod
    def _strategy_dsl(style: str, market: str, params: dict[str, Any]) -> dict[str, Any]:
        market_condition = {"op": "compare", "left": {"name": "security", "field": "is_a_share", "params": {}}, "right": 1 if market == "A" else 0, "comparator": "=="}
        universe = {
            "id": "trained-universe", "scope": "universe", "timeframe": "day", "daily_bar_mode": "completed",
            "condition": {"op": "all", "conditions": [
                market_condition,
                {"op": "compare", "left": {"name": "security", "field": "is_st", "params": {}}, "right": 0, "comparator": "=="},
                {"op": "compare", "left": {"name": "security", "field": "trade_status", "params": {}}, "right": 1, "comparator": "=="},
                {"op": "compare", "left": {"name": "security", "field": "listed_trading_days", "params": {}}, "right": 250, "comparator": ">="},
                {"op": "compare", "left": {"name": "candle", "field": "one_word_limit_down", "params": {}}, "right": 0, "comparator": "=="},
            ]},
            "action": "add_candidate", "label": "训练版市场与可交易股票池",
        }
        if style == "left":
            low_window = int(params.get("low_window", LEFT_LOW_WINDOW))
            volume_left = (
                {"name": "volume", "field": "value", "params": {}}
                if market == "A"
                else {"name": "volume", "field": "value", "params": {"window_op": "mean", "window": 5}}
            )
            conditions = [
                {"op": "within_pct", "left": {"name": "price", "field": "close", "params": {"lag": 1}}, "right": {"name": "price", "field": "close", "params": {"window_op": "min", "window": low_window, "lag": 1}}, "lower": 0, "upper": params["low_proximity_pct"]},
                {"op": "compare", "left": {"name": "rsi", "field": "value", "params": {"period": 14, "lag": 1}}, "right": params["rsi_max"], "comparator": "<"},
                {"op": "compare", "left": {"name": "macd", "field": "histogram", "params": {"lag": 1}}, "right": 0, "comparator": "<"},
                {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": {"name": "price", "field": "close", "params": {"lag": 1}}, "comparator": ">"},
                {"op": "compare", "left": {"name": "macd", "field": "histogram", "params": {}}, "right": {"name": "macd", "field": "histogram", "params": {"lag": 1}}, "comparator": ">"},
                {"op": "relative_change", "left": {"name": "price", "field": "close", "params": {}}, "right": params["confirm_return_max"], "comparator": "<="},
                {"op": "relative_change", "left": {"name": "price", "field": "close", "params": {}}, "right": 0, "comparator": ">"},
                {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": {"name": "sma", "field": "value", "params": {"period": 200, "scale": params["trend_floor_ratio"]}}, "comparator": ">="},
                {"op": "compare", "left": {"name": "sma", "field": "value", "params": {"period": 120}}, "right": {"name": "sma", "field": "value", "params": {"period": 120, "lag": LEFT_TREND_SLOPE_DAYS, "scale": params["trend_slope_floor"]}}, "comparator": ">="},
                {"op": "compare", "left": {"name": "adx", "field": "value", "params": {"period": 14}}, "right": params["adx_max"], "comparator": "<="},
            ]
            conditions.insert(3, {
                "op": "ratio_pct", "left": volume_left,
                "right": {"name": "volume", "field": "value", "params": {"window_op": "mean", "window": 20}},
                "lower": 0, "upper": params["volume_ratio_max"] * 100,
            })
            if market == "A":
                divergence_window = int(params["divergence_window"])
                conditions.insert(4, {"op": "any", "conditions": [
                    {"op": "compare", "left": {"name": "boll", "field": "percent_b", "params": {"period": 20, "lag": 1}}, "right": params["boll_b_max"], "comparator": "<="},
                    {"op": "compare", "left": {"name": "macd", "field": "histogram", "params": {"lag": 1}}, "right": {"name": "macd", "field": "histogram", "params": {"window_op": "min", "window": divergence_window, "lag": 2}}, "comparator": ">"},
                ]})
            quality_label = "BOLL低位或背离确认" if market == "A" else "缩量确认"
            buy_label = f"靠近{low_window}日低点，RSI(14)超卖、MACD柱拐头且{quality_label}"
        else:
            volume_left = (
                {"name": "volume", "field": "value", "params": {}}
                if market == "A"
                else {"name": "volume", "field": "value", "params": {"window_op": "mean", "window": 5}}
            )
            breakout_entry = {"op": "all", "conditions": [
                {"op": "breakout", "left": {"name": "price", "field": "close", "params": {}}, "periods": int(params.get("breakout_window", 55)), "direction": "high"},
                {"op": "ratio_pct", "left": volume_left, "right": {"name": "volume", "field": "value", "params": {"window_op": "mean", "window": 20}}, "lower": params["volume_ratio_min"] * 100, "upper": 1000},
                {"op": "relative_change", "left": {"name": "price", "field": "close", "params": {}}, "right": params["signal_return_max"], "comparator": "<="},
            ]}
            conditions = [
                breakout_entry,
                {"op": "compare", "left": {"name": "sma", "field": "value", "params": {"period": 60}}, "right": {"name": "sma", "field": "value", "params": {"period": 120}}, "comparator": ">"},
                {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": {"name": "sma", "field": "value", "params": {"period": 200}}, "comparator": ">"},
                {"op": "compare", "left": {"name": "sma", "field": "value", "params": {"period": 120}}, "right": {"name": "sma", "field": "value", "params": {"period": 120, "lag": int(params["trend_slope_days"])}}, "comparator": ">"},
                {"op": "compare", "left": {"name": "rsi", "field": "value", "params": {"period": 14}}, "right": params["rsi_min"], "comparator": ">="},
                {"op": "compare", "left": {"name": "rsi", "field": "value", "params": {"period": 14}}, "right": params["rsi_max"], "comparator": "<="},
                {"op": "compare", "left": {"name": "adx", "field": "value", "params": {"period": 14}}, "right": params["adx_min"], "comparator": ">="},
                {"op": "compare", "left": {"name": "atr", "field": "percent", "params": {"period": 14}}, "right": params["atr_pct_max"], "comparator": "<="},
                {"op": "compare", "left": {"name": "macd", "field": "dif", "params": {}}, "right": {"name": "macd", "field": "dea", "params": {}}, "comparator": ">"},
            ]
            if params.get("require_weekly"):
                conditions.extend([
                    {"op": "compare", "timeframe": "week", "left": {"name": "macd", "field": "dif", "params": {}, "timeframe": "week"}, "right": {"name": "macd", "field": "dea", "params": {}, "timeframe": "week"}, "comparator": ">"},
                    {"op": "compare", "timeframe": "week", "left": {"name": "price", "field": "close", "params": {}, "timeframe": "week"}, "right": {"name": "boll", "field": "middle", "params": {"period": 20}, "timeframe": "week"}, "comparator": ">"},
                ])
            if market == "A":
                conditions.append({
                    "op": "compare",
                    "left": {"name": "sma", "field": "value", "params": {"period": 200}},
                    "right": {"name": "sma", "field": "value", "params": {"period": 200, "lag": 60}},
                    "comparator": ">",
                })
            buy_label = "训练版右侧放量突破、长期均线向上、波动受控及日周趋势确认"
        # 自动待定池必须代表“完整入场信号”，而不是整个基础市场池。
        # 候选规则仍保留同一条件，既可生成建议建仓信号，也能正确处理
        # 用户手动加入待定池的股票。
        entry_condition = {"op": "all", "conditions": conditions}
        position_fraction = min(0.10, max(0.001, settings.max_security_exposure_pct / 100))
        strategy_fraction = min(1.0, max(position_fraction, settings.max_strategy_exposure_pct / 100))
        target_position_pct = round(position_fraction / strategy_fraction * 100, 2)
        universe["condition"]["conditions"].append(entry_condition)
        universe["label"] = "训练版市场过滤与完整入场条件"
        rules = [universe, {
            "id": "trained-buy", "scope": "candidate", "timeframe": "mixed", "daily_bar_mode": "completed",
            "condition": entry_condition, "action": "signal_buy", "target_position_pct": target_position_pct, "label": buy_label,
        }]
        if style == "left":
            rules.extend([{
                "id": "trained-left-stop", "scope": "position", "timeframe": "day", "daily_bar_mode": "completed",
                "condition": {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": {"name": "position", "field": "atr_stop_price", "params": {"period": 14, "multiple": params["stop_atr"]}}, "comparator": "<"},
                "action": "signal_stop", "target_position_pct": 0, "label": "左侧建仓后的ATR初始止损",
            }, {
                "id": "trained-left-target", "scope": "position", "timeframe": "day", "daily_bar_mode": "completed",
                "condition": {"op": "any", "conditions": [
                    {"op": "compare", "left": {"name": "rsi", "field": "value", "params": {"period": LEFT_RSI_PERIOD}}, "right": params["exit_rsi"], "comparator": ">="},
                    {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": {"name": "boll", "field": "middle", "params": {"period": 20}}, "comparator": ">="},
                ]}, "action": "signal_exit", "target_position_pct": 0, "label": "左侧均值回归止盈",
            }])
        else:
            rules.append({
                "id": "trained-right-atr-stop", "scope": "position", "timeframe": "day", "daily_bar_mode": "completed",
                "condition": {"op": "any", "conditions": [
                    {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": {"name": "position", "field": "atr_stop_price", "params": {"period": 14, "multiple": params["stop_atr"]}}, "comparator": "<"},
                    {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": {"name": "position", "field": "trailing_atr_stop_price", "params": {"period": 14, "multiple": params["trail_atr"]}}, "comparator": "<"},
                ]}, "action": "signal_stop", "target_position_pct": 0, "label": "ATR初始及移动止损",
            })
            if market == "A":
                rules.append({
                    "id": "trained-right-channel-exit", "scope": "position", "timeframe": "day", "daily_bar_mode": "completed",
                    "condition": {"op": "all", "conditions": [
                        {"op": "compare", "left": {"name": "position", "field": "holding_trading_days", "params": {}}, "right": 6, "comparator": ">="},
                        {"op": "compare", "left": {"name": "price", "field": "close", "params": {}}, "right": {"name": "price", "field": "close", "params": {"window_op": "min", "window": int(params.get("channel_exit_window", 20)), "lag": 1}}, "comparator": "<"},
                    ]}, "action": "signal_exit", "target_position_pct": 0, "label": "A股长期突破的低点通道退出",
                })
            if params.get("require_weekly"):
                rules.append({
                    "id": "trained-right-weekly-exit", "scope": "position", "timeframe": "mixed", "daily_bar_mode": "completed",
                    "condition": {"op": "all", "conditions": [
                        {"op": "compare", "left": {"name": "position", "field": "holding_trading_days", "params": {}}, "right": 6, "comparator": ">="},
                        {"op": "any", "conditions": [
                            {"op": "compare", "timeframe": "week", "left": {"name": "macd", "field": "dif", "params": {}, "timeframe": "week"}, "right": {"name": "macd", "field": "dea", "params": {}, "timeframe": "week"}, "comparator": "<="},
                            {"op": "compare", "timeframe": "week", "left": {"name": "price", "field": "close", "params": {}, "timeframe": "week"}, "right": {"name": "boll", "field": "middle", "params": {"period": 20}, "timeframe": "week"}, "comparator": "<="},
                        ]},
                    ]}, "action": "signal_exit", "target_position_pct": 0, "label": "持仓六日后的周线趋势破坏退出",
                })
        rules.append({
            "id": "trained-max-hold", "scope": "position", "timeframe": "day", "daily_bar_mode": "completed",
            "condition": {"op": "compare", "left": {"name": "position", "field": "holding_trading_days", "params": {}}, "right": int(params["max_hold"]), "comparator": ">="},
            "action": "signal_exit", "target_position_pct": 0, "label": "达到训练参数最大持仓交易日退出",
        })
        return {"schema_version": 1, "rules": rules}

    def approve(self, track: str) -> dict[str, Any]:
        champion = self.db.one("SELECT * FROM strategy_champions WHERE track=?", (track,))
        if not champion:
            raise LookupError("训练冠军不存在")
        gates = self.db.load(champion["gates_json"], {})
        if not gates.get("passed"):
            raise ValueError("训练冠军未通过全部晋级门槛")
        style, market = TRACKS[track]
        params = self.db.load(champion["params_json"], {})
        dsl = self._strategy_dsl(style, market, params)
        name = f"训练冠军·{'左侧' if style == 'left' else '右侧'}·{'A股' if market == 'A' else '港股'}"
        if champion.get("strategy_id"):
            strategy = self.repo.activate_version(champion["strategy_id"], name, dsl, ["由训练平台通过样本外和压力测试后人工批准进入模拟盘。"])
            if strategy:
                strategy = self.repo.update_strategy(strategy["id"], name, True)
        else:
            strategy = self.repo.create_strategy(name, name, dsl, ["由训练平台通过样本外和压力测试后人工批准进入模拟盘。"])
        if not strategy:
            raise LookupError("训练冠军关联策略不存在")
        now = utcnow()
        self.db.execute(
            "UPDATE strategy_champions SET strategy_id=?,status='paper',approved_at=?,paper_started_at=?,updated_at=? WHERE track=?",
            (strategy["id"], now, now, now, track),
        )
        return self.champion(track)

    def publish_research(self, track: str) -> dict[str, Any]:
        """Publish a failed champion as a scanner without trading authority.

        A research publication intentionally leaves the champion's status and
        gates untouched.  It can populate candidates and evaluate manually
        recorded positions, but all strategy allocations are disabled so its
        signals cannot become broker order intents.
        """
        champion = self.db.one("SELECT * FROM strategy_champions WHERE track=?", (track,))
        if not champion:
            raise LookupError("训练研究结果不存在")
        gates = self.db.load(champion["gates_json"], {})
        if gates.get("passed"):
            raise ValueError("该策略已通过全部门槛，请使用冠军批准流程")
        failed_data_gates = [key for key in DATA_GATE_NAMES if not gates.get(key)]
        if failed_data_gates:
            labels = "、".join(GATE_LABELS.get(key, key) for key in failed_data_gates)
            raise ValueError(f"数据门槛未满足，不能发布研究扫描：{labels}")
        style, market = TRACKS[track]
        params = self.db.load(champion["params_json"], {})
        dsl = self._strategy_dsl(style, market, params)
        failed_strategy_gates = [
            key for key in gates.get("reasons", []) if key not in DATA_GATE_NAMES
        ]
        failed_labels = "、".join(
            GATE_LABELS.get(key, key) for key in failed_strategy_gates
        ) or "严格冠军门槛"
        validation_tier = self._validation_tier(gates)
        prefix = "训练观察" if validation_tier == "paper_observation" else "训练研究"
        name = f"{prefix}·{'左侧' if style == 'left' else '右侧'}·{'A股' if market == 'A' else '港股'}"
        description = (
            f"未通过严格冠军门槛（{failed_labels}）；"
            "仅用于自动选股和持仓卖出提示，未配置资金，不生成交易订单。"
        )
        explanation = [
            description,
            "研究策略保留训练得到的完整入场、止损、止盈和最长持仓规则。",
        ]
        strategy = None
        if champion.get("strategy_id"):
            strategy = self.repo.activate_version(
                champion["strategy_id"], description, dsl, explanation,
            )
            if strategy:
                strategy = self.repo.update_strategy(strategy["id"], name, True)
        if not strategy:
            strategy = self.repo.create_strategy(name, description, dsl, explanation)
        now = utcnow()
        self.db.execute(
            "UPDATE strategy_allocations SET enabled=0,updated_at=? WHERE strategy_id=?",
            (now, strategy["id"]),
        )
        self.db.execute(
            "UPDATE strategy_champions SET strategy_id=?,updated_at=? WHERE track=?",
            (strategy["id"], now, track),
        )
        self._event(
            champion["campaign_id"], "research_strategy_published",
            f"{name}已接入系统研究扫描；资金配额保持禁用",
            {"strategy_id": strategy["id"], "failed_gates": failed_strategy_gates},
            level="warning",
        )
        return self.champion(track)

    def daily_check(self) -> list[dict[str, Any]]:
        today = date.today().isoformat()
        results = []
        market_quality = self.data_quality()["markets"]
        for track in TRACKS:
            champion = self.db.one("SELECT * FROM strategy_champions WHERE track=?", (track,))
            quality = market_quality[TRACKS[track][1]]
            status = (
                "data_building" if not quality["ready_for_training"]
                else "no_champion" if not champion
                else "healthy" if champion["status"] in {"candidate_ready", "paper"}
                else "research_only"
            )
            summary = {"champion_status": None if not champion else champion["status"], "data_quality": quality}
            self.db.execute(
                """INSERT INTO training_daily_checks(id,track,check_date,status,summary_json,created_at)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(track,check_date) DO UPDATE SET status=excluded.status,summary_json=excluded.summary_json,created_at=excluded.created_at""",
                (str(uuid.uuid4()), track, today, status, self.db.dump(summary), utcnow()),
            )
            results.append({"track": track, "status": status, "summary": summary})
        return results

    def campaign(self, campaign_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM training_campaigns WHERE id=?", (campaign_id,))
        if not row:
            raise LookupError("训练任务不存在")
        return {
            **row, "config": self.db.load(row.get("config_json"), {}), "summary": self.db.load(row.get("summary_json"), {}),
            "trials": [{**item, "params": self.db.load(item.get("params_json"), {}), "summary": self.db.load(item.get("summary_json"), {}), "folds": self.db.load(item.get("folds_json"), []), "gates": self.db.load(item.get("gates_json"), {})} for item in self.db.all("SELECT * FROM training_trials WHERE campaign_id=? ORDER BY score DESC LIMIT 100", (campaign_id,))],
            "events": [{**item, "detail": self.db.load(item.get("detail_json"), {})} for item in self.db.all("SELECT * FROM training_events WHERE campaign_id=? ORDER BY created_at DESC LIMIT 200", (campaign_id,))],
        }

    def list_campaigns(self, limit: int = 50) -> list[dict[str, Any]]:
        return [{**row, "summary": self.db.load(row.get("summary_json"), {})} for row in self.db.all("SELECT * FROM training_campaigns ORDER BY created_at DESC LIMIT ?", (limit,))]

    def champion(self, track: str) -> dict[str, Any]:
        row = self.db.one(
            """SELECT ch.*,t.score,t.summary_json
               FROM strategy_champions ch
               JOIN training_trials t ON t.id=ch.trial_id
               WHERE ch.track=?""",
            (track,),
        )
        if not row:
            raise LookupError("训练冠军不存在")
        summary = self.db.load(row.get("summary_json"), {})
        metric_names = (
            "trades", "total_return_pct", "annual_return_pct",
            "max_drawdown_pct", "sharpe", "average_gross_exposure_pct",
        )
        compact = lambda source: {
            key: source.get(key) for key in metric_names if key in source
        }
        performance = {
            "development": compact(summary.get("development") or {}),
            "holdout": compact(summary.get("holdout") or {}),
            "stress": compact(summary.get("stress") or {}),
            "benchmark": compact(summary.get("benchmark") or {}),
            "annual_excess_return_pct": summary.get("annual_excess_return_pct"),
            "dsr_probability": summary.get("dsr_probability"),
        }
        gates = self.db.load(row.get("gates_json"), {})
        return {
            **row,
            "params": self.db.load(row.get("params_json"), {}),
            "gates": gates,
            "performance": performance,
            "validation_tier": self._validation_tier(gates),
        }

    def dashboard(self) -> dict[str, Any]:
        quality = self.data_quality()
        champions = []
        for track in TRACKS:
            try:
                champion = self.champion(track)
            except LookupError:
                champion = {"track": track, "status": "not_trained", "params": {}, "gates": {}}
            market_quality = quality["markets"][TRACKS[track][1]]
            gate_reasons = champion.get("gates", {}).get("reasons", [])
            champion["market"] = TRACKS[track][1]
            champion["data_ready"] = market_quality["ready_for_training"]
            champion["data_blockers"] = market_quality["blocking_reasons"]
            champion["strategy_blockers"] = [reason for reason in gate_reasons if reason not in DATA_GATE_NAMES]
            champion["evaluation_status"] = "evaluated" if market_quality["ready_for_training"] else "pending_data"
            champions.append(champion)
        return {"data_quality": quality, "campaigns": self.list_campaigns(20), "champions": champions}


training_service = TrainingService()
