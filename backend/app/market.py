from __future__ import annotations

import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .config import settings
from .database import Database, db, utcnow
from .logging_config import logger
from .providers import AkshareProvider, BaoStockProvider, SecurityInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def market_session_state(market: str, now: datetime | None = None) -> str:
    local = (now or datetime.now(SHANGHAI_TZ)).astimezone(SHANGHAI_TZ)
    if local.weekday() >= 5:
        return "closed"
    minute = local.hour * 60 + local.minute
    if market in ("SH", "SZ"):
        if 570 <= minute < 690 or 780 <= minute < 900:
            return "open"
        if 690 <= minute < 780:
            return "lunch"
    elif market == "HK":
        if 570 <= minute < 720 or 780 <= minute < 960:
            return "open"
        if 720 <= minute < 780:
            return "lunch"
    return "closed"


class MarketService:
    def __init__(self, database: Database = db):
        self.db = database
        self._batch_a_provider: BaoStockProvider | None = None
        self._batch_hk_provider: AkshareProvider | None = None
        self._backfill_lock = threading.Lock()
        # Market refresh and history backfill both write the same SQLite file.
        # Serialize maintenance work so their CPU and write bursts do not overlap.
        self._maintenance_lock = threading.RLock()

    def _provider(self) -> AkshareProvider:
        if settings.market_data_mode != "akshare":
            raise RuntimeError("当前为演示数据模式；将 MARKET_DATA_MODE 设置为 akshare 后可同步")
        return AkshareProvider(settings.market_backfill_retries)

    @staticmethod
    def _desired_start() -> str:
        days = settings.market_history_years * 365 + settings.market_history_years // 4
        return (date.today() - timedelta(days=days)).isoformat()

    def sync_reference_universe(self) -> dict:
        with self._maintenance_lock:
            return self._sync_reference_universe_unlocked()

    def _sync_reference_universe_unlocked(self) -> dict:
        """Import current and delisted A-share master data from BaoStock."""
        started = time.monotonic()
        with BaoStockProvider() as provider:
            records = provider.fetch_security_master()
        for item in records:
            item["reference_source"] = "baostock"
        active_hk_codes = {
            row["code"] for row in self.db.all("SELECT code FROM securities WHERE market='HK' AND is_active=1")
        }
        hk_delisted = [
            item for item in self._provider().fetch_hk_delisted_master()
            if item["code"] not in active_hk_codes
        ]
        for item in hk_delisted:
            item["reference_source"] = "hkex_delisted_list"
        records.extend(hk_delisted)
        with self.db.transaction() as connection:
            for item in records:
                connection.execute(
                    """INSERT INTO securities(
                           id,market,code,name,currency,security_type,is_active,
                           listing_date,delisting_date,reference_source
                       ) VALUES(?,?,?,?,?,'stock',?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET name=excluded.name,is_active=excluded.is_active,
                         listing_date=excluded.listing_date,delisting_date=excluded.delisting_date,
                         reference_source=excluded.reference_source""",
                    (item["id"], item["market"], item["code"], item["name"], item["currency"],
                     item["is_active"], item["listing_date"], item["delisting_date"], item["reference_source"]),
                )
        self._refresh_backfill_counts()
        result = {
            "securities": len(records),
            "inactive": sum(not item["is_active"] for item in records),
            "hk_delisted": len(hk_delisted),
            "duration_seconds": round(time.monotonic() - started, 2),
        }
        logger.info("Reference universe sync succeeded", extra={"event": "reference_universe_synced", **result})
        return result

    def _refresh_backfill_counts(self) -> None:
        desired_start = self._desired_start()
        self.db.execute(
            """UPDATE market_status SET backfilled=(
                   SELECT COUNT(*) FROM securities s JOIN market_backfill_state m ON m.security_id=s.id
                   WHERE s.market=market_status.market AND s.is_active=1
                     AND m.status='complete' AND m.desired_start<=?
               )""",
            (desired_start,),
        )

    def sync_universe_and_quotes(self) -> dict:
        with self._maintenance_lock:
            return self._sync_universe_and_quotes_unlocked()

    def _sync_universe_and_quotes_unlocked(self) -> dict:
        started = time.monotonic()
        logger.info("Market universe sync started", extra={"event": "market_sync_started"})
        provider = self._provider()
        securities, quotes = provider.fetch_universe_and_quotes()
        now = utcnow()
        with self.db.transaction() as connection:
            for security in securities:
                connection.execute(
                    """INSERT INTO securities(id,market,code,name,currency,security_type,is_active)
                       VALUES(?,?,?,?,?,'stock',1)
                       ON CONFLICT(id) DO UPDATE SET name=excluded.name,currency=excluded.currency,
                         is_active=CASE
                           WHEN securities.delisting_date IS NOT NULL
                            AND securities.delisting_date<=date('now','+8 hours') THEN 0
                           ELSE 1
                         END""",
                    (security.id, security.market, security.code, security.name, security.currency),
                )
            for quote in quotes:
                connection.execute(
                    "UPDATE securities SET latest_price=?,quote_time=? WHERE id=?",
                    (quote.price, quote.quote_time, quote.security_id),
                )
                trade_date = quote.quote_time[:10]
                connection.execute(
                    """INSERT INTO bars(security_id,trade_date,open,high,low,close,volume,is_provisional)
                       VALUES(?,?,?,?,?,?,?,1)
                       ON CONFLICT(security_id,trade_date) DO UPDATE SET open=excluded.open,high=excluded.high,
                         low=excluded.low,close=excluded.close,volume=excluded.volume,is_provisional=1""",
                    (quote.security_id, trade_date, quote.open, quote.high, quote.low, quote.price, quote.volume),
                )
            for market in ("SH", "SZ", "HK"):
                total = sum(1 for item in securities if item.market == market)
                quote_time = max((item.quote_time for item in quotes if item.market == market), default=None)
                connection.execute(
                    """INSERT INTO market_status(market,state,quote_time,last_sync_at,data_status,message,initialized,backfilled,total)
                       VALUES(?,?,?,?,?,'公开行情源 · 历史数据逐批回填',1,0,?)
                       ON CONFLICT(market) DO UPDATE SET state=excluded.state,quote_time=excluded.quote_time,last_sync_at=excluded.last_sync_at,
                         data_status=excluded.data_status,message=excluded.message,initialized=1,total=excluded.total""",
                    (market, market_session_state(market), quote_time, now, "live", total),
                )
        self._refresh_backfill_counts()
        result = {"securities": len(securities), "quotes": len(quotes), "synced_at": now}
        logger.info(
            "Market universe sync succeeded",
            extra={
                "event": "market_sync_succeeded",
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "securities": len(securities),
                "quotes": len(quotes),
            },
        )
        return result

    def refresh_for_strategy_scan(
        self,
        now: datetime | None = None,
        max_age_minutes: int = 5,
    ) -> dict:
        """为手动策略扫描准备最新行情。

        交易时段保留当日 provisional K 线；当地交易日收盘后，将本次报价
        固化为当日已完成K线。五分钟内重复执行多个策略时复用同一份行情快照。
        """
        if settings.market_data_mode != "akshare":
            return {"refreshed": False, "mode": settings.market_data_mode, "finalized": 0}
        local_now = (now or datetime.now(SHANGHAI_TZ)).astimezone(SHANGHAI_TZ)
        utc_now = local_now.astimezone(timezone.utc)
        # 非交易日和开盘前的“最新”是上一个已完成交易日；
        # 此时不能把前收盘价写成今天的伪K线。
        if local_now.weekday() >= 5 or local_now.hour < 9:
            return {
                "refreshed": False,
                "finalized": 0,
                "trade_date": local_now.date().isoformat(),
                "reason": "non_trading_time",
            }
        statuses = self.db.all("SELECT market,last_sync_at,quote_time FROM market_status ORDER BY market")
        freshness = timedelta(minutes=max(1, max_age_minutes))

        def recently_synced(value: str | None) -> bool:
            if not value:
                return False
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return utc_now - parsed.astimezone(timezone.utc) <= freshness
            except ValueError:
                return False

        minute = local_now.hour * 60 + local_now.minute
        completed_for_day = (
            minute >= 16 * 60
            and len(statuses) == 3
            and all(str(row.get("quote_time") or "")[:10] == local_now.date().isoformat() for row in statuses)
        )
        fresh = completed_for_day or (
            len(statuses) == 3
            and all(recently_synced(row.get("last_sync_at")) for row in statuses)
        )
        sync_result = None if fresh else self.sync_universe_and_quotes()
        completed_markets: list[str] = []
        if local_now.weekday() < 5 and not completed_for_day:
            if minute >= 15 * 60:
                completed_markets.extend(("SH", "SZ"))
            if minute >= 16 * 60:
                completed_markets.append("HK")
        finalized = self.finalize_daily_bars(
            tuple(completed_markets),
            local_now.date().isoformat(),
        ) if completed_markets else 0
        return {
            "refreshed": not fresh,
            "synced": sync_result,
            "finalized": finalized,
            "trade_date": local_now.date().isoformat(),
        }

    def finalize_daily_bars(self, markets: tuple[str, ...], trade_date: str | None = None) -> int:
        """Mark the post-close quote snapshot as the completed daily bar for the given markets."""
        if not markets:
            return 0
        completed_date = trade_date or datetime.now(SHANGHAI_TZ).date().isoformat()
        placeholders = ",".join("?" for _market in markets)
        with self.db.transaction() as connection:
            cursor = connection.execute(
                f"""UPDATE bars SET is_provisional=0
                    WHERE trade_date=? AND is_provisional=1
                      AND security_id IN (SELECT id FROM securities WHERE market IN ({placeholders}))""",
                (completed_date, *markets),
            )
            connection.execute(
                f"""UPDATE positions SET entry_day_low=(
                       SELECT b.low FROM bars b
                       WHERE b.security_id=positions.security_id AND b.trade_date=positions.entry_trade_date
                   )
                   WHERE entry_trade_date=?
                     AND security_id IN (SELECT id FROM securities WHERE market IN ({placeholders}))""",
                (completed_date, *markets),
            )
        return cursor.rowcount

    def backfill_security(
        self,
        security_id: str,
        refresh_progress: bool = True,
        a_provider: BaoStockProvider | None = None,
        hk_provider: AkshareProvider | None = None,
    ) -> int:
        row = self.db.one("SELECT * FROM securities WHERE id=?", (security_id,))
        if not row:
            raise LookupError("股票不存在")
        security = SecurityInfo(row["id"], row["market"], row["code"], row["name"], row["currency"])
        end = date.today()
        market_status = self.db.one(
            "SELECT quote_time FROM market_status WHERE market=?", (security.market,)
        ) or {}
        freshness_target_date = str(market_status.get("quote_time") or "")[:10] or end.isoformat()
        desired_start = date.fromisoformat(self._desired_start())
        state = self.db.one(
            "SELECT * FROM market_backfill_state WHERE security_id=?", (security_id,)
        ) or {}
        incremental = state.get("status") == "complete" and bool(state.get("actual_end"))
        # Include the last stored day as an overlap. Forward-adjusted histories
        # can change scale after a dividend, split or consolidation; the overlap
        # lets us rescale the older prefix before appending new bars.
        start = date.fromisoformat(str(state["actual_end"])) if incremental else desired_start
        attempts = int(state.get("attempts") or 0) + 1
        owned_a_provider = None
        try:
            if security.market in ("SH", "SZ"):
                a_provider = a_provider or self._batch_a_provider
                if a_provider is None:
                    candidate_provider = BaoStockProvider()
                    a_provider = candidate_provider.__enter__()
                    owned_a_provider = candidate_provider
                bars = a_provider.fetch_daily_bars(security, start.isoformat(), end.isoformat())
            else:
                bars = (hk_provider or self._batch_hk_provider or self._provider()).fetch_daily_bars(security, start.isoformat(), end.isoformat())
        except Exception as error:
            delay_minutes = min(1440, 5 * (2 ** min(attempts - 1, 8)))
            next_retry = (datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)).isoformat()
            self.db.execute(
                """INSERT INTO market_backfill_state(
                       security_id,desired_start,status,attempts,last_error,next_retry_at,updated_at
                   ) VALUES(?,?,'failed',?,?,?,?)
                   ON CONFLICT(security_id) DO UPDATE SET desired_start=excluded.desired_start,
                     status='failed',attempts=excluded.attempts,last_error=excluded.last_error,
                     next_retry_at=excluded.next_retry_at,updated_at=excluded.updated_at""",
                (security_id, start.isoformat(), attempts, str(error)[:500], next_retry, utcnow()),
            )
            raise
        finally:
            if owned_a_provider is not None:
                owned_a_provider.__exit__(None, None, None)
        if bars:
            first_new_date = bars[0]["trade_date"]
            rows = [
                (security_id, item["trade_date"], item["open"], item["high"], item["low"], item["close"], item["volume"],
                 item.get("trade_status"), item.get("is_st"), item.get("data_source"))
                for item in bars
            ]
            with self.db.transaction() as connection:
                overlap = connection.execute(
                    "SELECT close FROM bars WHERE security_id=? AND trade_date=?",
                    (security_id, first_new_date),
                ).fetchone()
                old_overlap_close = float(overlap["close"]) if overlap and overlap["close"] else 0.0
                new_overlap_close = float(bars[0]["close"] or 0)
                adjustment_factor = (
                    new_overlap_close / old_overlap_close
                    if incremental and old_overlap_close > 0 and new_overlap_close > 0 else 1.0
                )
                if math.isfinite(adjustment_factor) and adjustment_factor > 0 and not math.isclose(adjustment_factor, 1.0):
                    connection.execute(
                        """UPDATE bars SET open=open*?,high=high*?,low=low*?,close=close*?
                           WHERE security_id=? AND trade_date<? AND is_provisional=0""",
                        (adjustment_factor, adjustment_factor, adjustment_factor, adjustment_factor,
                         security_id, first_new_date),
                    )
                connection.execute(
                    "DELETE FROM bars WHERE security_id=? AND trade_date BETWEEN ? AND ?",
                    (security_id, first_new_date, end.isoformat()),
                )
                connection.executemany(
                    """INSERT INTO bars(
                           security_id,trade_date,open,high,low,close,volume,is_provisional,
                           trade_status,is_st,data_source
                       ) VALUES(?,?,?,?,?,?,?,0,?,?,?)""",
                    rows,
                )
                connection.execute(
                    """UPDATE securities SET
                           listing_date=COALESCE(listing_date,?),
                           delisting_date=CASE WHEN is_active=0 THEN COALESCE(delisting_date,?) ELSE delisting_date END
                       WHERE id=?""",
                    (bars[0]["trade_date"], bars[-1]["trade_date"], security_id),
                )
                stats = connection.execute(
                    """SELECT MIN(trade_date) AS actual_start,MAX(trade_date) AS actual_end,
                              COUNT(*) AS bar_count
                       FROM bars WHERE security_id=? AND is_provisional=0 AND data_source IS NOT NULL""",
                    (security_id,),
                ).fetchone()
        else:
            stats = None
        source = bars[0].get("data_source") if bars else ("baostock_qfq" if security.market in ("SH", "SZ") else "akshare_qfq")
        has_existing_history = int(state.get("bar_count") or 0) > 0
        status = "complete" if bars or has_existing_history else "no_data"
        actual_start = stats["actual_start"] if stats else state.get("actual_start")
        actual_end = stats["actual_end"] if stats else state.get("actual_end")
        bar_count = int(stats["bar_count"] or 0) if stats else int(state.get("bar_count") or 0)
        retry_at = None if bars and actual_end == end.isoformat() else (
            datetime.now(timezone.utc) + timedelta(days=1 if has_existing_history else 7)
        ).isoformat()
        self.db.execute(
            """INSERT INTO market_backfill_state(
                   security_id,desired_start,actual_start,actual_end,bar_count,status,attempts,
                   source,last_error,next_retry_at,freshness_target_date,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(security_id) DO UPDATE SET desired_start=excluded.desired_start,
                 actual_start=excluded.actual_start,actual_end=excluded.actual_end,
                 bar_count=excluded.bar_count,status=excluded.status,attempts=excluded.attempts,
                 source=excluded.source,last_error=NULL,next_retry_at=excluded.next_retry_at,
                 freshness_target_date=excluded.freshness_target_date,updated_at=excluded.updated_at""",
            (security_id, desired_start.isoformat(), actual_start,
             actual_end, bar_count, status,
             attempts, source, None, retry_at, freshness_target_date, utcnow()),
        )
        if refresh_progress:
            self._refresh_backfill_counts()
        return len(bars)

    def backfill_next_batch(self, limit: int | None = None) -> dict:
        with self._maintenance_lock:
            with self._backfill_lock:
                return self._backfill_next_batch(limit)

    def _backfill_next_batch(self, limit: int | None = None) -> dict:
        if settings.market_data_mode != "akshare":
            return {"processed": 0, "bars": 0}
        desired_start = self._desired_start()
        today = date.today()
        fallback_end = today
        while fallback_end.weekday() >= 5:
            fallback_end -= timedelta(days=1)
        batch = self.db.all(
            """WITH eligible AS (
                   SELECT s.*,COALESCE(m.attempts,0) AS attempts
                   FROM securities s
                   LEFT JOIN market_backfill_state m ON m.security_id=s.id
                   LEFT JOIN market_status ms ON ms.market=s.market
                   WHERE s.security_type='stock'
                     AND (m.security_id IS NULL OR m.desired_start>? OR m.status IN ('pending','failed','no_data')
                          OR (s.is_active=1 AND m.status='complete'
                              AND CASE WHEN ms.last_sync_at IS NOT NULL
                                   THEN (COALESCE(m.freshness_target_date,'')<substr(ms.quote_time,1,10)
                                         OR COALESCE(m.actual_end,'')<date(substr(ms.quote_time,1,10),'-30 days'))
                                   ELSE COALESCE(m.actual_end,'')<? END))
                     AND (m.next_retry_at IS NULL OR m.next_retry_at<=?)
               ), ranked AS (
                   SELECT eligible.*,ROW_NUMBER() OVER(PARTITION BY market,is_active ORDER BY attempts,code) AS market_rank
                   FROM eligible
               )
               SELECT * FROM ranked
               ORDER BY market_rank,is_active DESC,CASE market WHEN 'SH' THEN 1 WHEN 'SZ' THEN 2 ELSE 3 END,code
               LIMIT ?""",
            (desired_start, fallback_end.isoformat(), utcnow(), limit or settings.backfill_batch_size),
        )
        logger.info(
            "Market history backfill batch started",
            extra={"event": "market_backfill_started", "securities": len(batch)},
        )
        processed = 0
        total_bars = 0

        def record_failure(security: dict, error: Exception) -> None:
            logger.warning(
                "Market history backfill failed",
                extra={
                    "event": "market_backfill_security_failed",
                    "security_id": security["id"],
                    "error_type": type(error).__name__,
                    "error_detail": str(error)[:500],
                },
            )
            self.db.execute(
                "UPDATE market_status SET message=? WHERE market=?",
                (f"{security['id']} 回填失败：{str(error)[:140]}", security["market"]),
            )

        with BaoStockProvider() as a_provider:
            hk_provider = self._provider()
            self._batch_a_provider, self._batch_hk_provider = a_provider, hk_provider
            try:
                a_batch = [security for security in batch if security["market"] in ("SH", "SZ")]
                hk_batch = [security for security in batch if security["market"] == "HK"]
                # BaoStock owns one authenticated cursor and is very fast for a
                # short incremental window, so keep A shares sequential. HK
                # history is independent HTTP I/O per code; bounded concurrency
                # turns a multi-hour refresh into minutes without overwhelming
                # either the public source or SQLite's serialized writer.
                with ThreadPoolExecutor(
                    max_workers=min(settings.backfill_max_concurrency, max(1, len(hk_batch)))
                ) as executor:
                    hk_futures = {
                        executor.submit(self.backfill_security, security["id"], False): security
                        for security in hk_batch
                    }
                    for security in a_batch:
                        try:
                            total_bars += self.backfill_security(security["id"], refresh_progress=False)
                            processed += 1
                        except Exception as error:
                            record_failure(security, error)
                    for future in as_completed(hk_futures):
                        security = hk_futures[future]
                        try:
                            total_bars += future.result()
                            processed += 1
                        except Exception as error:
                            record_failure(security, error)
            finally:
                self._batch_a_provider = self._batch_hk_provider = None
        self._refresh_backfill_counts()
        result = {"processed": processed, "bars": total_bars}
        logger.info(
            "Market history backfill batch finished",
            extra={"event": "market_backfill_finished", "processed": processed, "bars": total_bars},
        )
        return result

    def backfill_progress(self) -> dict:
        desired_start = self._desired_start()
        markets = {}
        for market in ("SH", "SZ", "HK"):
            row = self.db.one(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN s.is_active=1 THEN 1 ELSE 0 END) AS active_total,
                          SUM(CASE WHEN s.is_active=0 THEN 1 ELSE 0 END) AS inactive_total,
                          SUM(CASE WHEN s.is_active=1 AND m.status='complete' AND m.desired_start<=? THEN 1 ELSE 0 END) AS active_complete,
                          SUM(CASE WHEN s.is_active=0 AND m.status='complete' AND m.desired_start<=? THEN 1 ELSE 0 END) AS inactive_complete,
                          SUM(CASE WHEN m.status='failed' THEN 1 ELSE 0 END) AS failed,
                          SUM(CASE WHEN m.status='no_data' THEN 1 ELSE 0 END) AS no_data
                   FROM securities s LEFT JOIN market_backfill_state m ON m.security_id=s.id
                   WHERE s.market=? AND s.security_type='stock'""",
                (desired_start, desired_start, market),
            ) or {}
            active_total = int(row.get("active_total") or 0)
            active_complete = int(row.get("active_complete") or 0)
            markets[market] = {
                **{key: int(row.get(key) or 0) for key in (
                    "total", "active_total", "inactive_total", "active_complete",
                    "inactive_complete", "failed", "no_data",
                )},
                "active_progress_pct": active_complete / active_total * 100 if active_total else 0,
            }
        return {"desired_start": desired_start, "history_years": settings.market_history_years, "markets": markets}


market_service = MarketService()
