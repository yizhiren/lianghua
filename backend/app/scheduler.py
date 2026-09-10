from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .config import settings
from .logging_config import logger
from .market import market_service, market_session_state
from .strategy_service import strategy_service
from .trading import trading_service
from .training import training_service


TZ = ZoneInfo("Asia/Shanghai")


def _slot() -> str:
    return datetime.now(TZ).strftime("scheduled-%Y%m%d-%H%M")


async def scheduled_scan() -> None:
    if not any(market_session_state(market) == "open" for market in ("SH", "SZ", "HK")):
        return
    if settings.market_data_mode == "akshare":
        await asyncio.to_thread(market_service.sync_universe_and_quotes)
    await asyncio.to_thread(strategy_service.execute_all, _slot())


async def scheduled_history_backfill() -> None:
    if settings.market_data_mode == "akshare":
        await asyncio.to_thread(market_service.backfill_next_batch)


async def scheduled_reference_sync() -> None:
    if settings.market_data_mode == "akshare":
        await asyncio.to_thread(market_service.sync_reference_universe)


async def closing_scan(markets: tuple[str, ...]) -> None:
    if settings.market_data_mode == "akshare":
        await asyncio.to_thread(market_service.sync_universe_and_quotes)
        await asyncio.to_thread(market_service.finalize_daily_bars, markets)
    await asyncio.to_thread(strategy_service.execute_all, _slot())


async def scheduled_close_catchup() -> None:
    """收盘后补偿任务。

    如果应用错过了 15:01/16:01 固定任务（例如收盘后才启动），
    启动时和之后每15分钟尝试补做当日行情与策略快照。
    """
    local = datetime.now(TZ)
    if local.weekday() >= 5 or local.hour * 60 + local.minute < 16 * 60 + 5:
        return
    slot = f"daily-close-{local:%Y%m%d}"
    try:
        await asyncio.to_thread(market_service.refresh_for_strategy_scan)
        await asyncio.to_thread(strategy_service.execute_all, slot)
    except Exception as error:
        logger.warning(
            "Daily close catch-up skipped because market refresh failed",
            extra={
                "event": "daily_close_catchup_failed",
                "error_type": type(error).__name__,
                "error_detail": str(error)[:500],
            },
        )


async def scheduled_trading() -> None:
    await asyncio.to_thread(trading_service.process)


async def scheduled_training_check() -> None:
    await asyncio.to_thread(training_service.daily_check)
    due_tracks = await asyncio.to_thread(training_service.formal_training_due_tracks)
    if due_tracks:
        try:
            await asyncio.to_thread(
                training_service.start_campaigns,
                due_tracks,
                settings.training_weekly_budget,
                "scheduled",
            )
        except RuntimeError:
            return


async def scheduled_weekly_training() -> None:
    ready_tracks = await asyncio.to_thread(training_service.ready_tracks)
    if not ready_tracks:
        return
    try:
        await asyncio.to_thread(
            training_service.start_campaigns,
            ready_tracks,
            settings.training_weekly_budget,
            "scheduled",
        )
    except RuntimeError:
        return


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=TZ)
    scheduler.add_job(
        scheduled_close_catchup,
        IntervalTrigger(minutes=15, timezone=TZ),
        next_run_time=datetime.now(TZ),
        id="daily-close-catchup",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        scheduled_trading,
        IntervalTrigger(seconds=15, timezone=TZ),
        id="trading-order-loop",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    scheduler.add_job(
        scheduled_scan,
        CronTrigger(minute=f"*/{max(1, settings.scan_interval_minutes)}", hour="9-16", day_of_week="mon-fri", timezone=TZ),
        id="intraday-scan",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        scheduled_history_backfill,
        CronTrigger(minute=f"*/{max(1, settings.scan_interval_minutes)}", timezone=TZ),
        id="history-backfill",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        scheduled_reference_sync,
        CronTrigger(hour=1, minute=0, day_of_week="sun", timezone=TZ),
        id="reference-universe-sync",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        closing_scan,
        CronTrigger(hour=15, minute=1, day_of_week="mon-fri", timezone=TZ),
        args=[("SH", "SZ")],
        id="a-close",
        max_instances=1,
    )
    scheduler.add_job(
        scheduled_training_check,
        CronTrigger(hour=17, minute=30, day_of_week="mon-fri", timezone=TZ),
        id="training-daily-check",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        scheduled_weekly_training,
        CronTrigger(hour=2, minute=0, day_of_week="sat", timezone=TZ),
        id="training-weekly-campaign",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        closing_scan,
        CronTrigger(hour=16, minute=1, day_of_week="mon-fri", timezone=TZ),
        args=[("HK",)],
        id="hk-close",
        max_instances=1,
    )
    return scheduler
