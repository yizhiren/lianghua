from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .backtest import backtest_service
from .config import settings
from .database import db
from .dialogue_service import dialogue_service
from .dsl import translate_description, validate_dsl, validation_errors
from .market import market_service
from .logging_config import logger, request_id_context
from .repository import repo
from .scheduler import create_scheduler
from .schemas import (
    BacktestRequest,
    DraftActivate,
    DraftRequest,
    MarketSyncRequest,
    PositionAdd,
    PositionIncrease,
    StockAdd,
    StrategyCreate,
    StrategyDialogueCreate,
    StrategyDialogueMessage,
    StrategyUpdate,
    TradingAllocationUpdate,
    TradingArmRequest,
    TradingControlsUpdate,
    TrainingCampaignRequest,
)
from .seed import seed_demo
from .strategy_service import strategy_service
from .trading import trading_service
from .training import training_service


scheduler = create_scheduler()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.initialize()
    seed_demo()
    trading_service.initialize()
    training_service.initialize()
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(
    title="量策 · 股票策略跟踪 API",
    version="0.1.0",
    description="沪深A股与港股策略筛选、模拟持仓和信号跟踪服务。",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_log_middleware(request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request_id_token = request_id_context.set(request_id)
    started = time.monotonic()
    try:
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "HTTP request crashed",
                extra={
                    "event": "http_request_failed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                },
            )
            raise
        response.headers["x-request-id"] = request_id
        logger.info(
            "HTTP request completed",
            extra={
                "event": "http_request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
            },
        )
        return response
    finally:
        request_id_context.reset(request_id_token)


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": settings.market_data_mode, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/dashboard")
def dashboard():
    return {
        **repo.dashboard(), "trading": trading_service.dashboard(),
        "backtesting": backtest_service.dashboard(), "training": training_service.dashboard(),
    }


@app.get("/api/strategies")
def strategies():
    return repo.list_strategies()


@app.post("/api/strategies", status_code=201)
async def create_strategy(body: StrategyCreate):
    dsl, explanation, _source = await translate_description(body.description)
    return repo.create_strategy(body.name, body.description, dsl, explanation)


@app.patch("/api/strategies/{strategy_id}")
def update_strategy(strategy_id: str, body: StrategyUpdate):
    result = repo.update_strategy(strategy_id, body.name, body.active)
    if not result:
        raise HTTPException(404, "策略不存在")
    return result


@app.delete("/api/strategies/{strategy_id}", status_code=204)
def delete_strategy(strategy_id: str):
    if not repo.archive_strategy(strategy_id):
        raise HTTPException(404, "策略不存在")


@app.post("/api/strategies/{strategy_id}/execute")
async def execute_strategy(strategy_id: str):
    if settings.market_data_mode == "akshare":
        try:
            await asyncio.to_thread(market_service.refresh_for_strategy_scan)
        except Exception as error:
            raise HTTPException(503, f"行情刷新失败，策略未执行：{str(error)[:240]}") from error
    try:
        return await asyncio.to_thread(strategy_service.execute, strategy_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/strategy-drafts/translate")
async def translate(body: DraftRequest):
    try:
        dsl, explanation, source = await translate_description(body.description)
        return {"dsl": dsl, "explanation": explanation, "source": source, "valid": True, "errors": []}
    except Exception as error:
        raise HTTPException(422, f"策略转换失败：{error}") from error


@app.post("/api/strategy-dialogues", status_code=201)
async def create_strategy_dialogue(body: StrategyDialogueCreate):
    try:
        return await dialogue_service.create(body.name, body.description, body.strategy_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.get("/api/strategy-dialogues/{dialogue_id}")
def get_strategy_dialogue(dialogue_id: str):
    result = dialogue_service.get(dialogue_id)
    if not result:
        raise HTTPException(404, "策略对话不存在")
    return result


@app.post("/api/strategy-dialogues/{dialogue_id}/messages")
async def add_strategy_dialogue_message(dialogue_id: str, body: StrategyDialogueMessage):
    try:
        return await dialogue_service.add_message(dialogue_id, body.content)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.post("/api/strategy-dialogues/{dialogue_id}/retry")
async def retry_strategy_dialogue(dialogue_id: str):
    try:
        return await dialogue_service.retry(dialogue_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.post("/api/strategy-dialogues/{dialogue_id}/compile")
def compile_strategy_dialogue(dialogue_id: str):
    try:
        return dialogue_service.compile(dialogue_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, f"规则化文本编译失败：{error}") from error


@app.post("/api/strategy-dialogues/{dialogue_id}/activate")
def activate_strategy_dialogue(dialogue_id: str):
    try:
        return dialogue_service.activate(dialogue_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, f"策略启用失败：{error}") from error


@app.post("/api/strategies/{strategy_id}/versions/activate")
def activate_version(strategy_id: str, body: DraftActivate):
    errors = validation_errors(body.dsl)
    if errors:
        raise HTTPException(422, detail={"message": "DSL校验失败", "errors": errors})
    validate_dsl(body.dsl)
    result = repo.activate_version(strategy_id, body.description, body.dsl, body.explanation)
    if not result:
        raise HTTPException(404, "策略不存在")
    return result


@app.get("/api/securities")
def search_securities(query: str = Query(min_length=1, max_length=40), limit: int = Query(20, ge=1, le=50)):
    return repo.search_securities(query, limit)


@app.post("/api/strategies/{strategy_id}/candidates")
def add_candidate(strategy_id: str, body: StockAdd):
    try:
        return repo.add_candidate(strategy_id, body.security_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.delete("/api/strategies/{strategy_id}/candidates/{security_id}", status_code=204)
def delete_candidate(strategy_id: str, security_id: str):
    if not repo.delete_candidate(strategy_id, security_id):
        raise HTTPException(404, "待定股票不存在")


@app.post("/api/strategies/{strategy_id}/candidates/{security_id}/promote")
def promote_candidate(strategy_id: str, security_id: str, body: PositionAdd):
    if body.security_id != security_id:
        raise HTTPException(422, "股票标识不一致")
    try:
        return repo.add_position(strategy_id, security_id, body.quantity, body.avg_cost, body.opened_at.isoformat() if body.opened_at else None)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except Exception as error:
        raise HTTPException(409, "该股票已在持仓列表") from error


@app.post("/api/strategies/{strategy_id}/positions")
def add_position(strategy_id: str, body: PositionAdd):
    try:
        return repo.add_position(strategy_id, body.security_id, body.quantity, body.avg_cost, body.opened_at.isoformat() if body.opened_at else None)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except Exception as error:
        raise HTTPException(409, "该股票已在持仓列表") from error


@app.post("/api/strategies/{strategy_id}/positions/{security_id}/adds")
def add_to_position(strategy_id: str, security_id: str, body: PositionIncrease):
    try:
        return repo.add_to_position(
            strategy_id,
            security_id,
            body.quantity,
            body.price,
            body.occurred_at.isoformat() if body.occurred_at else None,
        )
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.delete("/api/strategies/{strategy_id}/positions/{security_id}", status_code=204)
def delete_position(strategy_id: str, security_id: str):
    if not repo.delete_position(strategy_id, security_id):
        raise HTTPException(404, "持仓股票不存在")


@app.post("/api/strategies/{strategy_id}/positions/{security_id}/return")
def return_position(strategy_id: str, security_id: str):
    if not repo.delete_position(strategy_id, security_id, True):
        raise HTTPException(404, "持仓股票不存在")
    return repo.get_strategy(strategy_id)


@app.get("/api/runs")
def runs(limit: int = Query(50, ge=1, le=200)):
    return db.all("SELECT * FROM strategy_runs ORDER BY started_at DESC LIMIT ?", (limit,))


@app.get("/api/signals")
def signals(limit: int = Query(100, ge=1, le=500)):
    return db.all("SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,))


@app.post("/api/backtests", status_code=202)
def create_backtest(body: BacktestRequest, background_tasks: BackgroundTasks):
    try:
        started, dsl = backtest_service.start(body)
        background_tasks.add_task(backtest_service.execute, started["id"], body, dsl)
        return started
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


@app.get("/api/backtests")
def backtests(limit: int = Query(20, ge=1, le=100)):
    return backtest_service.list_runs(limit)


@app.get("/api/backtests/{run_id}")
def backtest_detail(run_id: str):
    try:
        return backtest_service.detail(run_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.get("/api/data-quality")
def training_data_quality():
    return training_service.data_quality()


@app.post("/api/training/campaigns", status_code=202)
def create_training_campaigns(body: TrainingCampaignRequest):
    try:
        return training_service.start_campaigns(body.tracks, body.budget, body.trigger_type)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error


@app.get("/api/training/campaigns")
def training_campaigns(limit: int = Query(50, ge=1, le=200)):
    return training_service.list_campaigns(limit)


@app.get("/api/training/campaigns/{campaign_id}")
def training_campaign_detail(campaign_id: str):
    try:
        return training_service.campaign(campaign_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.post("/api/training/campaigns/{campaign_id}/cancel")
def cancel_training_campaign(campaign_id: str):
    try:
        return training_service.cancel(campaign_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/training/champions/{track}/approve")
def approve_training_champion(track: str):
    try:
        return training_service.approve(track)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/training/champions/{track}/publish-research")
def publish_training_research(track: str):
    try:
        return training_service.publish_research(track)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/training/daily-check")
def run_training_daily_check():
    return training_service.daily_check()


@app.get("/api/trading")
def trading_dashboard():
    return trading_service.dashboard()


@app.get("/api/trading/intents")
def trading_intents(limit: int = Query(100, ge=1, le=500)):
    return trading_service.list_intents(limit)


@app.put("/api/trading/allocations/{strategy_id}")
def update_trading_allocation(strategy_id: str, body: TradingAllocationUpdate):
    try:
        return trading_service.configure_allocation(strategy_id, body.currency, body.capital_limit, body.enabled)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/trading/intents/{intent_id}/approve")
def approve_trading_intent(intent_id: str):
    try:
        return trading_service.approve(intent_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/trading/intents/{intent_id}/reject")
def reject_trading_intent(intent_id: str):
    try:
        return trading_service.reject(intent_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/trading/intents/{intent_id}/cancel")
def cancel_trading_intent(intent_id: str):
    try:
        return trading_service.cancel(intent_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.post("/api/trading/process")
async def process_trading_orders():
    return await asyncio.to_thread(trading_service.process)


@app.patch("/api/trading/controls")
def update_trading_controls(body: TradingControlsUpdate):
    try:
        return trading_service.set_controls(body.entry_paused, body.emergency_stop)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.post("/api/trading/arm")
def arm_trading(body: TradingArmRequest):
    try:
        return trading_service.arm(body.hours)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/trading/reconcile")
def reconcile_trading_account():
    return trading_service.reconcile()


@app.get("/api/market-status")
def market_status():
    return db.all("SELECT * FROM market_status ORDER BY market")


@app.post("/api/market/sync", status_code=202)
def sync_market(body: MarketSyncRequest, background_tasks: BackgroundTasks):
    if settings.market_data_mode != "akshare":
        raise HTTPException(409, "当前为演示数据模式，请先配置 MARKET_DATA_MODE=akshare")
    background_tasks.add_task(market_service.sync_universe_and_quotes)
    if body.backfill_tracked:
        background_tasks.add_task(market_service.backfill_next_batch)
    return {"status": "accepted"}


@app.post("/api/market/backfill/{security_id}")
async def backfill(security_id: str):
    try:
        count = await asyncio.to_thread(market_service.backfill_security, security_id)
        return {"security_id": security_id, "bars": count}
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/market/reference-sync")
async def reference_sync():
    if settings.market_data_mode != "akshare":
        raise HTTPException(409, "当前为演示数据模式，请先配置 MARKET_DATA_MODE=akshare")
    return await asyncio.to_thread(market_service.sync_reference_universe)


@app.get("/api/market/backfill-status")
def market_backfill_status():
    return market_service.backfill_progress()


@app.post("/api/market/backfill-next", status_code=202)
def market_backfill_next(background_tasks: BackgroundTasks, limit: int = Query(30, ge=1, le=100)):
    if settings.market_data_mode != "akshare":
        raise HTTPException(409, "当前为演示数据模式，请先配置 MARKET_DATA_MODE=akshare")
    background_tasks.add_task(market_service.backfill_next_batch, limit)
    return {"status": "accepted", "limit": limit}


@app.get("/api/events")
async def events():
    async def stream():
        while True:
            payload = {"type": "heartbeat", "time": datetime.now(timezone.utc).isoformat()}
            yield f"event: heartbeat\ndata: {json.dumps(payload)}\n\n"
            await asyncio.sleep(15)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
