from __future__ import annotations

import math
import random
import uuid
from datetime import date, datetime, timedelta, timezone

from .config import settings
from .database import Database, db, utcnow
from .dsl import DEFAULT_DESCRIPTION, DEFAULT_DSL, DEFAULT_EXPLANATION


SECURITIES = [
    ("SH.600519", "SH", "600519", "贵州茅台", "CNY", 1488.20),
    ("SH.600036", "SH", "600036", "招商银行", "CNY", 42.18),
    ("SZ.000858", "SZ", "000858", "五粮液", "CNY", 131.62),
    ("SZ.300750", "SZ", "300750", "宁德时代", "CNY", 268.50),
    ("HK.00700", "HK", "00700", "腾讯控股", "HKD", 562.50),
    ("HK.09988", "HK", "09988", "阿里巴巴-W", "HKD", 154.10),
    ("HK.03690", "HK", "03690", "美团-W", "HKD", 108.70),
    ("HK.01810", "HK", "01810", "小米集团-W", "HKD", 54.25),
]


def _seed_bars(database: Database, security_id: str, base_price: float, phase: float) -> None:
    if database.one("SELECT 1 FROM bars WHERE security_id=? LIMIT 1", (security_id,)):
        return
    randomizer = random.Random(security_id)
    current = date.today() - timedelta(days=210)
    price = base_price * 0.82
    rows = []
    index = 0
    while current <= date.today():
        if current.weekday() < 5:
            drift = 0.0012 + 0.006 * math.sin(index / 9 + phase) + randomizer.uniform(-0.012, 0.012)
            open_price = max(0.1, price * (1 + randomizer.uniform(-0.006, 0.006)))
            close = max(0.1, price * (1 + drift))
            high = max(open_price, close) * (1 + randomizer.uniform(0.001, 0.012))
            low = min(open_price, close) * (1 - randomizer.uniform(0.001, 0.012))
            volume = randomizer.randint(500_000, 25_000_000)
            rows.append((security_id, current.isoformat(), open_price, high, low, close, volume, 0))
            price = close
            index += 1
        current += timedelta(days=1)
    database.executemany(
        "INSERT OR IGNORE INTO bars(security_id,trade_date,open,high,low,close,volume,is_provisional) VALUES(?,?,?,?,?,?,?,?)",
        rows,
    )


def seed_demo(database: Database = db, force: bool = False) -> None:
    if settings.market_data_mode != "demo" and not force:
        return
    now = utcnow()
    for index, (security_id, market, code, name, currency, price) in enumerate(SECURITIES):
        database.execute(
            """INSERT INTO securities(id,market,code,name,currency,security_type,is_active,latest_price,quote_time)
               VALUES(?,?,?,?,?,'stock',1,?,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name,latest_price=excluded.latest_price,quote_time=excluded.quote_time""",
            (security_id, market, code, name, currency, price, now),
        )
        _seed_bars(database, security_id, price, index * 0.8)

    for market, total in (("SH", 2385), ("SZ", 2996), ("HK", 2283)):
        database.execute(
            """INSERT INTO market_status(market,state,quote_time,last_sync_at,data_status,message,initialized,backfilled,total)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(market) DO UPDATE SET state=excluded.state,quote_time=excluded.quote_time,last_sync_at=excluded.last_sync_at,
                 data_status=excluded.data_status,message=excluded.message,initialized=excluded.initialized,total=excluded.total""",
            (market, "closed", now, now, "demo", "演示行情 · 配置 AKShare 后可同步公开市场数据", 1, min(8, total), total),
        )

    if database.one("SELECT 1 FROM strategies WHERE deleted_at IS NULL LIMIT 1"):
        return
    strategies = [
        ("strategy-momentum", "趋势动量", DEFAULT_DESCRIPTION),
        (
            "strategy-breakout",
            "强势突破",
            "从全量股票中筛选出日K RSI大于55的股票放入待定列表；待定股票出现KDJ金叉时高亮。",
        ),
    ]
    for strategy_id, name, description in strategies:
        database.execute(
            "INSERT INTO strategies(id,name,description,active,current_version,created_at,updated_at) VALUES(?,?,?,1,1,?,?)",
            (strategy_id, name, description, now, now),
        )
        dsl = DEFAULT_DSL
        explanation = DEFAULT_EXPLANATION
        if strategy_id == "strategy-breakout":
            dsl = {
                "schema_version": 1,
                "rules": [
                    {
                        "id": "rsi-strength",
                        "scope": "universe",
                        "timeframe": "day",
                        "condition": {"op": "compare", "left": {"name": "rsi", "field": "value", "params": {"period": 14}}, "right": 55, "comparator": ">"},
                        "action": "add_candidate",
                        "label": "日K RSI(14) > 55",
                    },
                    {
                        "id": "candidate-kdj-cross",
                        "scope": "candidate",
                        "timeframe": "day",
                        "condition": {"op": "cross_above", "left": {"name": "kdj", "field": "k", "params": {"period": 9}}, "right": {"name": "kdj", "field": "d", "params": {"period": 9}}},
                        "action": "highlight_candidate",
                        "label": "日K KDJ金叉",
                    },
                ],
            }
            explanation = ["全市场日K RSI(14)大于55时加入待定。", "待定股票日K K线上穿D线时高亮。"]
        database.execute(
            "INSERT INTO strategy_versions(id,strategy_id,version,description,dsl_json,explanation_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), strategy_id, 1, description, database.dump(dsl), database.dump(explanation), now),
        )

    candidates = [
        ("strategy-momentum", "SH.600036", "auto", 1, "日K MACD死叉 · 盘中暂定"),
        ("strategy-momentum", "HK.09988", "auto", 0, None),
        ("strategy-breakout", "SZ.300750", "auto", 1, "日K KDJ金叉 · 盘中暂定"),
        ("strategy-breakout", "HK.01810", "manual", 0, None),
    ]
    for strategy_id, security_id, added_by, active, reason in candidates:
        database.execute(
            "INSERT OR IGNORE INTO candidates(id,strategy_id,security_id,added_by,created_at,signal_active,signal_reason,signal_updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), strategy_id, security_id, added_by, now, active, reason, now if active else None),
        )
    positions = [
        ("strategy-momentum", "SH.600519", 100, 1452.30, 1, "日K MACD金叉 · 盘中暂定"),
        ("strategy-momentum", "HK.00700", 200, 528.40, 0, None),
        ("strategy-breakout", "SZ.000858", 500, 126.80, 0, None),
    ]
    for strategy_id, security_id, quantity, cost, active, reason in positions:
        position_id = str(uuid.uuid4())
        opened = (datetime.now(timezone.utc) - timedelta(days=23)).isoformat()
        database.execute(
            "INSERT OR IGNORE INTO positions(id,strategy_id,security_id,quantity,avg_cost,opened_at,created_at,signal_active,signal_reason,signal_updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (position_id, strategy_id, security_id, quantity, cost, opened, now, active, reason, now if active else None),
        )
        database.execute(
            "INSERT INTO position_events(id,strategy_id,security_id,event_type,quantity,price,occurred_at) VALUES(?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), strategy_id, security_id, "opened", quantity, cost, opened),
        )
