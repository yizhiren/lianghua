from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class StrategyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=4000)


class StrategyUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    active: Optional[bool] = None


class DraftRequest(BaseModel):
    description: str = Field(min_length=3, max_length=4000)


class DraftActivate(BaseModel):
    description: str = Field(min_length=3, max_length=4000)
    dsl: Dict[str, Any]
    explanation: List[str] = Field(default_factory=list)


class StrategyDialogueCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=3, max_length=8000)
    strategy_id: Optional[str] = None


class StrategyDialogueMessage(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class StockAdd(BaseModel):
    security_id: str = Field(min_length=3, max_length=32)


class PositionAdd(StockAdd):
    quantity: float = Field(gt=0)
    avg_cost: float = Field(gt=0)
    opened_at: Optional[datetime] = None


class PositionIncrease(BaseModel):
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    occurred_at: Optional[datetime] = None


class MarketSyncRequest(BaseModel):
    backfill_tracked: bool = True


class TradingAllocationUpdate(BaseModel):
    currency: Literal["CNY", "HKD"]
    capital_limit: float = Field(gt=0)
    enabled: bool = True


class TradingControlsUpdate(BaseModel):
    entry_paused: Optional[bool] = None
    emergency_stop: Optional[bool] = None


class TradingArmRequest(BaseModel):
    hours: int = Field(default=16, ge=1, le=24)


class BacktestRequest(BaseModel):
    strategy_id: str = Field(min_length=1, max_length=80)
    start_date: str
    end_date: str
    initial_cny: float = Field(default=1_000_000, gt=0)
    initial_hkd: float = Field(default=1_000_000, gt=0)
    default_position_pct: float = Field(default=10, gt=0, le=100)
    max_security_pct: float = Field(default=10, gt=0, le=100)
    commission_pct: float = Field(default=0.03, ge=0, le=5)
    min_commission: float = Field(default=5, ge=0)
    sell_tax_pct: float = Field(default=0.05, ge=0, le=5)
    slippage_pct: float = Field(default=0.1, ge=0, le=10)
    stop_loss_pct: Optional[float] = Field(default=3, gt=0, le=50)
    max_holding_days: Optional[int] = Field(default=60, ge=1, le=2500)
    force_close: bool = True
    max_securities: int = Field(default=2000, ge=1, le=10000)


class TrainingCampaignRequest(BaseModel):
    tracks: List[Literal["left-A", "right-A", "left-HK", "right-HK"]] = Field(
        default_factory=lambda: ["left-A", "right-A", "left-HK", "right-HK"]
    )
    budget: Optional[int] = Field(default=None, ge=4, le=1000)
    trigger_type: Literal["manual", "scheduled", "smoke"] = "manual"


class RuleIndicator(BaseModel):
    name: Literal["macd", "sma", "ema", "rsi", "kdj", "boll", "atr", "adx", "volatility", "price", "volume", "candle", "security", "position"]
    field: str = "value"
    params: Dict[str, Union[float, int, str]] = Field(default_factory=dict)
    timeframe: Optional[Literal["day", "week"]] = None


class RuleCondition(BaseModel):
    op: Literal[
        "all",
        "any",
        "not",
        "compare",
        "cross_above",
        "cross_below",
        "rising",
        "falling",
        "breakout",
        "within_pct",
        "ratio_pct",
        "relative_change",
        "fraction_of_recent",
        "consecutive",
    ]
    timeframe: Optional[Literal["day", "week"]] = None
    conditions: List["RuleCondition"] = Field(default_factory=list)
    left: Optional[RuleIndicator] = None
    right: Optional[Union[RuleIndicator, float, int]] = None
    comparator: Optional[Literal[">", ">=", "<", "<=", "==", "!="]] = None
    periods: int = Field(default=1, ge=1, le=250)
    direction: Optional[Literal["high", "low"]] = None
    lower: Optional[float] = None
    upper: Optional[float] = None

    @model_validator(mode="after")
    def validate_shape(self):
        if self.op in {"all", "any"} and not self.conditions:
            raise ValueError(f"{self.op} 至少需要一个子条件")
        if self.op in {"not", "consecutive"} and len(self.conditions) != 1:
            raise ValueError(f"{self.op} 必须且只能包含一个子条件")
        if self.op in {"within_pct", "ratio_pct"} and (self.left is None or not isinstance(self.right, RuleIndicator)):
            raise ValueError(f"{self.op} 需要 left 和指标类型的 right")
        if self.op in {"within_pct", "ratio_pct"} and (self.lower is None or self.upper is None or self.lower > self.upper):
            raise ValueError(f"{self.op} 需要有效的 lower 和 upper")
        return self


class StrategyRule(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    scope: Literal["universe", "candidate", "position"]
    timeframe: Literal["day", "week", "mixed"]
    condition: RuleCondition
    action: Literal[
        "add_candidate",
        "highlight_candidate",
        "highlight_position",
        "signal_buy",
        "signal_add",
        "signal_hold",
        "signal_reduce",
        "signal_exit",
        "signal_stop",
    ]
    label: str = Field(min_length=1, max_length=200)
    target_position_pct: Optional[float] = Field(default=None, ge=0, le=100)
    daily_bar_mode: Literal["latest", "completed"] = "latest"

    @field_validator("action")
    @classmethod
    def validate_scope_action(cls, action: str, info):
        scope = info.data.get("scope")
        allowed = {
            "universe": {"add_candidate"},
            "candidate": {"highlight_candidate", "signal_buy"},
            "position": {
                "highlight_position",
                "signal_add",
                "signal_hold",
                "signal_reduce",
                "signal_exit",
                "signal_stop",
            },
        }
        if scope and action not in allowed[scope]:
            raise ValueError(f"{scope} scope 不支持 {action}")
        return action


class StrategyDSL(BaseModel):
    schema_version: Literal[1] = 1
    rules: List[StrategyRule] = Field(min_length=1, max_length=20)


RuleCondition.model_rebuild()
