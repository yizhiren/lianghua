from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from backend.app import dialogue_service as dialogue_module
from backend.app.controlled_language import compile_controlled_text, render_controlled_text
from backend.app.database import Database
from backend.app.dialogue_service import StrategyDialogueService
from backend.app.dsl import LEFT_SIDE_DSL, RIGHT_SIDE_DSL, TWO_DAY_LEFT_SIDE_DSL, validate_dsl
from backend.app.repository import Repository
from backend.app.seed import seed_demo


def test_controlled_language_round_trip_preserves_right_side_rules():
    text = render_controlled_text("周线趋势回踩右侧交易", RIGHT_SIDE_DSL)
    name, compiled = compile_controlled_text(text)
    parsed = validate_dsl(compiled)
    assert name == "周线趋势回踩右侧交易"
    assert len(parsed.rules) == 8
    assert [rule.action for rule in parsed.rules] == [rule["action"] for rule in RIGHT_SIDE_DSL["rules"]]
    assert "条件：全部满足(" in text
    assert "目标仓位：45%" in text


def test_controlled_language_round_trip_preserves_left_side_dynamic_reference():
    text = render_controlled_text("左侧底背离", LEFT_SIDE_DSL)
    name, compiled = compile_controlled_text(text)
    parsed = validate_dsl(compiled)
    selector = parsed.rules[0].condition.conditions[2].right
    assert name == "左侧底背离"
    assert selector.params["select_by"] == "price.close"
    assert selector.params["select_tie"] == "first"
    assert parsed.rules[0].daily_bar_mode == "completed"
    assert "日K口径：仅使用已完成日K" in text
    assert "持仓.建仓日最低价[scale=0.97]" in text


def test_controlled_language_round_trip_preserves_lag_and_security_filters():
    text = render_controlled_text("T-1底背离与T日确认", TWO_DAY_LEFT_SIDE_DSL)
    name, compiled = compile_controlled_text(text)
    parsed = validate_dsl(compiled)
    conditions = parsed.rules[0].condition.conditions
    assert name == "T-1底背离与T日确认"
    assert conditions[0].left.name == "security"
    assert conditions[2].left.field == "listed_trading_days"
    assert conditions[6].left.params["lag"] == 1
    assert conditions[6].right.params == {"lag": 1, "window": 30, "window_op": "min"}
    assert "日K.一字跌停" in text
    assert "lag=1" in text


def test_controlled_language_round_trip_supports_position_add_snapshot_fields():
    dsl = {
        "schema_version": 1,
        "rules": [{
            "id": "add-below-last-add-after-rebound",
            "scope": "position",
            "timeframe": "day",
            "condition": {
                "op": "all",
                "conditions": [
                    {"op": "compare", "left": {"name": "position", "field": "first_add_rebound_confirmed"}, "right": 1, "comparator": "=="},
                    {"op": "compare", "left": {"name": "price", "field": "close"}, "right": {"name": "position", "field": "last_add_price"}, "comparator": "<"},
                    {"op": "compare", "left": {"name": "position", "field": "days_since_last_add"}, "right": 3, "comparator": ">="},
                    {"op": "compare", "left": {"name": "position", "field": "days_since_last_buy"}, "right": 3, "comparator": ">="},
                ],
            },
            "action": "signal_add",
            "label": "已确认反弹且价格低于前次补仓价",
        }],
    }
    text = render_controlled_text("状态补仓", dsl)
    name, compiled = compile_controlled_text(text)
    assert name == "状态补仓"
    conditions = compiled["rules"][0]["condition"]["conditions"]
    assert conditions[0]["left"]["field"] == "first_add_rebound_confirmed"
    assert conditions[1]["right"]["field"] == "last_add_price"
    assert conditions[2]["left"]["field"] == "days_since_last_add"
    assert conditions[3]["left"]["field"] == "days_since_last_buy"
    assert "持仓.首次补仓时反弹已确认" in text
    assert "持仓.前一次补仓价格" in text
    assert "持仓.距前一次补仓自然日数" in text
    assert "持仓.距上次买入或补仓自然日数" in text


def test_dialogue_requires_clarification_before_activation(tmp_path: Path):
    class FakeAIService(StrategyDialogueService):
        def __init__(self, database, repository):
            super().__init__(database, repository, use_ai=True)
            self.calls = 0

        async def _ai_normalize(self, dialogue_id, name, conversation, current_text):
            self.calls += 1
            if self.calls == 1:
                return "我会继续确认仍有歧义的量化参数。", ""
            return "参数已经确认。", render_controlled_text(name, RIGHT_SIDE_DSL)

    database = Database(tmp_path / "dialogue.db")
    database.initialize()
    seed_demo(database, force=True)
    repository = Repository(database)
    service = FakeAIService(database, repository)
    initial = (
        "周线DIFF向上且最好在零轴附近，日K RSI小于60，回踩5周均线时建仓40%-50%，"
        "缩量小阳线后加仓，周线红柱明显缩短时减仓，有效跌破10周线或亏损-5%至-7%止损。"
    )
    dialogue = asyncio.run(service.create("周线趋势回踩右侧交易", initial))
    assert dialogue["ready"] is False
    assert dialogue["validation_errors"] == []
    answers = (
        "确认：零轴附近按1%；回踩范围-2%至+2%；建仓45%；缩量按5日均量70%；"
        "小阳线实体1.5%；红柱缩短30%；连续2个日K跌破算有效；止损-7%。"
    )
    dialogue = asyncio.run(service.add_message(dialogue["id"], answers))
    assert dialogue["ready"] is True
    assert len(dialogue["compiled_dsl"]["rules"]) == 8
    activated = service.activate(dialogue["id"])
    assert activated["strategy"]["name"] == "周线趋势回踩右侧交易"
    assert activated["dialogue"]["status"] == "activated"


def test_every_dialogue_turn_calls_ai_with_full_context(tmp_path: Path):
    class FakeAIService(StrategyDialogueService):
        def __init__(self, database, repository):
            super().__init__(database, repository, use_ai=True)
            self.calls: list[list[dict[str, str]]] = []

        async def _ai_normalize(self, dialogue_id, name, conversation, current_text):
            self.calls.append(conversation)
            return f"这是AI第{len(self.calls)}轮回复，我会解释尚未确认的参数。", ""

    database = Database(tmp_path / "ai-dialogue.db")
    database.initialize()
    seed_demo(database, force=True)
    service = FakeAIService(database, Repository(database))
    dialogue = asyncio.run(
        service.create("AI对话验证", "周线DIFF在零轴附近，日K RSI小于60，回踩5周均线建仓40%-50%，亏损-5%至-7%止损。")
    )
    assert dialogue["ai_status"] == "success"
    assert dialogue["messages"][-1]["content"].startswith("这是AI第1轮回复")
    dialogue = asyncio.run(service.add_message(dialogue["id"], "？为什么都需要确认？"))
    assert len(service.calls) == 2
    assert {"role": "assistant", "content": "这是AI第1轮回复，我会解释尚未确认的参数。"} in service.calls[-1]
    assert {"role": "user", "content": "？为什么都需要确认？"} in service.calls[-1]
    assert dialogue["messages"][-1]["content"].startswith("这是AI第2轮回复")


def test_accepting_all_suggested_values_resolves_second_turn(tmp_path: Path):
    class FakeAIService(StrategyDialogueService):
        def __init__(self, database, repository):
            super().__init__(database, repository, use_ai=True)
            self.histories: list[list[dict[str, str]]] = []

        async def _ai_normalize(self, dialogue_id, name, conversation, current_text):
            self.histories.append(conversation)
            if len(self.histories) == 1:
                return "请确认建议的全部参数。", ""
            return "已采用全部建议值并生成规则。", render_controlled_text(name, RIGHT_SIDE_DSL)

    initial = (
        "周线DIFF上升且在零轴附近，日线RSI参数14小于60，股价回踩5周均线时建仓40%-50%；"
        "缩量十字星后加仓，周线红柱明显缩短时减仓，有效跌破10周均线或亏损-5%至-7%时止损。"
    )
    database = Database(tmp_path / "accept-all-suggestions.db")
    database.initialize()
    seed_demo(database, force=True)
    service = FakeAIService(database, Repository(database))

    first = asyncio.run(service.create("批量确认策略", initial))
    assert first["ready"] is False

    second = asyncio.run(service.add_message(first["id"], "都按照你建议的值来就行"))
    assert service.histories[1] == [
        {"role": "user", "content": initial},
        {"role": "assistant", "content": "请确认建议的全部参数。"},
        {"role": "user", "content": "都按照你建议的值来就行"},
    ]
    assert second["ai_status"] == "success"
    assert second["ready"] is True
    assert len(second["compiled_dsl"]["rules"]) == 8
    assert second["messages"][-1]["content"] == "已采用全部建议值并生成规则。"


def test_ai_failure_pauses_without_local_reply_or_validation(tmp_path: Path):
    class FailedAIService(StrategyDialogueService):
        async def _ai_normalize(self, dialogue_id, name, conversation, current_text):
            raise ValueError("模拟AI无有效响应")

    database = Database(tmp_path / "failed-ai-dialogue.db")
    database.initialize()
    seed_demo(database, force=True)
    service = FailedAIService(database, Repository(database), use_ai=True)

    dialogue = asyncio.run(service.create("失败不降级", "周线DIFF在零轴附近，回踩5周均线建仓40%-50%。"))

    assert dialogue["ai_status"] == "failed"
    assert dialogue["ready"] is False
    assert dialogue["compiled_dsl"] is None
    assert dialogue["validation_errors"] == []
    assert [message["role"] for message in dialogue["messages"]] == ["user"]
    assert "ValueError" in dialogue["ai_error"]


def test_manual_retry_reuses_conversation_and_can_recover(tmp_path: Path):
    class RecoveringAIService(StrategyDialogueService):
        def __init__(self, database, repository):
            super().__init__(database, repository, use_ai=True)
            self.calls = 0

        async def _ai_normalize(self, dialogue_id, name, conversation, current_text):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("第一次调用失败")
            return "AI重试成功，请确认这些量化参数。", ""

    database = Database(tmp_path / "retry-ai-dialogue.db")
    database.initialize()
    seed_demo(database, force=True)
    service = RecoveringAIService(database, Repository(database))

    failed = asyncio.run(service.create("可重试策略", "周线DIFF在零轴附近。"))
    assert failed["ai_status"] == "failed"
    assert len(failed["messages"]) == 1

    recovered = asyncio.run(service.retry(failed["id"]))
    assert service.calls == 2
    assert recovered["ai_status"] == "success"
    assert recovered["messages"][-1]["content"] == "AI重试成功，请确认这些量化参数。"
    assert recovered["validation_errors"] == []
    assert recovered["ready"] is False


def test_ai_transport_retries_before_returning_success(monkeypatch, tmp_path: Path):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            content = json.dumps({"assistant_message": "第三次请求成功", "normalized_text": ""})
            return {"choices": [{"message": {"content": content}}]}

    class FakeClient:
        calls = 0
        last_payload = None

        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            type(self).calls += 1
            type(self).last_payload = kwargs["json"]
            if type(self).calls < 3:
                raise httpx.ConnectError("模拟临时网络故障")
            return FakeResponse()

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(dialogue_module.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(dialogue_module.asyncio, "sleep", no_wait)
    original_attempts = dialogue_module.settings.ai_retry_attempts
    object.__setattr__(dialogue_module.settings, "ai_retry_attempts", 3)
    try:
        database = Database(tmp_path / "transport-retry.db")
        database.initialize()
        service = StrategyDialogueService(database, Repository(database), use_ai=True)
        history = [
            {"role": "user", "content": "原始策略"},
            {"role": "assistant", "content": "请确认参数"},
            {"role": "user", "content": "按你的建议处理"},
        ]
        assistant, normalized = asyncio.run(
            service._ai_normalize("dialogue-id", "策略", history, "")
        )
    finally:
        object.__setattr__(dialogue_module.settings, "ai_retry_attempts", original_attempts)

    assert FakeClient.calls == 3
    assert FakeClient.last_payload["messages"][1:] == history
    assert "不得建议忽略反弹验证" in FakeClient.last_payload["messages"][0]["content"]
    assert "距前一次补仓自然日数" in FakeClient.last_payload["messages"][0]["content"]
    assert "距上次买入或补仓自然日数" in FakeClient.last_payload["messages"][0]["content"]
    assert assistant == "第三次请求成功"
    assert normalized == ""
