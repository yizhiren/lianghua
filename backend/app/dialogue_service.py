from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import httpx

from .config import settings
from .controlled_language import compile_controlled_text, explanations_from_dsl, render_controlled_text
from .database import Database, db, utcnow
from .dsl import DEFAULT_DSL, LEFT_SIDE_DSL, RIGHT_SIDE_DSL, TWO_DAY_LEFT_SIDE_DSL
from .logging_config import logger
from .repository import Repository, repo


class StrategyDialogueService:
    def __init__(self, database: Database = db, repository: Repository = repo, use_ai: bool | None = None):
        self.db = database
        self.repo = repository
        self.use_ai = settings.ai_enabled if use_ai is None else use_ai

    def _message(self, dialogue_id: str, role: str, content: str) -> None:
        self.db.execute(
            "INSERT INTO strategy_dialogue_messages(id,dialogue_id,role,content,created_at) VALUES(?,?,?,?,?)",
            (str(uuid.uuid4()), dialogue_id, role, content, utcnow()),
        )

    def get(self, dialogue_id: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM strategy_dialogues WHERE id=?", (dialogue_id,))
        if not row:
            return None
        messages = self.db.all(
            "SELECT id,role,content,created_at FROM strategy_dialogue_messages WHERE dialogue_id=? ORDER BY created_at,id",
            (dialogue_id,),
        )
        return {
            **row,
            "compiled_dsl": self.db.load(row.get("compiled_dsl_json"), None),
            "explanation": self.db.load(row.get("explanation_json"), []),
            "validation_errors": self.db.load(row.get("validation_errors_json"), []),
            "messages": messages,
            "ready": row["status"] in {"ready", "activated"},
        }

    async def _ai_normalize(
        self,
        dialogue_id: str,
        name: str,
        conversation: list[dict[str, str]],
        current_text: str,
    ) -> tuple[str, str]:
        simple_example = render_controlled_text("简单策略示例", DEFAULT_DSL)
        advanced_example = render_controlled_text("复杂策略示例", RIGHT_SIDE_DSL)
        left_side_example = render_controlled_text("左侧底背离示例", LEFT_SIDE_DSL)
        two_day_left_side_example = render_controlled_text("T-1底背离与T日确认示例", TWO_DAY_LEFT_SIDE_DSL)
        system_prompt = (
            "你是交易策略需求分析师。请根据对话生成受控自然语言策略。只返回JSON对象，字段为assistant_message和normalized_text。"
            "不得创造未确认的阈值；仍有歧义时normalized_text返回空字符串并在assistant_message中逐项提问。"
            "你必须结合随后提供的完整多轮消息判断用户意图。"
            "只要用户明确或概括地接受你上一轮给出的建议值，或授权你按建议处理，就视为那些参数已经全部确认；"
            "此时禁止再次要求确认，必须直接把上一轮建议值写入完整normalized_text。"
            "如果用户输入问号或要求解释，应解释为什么需要确认以及如何回答，不要机械重复上一轮原文。"
            "规则化文本必须严格沿用示例的行结构和中文函数语法，条件必须写在单行。"
            "允许的条件函数：全部满足、任一满足、不满足、大于、大于等于、小于、小于等于、等于、不等于、"
            "上穿、下穿、连续上升、连续下降、突破近高、跌破近低、偏离率介于、比值百分比介于、"
            "相对前期变化、最近峰值占比、连续满足。"
            "允许的动作：加入待定、优选关注、持仓提醒、提示建仓、提示加仓、继续持有、提示减仓、提示清仓、止损清仓。"
            "允许的持仓状态指标：持仓.盈亏百分比、持仓.补仓次数、持仓.前一次补仓价格、持仓.距前一次补仓自然日数、"
            "持仓.距上次买入或补仓自然日数、持仓.建仓日最低价、"
            "持仓.首次补仓时20日最低价、持仓.首次补仓时最低点后最高价、持仓.首次补仓时反弹幅度百分比、"
            "持仓.首次补仓时反弹已确认。反弹已确认取值1或0，其固定定义为：首次补仓时回看此前20个已完成日K，"
            "以最后一次20日最低点为准，最低点之后的最高价达到最低价的1.05倍；不得改写成不考虑时间顺序的20日最高最低价差。"
            "上述首次补仓快照和反弹结果都由系统在记录首次补仓时预先计算并冻结，规则可以直接引用；"
            "不得声称规则语言无法引用历史最低点后的反弹，也不得建议忽略反弹验证。"
            "用户说补仓间隔至少N天时，若未特别说明交易日，按自然日处理，必须编译为大于等于(持仓.距前一次补仓自然日数,N)；"
            "没有前一次补仓记录时该指标为空且条件不命中。"
            "如果用户要求建仓后即使没有补仓也要计算间隔，或表达距上次买入/补仓至少N天，必须使用"
            "大于等于(持仓.距上次买入或补仓自然日数,N)：从未补仓时以建仓时间为起点，补仓后以最近补仓时间为起点。"
            "需要滚动指标时在指标参数中使用window_op=min/max/mean和window=N；需要把指标定位到另一序列极值日时使用"
            "select_by=price.close、select_op=min/max、select_window=N、select_tie=first/last；指标乘固定倍数使用scale。"
            "例如最早20日最低收盘价对应的DIF写成日K.MACD.DIFF[fast=12;select_by=price.close;select_op=min;"
            "select_tie=first;select_window=20;signal=9;slow=26]。这些参数均为系统原生能力，不得声称不支持。"
            "用户明确要求日线收盘后判断时，每条对应规则必须写“日K口径：仅使用已完成日K”。"
            "固定引用前N根K线时在指标参数中使用lag=N；lag在滚动计算之后生效，因此T-1时点的30日最低价必须让"
            "价格和滚动最低价两侧都使用lag=1。股票池可使用日K.沪深A股、日K.ST股票、日K.上市交易日数和"
            "日K.一字跌停；这些值分别用于限制沪深市场、排除ST/*ST、要求已完成日K数量和排除按所属板块跌停价封死的K线。"
            "补仓状态语法示例：全部满足(大于等于(持仓.距前一次补仓自然日数,3),"
            "等于(持仓.首次补仓时反弹已确认,1),偏离率介于(日K.收盘价,持仓.首次补仓时20日最低价,-3,3))。"
            "消除全部歧义后必须生成完整normalized_text；仍有歧义时normalized_text必须为空字符串。"
            f"\n简单语法示例：\n{simple_example}\n复杂语法示例：\n{advanced_example}"
            f"\n左侧底背离语法示例：\n{left_side_example}"
            f"\nT-1与T日组合语法示例：\n{two_day_left_side_example}"
            f"\n当前策略名：{name}\n当前规则化文本：\n{current_text}"
        )
        headers = {"Authorization": f"Bearer {settings.ai_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": settings.ai_model,
            "messages": [{"role": "system", "content": system_prompt}, *conversation],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 4096,
        }
        root = settings.ai_base_url if settings.ai_base_url.endswith("/v1") else f"{settings.ai_base_url}/v1"
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=settings.ai_timeout_seconds) as client:
            for attempt in range(1, settings.ai_retry_attempts + 1):
                started = time.monotonic()
                logger.info(
                    "AI request started",
                    extra={
                        "event": "ai_request_started",
                        "dialogue_id": dialogue_id,
                        "attempt": attempt,
                        "max_attempts": settings.ai_retry_attempts,
                        "model": settings.ai_model,
                    },
                )
                try:
                    response = await client.post(f"{root}/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                    parsed = json.loads(response.json()["choices"][0]["message"]["content"])
                    assistant = str(parsed.get("assistant_message", "")).strip()
                    normalized = str(parsed.get("normalized_text", "")).strip()
                    if not assistant:
                        raise ValueError("AI响应缺少assistant_message")
                    logger.info(
                        "AI request succeeded",
                        extra={
                            "event": "ai_request_succeeded",
                            "dialogue_id": dialogue_id,
                            "attempt": attempt,
                            "max_attempts": settings.ai_retry_attempts,
                            "duration_ms": round((time.monotonic() - started) * 1000, 2),
                            "status_code": response.status_code,
                            "model": settings.ai_model,
                        },
                    )
                    return assistant, normalized
                except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
                    last_error = error
                    status_code = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                    retryable = status_code is None or status_code in {408, 429} or status_code >= 500
                    logger.warning(
                        "AI request failed",
                        extra={
                            "event": "ai_request_failed",
                            "dialogue_id": dialogue_id,
                            "attempt": attempt,
                            "max_attempts": settings.ai_retry_attempts,
                            "duration_ms": round((time.monotonic() - started) * 1000, 2),
                            "status_code": status_code,
                            "error_type": type(error).__name__,
                            "error_detail": str(error)[:500],
                            "model": settings.ai_model,
                        },
                    )
                    if attempt >= settings.ai_retry_attempts or not retryable:
                        raise
                    await asyncio.sleep(settings.ai_retry_backoff_seconds * (2 ** (attempt - 1)))
        if last_error:
            raise last_error
        raise RuntimeError("AI请求未执行")

    def _save_result(
        self,
        dialogue_id: str,
        assistant: str,
        normalized_text: str,
        dsl: dict[str, Any] | None,
        errors: list[str],
        ai_status: str,
        ai_error: str | None = None,
    ) -> None:
        explanation = explanations_from_dsl(dsl) if dsl else []
        status = "ready" if dsl and not errors else "clarifying"
        self.db.execute(
            """UPDATE strategy_dialogues SET normalized_text=?,compiled_dsl_json=?,explanation_json=?,
               validation_errors_json=?,status=?,ai_status=?,ai_error=?,updated_at=? WHERE id=?""",
            (
                normalized_text,
                self.db.dump(dsl) if dsl else None,
                self.db.dump(explanation),
                self.db.dump(errors),
                status,
                ai_status,
                ai_error,
                utcnow(),
                dialogue_id,
            ),
        )
        self._message(dialogue_id, "assistant", assistant)

    def _save_ai_failure(self, dialogue_id: str, error: str, error_type: str, ai_status: str = "failed") -> None:
        self.db.execute(
            """UPDATE strategy_dialogues SET compiled_dsl_json=NULL,explanation_json='[]',validation_errors_json='[]',
               status='clarifying',ai_status=?,ai_error=?,updated_at=? WHERE id=?""",
            (ai_status, error, utcnow(), dialogue_id),
        )
        logger.error(
            "Dialogue paused because AI is unavailable",
            extra={
                "event": "dialogue_ai_unavailable",
                "dialogue_id": dialogue_id,
                "error_type": error_type,
                "error_detail": error[:500],
            },
        )

    async def _advance(self, dialogue_id: str) -> dict[str, Any]:
        dialogue = self.get(dialogue_id)
        if not dialogue:
            raise LookupError("策略对话不存在")
        conversation = [
            {"role": item["role"], "content": item["content"]}
            for item in dialogue["messages"]
            if item["role"] in {"user", "assistant"}
        ]
        ai_assistant = ""
        ai_normalized = ""
        ai_status = "disabled"
        ai_error = None
        if self.use_ai:
            ai_status = "requested"
            try:
                ai_assistant, ai_normalized = await self._ai_normalize(
                    dialogue_id, dialogue["strategy_name"], conversation, dialogue["normalized_text"]
                )
                ai_status = "success"
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
                ai_status = "failed"
                ai_error = f"{type(error).__name__}: {str(error)[:240]}"
        else:
            ai_error = "AI未配置，策略对话不能继续"
        if ai_status != "success":
            self._save_ai_failure(
                dialogue_id,
                ai_error or "AI请求失败",
                "AI_DISABLED" if not self.use_ai else (ai_error or "AI_ERROR").split(":", 1)[0],
                ai_status,
            )
            return self.get(dialogue_id) or {}
        assistant = ai_assistant.strip() or "规则已经消除歧义，可以编译为确定性DSL。请检查右侧规则化文本后确认启用。"
        normalized_text = ai_normalized
        if not normalized_text:
            logger.info(
                "Dialogue is waiting for AI clarification",
                extra={"event": "dialogue_clarification_required", "dialogue_id": dialogue_id},
            )
            self._save_result(
                dialogue_id,
                assistant,
                dialogue["normalized_text"],
                None,
                [],
                ai_status,
                ai_error,
            )
            return self.get(dialogue_id) or {}
        try:
            _name, dsl = compile_controlled_text(normalized_text)
            self._save_result(dialogue_id, assistant, normalized_text, dsl, [], ai_status, ai_error)
            logger.info(
                "Controlled language compiled successfully",
                extra={
                    "event": "controlled_language_compiled",
                    "dialogue_id": dialogue_id,
                    "rule_count": len(dsl.get("rules", [])),
                },
            )
        except ValueError as error:
            logger.warning(
                "Controlled language compilation failed",
                extra={
                    "event": "controlled_language_compile_failed",
                    "dialogue_id": dialogue_id,
                    "error_type": type(error).__name__,
                    "error_detail": str(error)[:500],
                },
            )
            self._save_result(
                dialogue_id,
                ai_assistant.strip() or f"当前规则化文本还不能编译：{error}。请补充或修正规则。",
                normalized_text,
                None,
                [str(error)],
                ai_status,
                ai_error,
            )
        return self.get(dialogue_id) or {}

    async def create(self, name: str, description: str, strategy_id: str | None = None) -> dict[str, Any]:
        if strategy_id and not self.repo.get_strategy(strategy_id):
            raise LookupError("策略不存在")
        dialogue_id = str(uuid.uuid4())
        now = utcnow()
        self.db.execute(
            """INSERT INTO strategy_dialogues(id,strategy_id,strategy_name,original_description,status,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?)""",
            (dialogue_id, strategy_id, name, description, "clarifying", now, now),
        )
        self._message(dialogue_id, "user", description)
        return await self._advance(dialogue_id)

    async def add_message(self, dialogue_id: str, content: str) -> dict[str, Any]:
        if not self.get(dialogue_id):
            raise LookupError("策略对话不存在")
        self._message(dialogue_id, "user", content)
        return await self._advance(dialogue_id)

    async def retry(self, dialogue_id: str) -> dict[str, Any]:
        if not self.get(dialogue_id):
            raise LookupError("策略对话不存在")
        logger.info("Dialogue AI retry requested", extra={"event": "dialogue_retry", "dialogue_id": dialogue_id})
        return await self._advance(dialogue_id)

    def compile(self, dialogue_id: str) -> dict[str, Any]:
        dialogue = self.get(dialogue_id)
        if not dialogue:
            raise LookupError("策略对话不存在")
        name, dsl = compile_controlled_text(dialogue["normalized_text"])
        explanation = explanations_from_dsl(dsl)
        self.db.execute(
            """UPDATE strategy_dialogues SET strategy_name=?,compiled_dsl_json=?,explanation_json=?,
               validation_errors_json='[]',status='ready',updated_at=? WHERE id=?""",
            (name, self.db.dump(dsl), self.db.dump(explanation), utcnow(), dialogue_id),
        )
        return self.get(dialogue_id) or {}

    def activate(self, dialogue_id: str) -> dict[str, Any]:
        dialogue = self.compile(dialogue_id)
        dsl = dialogue["compiled_dsl"]
        explanation = dialogue["explanation"]
        if dialogue.get("strategy_id"):
            strategy = self.repo.activate_version(
                dialogue["strategy_id"], dialogue["original_description"], dsl, explanation
            )
            if not strategy:
                raise LookupError("策略不存在")
        else:
            strategy = self.repo.create_strategy(
                dialogue["strategy_name"], dialogue["original_description"], dsl, explanation
            )
        self.db.execute(
            """UPDATE strategy_dialogues SET strategy_id=?,status='activated',activated_version=?,updated_at=? WHERE id=?""",
            (strategy["id"], strategy["current_version"], utcnow(), dialogue_id),
        )
        return {"dialogue": self.get(dialogue_id), "strategy": strategy}


dialogue_service = StrategyDialogueService()
