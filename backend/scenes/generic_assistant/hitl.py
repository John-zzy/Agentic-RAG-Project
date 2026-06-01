from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.prompts import PromptTemplate


GENERIC_ASSISTANT_HITL_SCOPE = "generic_assistant"
logger = logging.getLogger(__name__)


GENERIC_ASSISTANT_HITL_CLARIFICATION_PROMPT = PromptTemplate.from_template(
    """你是通用 RAG 系统里的 Human-in-the-Loop 建议生成器。

【输入】
用户原始问题：
{user_message}

RAG Judge 判断：
{follow_up_question}

【目标】
在 RAG 无法可靠继续召回时，生成一个给用户选择的澄清问题，以及 3 个可点击选项。
用户点击任一选项后，系统会把该选项的 value 直接作为新的用户补充内容提交给 resume，并继续召回。

【生成原则】
1. question 必须是自然问句，用来让用户选择下一步补充方向。
2. suggestions 必须根据“用户原始问题”和“RAG Judge 判断”生成，不能写成固定模板。
3. 每个 suggestion.value 必须是完整、具体、可直接提交的补充内容，不能是占位符、待填写模板或“请补充...”。
4. 只能使用输入中已经明确出现的信息；不要引入输入之外的领域、产品、技术、岗位、订单、商品或业务对象。
5. 你不知道知识库里实际有哪些文档，因此不要假设知识库包含某个具体领域资料。
6. 如果用户原始问题缺少明确主题，value 应使用中立的当前知识库/已挂载知识源表达，让召回继续围绕可用知识范围探索。
7. label 要短，description 说明为什么这个选项有助于继续召回。
8. 只返回 JSON，不要返回 Markdown、解释或多余文本。

【输出 JSON 契约】
返回一个合法 JSON object，字段名和字符串必须使用英文双引号：
- question: string
- suggestions: array，长度必须为 3
- suggestions[].label: string
- suggestions[].description: string
- suggestions[].value: string
"""
)


class GenericAssistantHitlTurn(Protocol):
    """generic HITL 只需要读取的一轮聊天信息。"""

    session_id: str
    request_id: str
    user_message: str
    final_decision: str | None
    follow_up_question: str | None
    answer_mode: str
    scene_metadata: Any


class GenericAssistantHitlSuggestionModel(Protocol):
    """HITL 建议生成只依赖模型客户端的两个基础方法。"""

    def get_runnable(
            self,
            complexity: str = "simple",
            prompt_template: Any | None = None,
            *,
            output_parser: Any | None = None,
    ) -> Any:
        """创建可执行的模型链。"""
        ...

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> Any:
        """执行模型链并返回文本。"""
        ...


@dataclass(frozen=True)
class GenericAssistantHitlOptions:
    """generic_assistant 的 HITL 开关，默认不改变现有聊天行为。"""

    clarification_enabled: bool = False
    test_tools_enabled: bool = False

    def to_metadata(self) -> dict[str, Any]:
        """转换成 scene metadata，方便 ChatService 读取。"""
        return {
            "scope": GENERIC_ASSISTANT_HITL_SCOPE,
            "clarification_enabled": self.clarification_enabled,
            "test_tools_enabled": self.test_tools_enabled,
            "business_extensions_in_scope": False,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "GenericAssistantHitlOptions":
        """从 scene metadata 里读取 HITL 开关。"""
        raw_hitl = metadata.get("hitl")
        if not isinstance(raw_hitl, dict):
            return cls()
        if raw_hitl.get("scope") != GENERIC_ASSISTANT_HITL_SCOPE:
            return cls()
        return cls(
            clarification_enabled=bool(raw_hitl.get("clarification_enabled", False)),
            test_tools_enabled=bool(raw_hitl.get("test_tools_enabled", False)),
        )


@dataclass(frozen=True)
class GenericAssistantHitlWaitPlan:
    """创建 HITL 等待态前，generic_assistant 先整理好的等待计划。"""

    interrupt_id: str
    reason: str
    pending_action: str
    allowed_actions: tuple[str, ...]
    proposed_tool_call: dict[str, Any] | None = None
    suggested_responses: tuple[dict[str, Any], ...] = ()
    allow_freeform_response: bool = False
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenericAssistantHitlClarificationDraft:
    """模型或兜底逻辑生成的 HITL 澄清草稿。"""

    question: str
    suggestions: tuple[dict[str, Any], ...]
    source: str


class GenericAssistantHitlPlanner:
    """只负责判断 generic_assistant 什么时候需要等用户，以及等待内容是什么。"""

    def __init__(
            self,
            options: GenericAssistantHitlOptions | None = None,
            *,
            suggestion_model: GenericAssistantHitlSuggestionModel | None = None,
    ) -> None:
        self.options = options or GenericAssistantHitlOptions()
        self.suggestion_model = suggestion_model

    @classmethod
    def from_scene_metadata(
            cls,
            metadata: dict[str, Any],
            *,
            suggestion_model: GenericAssistantHitlSuggestionModel | None = None,
    ) -> "GenericAssistantHitlPlanner":
        """从场景 metadata 创建 planner。"""
        return cls(
            GenericAssistantHitlOptions.from_metadata(metadata),
            suggestion_model=suggestion_model,
        )

    def should_wait_for_clarification(self, turn: GenericAssistantHitlTurn) -> bool:
        """判断本轮 ask_user 是否要从直接追问改成等待用户补充。"""
        return (
            self.options.clarification_enabled
            and getattr(turn.scene_metadata, "scene", None) == GENERIC_ASSISTANT_HITL_SCOPE
            and turn.final_decision == "ask_user"
            and turn.answer_mode == "follow_up"
        )

    def build_clarification_wait(self, turn: GenericAssistantHitlTurn) -> GenericAssistantHitlWaitPlan:
        """为 ask_user 分支创建等待用户补充信息的计划。"""
        follow_up_question = turn.follow_up_question or "请补充更具体的文档主题、术语或查询范围。"
        draft = self.build_clarification_draft(
            user_message=turn.user_message,
            follow_up_question=follow_up_question,
        )
        return GenericAssistantHitlWaitPlan(
            interrupt_id=f"clarification-{uuid4().hex}",
            reason=draft.question,
            pending_action="clarification",
            allowed_actions=("respond", "reject"),
            suggested_responses=draft.suggestions,
            allow_freeform_response=True,
            metadata={
                "hitl_trigger": "ask_user",
                "judge_follow_up_question": follow_up_question,
                "suggestion_source": draft.source,
            },
        )

    def build_clarification_draft(
        self,
        *,
        user_message: str,
        follow_up_question: str,
    ) -> GenericAssistantHitlClarificationDraft:
        """优先让模型生成追问和选项，失败时用保守兜底。"""
        if self.suggestion_model is not None:
            try:
                raw_output = self._invoke_suggestion_model(
                    user_message=user_message,
                    follow_up_question=follow_up_question,
                )
                draft = self._parse_model_clarification_draft(
                    raw_output,
                    user_message=user_message,
                    follow_up_question=follow_up_question,
                )
                if draft is not None:
                    return draft
            except Exception as exc:  # pragma: no cover - 真实模型异常只需要降级
                logger.warning("Generic HITL suggestion model failed: %s", exc)

        return GenericAssistantHitlClarificationDraft(
            question="我现在缺少可检索的信息。你能补充哪一类内容？",
            suggestions=self._build_fallback_clarification_suggestions(
                user_message=user_message,
                follow_up_question=follow_up_question,
            ),
            source="fallback",
        )

    def _invoke_suggestion_model(
            self,
            *,
            user_message: str,
            follow_up_question: str,
    ) -> Any:
        """调用模型生成 HITL 追问和候选补充方向。"""
        if self.suggestion_model is None:
            return None
        runnable = self.suggestion_model.get_runnable(
            complexity="simple",
            prompt_template=GENERIC_ASSISTANT_HITL_CLARIFICATION_PROMPT,
        )
        return self.suggestion_model.invoke_runnable(
            runnable,
            {
                "user_message": user_message,
                "follow_up_question": follow_up_question,
            },
        )

    def _parse_model_clarification_draft(
            self,
            raw_output: Any,
            *,
            user_message: str,
            follow_up_question: str,
    ) -> GenericAssistantHitlClarificationDraft | None:
        """把模型 JSON 输出整理成前端可以直接消费的结构。"""
        payload = self._load_json_object(raw_output)
        if payload is None:
            return None
        question = self._normalize_question(payload.get("question"))
        suggestions = self._normalize_model_suggestions(
            payload.get("suggestions"),
            follow_up_question=follow_up_question,
        )
        if not question or not suggestions:
            return None
        return GenericAssistantHitlClarificationDraft(
            question=question,
            suggestions=suggestions,
            source="model",
        )

    def _load_json_object(self, raw_output: Any) -> dict[str, Any] | None:
        """兼容模型只返回 JSON 或在 JSON 外包了一层说明的情况。"""
        text = str(raw_output or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                payload = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None

    def _normalize_question(self, value: Any) -> str:
        """把模型给出的追问统一成真正的问句。"""
        question = " ".join(str(value or "").split())
        if not question:
            return ""
        if not question.endswith(("?", "？")):
            question = question.rstrip("。；;，,") + "？"
        return question

    def _normalize_model_suggestions(
            self,
            value: Any,
            *,
            follow_up_question: str,
    ) -> tuple[dict[str, Any], ...]:
        """过滤模型建议项，并补齐可审计的来源信息。"""
        if not isinstance(value, list):
            return ()
        suggestions: list[dict[str, Any]] = []
        for index, item in enumerate(value[:3]):
            if not isinstance(item, dict):
                continue
            label = " ".join(str(item.get("label") or "").split())
            description = " ".join(str(item.get("description") or "").split())
            response_template = " ".join(str(item.get("value") or "").split())
            if not label or not response_template:
                continue
            suggestions.append(
                {
                    "suggestion_id": f"model_clarify_{index + 1}",
                    "label": label,
                    "value": response_template,
                    "description": description or "请选择后补充具体内容。",
                    "metadata": {
                        "judge_follow_up_question": follow_up_question,
                        "source": "model",
                    },
                }
            )
        return tuple(suggestions)

    def _build_fallback_clarification_suggestions(
        self,
        *,
        user_message: str,
        follow_up_question: str,
    ) -> tuple[dict[str, Any], ...]:
        """模型不可用时的保底选项，保证点击后也能继续召回。"""
        short_query = " ".join(user_message.strip().split())[:40] or "当前问题"
        return (
            {
                "suggestion_id": "clarify_topic",
                "label": "查询相关主题",
                "value": f"我想查询与「{short_query}」相关的具体文档主题和说明。",
                "description": "把原始问题改成更适合文档召回的主题查询。",
                "metadata": {
                    "judge_follow_up_question": follow_up_question,
                    "source": "fallback",
                },
            },
            {
                "suggestion_id": "clarify_term",
                "label": "查询关键术语",
                "value": f"我想查询「{short_query}」涉及的关键术语、功能名、流程名或错误码说明。",
                "description": "把问题转成关键词检索，更容易命中文档片段。",
                "metadata": {
                    "judge_follow_up_question": follow_up_question,
                    "source": "fallback",
                },
            },
            {
                "suggestion_id": "clarify_scope",
                "label": "查询业务范围",
                "value": f"请在当前挂载知识源中查询「{short_query}」相关的业务规则、操作流程或限制条件。",
                "description": "限定到业务知识范围，减少无关召回。",
                "metadata": {
                    "judge_follow_up_question": follow_up_question,
                    "source": "fallback",
                },
            },
        )

    def build_write_tool_wait(
        self,
        *,
        tool_name: str,
        operation: str,
        args: dict[str, Any],
    ) -> GenericAssistantHitlWaitPlan:
        """为 generic 写操作创建审批等待计划，工具不会在 approve 前执行。"""
        return self._build_tool_wait(
            pending_action="tool_approval",
            tool_name=tool_name,
            operation=operation,
            args=args,
            risk_level="medium",
        )

    def build_external_api_wait(
        self,
        *,
        tool_name: str,
        operation: str,
        args: dict[str, Any],
    ) -> GenericAssistantHitlWaitPlan:
        """为 generic 外部 API 操作创建审批等待计划，reject 时不会调用外部 API。"""
        return self._build_tool_wait(
            pending_action="external_api_approval",
            tool_name=tool_name,
            operation=operation,
            args=args,
            risk_level="high",
        )

    def _build_tool_wait(
        self,
        *,
        pending_action: str,
        tool_name: str,
        operation: str,
        args: dict[str, Any],
        risk_level: str,
    ) -> GenericAssistantHitlWaitPlan:
        """整理工具审批等待态需要展示和恢复的信息。"""
        return GenericAssistantHitlWaitPlan(
            interrupt_id=f"{pending_action}-{uuid4().hex}",
            reason=f"工具 {tool_name} 将执行 {operation}，需要用户确认。",
            pending_action=pending_action,
            allowed_actions=("approve", "reject"),
            proposed_tool_call={
                "tool_name": tool_name,
                "operation": operation,
                "args": dict(args),
                "risk_level": risk_level,
            },
            suggested_responses=(),
            allow_freeform_response=False,
            metadata={"hitl_trigger": pending_action},
        )
