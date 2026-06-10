from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import ast
import json
from threading import Event, Lock
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from backend.application.runtime.api.app import create_app
from backend.application.runtime.api.chat.schemas import ChatRequest
from backend.application.runtime import ChatService, SceneChatService, build_default_scene_registry
from backend.platform.agent_runtime.core.contracts import ReActAction, ReActRun, ReActTurn, ToolObservation
from backend.platform.agent_runtime.plan.planner import PlanDraft, PlanStepDraft
from backend.platform.agent_runtime.tooling.rag import AGENTIC_RAG_TOOL_NAME, NATIVE_RAG_TOOL_NAME
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.config.settings import AppSettings
from backend.platform.knowledge.repositories import VectorStoreFactory
from backend.platform.memory.base.session_store import SQLiteSessionStore
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.rag.contracts import RetrievalResult
from backend.platform.rag.retrieval.documents import DocumentChunkRetrievalResult
from backend.platform.rag.retrieval.documents.service import DocumentRetrievalService
from backend.platform.search_foundation import VectorSearchResult, VectorStoreDocument
from backend.platform.workflow.langgraph.config import DEFAULT_RUNTIME_CHECKPOINT_NS
from backend.platform.models.llm.guards import JsonSchemaGuard
from backend.scenes.base import SceneDefinition, SceneFallbackPolicy, SceneRetrievalPolicy
from backend.scenes.ecommerce.knowledge_service import create_knowledge_service
from backend.tests.test_support import make_test_runtime_dir


def _result(
    doc_id: str,
    content: str,
    score: float,
    metadata: dict[str, Any],
) -> VectorSearchResult:
    return VectorSearchResult(
        document=VectorStoreDocument(
            id=doc_id,
            content=content,
            metadata=metadata,
        ),
        score=score,
    )


def _build_document_retrieval_service(app_settings: AppSettings) -> DocumentRetrievalService:
    return DocumentRetrievalService(
        app_settings=app_settings,
        vector_repository=VectorStoreFactory.create_document_chunk_vector_repository(app_settings),
        chunk_source=VectorStoreFactory.create_active_document_chunk_source(app_settings),
    )


class FakeKnowledgeService:
    def __init__(
        self,
        products: list[VectorSearchResult] | None = None,
        reviews: list[VectorSearchResult] | None = None,
        documents: list[VectorSearchResult] | None = None,
    ) -> None:
        self._products = products or []
        self._reviews = reviews or []
        del documents

    def search_products(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        return self._products

    def search_reviews(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        return self._reviews

    def search_orders(self, query: str, top_k: int | None = None) -> list[VectorSearchResult]:
        return []

class FakeDocumentRetrievalService:
    def __init__(self, documents: list[VectorSearchResult] | None = None) -> None:
        self._documents = documents or []
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: str = "hybrid",
    ) -> list[DocumentChunkRetrievalResult]:
        del namespace
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "minimum_relevance": minimum_relevance,
                "recall_strategy": recall_strategy,
            }
        )
        documents = self._documents
        if minimum_relevance is not None:
            documents = [
                result
                for result in documents
                if result.score is None or float(result.score) >= minimum_relevance
            ]
        return [
            DocumentChunkRetrievalResult(
                document=result.document,
                score=result.score,
                vector_score=result.score,
                vector_rank=index,
                matched_by=["vector"],
            )
            for index, result in enumerate(documents[:top_k], start=1)
        ]


class QueryFilteredFakeDocumentRetrievalService(FakeDocumentRetrievalService):
    def __init__(
        self,
        documents: list[VectorSearchResult] | None = None,
        *,
        allowed_query_terms: tuple[str, ...] = (),
    ) -> None:
        super().__init__(documents=documents)
        self._allowed_query_terms = allowed_query_terms

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        minimum_relevance: float | None = None,
        recall_strategy: str = "hybrid",
    ) -> list[DocumentChunkRetrievalResult]:
        normalized_query = query.lower()
        if not any(term.lower() in normalized_query for term in self._allowed_query_terms):
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "minimum_relevance": minimum_relevance,
                    "recall_strategy": recall_strategy,
                }
            )
            return []
        return super().retrieve(
            query=query,
            top_k=top_k,
            namespace=namespace,
            minimum_relevance=minimum_relevance,
            recall_strategy=recall_strategy,
        )


class FakeModel:
    def __init__(
        self,
        answer: str = "mock-answer",
        stream_chunks: list[str] | None = None,
        plan_draft: PlanDraft | None = None,
    ) -> None:
        self.answer = answer
        self.get_runnable_calls: list[str] = []
        self.invoke_runnable_calls: list[dict[str, Any]] = []
        self.stream_runnable_calls: list[dict[str, Any]] = []
        self.history_recorder = HistoryRecorder()
        self.stream_chunks = stream_chunks or [answer]
        self.plan_draft = plan_draft

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ):
        del output_parser
        if _is_react_selector_prompt(prompt_template):
            return FakeReactSelectorRunnable()
        self.get_runnable_calls.append(complexity)
        if prompt_template is None:
            return FakeAnswerRunnable(
                answer=self.answer,
                stream_chunks=self.stream_chunks,
                history_recorder=self.history_recorder,
            )
        return prompt_template | FakeAnswerRunnable(
            answer=self.answer,
            stream_chunks=self.stream_chunks,
            history_recorder=self.history_recorder,
        )

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        if isinstance(runnable, FakeReactSelectorRunnable):
            return runnable.invoke(input, config=config)
        self.invoke_runnable_calls.append({"runnable": runnable, "input": input, "config": config})
        return runnable.invoke(input, config=config)

    def invoke_json_schema(
        self,
        runnable: Any,
        input: Any,
        *,
        schema_model: type[Any],
        schema_source: str,
        config: Any | None = None,
        complexity: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        del complexity
        raw_output = self.invoke_runnable(runnable, input, config=config)
        return JsonSchemaGuard().validate(
            raw_output,
            schema_model=schema_model,
            source=schema_source,
            metadata=metadata,
        )

    def stream_runnable(
        self,
        runnable: Any,
        input: Any,
        *,
        config: Any | None = None,
    ):
        self.stream_runnable_calls.append({"runnable": runnable, "input": input, "config": config})
        yield from runnable.stream(input, config=config)

    def build_chat_model_for_complexity(
        self,
        complexity: str = "simple",
    ):
        del complexity
        return _FakeProviderChatModel(answer=self.answer, plan_draft=self.plan_draft)

    def get_chat_model_provider(self) -> Any:
        return lambda complexity="simple": self.build_chat_model_for_complexity(complexity)


class _FakeProviderChatModel(BaseChatModel):
    answer: str
    plan_draft: PlanDraft | None = None
    bound_tool_names: list[str] = []
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-provider-chat-model"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        del tool_choice, kwargs
        object.__setattr__(
            self,
            "bound_tool_names",
            [str(getattr(tool, "name", tool)) for tool in tools],
        )
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableSerializable[Any, Any]:
        del kwargs
        if schema is PlanDraft:
            return _FakePlanStructuredModel(plan_draft=self.plan_draft)
        return super().with_structured_output(schema)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        next_count = self.call_count + 1
        object.__setattr__(self, "call_count", next_count)
        if self.bound_tool_names and next_count == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.bound_tool_names[0],
                        "args": {"query": _latest_user_message(messages)},
                        "id": "call-fake-provider-1",
                    }
                ],
            )
        else:
            message = AIMessage(content=self.answer)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _latest_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type == "human":
            return str(message.content)
    return "test query"


class _FakePlanStructuredModel(RunnableSerializable[Any, PlanDraft]):
    plan_draft: PlanDraft | None = None

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> PlanDraft:
        del config, kwargs
        if self.plan_draft is not None:
            return self.plan_draft
        prompt_text = _prompt_text(input)
        tool_name = _first_plan_allowed_tool(prompt_text)
        return PlanDraft(
            steps=[
                PlanStepDraft(
                    step_id="step-1",
                    goal="调用允许的检索工具收集证据。",
                    tool_name=tool_name,
                    input=_plan_default_input(prompt_text, tool_name=tool_name),
                )
            ],
            rationale_summary="测试默认计划：选择一个可执行 RAG 工具。",
        )


class FakeRewriteModel:
    def __init__(self, output: str) -> None:
        self.output = output
        self.get_runnable_calls: list[str] = []
        self.invoke_runnable_calls: list[dict[str, Any]] = []

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ) -> object:
        del prompt_template, output_parser
        self.get_runnable_calls.append(complexity)
        return object()

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        self.invoke_runnable_calls.append({"runnable": runnable, "input": input, "config": config})
        return self.output

    def invoke_json_schema(
        self,
        runnable: Any,
        input: Any,
        *,
        schema_model: type[Any],
        schema_source: str,
        config: Any | None = None,
        complexity: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        del complexity
        raw_output = self.invoke_runnable(runnable, input, config=config)
        return JsonSchemaGuard().validate(
            raw_output,
            schema_model=schema_model,
            source=schema_source,
            metadata=metadata,
        )


class FakeAnswerRunnable(RunnableSerializable[Any, str]):
    answer: str
    stream_chunks: list[str]
    history_recorder: Any

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> str:
        del config, kwargs
        self._record_prompt_messages(input)
        return self.answer

    def stream(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ):
        del config, kwargs
        self._record_prompt_messages(input)
        for chunk in self.stream_chunks:
            yield chunk

    def _record_prompt_messages(self, input: Any) -> None:
        if isinstance(input, ChatPromptValue):
            messages = input.to_messages()
        elif isinstance(input, list) and all(isinstance(message, BaseMessage) for message in input):
            messages = list(input)
        else:
            return

        history_messages = [
            message
            for message in messages[:-1]
            if isinstance(message, BaseMessage) and message.type in {"human", "ai"}
        ]
        self.history_recorder.snapshots.append(history_messages)


class FakeReactSelectorRunnable(RunnableSerializable[Any, str]):
    """测试用 LLM 调度器：只为 ReAct selector 返回结构化 action。"""

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> str:
        del config, kwargs
        if isinstance(input, dict):
            return self._invoke_from_variables(input)
        prompt_text = _prompt_text(input)
        if _react_previous_turn_count(prompt_text) > 0:
            return json.dumps(
                {
                    "action_type": "final_answer",
                    "rationale_summary": "已有工具观察，进入最终汇总。",
                },
                ensure_ascii=False,
            )
        tool_name = _first_react_preferred_tool(prompt_text)
        return json.dumps(
            {
                "action_type": "tool_call",
                "tool_name": tool_name,
                "input": {},
                "rationale_summary": "首轮先调用允许的 RAG 工具。",
            },
            ensure_ascii=False,
        )

    def _invoke_from_variables(self, variables: dict[str, Any]) -> str:
        previous_turns = _loads_json_value(variables.get("react_previous_turns_json"))
        if isinstance(previous_turns, list) and previous_turns:
            return json.dumps(
                {
                    "action_type": "final_answer",
                    "rationale_summary": "已有工具观察，进入最终汇总。",
                },
                ensure_ascii=False,
            )
        policy = _loads_json_value(variables.get("react_scene_policy_json"))
        tool_name = AGENTIC_RAG_TOOL_NAME
        if isinstance(policy, dict):
            preferred_tools = policy.get("preferred_tools")
            if isinstance(preferred_tools, list) and preferred_tools:
                tool_name = str(preferred_tools[0])
        return json.dumps(
            {
                "action_type": "tool_call",
                "tool_name": tool_name,
                "input": {},
                "rationale_summary": "首轮先调用允许的 RAG 工具。",
            },
            ensure_ascii=False,
        )


class DirectAnswerSelectorRunnable(RunnableSerializable[Any, str]):
    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> str:
        del input, config, kwargs
        return json.dumps(
            {
                "action_type": "final_answer",
                "rationale_summary": "无需检索知识库，直接回答。",
            },
            ensure_ascii=False,
        )


class DirectAnswerModel(FakeModel):
    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ):
        if _is_react_selector_prompt(prompt_template):
            return DirectAnswerSelectorRunnable()
        if prompt_template is None:
            raise AssertionError("direct answer should use an explicit prompt template")
        return super().get_runnable(
            complexity=complexity,
            prompt_template=prompt_template,
            output_parser=output_parser,
        )

    def build_chat_model_for_complexity(
        self,
        complexity: str = "simple",
    ):
        del complexity
        return _DirectAnswerChatModel(answer=self.answer)


class _DirectAnswerChatModel(_FakeProviderChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        del tools, tool_choice, kwargs
        object.__setattr__(self, "bound_tool_names", [])
        return self


def _is_react_selector_prompt(prompt_template: Any | None) -> bool:
    template = str(getattr(prompt_template, "template", "") or "")
    return "REACT_SELECTOR" in template


def _prompt_text(input: Any) -> str:
    if isinstance(input, ChatPromptValue):
        return "\n".join(str(message.content) for message in input.to_messages())
    if isinstance(input, list) and all(isinstance(message, BaseMessage) for message in input):
        return "\n".join(str(message.content) for message in input)
    if isinstance(input, list) and all(isinstance(message, dict) for message in input):
        return "\n".join(str(message.get("content") or "") for message in input)
    return str(input)


def _first_plan_allowed_tool(prompt_text: str) -> str:
    tools = _plan_allowed_tools(prompt_text)
    for preferred_tool in (AGENTIC_RAG_TOOL_NAME, NATIVE_RAG_TOOL_NAME):
        if preferred_tool in tools:
            return preferred_tool
    return tools[0] if tools else AGENTIC_RAG_TOOL_NAME


def _plan_default_input(prompt_text: str, *, tool_name: str) -> dict[str, Any]:
    for tool in _plan_tool_payloads(prompt_text):
        if str(tool.get("name") or "") != tool_name:
            continue
        default_input = tool.get("default_input")
        if isinstance(default_input, dict) and default_input:
            return dict(default_input)
    return {"query": _plan_user_goal(prompt_text)}


def _plan_allowed_tools(prompt_text: str) -> list[str]:
    return [
        str(tool.get("name") or "")
        for tool in _plan_tool_payloads(prompt_text)
        if tool.get("name")
    ]


def _plan_tool_payloads(prompt_text: str) -> list[dict[str, Any]]:
    raw_value = _line_after_label(prompt_text, "Allowed tools and default inputs:")
    if not raw_value:
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def _plan_user_goal(prompt_text: str) -> str:
    return _line_after_label(prompt_text, "User goal:") or "test query"


def _line_after_label(prompt_text: str, label: str) -> str:
    start = prompt_text.find(label)
    if start < 0:
        return ""
    start += len(label)
    lines = prompt_text[start:].splitlines()
    return lines[0].strip() if lines else ""


def _react_previous_turn_count(prompt_text: str) -> int:
    payload = _json_after_label(prompt_text, "历史 turn 摘要：")
    return len(payload) if isinstance(payload, list) else 0


def _first_react_preferred_tool(prompt_text: str) -> str:
    policy = _json_after_label(prompt_text, "scene 策略：")
    if isinstance(policy, dict):
        preferred_tools = policy.get("preferred_tools")
        if isinstance(preferred_tools, list) and preferred_tools:
            return str(preferred_tools[0])
    allowed_tools = _json_after_label(prompt_text, "允许工具：")
    if isinstance(allowed_tools, list) and allowed_tools:
        return str(allowed_tools[0])
    return AGENTIC_RAG_TOOL_NAME


def _json_after_label(prompt_text: str, label: str) -> Any:
    start = prompt_text.find(label)
    if start < 0:
        return None
    start += len(label)
    lines = prompt_text[start:].splitlines()
    raw_json = ""
    for line in lines:
        stripped = line.strip()
        if stripped:
            raw_json = stripped
            break
    if not raw_json:
        return None
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        return None


def _loads_json_value(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


class HistoryRecorder:
    def __init__(self) -> None:
        self.snapshots: list[list[BaseMessage]] = []


def _parse_sse_events(raw_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_payload = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            if line.startswith("data: "):
                data_payload = line[len("data: ") :]
        events.append({"event": event_name, "data": data_payload})
    return events


def _assert_display_sse_events(
    events: list[dict[str, Any]],
    *,
    final_event: str = "done",
) -> None:
    event_names = [event["event"] for event in events]
    assert event_names[0] == "start"
    assert event_names[-1] == final_event
    assert "history" not in event_names
    assert "tool" not in event_names
    assert set(event_names) <= {"start", "thinking", "chunk", "waiting_user", "done", "error"}


def _event_payloads(events: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [
        json.loads(event["data"])
        for event in events
        if event["event"] == event_name and event["data"]
    ]


def _build_chat_service(
    test_name: str,
    knowledge_service: FakeKnowledgeService,
    document_retrieval_service: FakeDocumentRetrievalService,
    model: FakeModel,
) -> SceneChatService:
    runtime_dir = make_test_runtime_dir(test_name)
    files_root = runtime_dir / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    for result in document_retrieval_service._documents:
        source_path = result.document.metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            continue
        file_path = files_root / source_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(result.document.content, encoding="utf-8")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={
            "active_scene": "generic_assistant",
        },
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        }
    )
    return SceneChatService(
        scene_registry=build_default_scene_registry(
            app_settings=app_settings,
            knowledge_service=knowledge_service,  # type: ignore[arg-type]
            document_retrieval_service=document_retrieval_service,  # type: ignore[arg-type]
        ),
        app_settings=app_settings,
        knowledge_service=knowledge_service,  # type: ignore[arg-type]
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )


class _ForcedDecision:
    def __init__(
        self,
        *,
        next_action: str,
        is_sufficient: bool,
        follow_up_question: str | None = None,
    ) -> None:
        self.next_action = next_action
        self.is_sufficient = is_sufficient
        self.follow_up_question = follow_up_question


class _ForcedOutcome:
    def __init__(
        self,
        *,
        documents: list[Document],
        success: bool | None,
        exit_reason: str,
        final_decision: _ForcedDecision | None = None,
        follow_up_question: str | None = None,
        include_success: bool = True,
        include_final_decision: bool = True,
    ) -> None:
        self.documents = documents
        self.exit_reason = exit_reason
        self.rounds: list[object] = []
        self.follow_up_question = follow_up_question
        if include_success:
            self.success = success
        if include_final_decision:
            self.final_decision = final_decision


class _ForcedOutcomeRetriever:
    def __init__(self, outcome: _ForcedOutcome) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def retrieve_with_trace(self, query: str, *, candidate_tools: tuple[str, ...], **kwargs: Any):
        self.calls.append(
            {
                "query": query,
                "candidate_tools": candidate_tools,
                "kwargs": kwargs,
            }
        )
        return self.outcome


def _forced_document() -> Document:
    return Document(
        page_content="强制测试文档：这是可引用证据。",
        metadata={
            "document_id": "doc-forced",
            "source_path": "forced.md",
            "namespace": "documents",
            "chunk_id": "chunk-forced",
            "chunk_index": 0,
            "score": 0.88,
        },
    )


def _build_forced_outcome_chat_service(
    test_name: str,
    *,
    outcome: _ForcedOutcome,
    model: FakeModel,
) -> ChatService:
    runtime_dir = make_test_runtime_dir(test_name)
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={"active_scene": "forced_scene"},
        session={"sqlite_path": sqlite_path, "window_size": 3},
    )
    retriever = _ForcedOutcomeRetriever(outcome)
    scene_definition = SceneDefinition(
        scene="forced_scene",
        name="Forced Scene",
        description="Forced outcome scene for runtime boundary tests.",
        build_retriever=lambda: retriever,  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda mounted: ("forced_tool",),
        system_prompt="你是测试助手。",
        fallback_policy=SceneFallbackPolicy(
            no_hit_message="测试兜底：没有可用证据，请补充信息。",
        ),
        infer_complexity=lambda message: "simple",
        retrieval_policy=SceneRetrievalPolicy(),
    )
    return ChatService(
        scene_definition=scene_definition,
        app_settings=app_settings,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )


class _MultiStepPlanRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls.append({"method": "retrieve", "query": query, "kwargs": kwargs})
        document = _agent_runtime_document(
            document_id="doc-plan-native",
            chunk_id="chunk-plan-native",
            content="Plan 第一步证据：Native RAG 找到了计划背景。",
            source_path="plan-native.md",
            score=0.91,
        )
        return RetrievalResult.ok(
            tool_name="knowledge_document_search",
            query=query,
            records=[
                {
                    "citation_id": "chunk-plan-native",
                    "content": document.page_content,
                    "metadata": dict(document.metadata),
                }
            ],
            documents=[document],
            metadata={
                "document_retrieval_trace": {
                    "raw_candidates_count": 1,
                    "filtered_candidates_count": 1,
                }
            },
        )

    def retrieve_with_trace(self, query: str, *, candidate_tools: tuple[str, ...], **kwargs: Any):
        self.calls.append(
            {
                "method": "retrieve_with_trace",
                "query": query,
                "candidate_tools": candidate_tools,
                "kwargs": kwargs,
            }
        )
        return _ForcedOutcome(
            documents=[
                _agent_runtime_document(
                    document_id="doc-plan-agentic",
                    chunk_id="chunk-plan-agentic",
                    content="Plan 第二步证据：Agentic RAG 补充了执行结论。",
                    source_path="plan-agentic.md",
                    score=0.92,
                )
            ],
            success=True,
            exit_reason="sufficient",
            final_decision=_ForcedDecision(
                next_action="finish",
                is_sufficient=True,
            ),
        )


def _agent_runtime_document(
    *,
    document_id: str,
    chunk_id: str,
    content: str,
    source_path: str,
    score: float,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "source_path": source_path,
            "namespace": "documents",
            "is_managed_document": True,
            "chunk_id": chunk_id,
            "chunk_index": 0,
            "score": score,
            "vector_score": score,
            "vector_rank": 1,
            "matched_by": ["vector"],
        },
    )


def test_chat_api_success_path() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="P001 手机，续航强，电池 5000mAh。",
                score=0.92,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "doc.txt",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-doc-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="推荐 P001，续航表现较好。")
    service = _build_chat_service("chat-api-success", knowledge, document_retrieval_service, model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "推荐续航好的手机"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["session_id"]
    assert payload["request_id"]
    assert payload["answer"] == "推荐 P001，续航表现较好。\n\n参考来源：[1]"
    assert payload["knowledge_used"] is True
    assert payload["scene"] == "generic_assistant"
    assert payload["agent"] is None
    assert len(payload["citations"]) == 1
    assert payload["citations"][0] == {
        "index": 1,
        "citation_id": "chunk-doc-1",
        "namespace": "documents",
        "source_kind": "document_chunk",
        "source_name": "doc.txt",
        "source_path": "doc.txt",
        "document_id": "doc-1",
        "chunk_id": "chunk-doc-1",
        "chunk_index": 0,
        "snippet": "P001 手机，续航强，电池 5000mAh。",
        "score": 0.92,
        "vector_score": 0.92,
        "keyword_score": None,
        "vector_rank": 1,
        "keyword_rank": None,
        "rerank_score": None,
        "matched_by": ["vector"],
        "rank": 1,
    }
    trace = payload["retrieval_trace"]
    assert trace["original_query"] == "推荐续航好的手机"
    assert trace["final_query"] == "推荐续航好的手机"
    assert trace["rewritten_query"] is None
    assert trace["tool_call_count"] == 1
    assert trace["candidate_tools"] == ["knowledge_document_search"]
    assert trace["exit_reason"] == "sufficient"
    assert trace["final_decision"] == "answer_with_evidence"
    assert trace["success"] is True
    assert trace["follow_up_question"] is None
    assert trace["knowledge_used"] is True
    assert trace["raw_candidates_count"] == 1
    assert trace["filtered_candidates_count"] == 1
    assert trace["citations"] == payload["citations"]
    assert trace["top_k_chunks"][0]["citation_id"] == "chunk-doc-1"
    assert trace["top_k_chunks"][0]["chunk_id"] == "chunk-doc-1"
    assert trace["top_k_chunks"][0]["score"] == 0.92
    assert trace["rounds"][0]["tool_name"] == "knowledge_document_search"
    assert trace["rounds"][0]["raw_candidates_count"] == 1
    assert trace["rounds"][0]["filtered_candidates_count"] == 1
    assert trace["rounds"][0]["rerank"] == {
        "enabled": False,
        "provider": None,
        "model": None,
        "applied": False,
        "input_count": 1,
        "output_count": 1,
        "top_n": None,
        "fallback_reason": None,
        "error": None,
    }
    saved_session = service.session_store.get_session(payload["session_id"])
    assert saved_session is not None
    assert saved_session.mounted_knowledge_sources == ("documents",)
    assert model.get_runnable_calls == ["simple"]
    assert len(model.invoke_runnable_calls) == 1


def test_chat_api_prepare_turn_defers_agent_runtime_to_chat_graph() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-graph",
                content="Graph orchestration evidence.",
                score=0.92,
                metadata={
                    "document_id": "doc-graph",
                    "source_path": "graph.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-graph",
                    "chunk_index": 0,
                },
            )
        ]
    )
    service = _build_chat_service(
        "chat-api-prepare-defers-agent-runtime",
        knowledge,
        document_retrieval_service,
        FakeModel(answer="Graph evidence answer.[1]"),
    )

    prepared = service._get_scene_service("generic_assistant")._prepare_chat_turn(
        ChatRequest(message="use graph evidence")
    )

    assert prepared.react_run is None
    assert prepared.plan_run is None
    assert prepared.retrieval_trace.exit_reason == "pending_chat_graph"
    assert document_retrieval_service.calls == []


def test_chat_api_generic_react_can_direct_answer_without_rag() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-unused",
                content="This document should not be retrieved.",
                score=0.99,
                metadata={
                    "document_id": "doc-unused",
                    "source_path": "unused.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-unused",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = DirectAnswerModel(answer="你好，我可以直接帮你。")
    service = _build_chat_service(
        "chat-api-generic-direct-answer",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "你好"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "你好，我可以直接帮你。"
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert payload["retrieval_trace"]["final_decision"] == "direct_answer"
    assert payload["retrieval_trace"]["tool_call_count"] == 0
    assert document_retrieval_service.calls == []


def test_chat_api_rag_evidence_is_produced_through_tool_executor(monkeypatch) -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-tool-executor",
                content="ToolExecutor 负责执行顶层 RAG 工具。",
                score=0.94,
                metadata={
                    "document_id": "doc-tool-executor",
                    "source_path": "tool-executor.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-tool-executor",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="ToolExecutor 已返回证据。")
    calls: list[dict[str, Any]] = []
    original_execute = ToolExecutor.execute

    def _spy_execute(self: ToolExecutor, **kwargs: Any):
        calls.append(dict(kwargs))
        return original_execute(self, **kwargs)

    monkeypatch.setattr(ToolExecutor, "execute", _spy_execute)
    service = _build_chat_service(
        "chat-api-tool-executor-rag-path",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "说明 ToolExecutor"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["knowledge_used"] is True
    assert calls
    assert calls[0]["tool_name"] == "agentic_rag_search"
    assert calls[0]["input_payload"]["query"] == "说明 ToolExecutor"


def test_chat_api_sse_success_path_returns_structured_events() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="P001 手机，续航强，电池 5000mAh。",
                score=0.92,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "doc.txt",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-doc-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(
        answer="推荐 P001，续航表现较好。[1]",
        stream_chunks=["推荐 P001，", "续航表现较好。[1]"],
    )
    service = _build_chat_service("chat-api-sse-success", knowledge, document_retrieval_service, model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "推荐续航好的手机", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    _assert_display_sse_events(events)

    start_data = events[0]["data"]
    chunk_payloads = _event_payloads(events, "chunk")
    done_data = events[-1]["data"]
    assert start_data
    assert chunk_payloads
    assert done_data

    start_payload = json.loads(start_data)
    done_payload = json.loads(done_data)
    assert start_payload["session_id"]
    assert start_payload["request_id"]
    assert start_payload["knowledge_used"] is True
    assert start_payload["agent_mode"] == "react"
    assert start_payload["state"] == "running"
    assert start_payload["state_event"] == "run_start"
    assert start_payload["observability"]["provider"] == "FakeModel"
    assert start_payload["observability"]["complexity"] == "simple"
    assert start_payload["observability"]["retry_count"] == 0
    assert start_payload["observability"]["tools"] == [
        {"tool_name": AGENTIC_RAG_TOOL_NAME, "tool_status": "succeeded"}
    ]
    assert "".join(payload["delta"] for payload in chunk_payloads) == (
        "推荐 P001，续航表现较好。[1]"
    )
    assert done_payload["answer"] == "推荐 P001，续航表现较好。[1]"
    assert done_payload["knowledge_used"] is True
    assert done_payload["scene"] == "generic_assistant"
    assert done_payload["state"] == "succeeded"
    assert done_payload["final_state"] == "succeeded"
    assert done_payload["run_id"]
    assert len(done_payload["citations"]) == 1
    assert done_payload["retrieval_trace"]["tool_call_count"] == 1
    assert done_payload["retrieval_trace"]["final_decision"] == "answer_with_evidence"
    assert done_payload["retrieval_trace"]["top_k_chunks"][0]["citation_id"] == "chunk-doc-1"
    assert done_payload["retrieval_trace"]["citations"] == done_payload["citations"]
    assert done_payload["observability"] == start_payload["observability"]
    assert "prompt" not in json.dumps(done_payload, ensure_ascii=False)
    assert "full_history" not in json.dumps(done_payload, ensure_ascii=False)
    assert "tool_args" not in json.dumps(done_payload, ensure_ascii=False)
    saved_turns, total_turns = service.session_store.get_session_detail(
        done_payload["session_id"],
        limit=10,
    )
    assert total_turns == 1
    assert saved_turns[0].assistant_answer == "推荐 P001，续航表现较好。[1]"
    assert tuple(
        event.status
        for event in service.graph_runtime.lifecycle.events(done_payload["run_id"])
    ) == ("created", "running", "succeeded")
    restored = service.graph_runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": done_payload["session_id"],
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )
    assert restored is not None
    assert restored.checkpoint["channel_values"]["status"] == "succeeded"
    assert restored.checkpoint["channel_values"]["run_id"] == done_payload["run_id"]
    assert restored.checkpoint["channel_values"]["agent_mode"] == "react"
    assert restored.checkpoint["channel_values"]["retrieval_trace"] == done_payload["retrieval_trace"]
    assert restored.checkpoint["channel_values"]["citations"] == done_payload["citations"]
    assert restored.checkpoint["channel_values"]["react_run"]["turns"][0]["observation"]["trace"][
        "retrieval_trace"
    ]["final_decision"] == "answer_with_evidence"
    assert model.get_runnable_calls == ["simple"]
    assert len(model.invoke_runnable_calls) == 1
    assert model.stream_runnable_calls == []


def test_chat_api_stream_done_matches_non_stream_projection_for_stable_fields() -> None:
    def _build_service(test_name: str) -> SceneChatService:
        document_retrieval_service = FakeDocumentRetrievalService(
            documents=[
                _result(
                    doc_id="doc-consistency",
                    content="AeroPhone X 电池 5000mAh，当前有货。",
                    score=0.94,
                    metadata={
                        "document_id": "doc-consistency",
                        "source_path": "consistency.md",
                        "namespace": "documents",
                        "is_managed_document": True,
                        "chunk_id": "chunk-consistency",
                        "chunk_index": 0,
                    },
                )
            ]
        )
        return _build_chat_service(
            test_name,
            FakeKnowledgeService(),
            document_retrieval_service,
            FakeModel(answer="AeroPhone X 当前有货，电池 5000mAh。[1]"),
        )

    non_stream_app = create_app(
        chat_service=_build_service("chat-api-non-stream-consistency")
    )
    stream_app = create_app(chat_service=_build_service("chat-api-stream-consistency"))

    with TestClient(non_stream_app) as client:
        non_stream_response = client.post(
            "/chat",
            json={"message": "AeroPhone X 续航和库存", "stream": False},
        )
    with TestClient(stream_app) as client:
        stream_response = client.post(
            "/chat",
            json={"message": "AeroPhone X 续航和库存", "stream": True},
        )

    assert non_stream_response.status_code == 200
    assert stream_response.status_code == 200
    non_stream_payload = non_stream_response.json()
    stream_events = _parse_sse_events(stream_response.text)
    _assert_display_sse_events(stream_events)
    done_payload = json.loads(stream_events[-1]["data"])

    for field_name in (
        "answer",
        "knowledge_used",
        "state",
        "final_state",
        "state_event",
        "citations",
    ):
        assert done_payload[field_name] == non_stream_payload[field_name]
    assert done_payload["retrieval_trace"] == non_stream_payload["retrieval_trace"]
    assert done_payload["observability"]["tools"] == [
        {"tool_name": AGENTIC_RAG_TOOL_NAME, "tool_status": "succeeded"}
    ]


def test_chat_api_sse_plan_request_uses_plan_step_tool_stage() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-plan",
                content="Planner Runtime 支持步骤化执行和最终汇总。",
                score=0.93,
                metadata={
                    "document_id": "doc-plan",
                    "source_path": "plan.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-plan",
                    "chunk_index": 0,
                },
            )
        ]
    )
    service = _build_chat_service(
        "chat-api-sse-plan-stage",
        knowledge,
        document_retrieval_service,
        FakeModel(answer="Planner Runtime 支持步骤化执行。[1]"),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "请分步骤制定计划并汇总 Planner Runtime", "stream": True},
        )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    _assert_display_sse_events(events)
    start_payload = json.loads(events[0]["data"])
    done_payload = json.loads(events[-1]["data"])
    assert start_payload["agent_mode"] == "plan"
    assert done_payload["final_state"] == "succeeded"
    assert done_payload["retrieval_trace"]["rounds"][0]["tool_name"] == "knowledge_document_search"
    restored = service.graph_runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": done_payload["session_id"],
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )
    assert restored is not None
    assert restored.checkpoint["channel_values"]["agent_mode"] == "plan"
    assert restored.checkpoint["channel_values"]["react_run"] is None
    assert restored.checkpoint["channel_values"]["plan_run"]["workflow_status"] == "succeeded"
    assert restored.checkpoint["channel_values"]["plan_run"]["steps"][0]["status"] == "succeeded"
    assert restored.checkpoint["channel_values"]["current_step_id"] is None
    assert restored.checkpoint["channel_values"]["current_turn_id"] is None
    assert restored.checkpoint["channel_values"]["tool_observation"]["tool_name"] == "agentic_rag_search"
    planner_metadata = restored.checkpoint["channel_values"]["plan_run"]["metadata"]["planner"]
    assert planner_metadata["step_source"] == "llm_structured_output"


def test_chat_api_plan_mode_uses_scene_policy_multi_step_plan() -> None:
    runtime_dir = make_test_runtime_dir("chat-api-plan-multi-step")
    sqlite_path = runtime_dir / "chat-sessions.db"
    retriever = _MultiStepPlanRetriever()
    scene_definition = SceneDefinition(
        scene="plan_scene",
        name="Plan Scene",
        description="Use explicit multi-step plan policy.",
        build_retriever=lambda: retriever,  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda mounted: ("knowledge_document_search",),
        system_prompt="你是测试助手。",
        fallback_policy=SceneFallbackPolicy(no_hit_message="没有证据。"),
        infer_complexity=lambda _: "complex",
        retrieval_policy=SceneRetrievalPolicy(min_relevance_score=0.0),
        metadata={
            "agent_runtime": {
                "plan": {
                    "plan_steps": [
                        {
                            "step_id": "collect-native",
                            "goal": "先做 native 检索",
                            "tool_name": NATIVE_RAG_TOOL_NAME,
                            "input": {"query": "第一步"},
                        },
                        {
                            "step_id": "collect-agentic",
                            "goal": "再做 agentic 检索",
                            "tool_name": AGENTIC_RAG_TOOL_NAME,
                            "input": {"query": "第二步"},
                            "depends_on": ["collect-native"],
                        },
                    ]
                }
            }
        },
    )
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=AppSettings(
            data_dir=runtime_dir,
            app={"active_scene": "plan_scene"},
            session={"sqlite_path": sqlite_path, "window_size": 3},
        ),
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(
            answer="按计划汇总完成。",
            plan_draft=PlanDraft(
                steps=[
                    PlanStepDraft(
                        step_id="collect-native",
                        goal="先做 native 检索",
                        tool_name=NATIVE_RAG_TOOL_NAME,
                        input={"query": "第一步"},
                    ),
                    PlanStepDraft(
                        step_id="collect-agentic",
                        goal="再做 agentic 检索",
                        tool_name=AGENTIC_RAG_TOOL_NAME,
                        input={"query": "第二步"},
                        depends_on=["collect-native"],
                    ),
                ],
                rationale_summary="测试多步计划由 LLM structured output 返回。",
            ),
        ),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "请分步骤完成计划任务"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is True
    assert [citation["citation_id"] for citation in payload["citations"]] == [
        "chunk-plan-native",
        "chunk-plan-agentic",
    ]
    assert payload["retrieval_trace"]["top_k_chunks"][0]["citation_id"] == "chunk-plan-native"
    assert payload["retrieval_trace"]["top_k_chunks"][1]["citation_id"] == "chunk-plan-agentic"
    restored = service.graph_runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": payload["session_id"],
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )
    assert restored is not None
    plan_run = restored.checkpoint["channel_values"]["plan_run"]
    assert plan_run["workflow_status"] == "succeeded"
    assert [step["step_id"] for step in plan_run["steps"]] == ["collect-native", "collect-agentic"]
    assert [step["status"] for step in plan_run["steps"]] == ["succeeded", "succeeded"]
    assert plan_run["metadata"]["execution_order"] == ["collect-native", "collect-agentic"]
    assert len(plan_run["observations"]) == 2
    assert plan_run["metadata"]["planner"]["step_source"] == "llm_structured_output"
    assert retriever.calls[0]["method"] == "retrieve"
    assert retriever.calls[1]["method"] == "retrieve_with_trace"


def test_chat_api_react_aggregation_uses_all_successful_observations(monkeypatch) -> None:
    service = _build_chat_service(
        "chat-api-react-aggregate-all-observations",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(answer="汇总两条证据。"),
    )
    app = create_app(chat_service=service)

    first_observation = ToolObservation(
        tool_name=AGENTIC_RAG_TOOL_NAME,
        success=True,
        output={
            "documents": [
                {
                    "page_content": "第一条证据：制度范围。",
                    "metadata": {
                        "document_id": "doc-react-1",
                        "source_path": "react-1.md",
                        "namespace": "documents",
                        "chunk_id": "chunk-react-1",
                        "chunk_index": 0,
                        "score": 0.91,
                        "vector_score": 0.91,
                        "vector_rank": 1,
                        "matched_by": ["vector"],
                    },
                }
            ]
        },
        result_summary="第一轮检索成功。",
        citations=[{"citation_id": "chunk-react-1"}],
        trace={
            "retrieval_trace": {
                "original_query": "聚合测试",
                "final_query": "聚合测试",
                "tool_call_count": 1,
                "candidate_tools": ["knowledge_document_search"],
                "exit_reason": "sufficient",
                "final_decision": "answer_with_evidence",
                "success": True,
                "raw_candidates_count": 1,
                "filtered_candidates_count": 1,
                "rounds": [{"round_index": 1, "tool_name": "knowledge_document_search", "query": "聚合测试", "decision": "finish", "is_sufficient": True, "result_count": 1, "document_count": 1, "success": True}],
            }
        },
        metadata={"final_decision": "answer_with_evidence", "knowledge_used": True},
    )
    second_observation = ToolObservation(
        tool_name=NATIVE_RAG_TOOL_NAME,
        success=True,
        output={
            "documents": [
                {
                    "page_content": "第二条证据：实施细节。",
                    "metadata": {
                        "document_id": "doc-react-2",
                        "source_path": "react-2.md",
                        "namespace": "documents",
                        "chunk_id": "chunk-react-2",
                        "chunk_index": 0,
                        "score": 0.92,
                        "vector_score": 0.92,
                        "vector_rank": 1,
                        "matched_by": ["vector"],
                    },
                }
            ]
        },
        result_summary="第二轮检索成功。",
        citations=[{"citation_id": "chunk-react-2"}],
        trace={
            "retrieval_trace": {
                "original_query": "聚合测试",
                "final_query": "聚合测试",
                "tool_call_count": 1,
                "candidate_tools": ["knowledge_document_search"],
                "exit_reason": "sufficient",
                "final_decision": "answer_with_evidence",
                "success": True,
                "raw_candidates_count": 1,
                "filtered_candidates_count": 1,
                "rounds": [{"round_index": 1, "tool_name": "knowledge_document_search", "query": "聚合测试", "decision": "finish", "is_sufficient": True, "result_count": 1, "document_count": 1, "success": True}],
            }
        },
        metadata={"final_decision": "answer_with_evidence", "knowledge_used": True},
    )
    react_run = ReActRun(
        react_run_id="react-aggregate",
        session_id="session-react-aggregate",
        request_id="request-react-aggregate",
        user_goal="聚合测试",
        workflow_status="succeeded",
        turns=[
            ReActTurn(
                turn_id="turn-1",
                round_index=1,
                goal="聚合测试",
                action=ReActAction(action_type="tool_call", tool_name=AGENTIC_RAG_TOOL_NAME),
                status="succeeded",
                tool_name=AGENTIC_RAG_TOOL_NAME,
                observation=first_observation,
                result_summary=first_observation.result_summary,
            ),
            ReActTurn(
                turn_id="turn-2",
                round_index=2,
                goal="聚合测试",
                action=ReActAction(action_type="tool_call", tool_name=NATIVE_RAG_TOOL_NAME),
                status="succeeded",
                tool_name=NATIVE_RAG_TOOL_NAME,
                observation=second_observation,
                result_summary=second_observation.result_summary,
            ),
        ],
        observations=[first_observation, second_observation],
        final_answer="汇总两条证据。",
        result_summary="汇总两条证据。",
        metadata={"citations": [{"citation_id": "chunk-react-1"}, {"citation_id": "chunk-react-2"}]},
    )

    inner_service = service._get_scene_service("generic_assistant")
    original_build_react_deps = inner_service._build_react_deps

    def _fake_build_react_deps(prepared: Any, state: Any) -> Any:
        seeded_run = react_run.model_copy(
            update={
                "session_id": prepared.session_id,
                "request_id": prepared.request_id,
                "user_goal": prepared.user_message,
            }
        )
        return replace(
            original_build_react_deps(prepared, state),
            initial_run=seeded_run,
        )

    monkeypatch.setattr(inner_service, "_build_react_deps", _fake_build_react_deps)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "聚合测试"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is True
    assert [citation["citation_id"] for citation in payload["citations"]] == [
        "chunk-react-1",
        "chunk-react-2",
    ]
    assert payload["retrieval_trace"]["tool_call_count"] == 2
    assert payload["retrieval_trace"]["raw_candidates_count"] == 2
    assert payload["retrieval_trace"]["filtered_candidates_count"] == 2
    assert [chunk["citation_id"] for chunk in payload["retrieval_trace"]["top_k_chunks"]] == [
        "chunk-react-1",
        "chunk-react-2",
    ]
    assert payload["answer"].endswith("参考来源：[1][2]")


def test_chat_api_validation_error_when_message_missing() -> None:
    service = _build_chat_service(
        "chat-api-validation-error",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_api_no_hit_fallback_sets_knowledge_used_false() -> None:
    model = FakeModel(answer="unused")
    service = _build_chat_service("chat-api-no-hit", FakeKnowledgeService(), FakeDocumentRetrievalService(), model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "火星基地快递多久到"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert "暂时没有检索到足够相关的文档知识" in payload["answer"]
    assert payload["scene"] == "generic_assistant"
    assert payload["agent"] is None
    assert model.get_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_no_hit_with_unrelated_docs_keeps_knowledge_unused(monkeypatch) -> None:
    rewrite_model = FakeRewriteModel(
        '{"query":"VOID-ALPHA-7788 secret handshake 数据模型","reason":"尝试添加泛词"}'
    )
    monkeypatch.setattr(
        "backend.scenes.generic_assistant.definition.default_model_client",
        rewrite_model,
    )
    model = FakeModel(answer="unused")
    document_retrieval_service = QueryFilteredFakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-unrelated",
                content="员工手册：远程访问需要开启 MFA，并遵守信息安全要求。",
                score=0.95,
                metadata={
                    "document_id": "doc-unrelated",
                    "source_path": "it-policy.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-unrelated",
                    "chunk_index": 0,
                },
            )
        ],
        allowed_query_terms=("MFA",),
    )
    service = _build_chat_service(
        "chat-api-no-hit-unrelated-docs",
        FakeKnowledgeService(),
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "VOID-ALPHA-7788 secret handshake?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert "暂时没有检索到足够相关的文档知识" in payload["answer"]
    assert payload["retrieval_trace"]["knowledge_used"] is False
    assert payload["retrieval_trace"]["citations"] == []
    assert payload["retrieval_trace"]["filtered_candidates_count"] == 0
    assert payload["retrieval_trace"]["rewritten_query"] == "VOID-ALPHA-7788 secret handshake?"
    assert [call["query"] for call in document_retrieval_service.calls] == [
        "VOID-ALPHA-7788 secret handshake?",
        "VOID-ALPHA-7788 secret handshake?",
    ]
    assert rewrite_model.get_runnable_calls == ["simple"]
    assert len(rewrite_model.invoke_runnable_calls) == 1
    assert model.get_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_sse_no_hit_uses_fallback_without_model_streaming() -> None:
    model = FakeModel(answer="unused", stream_chunks=["unused"])
    service = _build_chat_service(
        "chat-api-sse-no-hit",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "火星基地快递多久到", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    _assert_display_sse_events(events)

    chunk_payload = _event_payloads(events, "chunk")[0]
    done_payload = json.loads(events[-1]["data"])
    assert "暂时没有检索到足够相关的文档知识" in chunk_payload["delta"]
    assert done_payload["knowledge_used"] is False
    assert done_payload["citations"] == []
    assert done_payload["retrieval_trace"]["knowledge_used"] is False
    assert done_payload["retrieval_trace"]["filtered_candidates_count"] == 0
    assert model.get_runnable_calls == []
    assert model.stream_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_ask_user_returns_follow_up_without_answer_model() -> None:
    model = FakeModel(answer="unused")
    outcome = _ForcedOutcome(
        documents=[_forced_document()],
        success=False,
        exit_reason="ask_user",
        final_decision=_ForcedDecision(
            next_action="ask_user",
            is_sufficient=False,
            follow_up_question="请补充你要查询的文档主题。",
        ),
        follow_up_question="请补充更具体的问题。",
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-ask-user-boundary",
        outcome=outcome,
        model=model,
    )
    app = create_app(chat_service=service)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "帮我查一下"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "请补充更具体的问题。"
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    trace = payload["retrieval_trace"]
    assert trace["final_decision"] == "ask_user"
    assert trace["follow_up_question"] == "请补充更具体的问题。"
    assert trace["knowledge_used"] is False
    assert trace["citations"] == []
    assert trace["top_k_chunks"] == []
    assert model.get_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_sse_ask_user_matches_json_semantics_without_stream_model() -> None:
    model = FakeModel(answer="unused", stream_chunks=["unused"])
    outcome = _ForcedOutcome(
        documents=[_forced_document()],
        success=False,
        exit_reason="ask_user",
        final_decision=_ForcedDecision(
            next_action="ask_user",
            is_sufficient=False,
            follow_up_question="请说明需要查询的知识范围。",
        ),
        follow_up_question="请说明需要查询的知识范围。",
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-sse-ask-user-boundary",
        outcome=outcome,
        model=model,
    )
    app = create_app(chat_service=service)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "查资料", "stream": True})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    _assert_display_sse_events(events)
    chunk_payload = _event_payloads(events, "chunk")[0]
    done_payload = json.loads(events[-1]["data"])
    assert chunk_payload["delta"] == "请说明需要查询的知识范围。"
    assert done_payload["knowledge_used"] is False
    assert done_payload["citations"] == []
    assert done_payload["retrieval_trace"]["final_decision"] == "ask_user"
    assert model.get_runnable_calls == []
    assert model.stream_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_react_ask_user_without_observation_enters_hitl_wait(monkeypatch) -> None:
    model = FakeModel(answer="unused")
    service = _build_chat_service(
        "chat-api-react-ask-user-no-observation",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(documents=[]),
        model,
    )
    app = create_app(chat_service=service)

    inner_service = service._get_scene_service("generic_assistant")
    original_build_react_deps = inner_service._build_react_deps

    def _fake_build_react_deps(prepared: Any, state: Any) -> Any:
        session_id = prepared.session_id
        request_id = prepared.request_id
        message = prepared.user_message
        seeded_run = ReActRun(
            react_run_id=f"react-{request_id}",
            session_id=session_id,
            request_id=request_id,
            user_goal=message,
            workflow_status="waiting_user",
            turns=[
                ReActTurn(
                    turn_id="turn-1",
                    round_index=1,
                    goal=message,
                    action=ReActAction(
                        action_type="ask_user",
                        instruction="请补充需要查询的知识范围。",
                    ),
                    status="waiting_user",
                    result_summary="请补充需要查询的知识范围。",
                )
            ],
            observations=[],
            current_turn_id="turn-1",
            metadata={
                "hitl": {
                    "mode": "react",
                    "react_run_id": f"react-{request_id}",
                    "current_turn_id": "turn-1",
                    "user_prompt": "请补充需要查询的知识范围。",
                    "source": "react_action",
                }
            },
        )
        return replace(
            original_build_react_deps(prepared, state),
            initial_run=seeded_run,
        )

    monkeypatch.setattr(inner_service, "_build_react_deps", _fake_build_react_deps)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "查资料",
                "hitl_clarification_enabled": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "waiting_user"
    assert payload["knowledge_used"] is False
    assert payload["hitl"]["metadata"]["mode"] == "react"
    assert payload["hitl"]["metadata"]["current_turn_id"] == "turn-1"
    assert payload["retrieval_trace"]["tool_call_count"] == 0
    assert payload["retrieval_trace"]["final_decision"] == "ask_user"
    assert payload["retrieval_trace"]["follow_up_question"] == "请补充需要查询的知识范围。"
    assert model.stream_runnable_calls == []
    lifecycle_statuses = [
        event.status
        for event in inner_service.graph_runtime.lifecycle.events_for_request(
            payload["request_id"]
        )
    ]
    assert lifecycle_statuses == ["created", "running", "waiting_user"]


def test_chat_api_sse_hitl_reject_resume_returns_done_cancelled() -> None:
    model = FakeModel(answer="unused", stream_chunks=["unused"])
    service = _build_chat_service(
        "chat-api-sse-hitl-reject-resume",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(documents=[]),
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        wait_response = client.post(
            "/chat",
            json={
                "message": "查资料",
                "stream": True,
                "hitl_clarification_enabled": True,
            },
        )

    assert wait_response.status_code == 200
    wait_events = _parse_sse_events(wait_response.text)
    _assert_display_sse_events(wait_events, final_event="waiting_user")
    assert [event["event"] for event in wait_events] == [
        "start",
        "thinking",
        "thinking",
        "waiting_user",
    ]
    wait_payload = json.loads(wait_events[-1]["data"])
    assert wait_payload["state"] == "waiting_user"
    assert wait_payload["hitl"]["metadata"]["mode"] == "react"
    assert wait_payload["hitl"]["metadata"]["react_run_id"].startswith("react-")
    assert wait_payload["hitl"]["metadata"]["current_turn_id"] == "turn-1"

    with TestClient(app) as client:
        resume_response = client.post(
            "/chat/resume",
            json={
                "session_id": wait_payload["session_id"],
                "interrupt_id": wait_payload["hitl"]["interrupt_id"],
                "action": "reject",
                "payload": {"reason": "用户取消"},
                "stream": True,
            },
        )

    assert resume_response.status_code == 200
    resume_events = _parse_sse_events(resume_response.text)
    assert [event["event"] for event in resume_events] == ["resume", "done"]
    resume_payload = json.loads(resume_events[0]["data"])
    done_payload = json.loads(resume_events[1]["data"])
    assert resume_payload["action"] == "reject"
    assert resume_payload["state_event"] == "resume_reject"
    assert done_payload["status"] == "cancelled"
    assert done_payload["state"] == "cancelled"
    assert done_payload["final_state"] == "cancelled"
    assert done_payload["state_event"] == "resume_reject"
    assert done_payload["resume_payload"]["action"] == "reject"


def test_chat_api_stream_waiting_user_matches_non_stream_hitl_payload() -> None:
    def _build_wait_service(test_name: str) -> SceneChatService:
        return _build_chat_service(
            test_name,
            FakeKnowledgeService(),
            FakeDocumentRetrievalService(documents=[]),
            FakeModel(answer="unused", stream_chunks=["unused"]),
        )

    non_stream_app = create_app(
        chat_service=_build_wait_service("chat-api-non-stream-hitl-consistency")
    )
    stream_app = create_app(
        chat_service=_build_wait_service("chat-api-stream-hitl-consistency")
    )

    request_json = {
        "message": "查资料",
        "hitl_clarification_enabled": True,
    }
    with TestClient(non_stream_app) as client:
        non_stream_response = client.post("/chat", json={**request_json, "stream": False})
    with TestClient(stream_app) as client:
        stream_response = client.post("/chat", json={**request_json, "stream": True})

    assert non_stream_response.status_code == 200
    assert stream_response.status_code == 200
    non_stream_payload = non_stream_response.json()
    stream_events = _parse_sse_events(stream_response.text)
    _assert_display_sse_events(stream_events, final_event="waiting_user")
    wait_payload = json.loads(stream_events[-1]["data"])

    assert wait_payload["state"] == non_stream_payload["state"] == "waiting_user"
    assert wait_payload["final_state"] == non_stream_payload["final_state"] == "waiting_user"
    assert wait_payload["retrieval_trace"] == non_stream_payload["retrieval_trace"]
    assert wait_payload["hitl"]["pending_action"] == non_stream_payload["hitl"]["pending_action"]
    assert wait_payload["hitl"]["allowed_actions"] == non_stream_payload["hitl"]["allowed_actions"]
    assert wait_payload["hitl"]["metadata"]["mode"] == non_stream_payload["hitl"]["metadata"]["mode"]
    assert wait_payload["hitl"]["interrupt_id"]
    assert wait_payload["hitl"]["thread_id"] == wait_payload["session_id"]
    assert wait_payload["observability"]["tools"] == [
        {"tool_name": AGENTIC_RAG_TOOL_NAME, "tool_status": "waiting_user"}
    ]


def test_chat_api_max_rounds_with_documents_does_not_use_intermediate_citations() -> None:
    model = FakeModel(answer="unused")
    outcome = _ForcedOutcome(
        documents=[_forced_document()],
        success=False,
        exit_reason="max_rounds_reached",
        final_decision=_ForcedDecision(next_action="ask_user", is_sufficient=False),
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-max-rounds-boundary",
        outcome=outcome,
        model=model,
    )
    app = create_app(chat_service=service)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "复杂问题"})

    assert response.status_code == 200
    payload = response.json()
    assert "测试兜底" in payload["answer"]
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    trace = payload["retrieval_trace"]
    assert trace["final_decision"] == "max_rounds_reached"
    assert trace["knowledge_used"] is False
    assert trace["citations"] == []
    assert trace["top_k_chunks"] == []
    assert model.get_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_sse_max_rounds_with_documents_keeps_tool_done_consistent() -> None:
    model = FakeModel(answer="unused", stream_chunks=["unused"])
    outcome = _ForcedOutcome(
        documents=[_forced_document()],
        success=False,
        exit_reason="max_rounds_reached",
        final_decision=_ForcedDecision(next_action="ask_user", is_sufficient=False),
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-sse-max-rounds-boundary",
        outcome=outcome,
        model=model,
    )
    app = create_app(chat_service=service)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "复杂问题", "stream": True})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    done_payload = json.loads(events[-1]["data"])
    _assert_display_sse_events(events)
    assert done_payload["retrieval_trace"]["final_decision"] == "max_rounds_reached"
    assert done_payload["knowledge_used"] is False
    assert done_payload["citations"] == []
    assert model.get_runnable_calls == []
    assert model.stream_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_failed_outcome_with_documents_does_not_use_intermediate_citations() -> None:
    model = FakeModel(answer="unused")
    outcome = _ForcedOutcome(
        documents=[_forced_document()],
        success=False,
        exit_reason="sufficient",
        final_decision=_ForcedDecision(next_action="finish", is_sufficient=True),
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-retrieval-failed-boundary",
        outcome=outcome,
        model=model,
    )
    app = create_app(chat_service=service)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "失败但有候选"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert payload["retrieval_trace"]["final_decision"] == "retrieval_failed"
    assert payload["retrieval_trace"]["citations"] == []
    assert payload["retrieval_trace"]["top_k_chunks"] == []
    assert model.get_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_sse_failed_outcome_with_documents_keeps_tool_done_consistent() -> None:
    model = FakeModel(answer="unused", stream_chunks=["unused"])
    outcome = _ForcedOutcome(
        documents=[_forced_document()],
        success=False,
        exit_reason="sufficient",
        final_decision=_ForcedDecision(next_action="finish", is_sufficient=True),
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-sse-retrieval-failed-boundary",
        outcome=outcome,
        model=model,
    )
    app = create_app(chat_service=service)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "失败但有候选", "stream": True})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    done_payload = json.loads(events[-1]["data"])
    _assert_display_sse_events(events)
    assert done_payload["retrieval_trace"]["final_decision"] == "retrieval_failed"
    assert done_payload["knowledge_used"] is False
    assert done_payload["citations"] == []
    assert model.get_runnable_calls == []
    assert model.stream_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_no_evidence_decision_uses_fallback_without_citations() -> None:
    model = FakeModel(answer="unused")
    outcome = _ForcedOutcome(
        documents=[],
        success=True,
        exit_reason="sufficient",
        final_decision=_ForcedDecision(next_action="finish", is_sufficient=True),
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-no-evidence-boundary",
        outcome=outcome,
        model=model,
    )
    app = create_app(chat_service=service)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "没有证据"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert payload["retrieval_trace"]["final_decision"] == "no_evidence"
    assert payload["retrieval_trace"]["filtered_candidates_count"] == 0
    assert model.get_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_answer_decision_without_effective_citations_becomes_no_evidence() -> None:
    model = FakeModel(answer="unused")
    document_without_snippet = Document(
        page_content="",
        metadata={
            "document_id": "doc-empty",
            "source_path": "empty.md",
            "namespace": "documents",
            "chunk_id": "chunk-empty",
            "chunk_index": 0,
        },
    )
    outcome = _ForcedOutcome(
        documents=[document_without_snippet],
        success=True,
        exit_reason="sufficient",
        final_decision=_ForcedDecision(next_action="finish", is_sufficient=True),
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-answer-without-citations-boundary",
        outcome=outcome,
        model=model,
    )
    app = create_app(chat_service=service)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "空证据"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert payload["retrieval_trace"]["final_decision"] == "no_evidence"
    assert payload["retrieval_trace"]["citations"] == []
    assert model.get_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_sse_no_hit_with_unrelated_docs_keeps_done_without_citations(monkeypatch) -> None:
    rewrite_model = FakeRewriteModel(
        '{"query":"VOID-ALPHA-7788 secret handshake 常见问题","reason":"尝试添加泛词"}'
    )
    monkeypatch.setattr(
        "backend.scenes.generic_assistant.definition.default_model_client",
        rewrite_model,
    )
    model = FakeModel(answer="unused", stream_chunks=["unused"])
    document_retrieval_service = QueryFilteredFakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-unrelated",
                content="Support FAQ：普通支持请求通常会在 1 个工作日内得到首次响应。",
                score=0.95,
                metadata={
                    "document_id": "doc-unrelated",
                    "source_path": "support-faq.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-unrelated",
                    "chunk_index": 0,
                },
            )
        ],
        allowed_query_terms=("普通支持请求",),
    )
    service = _build_chat_service(
        "chat-api-sse-no-hit-unrelated-docs",
        FakeKnowledgeService(),
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "VOID-ALPHA-7788 secret handshake?", "stream": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    _assert_display_sse_events(events)

    chunk_payload = _event_payloads(events, "chunk")[0]
    done_payload = json.loads(events[-1]["data"])
    assert "暂时没有检索到足够相关的文档知识" in chunk_payload["delta"]
    assert done_payload["knowledge_used"] is False
    assert done_payload["citations"] == []
    assert done_payload["retrieval_trace"]["knowledge_used"] is False
    assert done_payload["retrieval_trace"]["filtered_candidates_count"] == 0
    assert done_payload["retrieval_trace"]["citations"] == []
    assert [call["query"] for call in document_retrieval_service.calls] == [
        "VOID-ALPHA-7788 secret handshake?",
        "VOID-ALPHA-7788 secret handshake?",
    ]
    assert rewrite_model.get_runnable_calls == ["simple"]
    assert len(rewrite_model.invoke_runnable_calls) == 1
    assert model.get_runnable_calls == []
    assert model.stream_runnable_calls == []
    assert model.invoke_runnable_calls == []


def test_chat_api_sse_error_path_keeps_runtime_event_order() -> None:
    class ErrorModel(FakeModel):
        def invoke_runnable(
            self,
            runnable: Any,
            input: Any,
            *,
            config: Any | None = None,
        ) -> str:
            if isinstance(runnable, FakeReactSelectorRunnable):
                return runnable.invoke(input, config=config)
            self.invoke_runnable_calls.append({"runnable": runnable, "input": input, "config": config})
            raise ValueError("Model returned empty content")

    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="P001 手机，续航强，电池 5000mAh。",
                score=0.92,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "doc.txt",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-doc-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = ErrorModel(answer="unused", stream_chunks=["unused"])
    service = _build_chat_service("chat-api-sse-error", knowledge, document_retrieval_service, model)
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "推荐续航好的手机", "stream": True})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["error"]
    error_payload = json.loads(events[-1]["data"])
    assert error_payload["code"] == "MODEL_EMPTY_RESPONSE"
    assert error_payload["message"] == "Model returned empty response."
    assert error_payload["request_id"] != "N/A"
    lifecycle_events = service.graph_runtime.lifecycle.events_for_request(
        error_payload["request_id"]
    )
    assert [event.status for event in lifecycle_events] == ["created", "running", "failed"]


def test_chat_api_non_stream_response_and_session_persistence_do_not_regress() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货，售价 4599 元。[1]")
    service = _build_chat_service(
        "chat-api-non-stream-regression",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "AeroPhone X 多少钱", "stream": False})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "AeroPhone X 当前有货，售价 4599 元。[1]"
    assert payload["knowledge_used"] is True
    turns, total_turns = service.session_store.get_session_detail(payload["session_id"], limit=10)
    assert total_turns == 1
    assert turns[0].assistant_answer == "AeroPhone X 当前有货，售价 4599 元。[1]"
    assert turns[0].retrieval_snippets[0]["citation_id"] == "chunk-manual-1"


def test_chat_request_context_is_isolated_for_concurrent_cached_service_invocations() -> None:
    class BlockingFirstInvokeModel(FakeModel):
        def __init__(self) -> None:
            super().__init__(answer="并发隔离回答。[1]")
            self.first_call_ready = Event()
            self.release_first_call = Event()
            self._call_lock = Lock()
            self._call_count = 0

        def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
            with self._call_lock:
                self._call_count += 1
                call_index = self._call_count
            if call_index == 1:
                self.first_call_ready.set()
                if not self.release_first_call.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release the first chat request.")
            return super().invoke_runnable(runnable, input, config=config)

    model = BlockingFirstInvokeModel()
    outcome = _ForcedOutcome(
        documents=[_forced_document()],
        success=True,
        exit_reason="sufficient",
        final_decision=_ForcedDecision(next_action="finish", is_sufficient=True),
    )
    service = _build_forced_outcome_chat_service(
        "chat-api-concurrent-request-context-isolation",
        outcome=outcome,
        model=model,
    )

    def chat(message: str):
        return service.chat(
            type(
                "Payload",
                (),
                {
                    "message": message,
                    "session_id": None,
                    "stream": False,
                },
            )()
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(chat, "first request")
        assert model.first_call_ready.wait(timeout=5)
        try:
            second_response = chat("second request")
        finally:
            model.release_first_call.set()
        first_response = first_future.result(timeout=5)

    first_messages, first_total = service.session_store.get_session_messages(
        first_response.session_id,
        limit=10,
    )
    second_messages, second_total = service.session_store.get_session_messages(
        second_response.session_id,
        limit=10,
    )

    assert first_total == 2
    assert second_total == 2
    assert [message.content for message in first_messages] == [
        "first request",
        first_response.answer,
    ]
    assert [message.content for message in second_messages] == [
        "second request",
        second_response.answer,
    ]
    assert {message.request_id for message in first_messages} == {first_response.request_id}
    assert {message.request_id for message in second_messages} == {second_response.request_id}
    assert first_response.request_id != second_response.request_id


def test_chat_api_uses_message_history_for_follow_up_turns() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货，售价 4599 元。[1]")
    service = _build_chat_service(
        "chat-api-message-history-follow-up",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "AeroPhone X 多少钱"})
        assert first.status_code == 200
        session_id = first.json()["session_id"]

        second = client.post(
            "/chat",
            json={"message": "那它现在有货吗", "session_id": session_id},
        )

    assert second.status_code == 200
    assert len(model.history_recorder.snapshots) >= 2
    second_history = model.history_recorder.snapshots[-1]
    assert len(second_history) == 2
    assert second_history[0].type == "human"
    assert second_history[0].content == "AeroPhone X 多少钱"
    assert second_history[1].type == "ai"
    assert second_history[1].content == "AeroPhone X 当前有货，售价 4599 元。[1]"

    messages = service.session_store.get_messages(session_id)
    assert len(messages) == 4
    assert messages[0].type == "human"
    assert messages[1].type == "ai"
    assert messages[2].type == "human"
    assert messages[3].type == "ai"


def test_chat_api_message_history_respects_window_size() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货，售价 4599 元。[1]")
    service = _build_chat_service(
        "chat-api-message-history-window",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        session_id: str | None = None
        for index in range(1, 4):
            response = client.post(
                "/chat",
                json={
                    "message": f"第{index}轮问题",
                    **({"session_id": session_id} if session_id else {}),
                },
            )
            assert response.status_code == 200
            session_id = response.json()["session_id"]

    assert session_id is not None
    latest_history = model.history_recorder.snapshots[-1]
    assert len(latest_history) == 4
    assert [message.content for message in latest_history] == [
        "第1轮问题",
        "AeroPhone X 当前有货，售价 4599 元。[1]",
        "第2轮问题",
        "AeroPhone X 当前有货，售价 4599 元。[1]",
    ]
    messages = service.session_store.get_messages(session_id)
    assert len(messages) == 6


def test_chat_api_sse_display_protocol_does_not_emit_history_event() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(
        answer="AeroPhone X 当前有货，售价 4599 元。[1]",
        stream_chunks=["AeroPhone X 当前", "有货，售价 4599 元。[1]"],
    )
    service = _build_chat_service(
        "chat-api-sse-history-window",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        session_id: str | None = None
        for index in range(1, 4):
            response = client.post(
                "/chat",
                json={
                    "message": f"第{index}轮问题",
                    "stream": index == 3,
                    **({"session_id": session_id} if session_id else {}),
                },
            )
            assert response.status_code == 200
            if index < 3:
                session_id = response.json()["session_id"]
            else:
                events = _parse_sse_events(response.text)

    assert session_id is not None
    _assert_display_sse_events(events)
    done_payload = json.loads(events[-1]["data"])
    assert done_payload["answer"] == "AeroPhone X 当前有货，售价 4599 元。[1]"


def test_session_management_endpoints() -> None:
    service = _build_chat_service(
        "chat-api-session-endpoints",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post("/sessions")
        assert create_response.status_code == 200
        create_payload = create_response.json()
        session_id = create_payload["session_id"]
        assert session_id
        assert create_payload["scene"] == "generic_assistant"
        assert create_payload["mounted_knowledge_sources"] == ["documents"]
        assert service.session_store.get_session(session_id) is not None

        empty_session_response = client.get(f"/sessions/{session_id}")
        assert empty_session_response.status_code == 200
        assert empty_session_response.json()["scene"] == "generic_assistant"
        assert empty_session_response.json()["mounted_knowledge_sources"] == ["documents"]
        assert empty_session_response.json()["total_messages"] == 0
        assert empty_session_response.json()["messages"] == []

        chat_response = client.post("/chat", json={"message": "你好", "session_id": session_id})
        assert chat_response.status_code == 200
        restored_checkpoint = service.graph_runtime.checkpointer.get_tuple(
            {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
                }
            }
        )
        assert restored_checkpoint is not None

        populated_session_response = client.get(f"/sessions/{session_id}")
        assert populated_session_response.status_code == 200
        payload = populated_session_response.json()
        assert payload["session_id"] == session_id
        assert payload["mounted_knowledge_sources"] == ["documents"]
        assert payload["total_messages"] == 2
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["type"] == "human"
        assert payload["messages"][0]["content"] == "你好"
        assert payload["messages"][0]["knowledge_used"] is None
        assert payload["messages"][1]["type"] == "ai"
        assert payload["messages"][1]["request_id"] == chat_response.json()["request_id"]
        assert payload["messages"][1]["content"] == chat_response.json()["answer"]
        assert payload["messages"][1]["knowledge_used"] is False
        assert payload["messages"][1]["citations"] == []

        delete_response = client.delete(f"/sessions/{session_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted_messages"] == 2

        after_delete_response = client.get(f"/sessions/{session_id}")
        assert after_delete_response.status_code == 200
        assert after_delete_response.json()["total_messages"] == 0
        assert service.graph_runtime.checkpointer.get_tuple(
            {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
                }
            }
        ) is None


def test_chat_api_rejects_expired_session() -> None:
    service = _build_chat_service(
        "chat-api-expired-session",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    service.session_store.create_session(
        session_id="expired-session",
        now="2026-04-23T00:00:00+00:00",
    )
    service.session_store.cleanup_expired_sessions(
        now="2026-04-23T01:00:00+00:00",
        timeout_minutes=30,
        limit=10,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "你好", "session_id": "expired-session"})

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "SESSION_EXPIRED"


def test_chat_api_rejects_unknown_session_id() -> None:
    service = _build_chat_service(
        "chat-api-missing-session",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "你好", "session_id": "missing-session"})

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["code"] == "SESSION_NOT_FOUND"


def test_list_scenes_endpoint_returns_available_scene_metadata() -> None:
    service = _build_chat_service(
        "chat-api-list-scenes",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.get("/scenes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_scene"] == "generic_assistant"
    assert [scene["scene"] for scene in payload["scenes"]] == [
        "generic_assistant",
        "ecommerce",
    ]
    assert payload["scenes"][0]["is_default"] is True
    assert payload["scenes"][1]["is_default"] is False


def test_create_session_rejects_unknown_scene() -> None:
    service = _build_chat_service(
        "chat-api-unknown-scene",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/sessions", json={"scene": "unknown_scene"})

    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["code"] == "UNKNOWN_SCENE"


def test_create_session_accepts_explicit_mounted_knowledge_sources() -> None:
    service = _build_chat_service(
        "chat-api-mounted-sources",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/sessions",
            json={
                "scene": "generic_assistant",
                "mounted_knowledge_sources": ["ecommerce", "documents", "documents"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mounted_knowledge_sources"] == ["documents", "ecommerce"]
    saved_session = service.session_store.get_session(payload["session_id"])
    assert saved_session is not None
    assert saved_session.mounted_knowledge_sources == ("documents", "ecommerce")


def test_create_session_rejects_unknown_mounted_knowledge_source() -> None:
    service = _build_chat_service(
        "chat-api-invalid-mounted-source",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/sessions",
            json={"mounted_knowledge_sources": ["documents", "unknown_source"]},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["code"] == "INVALID_MOUNTED_KNOWLEDGE_SOURCES"


def test_session_detail_returns_explicit_mounted_knowledge_sources() -> None:
    service = _build_chat_service(
        "chat-api-session-detail-mounted-sources",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    created = service.create_session(
        scene="generic_assistant",
        mounted_knowledge_sources=["documents", "ecommerce"],
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.get(f"/sessions/{created.session_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mounted_knowledge_sources"] == ["documents", "ecommerce"]


def test_chat_routes_by_session_scene_instead_of_global_default() -> None:
    service = _build_chat_service(
        "chat-api-session-scene-routing",
        FakeKnowledgeService(),
        FakeDocumentRetrievalService(),
        FakeModel(),
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post("/sessions", json={"scene": "ecommerce"})
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]

        chat_response = client.post(
            "/chat",
            json={"message": "Where is my order?", "session_id": session_id},
        )

    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["scene"] == "ecommerce"
    assert payload["agent"] == "shopping_agent"


def test_chat_only_uses_document_tools_when_session_mounts_documents_only() -> None:
    knowledge = FakeKnowledgeService(
        products=[
            _result(
                doc_id="product-1",
                content="AeroPhone X，库存充足。",
                score=0.95,
                metadata={"product_id": "P005"},
            )
        ]
    )
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="售后 FAQ：库存问题以系统实时状态为准。",
                score=0.91,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "faq.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-faq-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="请以文档说明为准。")
    service = _build_chat_service(
        "chat-api-documents-only-routing",
        knowledge,
        document_retrieval_service,
        model,
    )
    created = service.create_session(scene="generic_assistant", mounted_knowledge_sources=["documents"])
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "AeroPhone X 现在有货吗", "session_id": created.session_id},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is True
    assert payload["citations"][0]["namespace"] == "documents"
    assert payload["citations"][0]["source_kind"] == "document_chunk"
    assert "[1]" in payload["answer"]


def test_chat_can_route_to_ecommerce_tools_when_session_mounts_ecommerce() -> None:
    knowledge = FakeKnowledgeService(
        products=[
            _result(
                doc_id="product-1",
                content="AeroPhone X，库存充足。",
                score=0.95,
                metadata={"product_id": "P005"},
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货。")
    service = _build_chat_service(
        "chat-api-ecommerce-routing",
        knowledge,
        FakeDocumentRetrievalService(),
        model,
    )
    created = service.create_session(
        scene="generic_assistant",
        mounted_knowledge_sources=["documents", "ecommerce"],
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "AeroPhone X 现在有货吗", "session_id": created.session_id},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    trace = payload["retrieval_trace"]
    assert trace["final_decision"] == "max_rounds_reached"
    assert trace["knowledge_used"] is False
    assert trace["top_k_chunks"] == []
    round_tools = {round_trace["tool_name"] for round_trace in trace["rounds"]}
    assert "product_semantic_search" in round_tools
    assert "inventory_lookup" in round_tools


def test_chat_service_resolves_candidate_tools_from_scene_definition() -> None:
    observed_candidate_tools: list[tuple[str, ...]] = []

    class _TrackingRetriever:
        def retrieve_with_trace(self, query: str, *, candidate_tools: tuple[str, ...], **kwargs: Any):
            del query, kwargs
            observed_candidate_tools.append(candidate_tools)

            class _Outcome:
                documents: list[Document] = []
                exit_reason = "sufficient"
                success = True
                rounds: list[object] = []

            return _Outcome()

    scene_definition = SceneDefinition(
        scene="generic_assistant",
        name="Tracking Scene",
        description="Track candidate tool resolution source.",
        build_retriever=lambda: _TrackingRetriever(),  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda mounted_knowledge_sources: (
            "scene_documents_only",
            *(("scene_ecommerce_tool",) if "ecommerce" in mounted_knowledge_sources else ()),
        ),
        system_prompt="test",
        fallback_policy=SceneFallbackPolicy(no_hit_message="no hit"),
        infer_complexity=lambda _: "simple",
    )
    runtime_dir = make_test_runtime_dir("chat-service-scene-candidate-tools")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        session={"sqlite_path": sqlite_path, "window_size": 3},
    )
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=app_settings,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(answer="unused"),
    )
    created = service.session_store.create_session(
        session_id="scene-definition-candidate-tools",
        scene="generic_assistant",
        mounted_knowledge_sources=("documents", "ecommerce"),
    )

    response = service.chat(type("Payload", (), {
        "message": "hello",
        "session_id": created.session_id,
        "stream": False,
    })())

    assert response.knowledge_used is False
    assert observed_candidate_tools == [("scene_documents_only", "scene_ecommerce_tool")]


def test_chat_service_passes_through_scene_defined_custom_candidate_tools() -> None:
    observed_candidate_tools: list[tuple[str, ...]] = []

    class _TrackingRetriever:
        def retrieve_with_trace(self, query: str, *, candidate_tools: tuple[str, ...], **kwargs: Any):
            del query, kwargs
            observed_candidate_tools.append(candidate_tools)

            class _Outcome:
                documents: list[Document] = []
                exit_reason = "sufficient"
                success = True
                rounds: list[object] = []

            return _Outcome()

    scene_definition = SceneDefinition(
        scene="generic_assistant",
        name="Custom Candidate Scene",
        description="Pass through arbitrary candidate tools.",
        build_retriever=lambda: _TrackingRetriever(),  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda mounted_knowledge_sources: (
            "alpha_tool",
            *(("beta_extension_tool",) if "ecommerce" in mounted_knowledge_sources else ()),
        ),
        system_prompt="test",
        fallback_policy=SceneFallbackPolicy(no_hit_message="no hit"),
        infer_complexity=lambda _: "simple",
    )
    runtime_dir = make_test_runtime_dir("chat-service-custom-candidate-tools")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        session={"sqlite_path": sqlite_path, "window_size": 3},
    )
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=app_settings,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(answer="unused"),
    )
    created = service.session_store.create_session(
        session_id="custom-candidate-tools",
        scene="generic_assistant",
        mounted_knowledge_sources=("documents", "ecommerce"),
    )

    response = service.chat(type("Payload", (), {
        "message": "hello",
        "session_id": created.session_id,
        "stream": False,
    })())

    assert response.knowledge_used is False
    assert observed_candidate_tools == [("alpha_tool", "beta_extension_tool")]


def test_chat_service_passes_scene_retrieval_policy_to_agentic_retriever() -> None:
    observed_kwargs: list[dict[str, Any]] = []

    class _TrackingRetriever:
        def retrieve_with_trace(
            self,
            query: str,
            *,
            candidate_tools: tuple[str, ...],
            top_k: int | None = None,
            min_relevance_score: float | None = None,
            recall_strategy: str = "hybrid",
            rerank_enabled: bool = False,
            rerank_top_n: int | None = None,
        ):
            del query, candidate_tools
            observed_kwargs.append(
                {
                    "top_k": top_k,
                    "min_relevance_score": min_relevance_score,
                    "recall_strategy": recall_strategy,
                    "rerank_enabled": rerank_enabled,
                    "rerank_top_n": rerank_top_n,
                }
            )

            class _Outcome:
                documents: list[Document] = []
                exit_reason = "ask_user"
                success = False
                rounds: list[object] = []

            return _Outcome()

    scene_definition = SceneDefinition(
        scene="generic_assistant",
        name="Policy Scene",
        description="Track scene retrieval policy propagation.",
        build_retriever=lambda: _TrackingRetriever(),  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda _: ("knowledge_document_search",),
        system_prompt="test",
        fallback_policy=SceneFallbackPolicy(no_hit_message="no hit"),
        infer_complexity=lambda _: "simple",
        retrieval_policy=SceneRetrievalPolicy(
            top_k=2,
            min_relevance_score=0.91,
            recall_strategy="hybrid",
            no_hit_strategy="ask_user",
            rerank_enabled=True,
            rerank_top_n=1,
        ),
    )
    runtime_dir = make_test_runtime_dir("chat-service-scene-policy-propagation")
    sqlite_path = runtime_dir / "chat-sessions.db"
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=AppSettings(
            data_dir=runtime_dir,
            session={"sqlite_path": sqlite_path, "window_size": 3},
        ),
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(answer="unused"),
    )

    response = service.chat(type("Payload", (), {
        "message": "hello",
        "session_id": None,
        "stream": False,
        "top_k": 99,
    })())

    assert response.knowledge_used is False
    assert observed_kwargs == [
        {
            "top_k": 2,
            "min_relevance_score": 0.91,
            "recall_strategy": "hybrid",
            "rerank_enabled": True,
            "rerank_top_n": 1,
        }
    ]


def test_chat_no_hit_strategy_fallback_answer_uses_neutral_message() -> None:
    class _EmptySearchRetriever:
        def search(self, **kwargs: Any) -> list[Document]:
            del kwargs
            return []

    scene_definition = SceneDefinition(
        scene="generic_assistant",
        name="Fallback Answer Scene",
        description="Track no-hit strategy.",
        build_retriever=lambda: _EmptySearchRetriever(),  # type: ignore[return-value]
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda _: ("knowledge_document_search",),
        system_prompt="test",
        fallback_policy=SceneFallbackPolicy(
            no_hit_message="ask user fallback",
            neutral_no_hit_message="neutral no evidence fallback",
        ),
        infer_complexity=lambda _: "simple",
        retrieval_policy=SceneRetrievalPolicy(no_hit_strategy="fallback_answer"),
    )
    runtime_dir = make_test_runtime_dir("chat-no-hit-strategy-fallback-answer")
    sqlite_path = runtime_dir / "chat-sessions.db"
    service = ChatService(
        scene_definition=scene_definition,
        app_settings=AppSettings(
            data_dir=runtime_dir,
            session={"sqlite_path": sqlite_path, "window_size": 3},
        ),
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=FakeModel(answer="unused"),
    )

    response = service.chat(type("Payload", (), {
        "message": "hello",
        "session_id": None,
        "stream": False,
    })())

    assert response.knowledge_used is False
    assert response.citations == []
    assert response.answer == "neutral no evidence fallback"


def test_chat_scene_policy_min_relevance_preserves_no_hit_fallback() -> None:
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-low",
                content="低相关片段不应进入回答。",
                score=0.5,
                metadata={
                    "document_id": "doc-low",
                    "source_path": "low.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-low",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="unused")
    service = _build_chat_service(
        "chat-api-scene-min-relevance-no-hit",
        FakeKnowledgeService(),
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "低相关问题"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert document_retrieval_service.calls[-1]["minimum_relevance"] == 0.8


def test_chat_documents_and_ecommerce_mounts_do_not_force_extension_when_docs_are_sufficient() -> None:
    knowledge = FakeKnowledgeService(
        products=[
            _result(
                doc_id="product-1",
                content="AeroPhone X，库存充足。",
                score=0.95,
                metadata={"product_id": "P005"},
            )
        ]
    )
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="产品手册：AeroPhone X 价格 4599 元，电池 5000mAh。",
                score=0.97,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="根据产品手册，AeroPhone X 售价 4599 元，电池 5000mAh。")
    service = _build_chat_service(
        "chat-api-docs-sufficient-with-ecommerce-mounted",
        knowledge,
        document_retrieval_service,
        model,
    )
    created = service.create_session(
        scene="generic_assistant",
        mounted_knowledge_sources=["documents", "ecommerce"],
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "请根据产品手册说明 AeroPhone X 的价格和电池参数",
                "session_id": created.session_id,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is True
    assert {citation["namespace"] for citation in payload["citations"]} == {"documents"}


def test_session_detail_returns_message_view_with_assistant_metadata() -> None:
    knowledge = FakeKnowledgeService()
    document_retrieval_service = FakeDocumentRetrievalService(
        documents=[
            _result(
                doc_id="doc-1",
                content="AeroPhone X 当前有货，售价 4599 元。",
                score=0.93,
                metadata={
                    "document_id": "doc-1",
                    "source_path": "manual.md",
                    "namespace": "documents",
                    "is_managed_document": True,
                    "chunk_id": "chunk-manual-1",
                    "chunk_index": 0,
                },
            )
        ]
    )
    model = FakeModel(answer="AeroPhone X 当前有货，售价 4599 元。[1]")
    service = _build_chat_service(
        "chat-api-session-message-view",
        knowledge,
        document_retrieval_service,
        model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        chat_response = client.post("/chat", json={"message": "AeroPhone X 有货吗"})
        assert chat_response.status_code == 200
        chat_payload = chat_response.json()

        session_response = client.get(f"/sessions/{chat_payload['session_id']}")

    assert session_response.status_code == 200
    payload = session_response.json()
    assert payload["total_messages"] == 2
    assert [message["type"] for message in payload["messages"]] == ["human", "ai"]
    assistant_message = payload["messages"][1]
    assert assistant_message["request_id"] == chat_payload["request_id"]
    assert assistant_message["timestamp"]
    assert assistant_message["knowledge_used"] is True
    assert assistant_message["citations"][0]["citation_id"] == "chunk-manual-1"


def test_chat_api_real_runtime_filters_low_relevance_document_hits_for_greeting() -> None:
    runtime_dir = make_test_runtime_dir("chat-api-real-runtime-low-relevance")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={
            "active_scene": "generic_assistant",
        },
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        },
        vector_store={
            "provider": "chroma",
            "chroma": {"persist_directory": runtime_dir / ".chroma"},
        },
    )
    store = VectorStoreFactory.create(app_settings)
    store.ensure_document_indexes()
    store.upsert_document_chunks(
        [
            VectorStoreDocument(
                id="chunk-order-1",
                content=(
                    '{"carrier":"申通快递","status":"已签收","tracking_no":"ST0011223344CN",'
                    '"shipping_address":"重庆市渝中区解放碑步行街9号"}'
                ),
                metadata={
                    "document_id": "doc-order-1",
                    "source_path": "orders.json",
                    "namespace": "orders",
                    "chunk_id": "chunk-order-1",
                    "chunk_index": 0,
                    "is_active": True,
                },
            )
        ]
    )
    knowledge_service = create_knowledge_service(app_settings=app_settings, store=store)
    model = FakeModel(answer="unused")
    service = SceneChatService(
        scene_registry=build_default_scene_registry(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=_build_document_retrieval_service(app_settings),
        ),
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post(
            "/sessions",
            json={
                "scene": "generic_assistant",
                "mounted_knowledge_sources": ["documents"],
            },
        )
        assert create_response.status_code == 200
        response = client.post(
            "/chat",
            json={
                "message": "你好",
                "session_id": create_response.json()["session_id"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert "请补充" in payload["answer"]
    assert payload["retrieval_trace"]["final_decision"] == "ask_user"
    assert payload["retrieval_trace"]["follow_up_question"] == payload["answer"]
    assert model.get_runnable_calls == []


def test_chat_api_real_runtime_greeting_does_not_rewrite_into_faq_hit() -> None:
    runtime_dir = make_test_runtime_dir("chat-api-real-runtime-greeting-faq-no-rewrite")
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={
            "active_scene": "generic_assistant",
        },
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        },
        vector_store={
            "provider": "chroma",
            "chroma": {"persist_directory": runtime_dir / ".chroma"},
        },
    )
    store = VectorStoreFactory.create(app_settings)
    store.ensure_document_indexes()
    store.upsert_document_chunks(
        [
            VectorStoreDocument(
                id="chunk-faq-1",
                content="Support FAQ：如需查询已上传知识库，请提供具体文档主题、术语或章节名称。",
                metadata={
                    "document_id": "doc-faq-1",
                    "source_path": "support-faq.md",
                    "namespace": "documents",
                    "chunk_id": "chunk-faq-1",
                    "chunk_index": 0,
                    "is_active": True,
                    "is_managed_document": True,
                },
            )
        ]
    )
    knowledge_service = create_knowledge_service(app_settings=app_settings, store=store)
    model = FakeModel(answer="unused")
    service = SceneChatService(
        scene_registry=build_default_scene_registry(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=_build_document_retrieval_service(app_settings),
        ),
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post(
            "/sessions",
            json={
                "scene": "generic_assistant",
                "mounted_knowledge_sources": ["documents"],
            },
        )
        assert create_response.status_code == 200
        response = client.post(
            "/chat",
            json={
                "message": "你好",
                "session_id": create_response.json()["session_id"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert payload["retrieval_trace"]["knowledge_used"] is False
    assert payload["retrieval_trace"]["filtered_candidates_count"] == 0
    assert payload["retrieval_trace"]["top_k_chunks"] == []
    assert payload["retrieval_trace"]["citations"] == []
    assert "请补充" in payload["answer"]
    assert payload["retrieval_trace"]["final_decision"] == "ask_user"
    assert payload["retrieval_trace"]["follow_up_question"] == payload["answer"]
    assert model.get_runnable_calls == []


def test_chat_api_ignores_builtin_orders_json_in_documents_only_session() -> None:
    runtime_dir = make_test_runtime_dir("chat-api-ignore-builtin-orders")
    sqlite_path = runtime_dir / "chat-sessions.db"
    (runtime_dir / "files").mkdir(parents=True, exist_ok=True)
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={
            "active_scene": "generic_assistant",
        },
        session={
            "sqlite_path": sqlite_path,
            "window_size": 3,
        },
        vector_store={
            "provider": "chroma",
            "chroma": {"persist_directory": runtime_dir / ".chroma"},
        },
    )
    store = VectorStoreFactory.create(app_settings)
    store.ensure_document_indexes()
    store.upsert_document_chunks(
        [
            VectorStoreDocument(
                id="chunk-order-1",
                content='{"carrier":"EMS","status":"已签收","tracking_no":"EMS001"}',
                metadata={
                    "document_id": "doc-order-1",
                    "source_path": "orders.json",
                    "namespace": "orders",
                    "chunk_id": "chunk-order-1",
                    "chunk_index": 0,
                    "is_active": True,
                },
            )
        ]
    )
    knowledge_service = create_knowledge_service(app_settings=app_settings, store=store)
    model = FakeModel(answer="unused")
    service = SceneChatService(
        scene_registry=build_default_scene_registry(
            app_settings=app_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=_build_document_retrieval_service(app_settings),
        ),
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=model,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        create_response = client.post(
            "/sessions",
            json={
                "scene": "generic_assistant",
                "mounted_knowledge_sources": ["documents"],
            },
        )
        response = client.post(
            "/chat",
            json={
                "message": "你好",
                "session_id": create_response.json()["session_id"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_used"] is False
    assert payload["citations"] == []
    assert model.get_runnable_calls == []
