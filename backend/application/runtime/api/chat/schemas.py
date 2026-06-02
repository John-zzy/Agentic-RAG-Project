from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.platform.workflow.state_machine import WorkflowRunEvent, WorkflowRunState


class ChatRequest(BaseModel):
    """聊天接口请求体。"""

    message: str = Field(min_length=1, max_length=4000, description="用户本轮输入的问题。")
    session_id: str | None = Field(default=None, description="会话 ID，不传时由服务端自动创建。")
    hitl_clarification_enabled: bool = Field(
        default=False,
        description="是否把 generic_assistant 的 ask_user 分支转成 HITL 等待用户补充。",
    )
    stream: bool = Field(
        default=False,
        description="是否请求流式输出；为 true 时返回 SSE，仅最终回答阶段按 chunk 推送。",
    )


ChatRuntimeStatus = WorkflowRunState
HitlPendingAction = Literal["tool_approval", "external_api_approval", "clarification"]
HitlAllowedAction = Literal["approve", "edit", "reject", "respond"]
HitlResumeSource = Literal["suggested_response", "freeform", "system"]


class HitlSuggestedResponse(BaseModel):
    """需要用户补充信息时，后端给出的一个可选答案。"""

    suggestion_id: str = Field(description="建议项 ID，用来记录用户选了哪一项。")
    label: str = Field(description="建议项短标签，供前端按钮或列表展示。")
    value: str = Field(description="用户选择该建议后，实际提交的文本。")
    description: str | None = Field(default=None, description="建议项的可选说明。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="建议项扩展信息。")


class HitlResumePayload(BaseModel):
    """用户处理等待任务时提交的内容。"""

    action: HitlAllowedAction | None = Field(default=None, description="本次人工动作。")
    session_id: str | None = Field(default=None, description="本次 resume 所属会话 ID。")
    interrupt_id: str | None = Field(default=None, description="本次恢复的等待点 ID。")
    request_id: str | None = Field(default=None, description="本次 resume 请求 ID。")
    reason: str | None = Field(default=None, description="人工操作原因或备注。")
    edited_args: dict[str, Any] | None = Field(default=None, description="edit 动作后的工具参数。")
    response: str | None = Field(default=None, description="respond 动作的用户补充内容。")
    source: HitlResumeSource | None = Field(default=None, description="respond 内容来源。")
    suggestion_id: str | None = Field(default=None, description="被选择的建议项 ID。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="resume 扩展信息。")


class HitlState(BaseModel):
    """返回给前端的等待用户处理状态。"""

    interrupt_id: str = Field(description="当前等待点 ID，恢复时必须带回这个 ID。")
    thread_id: str = Field(description="LangGraph thread ID，当前等同于 session_id。")
    reason: str = Field(description="为什么需要用户处理。")
    pending_action: HitlPendingAction = Field(description="当前在等用户处理什么事情。")
    proposed_tool_call: dict[str, Any] | None = Field(
        default=None,
        description="待审批工具调用摘要；澄清场景为空。",
    )
    allowed_actions: list[HitlAllowedAction] = Field(
        default_factory=list,
        description="当前等待态允许执行的人类动作。",
    )
    suggested_responses: list[HitlSuggestedResponse] = Field(
        default_factory=list,
        description="ask_user 澄清场景的建议补充项；审批场景应为空。",
    )
    allow_freeform_response: bool = Field(
        default=False,
        description="是否允许用户自由输入补充信息。",
    )
    resume_payload: HitlResumePayload | None = Field(
        default=None,
        description="已恢复时记录的人工输入；初始等待态为空。",
    )


class ChatResumeRequest(BaseModel):
    """用户处理等待任务后，提交给 resume 接口的请求体。"""

    session_id: str = Field(description="需要恢复的会话 ID。")
    interrupt_id: str = Field(description="需要恢复的 HITL 等待点 ID。")
    action: HitlAllowedAction = Field(description="本次人工动作。")
    payload: HitlResumePayload = Field(
        default_factory=HitlResumePayload,
        description="本次人工动作携带的输入。",
    )
    stream: bool = Field(default=False, description="是否请求流式 resume 输出。")


class ChatResumeResponse(BaseModel):
    """resume 接口的响应体。"""

    session_id: str = Field(description="被恢复的会话 ID。")
    request_id: str = Field(description="本次 resume 请求 ID。")
    status: ChatRuntimeStatus = Field(description="resume 后的 runtime 状态。")
    state: ChatRuntimeStatus | None = Field(default=None, description="当前 workflow run 状态。")
    final_state: ChatRuntimeStatus | None = Field(default=None, description="终止时的 workflow 状态。")
    run_id: str | None = Field(default=None, description="本次 workflow run ID。")
    state_event: WorkflowRunEvent | None = Field(default=None, description="产生当前状态的状态机事件。")
    answer: str | None = Field(default=None, description="resume 后形成的最终说明或回答。")
    knowledge_used: bool = Field(default=False, description="resume 后的结果是否使用知识证据。")
    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="resume 后结果使用的引用；无证据或拒绝时为空。",
    )
    retrieval_trace: dict[str, Any] | None = Field(
        default=None,
        description="respond 继续检索时的 trace；审批 approve/reject 可为空。",
    )
    hitl: HitlState | None = Field(default=None, description="仍在等待时的 HITL 状态。")
    resume_payload: HitlResumePayload | None = Field(
        default=None,
        description="已接受的人工恢复输入，用于 resume 审计。",
    )


class SceneSummary(BaseModel):
    """场景列表项。"""

    scene: str = Field(description="场景唯一标识。")
    name: str = Field(description="场景展示名称。")
    description: str = Field(description="场景说明文案。")
    is_default: bool = Field(default=False, description="是否为默认场景。")


class SceneListResponse(BaseModel):
    """场景列表响应。"""

    default_scene: str = Field(description="当前系统默认场景。")
    scenes: list[SceneSummary] = Field(default_factory=list, description="可选场景列表。")


class Citation(BaseModel):
    """统一的回答引用信息。"""

    index: int = Field(ge=1, description="回答中展示的引用编号，从 1 开始。")
    citation_id: str = Field(description="引用的稳定 ID。")
    namespace: str = Field(description="引用所属命名空间。")
    source_kind: str = Field(description="来源类型，例如 document_chunk、product、order。")
    source_name: str = Field(description="前端展示用的来源名称。")
    source_path: str | None = Field(default=None, description="来源路径或来源主键。")
    document_id: str | None = Field(default=None, description="文档来源对应的文档 ID。")
    chunk_id: str | None = Field(default=None, description="文档分块 ID。")
    chunk_index: int | None = Field(default=None, description="文档分块序号。")
    snippet: str = Field(description="用于展示的命中文本片段。")
    score: float | None = Field(default=None, description="检索得分。")
    vector_score: float | None = Field(default=None, description="语义召回分数。")
    keyword_score: float | None = Field(default=None, description="关键词召回分数。")
    vector_rank: int | None = Field(default=None, description="语义召回原始排序位置，从 1 开始。")
    keyword_rank: int | None = Field(default=None, description="关键词召回原始排序位置，从 1 开始。")
    rerank_score: float | None = Field(default=None, description="真实 ReRank 成功应用后的重排分数。")
    matched_by: list[str] = Field(default_factory=list, description="命中来源，例如 vector、keyword。")
    rank: int = Field(ge=1, description="原始检索排序位置，从 1 开始。")


class RetrievalTraceTopChunk(BaseModel):
    """检索 trace 中的安全分块摘要，不包含完整正文。"""

    rank: int = Field(ge=1, description="最终候选排序位置，从 1 开始。")
    citation_id: str = Field(description="可与 citations 对齐的引用 ID。")
    document_id: str | None = Field(default=None, description="文档 ID。")
    chunk_id: str | None = Field(default=None, description="分块 ID。")
    chunk_index: int | None = Field(default=None, description="分块序号。")
    source_name: str = Field(description="来源展示名称。")
    source_path: str | None = Field(default=None, description="来源路径。")
    score: float | None = Field(default=None, description="融合或当前召回得分。")
    vector_score: float | None = Field(default=None, description="向量召回得分。")
    keyword_score: float | None = Field(default=None, description="关键词召回得分。")
    vector_rank: int | None = Field(default=None, description="向量召回排序。")
    keyword_rank: int | None = Field(default=None, description="关键词召回排序。")
    rerank_score: float | None = Field(default=None, description="真实 ReRank 成功应用后的重排分数。")
    matched_by: list[str] = Field(default_factory=list, description="命中来源。")


class RetrievalTraceRound(BaseModel):
    """Agentic RAG 单轮检索 trace。"""

    round_index: int = Field(ge=1, description="检索轮次，从 1 开始。")
    tool_name: str = Field(description="本轮调用的 retrieval tool 名称。")
    query: str = Field(description="本轮实际查询。")
    rewritten_query: str | None = Field(default=None, description="本轮触发改写后的下一轮查询。")
    decision: str = Field(description="本轮充分性判断动作。")
    is_sufficient: bool = Field(description="本轮判断是否已有足够证据。")
    reason: str | None = Field(default=None, description="本轮判断原因。")
    result_count: int = Field(ge=0, description="本轮工具返回 record 数量。")
    document_count: int = Field(ge=0, description="本轮工具返回 Document 数量。")
    success: bool = Field(description="本轮工具调用是否成功。")
    error: str | None = Field(default=None, description="本轮工具错误信息。")
    raw_candidates_count: int | None = Field(default=None, ge=0, description="过滤前候选数。")
    filtered_candidates_count: int | None = Field(default=None, ge=0, description="过滤后候选数。")
    top_k_chunks: list[RetrievalTraceTopChunk] = Field(
        default_factory=list,
        description="本轮过滤后的安全 top-k 分块摘要。",
    )
    rerank: dict[str, Any] | None = Field(default=None, description="本轮 ReRank trace 摘要。")


class RetrievalTrace(BaseModel):
    """一次 `/chat` 请求的检索链路 trace。"""

    original_query: str = Field(description="用户原始问题。")
    final_query: str = Field(description="检索结束时使用的查询。")
    rewritten_query: str | None = Field(default=None, description="最后一次 query rewrite 结果。")
    tool_call_count: int = Field(ge=0, description="实际 retrieval tool 调用次数。")
    candidate_tools: list[str] = Field(default_factory=list, description="本轮候选 retrieval tools。")
    exit_reason: str | None = Field(default=None, description="Agentic RAG 退出原因。")
    # 以下字段是向后兼容的观测字段，避免改变现有 `/chat` 必填响应契约。
    final_decision: str | None = Field(
        default=None,
        description=(
            "Runtime 归一化后的最终业务决策，例如 answer_with_evidence、ask_user、"
            "max_rounds_reached、no_evidence、retrieval_failed。"
        ),
    )
    success: bool | None = Field(
        default=None,
        description="聚合检索是否成功；旧式 retriever 未提供时为空。",
    )
    follow_up_question: str | None = Field(
        default=None,
        description="ask_user 分支可返回给用户的澄清问题；非追问场景为空。",
    )
    raw_candidates_count: int = Field(default=0, ge=0, description="聚合过滤前候选数。")
    filtered_candidates_count: int = Field(default=0, ge=0, description="聚合过滤后候选数。")
    top_k_chunks: list[RetrievalTraceTopChunk] = Field(
        default_factory=list,
        description="最终用于回答证据的安全 top-k 分块摘要。",
    )
    citations: list[Citation] = Field(default_factory=list, description="与响应一致的引用列表。")
    knowledge_used: bool = Field(description="本轮最终是否使用知识。")
    rounds: list[RetrievalTraceRound] = Field(default_factory=list, description="Agentic RAG 轮次 trace。")


class ChatResponse(BaseModel):
    """聊天接口响应体。"""

    session_id: str = Field(description="当前会话 ID。")
    request_id: str = Field(description="本次请求 ID。")
    answer: str = Field(description="最终回答文本，包含可见引用编号。")
    knowledge_used: bool = Field(description="本轮是否使用了知识检索结果。")
    scene: str = Field(description="本轮回答所属场景。")
    agent: str | None = Field(default=None, description="场景使用的代理标识，没有则为空。")
    status: ChatRuntimeStatus | None = Field(
        default=None,
        description="可选 runtime 状态；普通完成请求可为空，HITL 等待时为 waiting_user。",
    )
    state: ChatRuntimeStatus | None = Field(default=None, description="当前 workflow run 状态。")
    final_state: ChatRuntimeStatus | None = Field(default=None, description="终止时的 workflow 状态。")
    run_id: str | None = Field(default=None, description="本次 workflow run ID。")
    state_event: WorkflowRunEvent | None = Field(default=None, description="产生当前状态的状态机事件。")
    hitl: HitlState | None = Field(
        default=None,
        description="可选 HITL 等待态；非人工等待场景为空。",
    )
    citations: list[Citation] = Field(default_factory=list, description="结构化引用列表。")
    retrieval_trace: RetrievalTrace | None = Field(
        default=None,
        description="本轮检索链路 trace；仅用于本地可观测，不参与回答语义判断。",
    )


class SessionCreateResponse(BaseModel):
    """会话创建响应体。"""

    session_id: str = Field(description="新创建的会话 ID。")
    scene: str = Field(description="会话绑定的场景。")
    mounted_knowledge_sources: list[str] = Field(
        default_factory=list,
        description="当前会话允许使用的知识源列表。",
    )


class SessionCreateRequest(BaseModel):
    """会话创建请求体。"""

    scene: str | None = Field(default=None, description="要绑定的场景；不传时使用默认场景。")
    mounted_knowledge_sources: list[str] | None = Field(
        default=None,
        description="要挂载的知识源列表，例如 documents、ecommerce。",
    )


class SessionTurnResponse(BaseModel):
    """会话单轮响应体。"""

    request_id: str = Field(description="该轮对话的请求 ID。")
    user_message: str = Field(description="用户问题。")
    assistant_answer: str = Field(description="助手回答。")
    retrieval_snippets: list[dict[str, Any]] = Field(
        default_factory=list,
        description="该轮保存的引用片段列表，与 citations 契约兼容。",
    )
    timestamp: str = Field(description="该轮写入时间。")


class SessionMessageResponse(BaseModel):
    """会话单条 message 响应体。"""

    type: str = Field(description="消息类型，例如 human、ai。")
    content: str = Field(description="消息内容。")
    request_id: str = Field(description="所属请求 ID。")
    timestamp: str = Field(description="消息写入时间。")
    knowledge_used: bool | None = Field(
        default=None,
        description="仅 assistant 消息可用，表示该回答是否使用了知识检索。",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="仅 assistant 消息可用，表示该回答关联的结构化引用。",
    )


class SessionDetailResponse(BaseModel):
    """会话详情响应体。"""

    session_id: str = Field(description="会话 ID。")
    scene: str = Field(description="会话绑定的场景。")
    mounted_knowledge_sources: list[str] = Field(
        default_factory=list,
        description="该会话当前挂载的知识源列表。",
    )
    total_messages: int = Field(description="该会话历史总消息数。")
    messages: list[SessionMessageResponse] = Field(
        default_factory=list,
        description="最近的会话消息列表。",
    )


class SessionDeleteResponse(BaseModel):
    """会话删除响应体。"""

    session_id: str = Field(description="被删除的会话 ID。")
    deleted_messages: int = Field(description="被删除的消息数量。")
