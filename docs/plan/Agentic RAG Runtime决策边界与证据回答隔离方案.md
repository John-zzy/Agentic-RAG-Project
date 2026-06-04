# Agentic RAG Runtime 决策边界与证据回答隔离方案

## Summary

本计划对应 `PRD-20260530-01：Agentic RAG Runtime P0-1`，目标是修正 `AgenticRetrievalOutcome` 到 `/chat` 输出之间的决策消费边界。

当前项目已经具备 Agentic Retrieval、Hybrid Search、ReRank、`retrieval_trace`、SSE 和 Evaluation Harness。下一步风险不再是“能不能检索”，而是 runtime 能否把不确定的检索决策稳定地转成系统行为：

- 只有最终决策允许证据回答，且存在最终采纳证据时，才进入证据回答链。
- `ask_user` 必须返回澄清问题，不组装 citations，不标记 `knowledge_used=true`。
- `max_rounds_reached`、`success=false`、无有效证据必须降级或 no-hit fallback，不进入证据回答链。
- JSON 与 SSE 对同一个 retrieval outcome 的业务语义必须一致。

本计划严格对齐 README P0 第一项：

> Runtime 边界修正：严格消费 `AgenticRetrievalOutcome.success`、`final_decision` 和 `follow_up_question`，确保 `ask_user` / `max_rounds_reached` 不误入证据回答链。

本次不做 LangGraph Runtime、Workflow State Machine、Human-in-the-Loop、Planner / Executor、Tool Registry 平台化或 Agentic RAG Subgraph 迁移。

## Current Gap

当前主链路大致如下：

```text
AgenticRetriever.retrieve_with_trace()
  -> AgenticRetrievalOutcome
     success
     final_decision
     follow_up_question
     documents
     rounds
     exit_reason

RetrievalExecutor.retrieve()
  -> RetrievalExecutionResult
     documents
     tool_event
     retrieval_trace

ChatService._prepare_chat_turn()
  -> citations = citations_from_documents(documents)
  -> knowledge_used = len(citations) > 0

ChatService.chat / chat_stream
  -> knowledge_used=true 进入模型证据回答链
  -> knowledge_used=false 返回 fallback
```

主要缺口：

- `RetrievalExecutor` 没有把 `outcome.final_decision` 和 `outcome.follow_up_question` 作为 runtime 分支输入传出。
- `success` 只出现在 tool event 中，不参与是否允许证据回答的判断。
- `ChatService` 只用 `len(citations) > 0` 判断 `knowledge_used`。
- `ask_user` 分支没有专门返回 `follow_up_question`。
- `max_rounds_reached` 或 `success=false` 如果携带累计 documents，可能误入证据回答链。
- `retrieval_trace` 目前有 `exit_reason`，但没有稳定暴露归一化后的 `final_decision`。

按 PRD 6 条目标做实施前评估：

| 功能目标 | 当前状态 | 判断 |
| --- | --- | --- |
| 梳理 `AgenticRetrievalOutcome` 到 `/chat` 路径 | 部分满足 | 路径存在，但 `final_decision` / `follow_up_question` 未进入 API 语义 |
| 只有 `answer_with_evidence` 且有有效证据才进入证据回答链 | 未满足 | 当前只看 citations 数量 |
| `ask_user` 返回追问且不组 citations | 未满足 | 没有专门分支，`follow_up_question` 丢失 |
| `max_rounds_reached` / `success=false` / 无证据降级 | 部分满足 | 无证据 no-hit 已满足；有 documents 的失败边界不安全 |
| JSON 与 SSE 语义一致 | 部分满足 | 两者共享 `PreparedChatTurn`，但共享的是不完整门控语义 |
| 保留 normal-hit、no-hit、ReRank trace、Eval 回放 | 基线满足 | 已有测试与能力，修改后必须回归保护 |

实施前结论：当时有 5 条存在缺口，其中 3 条未满足、2 条部分满足、1 条为既有基线能力。

实施后复核结论：

| 功能目标 | 当前实现状态 | 判断 |
| --- | --- | --- |
| 梳理 `AgenticRetrievalOutcome` 到 `/chat` 路径 | `RetrievalExecutor` 读取 `success`、`final_decision`、`follow_up_question`，写入 `RetrievalExecutionResult`、`tool_event` 和 `retrieval_trace` | 已满足 |
| 只有 `answer_with_evidence` 且有有效证据才进入证据回答链 | `ChatService._prepare_chat_turn()` 使用 `final_decision + effective citations` 门控，非证据分支不调用 answer runnable | 已满足 |
| `ask_user` 返回追问且不组 citations | `answer_mode=follow_up` 直接返回追问文本，`knowledge_used=false`，`citations=[]` | 已满足 |
| `max_rounds_reached` / `success=false` / 无证据降级 | 统一进入 fallback 或 no-hit 降级，保留诊断 trace，清空最终采纳 citations | 已满足 |
| JSON 与 SSE 语义一致 | JSON、SSE `tool`、SSE `done` 共享同一份 `PreparedChatTurn` 决策结果 | 已满足 |
| 保留 normal-hit、no-hit、ReRank trace、Eval 回放 | 定向测试、回归测试和 OpenSpec 校验已通过 | 已满足 |

## Decision Model

在 application runtime 增加一个归一化的最终回答模式，不直接扩大底层 `RetrievalNextAction`。

当前 `SufficiencyDecision.next_action` 仍保持：

```text
finish | rewrite | switch_tool | ask_user
```

runtime 对外归一化为：

```text
answer_with_evidence | ask_user | max_rounds_reached | no_evidence | retrieval_failed
```

建议归一化规则：

```text
if outcome.success
   and outcome.final_decision.is_sufficient
   and outcome.final_decision.next_action == "finish"
   and effective_citations is not empty:
    final_decision = "answer_with_evidence"

elif outcome.exit_reason == "max_rounds_reached":
    final_decision = "max_rounds_reached"

elif outcome.final_decision.next_action == "ask_user"
   or outcome.exit_reason == "ask_user":
    final_decision = "ask_user"

elif outcome.success is false:
    final_decision = "retrieval_failed"

else:
    final_decision = "no_evidence"
```

证据回答门控固定为：

```text
can_answer_with_evidence =
    final_decision == "answer_with_evidence"
    and effective_citations is not empty
```

`knowledge_used` 固定等于 `can_answer_with_evidence`，不能再由“是否有中间 documents”单独决定。

## Runtime Branches

### 1. answer_with_evidence

触发条件：

- `outcome.success=true`
- 归一化 `final_decision=answer_with_evidence`
- 最终采纳证据非空，能映射出 citations

行为：

- 调用现有 RAG answer runnable。
- `knowledge_used=true`
- `citations` 为最终采纳证据。
- `retrieval_trace.knowledge_used=true`
- `retrieval_trace.citations` 与响应 `citations` 一致。
- `retrieval_trace.final_decision=answer_with_evidence`

### 2. ask_user

触发条件：

- `outcome.final_decision.next_action=ask_user`
- 或 `outcome.exit_reason=ask_user`

行为：

- 不调用证据回答模型。
- `answer` 使用 `outcome.follow_up_question`。
- 如果 `follow_up_question` 为空，使用 scene fallback 中等价的澄清问题。
- `knowledge_used=false`
- `citations=[]`
- `retrieval_trace.knowledge_used=false`
- `retrieval_trace.citations=[]`
- `retrieval_trace.final_decision=ask_user`
- 保留 `retrieval_trace.rounds`，用于解释为什么需要追问。

### 3. max_rounds_reached

触发条件：

- `outcome.exit_reason=max_rounds_reached`

行为：

- 不调用证据回答模型。
- 返回明确降级说明或 no-hit fallback。
- `knowledge_used=false`
- `citations=[]`
- `retrieval_trace.final_decision=max_rounds_reached`
- 保留 rounds、candidate counts、rerank trace，便于排查为什么检索耗尽。

### 4. retrieval_failed / no_evidence

触发条件：

- `outcome.success=false`
- 或无有效 citations
- 或非 agentic retriever 返回空 documents

行为：

- 不调用证据回答模型。
- 返回 no-hit fallback 或降级说明。
- `knowledge_used=false`
- `citations=[]`
- `retrieval_trace.final_decision=retrieval_failed | no_evidence`
- 无候选的 no-hit 场景保持 `retrieval_trace.filtered_candidates_count=0`；失败或耗尽轮次但存在中间候选时保留诊断计数。

## Key Changes

### 1. 扩展 RetrievalExecutionResult

在 `backend/application/runtime/service.py` 中扩展 `RetrievalExecutionResult`：

```python
@dataclass(frozen=True)
class RetrievalExecutionResult:
    documents: list[Document]
    tool_event: dict[str, Any]
    retrieval_trace: RetrievalTrace
    success: bool | None = None
    final_decision: str | None = None
    follow_up_question: str | None = None
```

对于 `retrieve_with_trace` 分支：

- 从 `AgenticRetrievalOutcome.success` 读取 `success`。
- 从 `AgenticRetrievalOutcome.final_decision` 归一化得到 `final_decision`。
- 从 `AgenticRetrievalOutcome.follow_up_question` 读取澄清问题。

对于旧 `search` / `BaseRetriever` 分支：

- 有 documents 时归一化为 `answer_with_evidence`。
- 无 documents 时归一化为 `no_evidence`。
- 保持兼容，不要求旧 retriever 实现 `AgenticRetrievalOutcome`。

### 2. 扩展 RetrievalTrace

在 `/chat` schema 的 `RetrievalTrace` 增加：

```python
final_decision: str | None = Field(default=None, description="runtime 归一化后的最终检索决策。")
success: bool | None = Field(default=None, description="Agentic Retrieval 聚合结果是否成功。")
follow_up_question: str | None = Field(default=None, description="需要用户补充信息时的澄清问题。")
```

其中 `follow_up_question` 可只在 `ask_user` 场景暴露。

### 3. 扩展 PreparedChatTurn

在 `PreparedChatTurn` 增加运行时分支信息：

```python
final_decision: str | None
follow_up_question: str | None
answer_mode: str
```

`answer_mode` 建议取值：

```text
evidence_answer | follow_up | fallback
```

这样 `chat()` 与 `chat_stream()` 可以共享同一套准备结果，不各自判断分支。

### 4. 改造 _prepare_chat_turn

将现有逻辑：

```python
citations = citations_from_documents(documents)
knowledge_used = len(citations) > 0
```

改为：

```python
citations = citations_from_documents(documents)
can_answer_with_evidence = (
    retrieval_result.final_decision == "answer_with_evidence"
    and len(citations) > 0
)

if can_answer_with_evidence:
    knowledge_used = True
    answer_mode = "evidence_answer"
else:
    knowledge_used = False
    citations = []
    documents = []
    answer_mode = "follow_up" if retrieval_result.final_decision == "ask_user" else "fallback"
```

注意：

- 可以保留 `retrieval_trace.rounds` 和每轮 trace。
- 最终响应层的 `citations` 必须只代表最终采纳证据。
- 如果没有进入证据回答链，最终 `top_k_chunks` 可以清空，避免前端误解为已采纳证据。

### 5. 改造 fallback / follow-up answer

新增或改造一个统一方法：

```python
def _build_non_evidence_answer(prepared: PreparedChatTurn) -> tuple[str, list[Citation]]:
    if prepared.answer_mode == "follow_up":
        return prepared.follow_up_question or default_follow_up_question, []
    return fallback_policy.message_for_strategy(...), []
```

非流式：

- `answer_mode=evidence_answer` 才调用 `_invoke_answer_template()`。
- 其他分支直接返回非证据答案。

流式：

- `answer_mode=evidence_answer` 才调用 `_stream_model_answer()`。
- 其他分支只发送一个 fallback/follow-up chunk。

### 6. 对齐 JSON 与 SSE

继续让 JSON 与 SSE 共用 `_prepare_chat_turn()`、`_build_tool_event()` 和 `_build_chat_response()`。

SSE 事件语义：

```text
start -> history -> tool -> chunk -> done
```

边界分支要求：

- `tool.knowledge_used == done.knowledge_used`
- `tool.citations == done.citations`
- `tool.retrieval_trace.final_decision == done.retrieval_trace.final_decision`
- `tool.retrieval_trace.knowledge_used == done.retrieval_trace.knowledge_used`

## Public Interfaces

`/chat` JSON 响应示例：

```json
{
  "answer": "请补充更具体的文档主题、术语，或说明你希望查询的业务知识范围。",
  "knowledge_used": false,
  "citations": [],
  "retrieval_trace": {
    "exit_reason": "ask_user",
    "final_decision": "ask_user",
    "success": false,
    "follow_up_question": "请补充更具体的文档主题、术语，或说明你希望查询的业务知识范围。",
    "knowledge_used": false,
    "citations": [],
    "rounds": []
  }
}
```

SSE 中：

- `tool.retrieval_trace.final_decision` 与 `done.retrieval_trace.final_decision` 一致。
- `tool.knowledge_used` 与 `done.knowledge_used` 一致。
- `ask_user` / `fallback` 分支不触发模型流式回答。

## Non-Goals

本次明确不做：

- 不引入 LangGraph Runtime。
- 不实现 graph state、`thread_id`、checkpointer 或 graph lifecycle。
- 不做 Human-in-the-Loop interrupt/resume。
- 不做 Workflow State Machine。
- 不做 Planner / Executor。
- 不迁移 `AgenticRetriever` while loop 为 LangGraph Subgraph。
- 不重构 Tool Registry。
- 不重构整个 `ChatService`。
- 不扩充 retrieval benchmark 样本。
- 不放宽 no-hit fallback、citations 可信边界或 ReRank 默认兼容行为。

## Test Plan

定向测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

回归测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_document_hybrid_retrieval.py backend\tests\test_eval_assets.py -q -c backend\tests\pytest.ini
```

新增或调整测试：

- `ask_user` 非流式：
  - response answer 等于 `follow_up_question` 或默认澄清问题。
  - `knowledge_used=false`
  - `citations=[]`
  - `retrieval_trace.final_decision=ask_user`
  - 不调用 answer runnable。
- `ask_user` SSE：
  - `tool` 与 `done` 的 `final_decision / knowledge_used / citations` 一致。
  - chunk 输出澄清问题。
  - 不调用 `stream_runnable()`。
- `max_rounds_reached`：
  - 即使 outcome 中有 documents，也不进入证据回答链。
  - `knowledge_used=false`
  - `citations=[]`
  - `retrieval_trace.final_decision=max_rounds_reached`
- `success=false`：
  - 即使 outcome 中有 documents，也不进入证据回答链。
  - 返回降级或 no-hit fallback。
  - 不产生 citations。
- `answer_with_evidence`：
  - 有效证据存在时保持现有回答、citations 和 citation markers。
  - `retrieval_trace.final_decision=answer_with_evidence`
  - `retrieval_trace.citations` 与响应 citations 一致。
- 既有 no-hit fallback：
  - 保持 `knowledge_used=false`
  - 保持 `citations=[]`
  - 不调用回答模型。
- ReRank trace：
  - 保留 `rounds[].rerank`
  - 不因决策门控丢失 ReRank 观测信息。
- Evaluation Harness：
  - minimal / retrieval benchmark 回放继续通过。
  - SSE 回放中的 `done.retrieval_trace` 语义与 JSON 一致。

## Acceptance

完成后必须满足：

1. `ask_user` 场景返回追问文本，`knowledge_used=false`，`citations=[]`，`retrieval_trace.final_decision=ask_user`。
2. `max_rounds_reached` 场景返回降级或 no-hit 说明，不进入证据回答生成链。
3. `success=false` 场景不进入证据回答生成链，即使中间存在 documents。
4. `answer_with_evidence` 场景在有效证据存在时仍正常返回答案、citations 和 `retrieval_trace`。
5. JSON 与 SSE 对同类边界场景的 `final_decision`、`knowledge_used`、`citations`、`retrieval_trace` 语义一致。
6. no-hit fallback、ReRank trace、Evaluation Harness 回放能力不回归。
7. 能清楚解释 runtime 边界：什么时候回答、什么时候追问、什么时候降级，以及为什么 citations 只能代表最终采纳证据。

## Assumptions

- `answer_with_evidence` 是 application runtime 的归一化决策，不要求本次修改底层 `RetrievalNextAction` literal。
- `citations` 只代表最终采纳证据，不代表所有检索中间候选。
- `retrieval_trace.rounds` 可以保留中间过程，但最终响应层 `citations` 和 `knowledge_used` 必须由 runtime 决策门控决定。
- 旧 `search` / `BaseRetriever` 分支继续兼容，以 documents 是否为空归一化为 `answer_with_evidence` 或 `no_evidence`。
- 本次不改变 session 数据库结构；如果需要持久化新增 trace 字段，后续单独设计。
- README roadmap 已将 P0 第一项更新为已完成，并补充 Runtime 边界 Demo Path。


