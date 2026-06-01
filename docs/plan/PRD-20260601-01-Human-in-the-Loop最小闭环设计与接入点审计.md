# PRD-20260601-01 Human-in-the-Loop 最小闭环设计与接入点审计

## Summary

本 PRD 的目标不是马上扩大 Agent Runtime 功能面，而是在现有 LangGraph Runtime 骨架之上，先为 `generic_assistant` 场景定义 Human-in-the-Loop 的最小闭环、状态边界、后端协议和演示路径。

当前项目已经具备 graph state、`thread_id=session_id`、SQLite checkpointer、graph run lifecycle 和 SSE event mapping 的基础，但仍没有 LangGraph `interrupt/resume`、`waiting_user` 状态、人工动作协议或 resume API。现有 `generic_assistant` 的 `ask_user` 只是 Agentic Retrieval 的一种非证据终止分支，会直接返回澄清问题，不会挂起 graph run，也不能用人工 payload 恢复同一次执行。

本设计将 HITL 收敛为一个最小可解释闭环：

```text
user request
  -> graph detects approval / clarification boundary
  -> interrupt
  -> runtime persists waiting_user checkpoint
  -> SSE/API exposes pending action
  -> human approve / reject / respond
  -> resume same thread
  -> done or error
```

实施同步：当前变更已经按“只扩展 `generic_assistant`”落地最小闭环。实现采用显式 `waiting_user` checkpoint 和 `/chat/resume` 应用层入口，不引入完整长期任务队列表；`edit` 仍只保留协议占位，不出现在首轮 generic 等待态的 `allowed_actions` 中。

## Scope

本 PRD 覆盖：

- 审计当前 LangGraph Runtime 代码入口和缺口。
- 功能设计暂时只扩展 `generic_assistant` 场景，不接入 `ecommerce` 或其他业务扩展场景。
- 定义最小 HITL state 字段。
- 定义三类 interrupt 触发场景：写操作工具、外部 API 调用、`ask_user` 澄清。
- 定义 `approve / edit / reject / respond` 四类动作语义，其中首轮闭环 `approve / reject / respond`，`edit` 只保留协议占位。
- 定义 `/chat` 与 SSE 如何暴露 `waiting_user` 和 resume 结果。
- 给出最小验证路径、测试计划、JD 证明点和简历素材草稿。

本 PRD 不覆盖：

- 不实现完整多 Agent / CloudAgent。
- 不设计复杂审批 UI。
- 不重写 RAG 主链、Hybrid Search、ReRank 或 Agentic Retrieval while loop。
- 不引入完整权限系统。
- 不实现长期任务队列或完整 Workflow State Machine。
- 不把 `ecommerce` 的订单、库存、投诉、退货等业务工具纳入本轮 HITL 设计。
- 不通过 `GenericAssistantBusinessExtension` 把业务场景工具反向塞入 generic 主线；首轮只验证 generic assistant 自身边界。

## Current Implementation Audit

当前实现入口：

| 能力 | 当前位置 | 状态 |
| --- | --- | --- |
| LangGraph application facade | `backend/application/runtime/graph_runtime.py` | 已有 `ChatGraphRuntime.invoke()`，但只编译单节点 answer graph |
| Runtime graph state | `backend/platform/workflow/langgraph/state.py` | 已扩展 `status/hitl/hitl_resume`，能保存 `waiting_user` |
| Graph config | `backend/platform/workflow/langgraph/config.py` | 已绑定 `thread_id=session_id`，`request_id/session_id` 写入 metadata |
| SQLite checkpointer | `backend/platform/workflow/langgraph/checkpointer.py` | 已支持 checkpoint、blobs、pending writes、thread delete |
| Graph run lifecycle | `backend/platform/workflow/langgraph/lifecycle.py` | 只有 `created/running/succeeded/failed` |
| SSE event mapping | `backend/application/runtime/stream_events.py` | 已映射 `start/history/tool/chunk/waiting_user/resume/done/error` |
| `/chat` 非流式 graph 接入 | `backend/application/runtime/service.py` | `_generate_answer()` 调用 `graph_runtime.invoke()` |
| `/chat` 流式路径 | `backend/application/runtime/service.py` | 已在 HITL 等待时返回 `waiting_user`，非 HITL 顺序保持兼容 |
| `/chat/resume` | `backend/application/runtime/api/chat/routes.py` | 已新增非流式与流式 resume 入口 |
| Agentic Retrieval ask_user | `backend/platform/rag/orchestration/agentic.py` | 默认仍是 follow-up fallback；HITL 开关开启时由 runtime 转换为 clarification wait |
| generic assistant HITL planner | `backend/scenes/generic_assistant/hitl.py` | 已新增 generic HITL 开关、等待计划和建议项生成 |
| generic assistant tool | `backend/scenes/generic_assistant/tools/knowledge_document_search.py` | 只读文档检索不触发 HITL |
| generic HITL test tools | `backend/scenes/generic_assistant/tools/hitl_test_tools.py` | 已新增 opt-in fake write / fake external API 工具 |
| ecommerce tools | `backend/scenes/ecommerce/tools/` | 存在订单、工单等业务工具，但不属于本 PRD 设计范围 |

实施前缺口：

| PRD 需求 | 当前状态 | 缺口 |
| --- | --- | --- |
| `interrupt_id` | 已满足 | 当前等待点写入 checkpoint，并在 resume 时校验 |
| `pending_action` | 已满足 | 支持 `tool_approval/external_api_approval/clarification` |
| `proposed_tool_call` | 已满足 | 审批等待态保存工具名、操作、参数和风险等级 |
| `allowed_actions` | 已满足 | 首轮暴露 `approve/reject` 或 `respond/reject`，不暴露 edit |
| `resume_payload` | 已满足 | resume 后记录 action、session、request、source、suggestion_id、response |
| `waiting_user` | 已满足 | graph state、JSON response 和 SSE 均可表达 |
| `approve/reject` | 已满足 | approve 执行当前 proposed tool；reject 跳过副作用并返回说明 |
| `respond` | 已满足 | 使用用户补充内容继续 generic 检索；无证据时走原有 fallback 边界 |

## Target HITL State

最小 state 不替代完整 Workflow State Machine，只为一次 graph interrupt/resume 保留可解释上下文。

建议在现有 `RuntimeGraphState` 上扩展：

```python
class HitlState(TypedDict, total=False):
    interrupt_id: str
    thread_id: str
    reason: str
    pending_action: str
    proposed_tool_call: dict[str, Any] | None
    allowed_actions: list[str]
    suggested_responses: list[dict[str, Any]]
    allow_freeform_response: bool
    resume_payload: dict[str, Any] | None
```

字段语义：

| 字段 | 语义 | 约束 |
| --- | --- | --- |
| `interrupt_id` | 单次等待点 ID | 同一 `thread_id` 内唯一，用于 resume 幂等校验 |
| `thread_id` | LangGraph thread ID | 等于业务 `session_id` |
| `reason` | 中断原因说明 | 面向审计和前端展示，不驱动权限判断 |
| `pending_action` | 等待中的动作类型 | 例如 `tool_approval`、`external_api_approval`、`clarification` |
| `proposed_tool_call` | 拟执行工具调用摘要 | 高风险工具和外部 API 必填；`ask_user` 可为空 |
| `allowed_actions` | 当前允许的人工动作 | 最小闭环必须包含 `approve/reject` 或 `respond` |
| `suggested_responses` | 后端给用户的澄清建议选项 | 仅用于 `pending_action=clarification`；审批场景应为空 |
| `allow_freeform_response` | 是否允许用户自由输入补充信息 | 澄清场景建议为 `true`，避免强迫用户只能点选 |
| `resume_payload` | 人工恢复输入 | 初次 interrupt 为空；resume 后写入动作、会话、请求、等待点和参数 |

建议将 HITL 结构嵌入 graph state：

```python
class RuntimeGraphState(TypedDict):
    ...
    status: Literal["running", "waiting_user", "succeeded", "failed"]
    hitl: HitlState | None
```

当前 PRD 只要求最小等待态，因此 `status` 不需要扩展到完整 `created/planning/retrying/cancelled`。

## Interrupt Trigger Scenarios

### 1. 写操作工具

触发条件：

- `generic_assistant` 场景内新增或暴露的工具声明为 write operation，例如知识文档发布、知识条目标注、会话摘要写入或演示用 write tool。
- 工具执行会改变 generic assistant 可管理的本地状态、知识库状态或用户可见结果。
- 当前 `knowledge_document_search` 是只读检索工具，不应触发审批。

interrupt payload：

```json
{
  "pending_action": "tool_approval",
  "reason": "Tool call changes external state.",
  "proposed_tool_call": {
    "tool_name": "generic_knowledge_document_publish",
    "operation": "write",
    "args_preview": {},
    "risk_level": "high"
  },
  "allowed_actions": ["approve", "reject"]
}
```

边界：

- approve 后执行原工具调用。
- reject 后跳过工具调用，进入拒绝后的安全回答或 error。
- edit 首轮不暴露在 `allowed_actions`；后续如果实现，再设计参数编辑、schema diff 和重新校验。

### 2. 外部 API 调用

触发条件：

- `generic_assistant` 场景内新增的 generic 外部 API 工具访问第三方服务，且调用不可完全由本地只读检索解释。
- 例如 webhook、文档同步、通知发送、外部知识源刷新等 generic assistant 可解释的 demo tool。
- 不包含 `ecommerce` 的支付、订单、CRM、投诉工单、退货工单或库存工具。

interrupt payload：

```json
{
  "pending_action": "external_api_approval",
  "reason": "External API call requires human approval.",
  "proposed_tool_call": {
    "tool_name": "generic_external_webhook_call",
    "operation": "external_api",
    "args_preview": {},
    "risk_level": "medium"
  },
  "allowed_actions": ["approve", "reject"]
}
```

边界：

- 只读外部查询可以先不进入 HITL，除非 scene policy 明确要求。
- 外部 API 审批关注执行风险；不等同于用户澄清。

### 3. `ask_user` 澄清

触发条件：

- Agentic Retrieval 判断需要用户补充信息。
- no-hit fallback 策略为 `ask_user`。
- query rewrite 达到限制，继续检索不可解释。

interrupt payload：

```json
{
  "pending_action": "clarification",
  "reason": "More user input is required before continuing.",
  "proposed_tool_call": null,
  "allowed_actions": ["respond", "reject"],
  "suggested_responses": [
    {
      "suggestion_id": "topic_scope",
      "label": "限定文档主题",
      "value": "我想查询安全合规政策相关内容。"
    },
    {
      "suggestion_id": "term_scope",
      "label": "限定术语",
      "value": "请围绕权限审批流程继续检索。"
    }
  ],
  "allow_freeform_response": true
}
```

边界：

- `ask_user` 澄清不是审批工具风险，而是补充任务输入。
- `suggested_responses` 类似 Codex 的建议选项，只帮助用户快速补充上下文，不代表系统替用户做决定。
- 用户可以选择建议项，也可以自由输入；自由输入应通过同一 `respond` 动作恢复。
- respond 后将用户补充内容写入 `resume_payload`，同一 graph thread 继续执行。
- reject 表示用户不提供补充信息，runtime 应输出可解释 fallback，而不是继续伪造证据。

## Human Actions

| 动作 | 适用场景 | 对 graph state 的影响 | 后续执行 |
| --- | --- | --- | --- |
| `approve` | 工具审批、外部 API 审批 | `resume_payload.action=approve`，清空当前等待态 | 执行原 `proposed_tool_call`，成功后继续图 |
| `edit` | 工具审批、外部 API 审批 | 首轮不暴露在 `allowed_actions`，提交时返回 unsupported | 后续再设计参数编辑、schema diff 和重新校验 |
| `reject` | 所有等待态 | `resume_payload.action=reject`，记录拒绝原因 | 不执行工具；返回安全终止回答或 error |
| `respond` | `ask_user` 澄清 | `resume_payload.action=respond`，写入用户补充文本、来源和可选 `suggestion_id` | 将补充输入并入当前任务，继续检索或回答 |

本 PRD 首轮实现建议只完成 `approve/reject/respond`，将 `edit` 保留为协议字段和测试占位。原因是 edit 需要参数 schema diff、重新校验和 UI 输入能力，容易把最小闭环扩大成复杂表单系统。

## Approve Flow

最小流程：

```text
POST /chat
  -> graph reaches risky tool node
  -> interrupt with proposed_tool_call
  -> checkpoint persisted under thread_id=session_id
  -> response/SSE exposes status=waiting_user and interrupt_id

POST /chat/resume
  action=approve
  interrupt_id=...
  -> runtime validates thread_id + interrupt_id
  -> graph resumes with Command(resume={action:"approve"})
  -> tool executes
  -> graph produces done or error
```

approve 语义：

- approve 只批准当前 `interrupt_id` 对应的 `proposed_tool_call`。
- approve 不应批准后续新工具调用；后续高风险动作必须再次 interrupt。
- approve 后 `allowed_actions` 应清空或从 state 中移除。
- 如果 resume 时 checkpoint 不存在、interrupt_id 不匹配或动作不在 `allowed_actions`，返回业务错误。

## Reject Flow

最小流程：

```text
POST /chat
  -> graph reaches risky tool node or clarification boundary
  -> interrupt
  -> response/SSE exposes waiting_user

POST /chat/resume
  action=reject
  reason="用户拒绝执行"
  -> runtime records resume_payload
  -> skips proposed tool call
  -> graph returns safe final answer or error
```

reject 语义：

- 对工具审批：不执行工具，并向用户说明该操作未被执行。
- 对外部 API：不调用外部 API，避免副作用。
- 对 `ask_user`：不继续追问，返回 no-hit/fallback 结果。
- reject 不等同于系统异常；只有当业务要求必须执行该动作才能完成任务时，才进入 error。

## API Contract Sketch

不改变现有 `/chat` 的默认行为。新增 resume 入口建议由 application runtime 持有：

```http
POST /chat/resume
```

请求体：

```json
{
  "session_id": "session-1",
  "interrupt_id": "interrupt-1",
  "action": "approve",
  "payload": {
    "reason": "approved by operator",
    "edited_args": null,
    "response": null,
    "suggestion_id": null,
    "source": null
  },
  "stream": false
}
```

`/chat` 等待态沿用 `ChatResponse`，但需要额外暴露：

```json
{
  "session_id": "session-1",
  "request_id": "req-1",
  "status": "waiting_user",
  "hitl": {
    "interrupt_id": "interrupt-1",
    "thread_id": "session-1",
    "reason": "Tool call changes external state.",
    "pending_action": "tool_approval",
    "proposed_tool_call": {},
    "allowed_actions": ["approve", "reject"],
    "suggested_responses": [],
    "allow_freeform_response": false,
    "resume_payload": null
  }
}
```

兼容策略：

- 非 HITL 请求保持现有 `ChatResponse` 字段不变。
- `status/hitl` 可作为新增可选字段，避免破坏现有前端和 eval harness。
- `/chat/resume` 使用 `ChatResumeResponse`，返回 `status/answer/knowledge_used/citations/retrieval_trace/hitl/resume_payload`，让客户端能看到恢复后的最终说明或回答。

## SSE Contract Sketch

当前 SSE 事件只有：

```text
start -> history -> tool -> chunk... -> done
start -> history -> tool -> error
```

HITL 最小扩展：

```text
start -> history -> tool -> waiting_user
resume -> done
resume -> error
```

新增业务事件：

| 事件 | payload |
| --- | --- |
| `waiting_user` | `session_id/request_id/status/hitl` |
| `resume` | `session_id/request_id/interrupt_id/action` |

约束：

- 不向前端暴露 LangGraph 原始 event name。
- `waiting_user` 必须包含 `interrupt_id` 和 `allowed_actions`。
- `/chat/resume` 流式路径的 `done` 使用 `ChatResumeResponse`；`respond` 继续检索时会携带新的 `retrieval_trace`，`approve/reject` 可不带 trace。
- reject 路径也应发 `done`，除非 runtime 发生系统异常。

## Minimal Demo Path

### Path A: approve generic assistant 写操作工具

```text
用户：请把这份文档发布到知识库
runtime：构造 proposed_tool_call=generic_knowledge_document_publish
graph：interrupt，status=waiting_user
SSE：waiting_user，allowed_actions=["approve","reject"]
人工：approve
runtime：resume same thread
tool：执行 publish
graph：done
```

预期结果：

- checkpoint 中能恢复同一 `thread_id`。
- `resume_payload.action=approve` 可追溯。
- 工具只在 approve 后执行。

### Path B: reject generic assistant 外部 API

```text
用户：把这次知识库同步结果发到外部 webhook
runtime：构造 proposed_tool_call=generic_external_webhook_call
graph：interrupt，status=waiting_user
人工：reject，reason="暂不创建"
runtime：resume same thread
graph：跳过工具，生成未执行说明
done：knowledge_used=false 或按任务结果返回
```

预期结果：

- 外部 API 未被调用。
- `done` 明确说明操作被拒绝且未执行。
- 不进入伪成功状态。

### Path C: respond ask_user 澄清

```text
用户：帮我查那个政策
retrieval：ask_user，需要用户明确政策名称
graph：interrupt，pending_action=clarification
SSE/API：给出 suggested_responses，并允许自由输入
人工/用户：respond，选择建议“安全合规政策”或自由补充
runtime：resume same thread
retrieval：用补充信息继续检索
done：有证据回答或 no-hit fallback
```

预期结果：

- `ask_user` 不再只是一次性 fallback 文本，而是可恢复任务等待点。
- 建议项只用于降低用户补充成本；用户的最终选择或自由输入才进入 `resume_payload`。
- respond 和 approve/reject 属于同一 resume 协议，但语义不同。

## Test Plan

定向测试建议：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_checkpointer.py backend\tests\test_langgraph_runtime.py -q -c backend\tests\pytest.ini
```

HITL 新增测试建议：

| 测试 | 目的 |
| --- | --- |
| `test_hitl_interrupt_state_contains_required_fields` | 校验 `interrupt_id/thread_id/reason/pending_action/allowed_actions` |
| `test_hitl_tool_approval_waits_before_write_tool` | 写工具在 approve 前不执行 |
| `test_hitl_approve_resumes_same_thread_and_executes_tool` | approve 恢复同一 checkpoint |
| `test_hitl_reject_skips_tool_and_returns_done` | reject 不执行工具且返回可解释结果 |
| `test_hitl_respond_resumes_ask_user_clarification` | respond 将澄清输入并入后续检索 |
| `test_hitl_clarification_exposes_suggested_responses` | ask_user 等待态暴露建议选项和自由输入开关 |
| `test_hitl_respond_records_selected_suggestion_source` | respond 记录建议项来源和 `suggestion_id` |
| `test_hitl_sse_waiting_user_then_resume_done` | SSE 包含 waiting_user/resume/done |
| `test_hitl_resume_rejects_unknown_interrupt_id` | 防止错误或过期等待点被恢复 |

回归测试建议：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py backend\tests\test_agentic_retrieval.py backend\tests\test_eval_assets.py -q -c backend\tests\pytest.ini
```

必须防回归的现有路径：

- normal-hit：有证据回答仍返回 citations 和 `knowledge_used=true`。
- no-hit：无证据时不伪造 citation，`knowledge_used=false`。
- SSE：现有 `start/history/tool/chunk/done/error` 语义不变。
- runtime：graph checkpoint、thread cleanup、lifecycle 不退化。
- `ask_user/max_rounds_reached`：未启用 HITL 时仍不误入证据回答链。

## Acceptance Checklist

- [x] 能指出 LangGraph Runtime 当前实现入口和关键状态字段。
- [x] 有 HITL 最小 state 结构，覆盖 `interrupt_id/thread_id/reason/pending_action/proposed_tool_call/allowed_actions/resume_payload`。
- [x] 明确写操作工具、外部 API、`ask_user` 三类 interrupt 场景。
- [x] 明确 approve/reject/respond/edit 语义边界。
- [x] approve 路径能证明工具只在人工批准后执行。
- [x] reject 路径能证明工具或外部 API 没有执行。
- [x] ask_user 澄清能解释为补充输入等待点，而不是工具审批。
- [x] ask_user 澄清能给出建议选项，并允许用户自由输入。
- [x] SSE 能表达 `waiting_user` 和 resume 结果。
- [x] 测试计划覆盖 normal-hit、no-hit、SSE 和 runtime 边界。
- [x] 有可用于面试讲解的 JD 证明点和简历素材。

## JD Proof Points

可用于 Agent Runtime 岗位讲解的证明点：

- 不是普通弹窗确认，而是基于 LangGraph checkpoint 的可恢复执行边界。
- `thread_id=session_id` 保证用户会话、graph state 和 resume 请求可关联。
- `interrupt_id` 让单个等待点可审计、可幂等校验、可拒绝过期恢复。
- `proposed_tool_call` 将工具执行前的参数摘要暴露出来，便于人工判断风险。
- `allowed_actions` 将 UI 能做什么交给后端协议约束，而不是前端硬编码。
- `approve/reject/respond` 将工具审批和用户澄清统一到 resume 协议，但保留语义边界。
- `suggested_responses` 让 ask_user 节点像 Codex 一样给出可点选建议，同时通过自由输入保留用户控制权。
- SSE 的 `waiting_user/resume/done/error` 让长链路 Agent 执行可观察、可回放。

## Resume Draft

简历素材草稿：

```text
设计并接入 `generic_assistant` 场景的 Agent Runtime Human-in-the-Loop 最小闭环：基于 LangGraph thread checkpoint 定义 interrupt/resume 协议，抽象 waiting_user 状态、interrupt_id、proposed_tool_call、allowed_actions、suggested_responses 与 resume_payload，覆盖 generic 写操作工具审批、generic 外部 API 审批和 ask_user 澄清三类场景；为澄清节点提供类似 Codex 的建议选项和自由输入恢复路径，设计 approve/reject/respond 动作语义、SSE waiting/resume 事件和回归测试矩阵，保证高风险工具调用可审计、可拒绝、可恢复，而不是前端一次性弹窗确认。
```

## Implementation Notes For Follow-up PRD

建议后续实现分四步：

1. 扩展 state/schema/event，只新增可选字段和新事件，不改变现有 `/chat` 默认响应。
2. 在 graph runtime 中接入 LangGraph interrupt/resume，先用 fake write tool 做 smoke test。
3. 将 `ask_user` 从一次性 fallback 升级为可选 HITL clarification interrupt，保留兼容开关。
4. 再接入 `generic_assistant` 场景内的真实或 fake 写工具 / 外部 API 工具，补 approve/reject 端到端测试。

风险：

- 直接把所有 `ask_user` 改成 interrupt 会影响现有 no-hit fallback 和 eval harness，需要兼容开关或分阶段迁移。
- `edit` 若首轮实现会引入工具参数 schema、表单编辑和重新校验复杂度，建议先做协议占位。
- lifecycle 若过早扩展完整 Workflow State Machine，会扩大本 PRD 范围；首轮只需要 `waiting_user`。
- 若误把 `ecommerce` 工具纳入本轮，会把 HITL 设计变成业务工具权限设计，偏离 generic assistant 最小闭环目标。
