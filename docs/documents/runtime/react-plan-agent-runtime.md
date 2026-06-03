# ReAct / Plan Agent Runtime

本说明用尽量直白的方式解释当前 `/chat` 怎么跑起来。

可以先把系统理解成三层：

1. `/chat` 是入口，负责接收用户问题。
2. 顶层 Agent 负责决定“现在要不要拆步骤、要调用哪个工具”。
3. RAG 只是 Agent 可以调用的一个工具，负责查资料并返回证据。

所以现在的边界是：简单问题走 ReAct，像“先想一下，再调用一个工具”；复杂问题走 Plan，像“先列步骤，再一步步执行”。RAG 与 Agentic RAG 仍属于 `platform.rag` 的 Modular RAG 能力，在顶层 Agent 中通过 `ToolExecutor` 调用。Agentic Retrieval 自己内部的多轮检索过程只保存在 `retrieval_trace.rounds`，不会被当成顶层 Agent 的多个步骤。

## 当前运行图

- 主链路图：[SVG](./main-chat-agent-runtime-flow.svg) / [Mermaid](./main-chat-agent-runtime-flow.mmd)

当前 `/chat` 的边界是：`application.runtime` 负责按 session/scene 组装一次聊天运行；`platform.agent_runtime` 负责顶层 ReAct / Plan 编排合同和工具执行；`platform.rag` 负责 Modular RAG 与 Agentic Retrieval 内部检索轮次。也就是说，Agentic RAG 是顶层 Agent 可调用的工具能力，不再代表整个 `/chat` 执行入口。

## 模式选择

`ChatService` 在准备每轮请求时选择 `agent_mode`。可以把它理解成一个分流开关：

- `react`：默认模式，适合简单知识问答、单目标请求，例如“这个文档说了什么？”。
- `plan`：适合复杂请求，例如“先查资料，再对比，再汇总成计划”，或者需要审批、确认、多目标汇总的任务。

模式选择结果会写入 SSE `start.agent_mode`、`tool.agent_mode` 和 LangGraph checkpoint。普通 `/chat` 不再把检索写死在入口前面；文档证据来自 ReAct 或 Plan 调用 RAG 工具后的结果。

## 主链路分工

1. 先按 session 找到当前场景，例如通用助手或电商助手。
2. 再读取这次会话允许使用哪些知识源，例如只查 documents，或同时查 ecommerce。
3. 然后判断问题难度：简单问题用 ReAct，复杂问题用 Plan。
4. Agent 通过 `ToolExecutor` 调工具。这个执行器只允许调用当前场景白名单里的工具。
5. 如果调用的是 RAG 工具，RAG 内部再负责改写问题、查资料、重排、判断证据是否足够。
6. 工具结果整理成引用、检索 trace 和 Agent 执行记录。
7. 最后交给 LangGraph 保存状态、生成回答，并写入聊天历史。

## Checkpoint 字段

`RuntimeGraphState` 保留原有回答字段，并新增以下可选 orchestration 字段：

| 字段 | 用途 |
| --- | --- |
| `agent_mode` | 当前顶层模式，`react` 或 `plan`。旧 checkpoint 可缺省。 |
| `react_run` | ReAct 顶层运行审计结构，包含 turn、tool observation、final answer。 |
| `plan_run` | Plan 顶层运行审计结构，包含 step、依赖、tool observation、final answer。 |
| `current_turn_id` | ReAct 等待或重试时的恢复点。 |
| `current_step_id` | Plan 等待或重试时的恢复点。 |
| `current_tool_call` | 当前待审批或执行中的工具调用摘要。 |

旧 checkpoint 缺少这些字段时仍按 `None` 读取。初学者可以简单理解为：老状态没有这些新字段也不会崩。

## HITL 语义

HITL 指 Human-in-the-Loop，也就是“中途需要用户确认或补充信息”。它仍由 run-level workflow state 表达：

- ReAct clarification wait 写入 `hitl.metadata.mode=react`、`react_run_id`、`current_turn_id`。
- Plan clarification 或 approval wait 写入 `hitl.metadata.mode=plan`、`plan_run_id`、`current_step_id`。
- `respond` / `approve` 只恢复当前匹配的等待点。
- `reject` 转为 `cancelled`，等待中的 turn 或 step 也标记为 `cancelled`，不会执行 proposed tool side effect。

Workflow state 只表示这一整次运行的状态。ReAct 的 turn、Plan 的 step、工具结果和 RAG 内部检索轮次，都保存在各自的 payload 里。

## SSE 兼容

SSE 事件名保持现有协议：

```text
start -> history -> tool -> chunk -> done
start -> history -> tool -> waiting_user
resume -> done
error
```

顶层 Agent 进度复用 `tool` 事件 payload：

- ReAct：`stage=react_turn`，包含 `react_run_id`、`turn_id`、`turn_status`、`action`、`tool_name`。
- Plan：`stage=plan_step`，包含 `plan_run_id`、`step_id`、`step_status`、`tool_name`。
- RAG 细节仍放在 `retrieval_trace` 以及 observation 的 nested trace 中。

## 最小验证样例

1. 简单 ReAct 成功：用户问一个单目标文档问题，SSE `tool.stage=react_turn`，checkpoint 写入 `agent_mode=react` 与 `react_run.turns[0]`。
2. ReAct RAG trace：RAG 工具 observation 的 `trace.retrieval_trace.rounds` 保存召回轮次，顶层只有一个 ReAct turn。
3. Plan 多步语义：用户显式要求“分步骤制定计划并汇总”，SSE `tool.stage=plan_step`，checkpoint 写入 `agent_mode=plan` 与 `plan_run.steps`。
4. Plan retry：平台层 `PlanExecutor` 对 retryable tool error 执行 `retrying -> running`，重试耗尽后进入 `failed`。
5. Plan HITL：Plan approval wait 保存 `current_step_id`；用户 `reject` 后 run 与 step 均为 `cancelled`，不执行 proposed tool。
