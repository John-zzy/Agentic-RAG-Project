# PRD-20260603-01 Planner / Executor 深化架构与方案设计

## 0. 阅读说明

这份文档回答两个问题：

1. 现在项目里的 Planner / Executor 做到了什么，还缺什么。
2. 后续应该怎么把它做成“能拆计划、按依赖执行、沉淀结果、支持失败恢复和人工介入”的多步任务执行能力。

先用一句话概括：

> 当前项目已经有 Planner / Executor 的基础骨架，但 `/chat` 里的 plan 模式还主要是“把 RAG 检索包装成一个 plan step”。它还没有真正体现复杂任务的多步骤拆解、依赖执行、跨步骤汇总和完整恢复能力。

为了方便阅读，下面先解释几个关键词。

| 术语 | 小白理解 | 在项目里的含义 |
| --- | --- | --- |
| Planner | 先列计划的人 | 根据用户目标生成 `PlanRun` 和多个 `PlanStep`。 |
| Executor | 按计划做事的人 | 找到当前能执行的 step，调用工具，记录结果。 |
| PlanRun | 一次完整计划任务 | 保存用户目标、步骤列表、运行状态、最终结果。 |
| PlanStep | 计划中的一个步骤 | 保存步骤目标、要调用的工具、输入、依赖和执行结果。 |
| ToolObservation | 工具执行后的观察结果 | 保存工具是否成功、输出、引用、错误、是否需要用户补充信息。 |
| HITL | Human-in-the-Loop | 执行中需要用户确认、审批或补充信息。 |
| Workflow Status | 整个任务的状态 | 例如 `running`、`waiting_user`、`failed`、`succeeded`。 |

## 1. 当前结论

按 PRD 的 6 个需求目标统计：

| 分类 | 数量 | 说明 |
| --- | ---: | --- |
| 已满足 | 1 / 6 | 当前代码入口和职责边界已经能明确指出。 |
| 部分满足 | 5 / 6 | 合同、执行器、重试、HITL 接入点已有，但还没有完整贯通到 `/chat` 的多步计划执行。 |
| 完全未满足 | 0 / 6 | 没有完全空白的能力项。 |
| 尚未完全满足 | 5 / 6 | 这些是后续实现重点。 |

如果拆成更细的 18 个功能点：

| 分类 | 数量 | 代表功能点 |
| --- | ---: | --- |
| 已满足 | 8 / 18 | 基础合同、DAG 校验、选择可执行步骤、step observation、retry/fail、基础 HITL、final synthesis 输入、SSE `plan_step`。 |
| 部分满足 | 6 / 18 | 计划生成语义、步骤拆解、依赖状态传播、run 级结果沉淀、最终回答贯通、resume 后继续执行。 |
| 未满足 | 4 / 18 | `PlanRun.context_summary`、run-level `observations`、`blocked` step 语义、`/chat` 多步骤 plan 生成与聚合。 |
| 尚未完全满足 | 10 / 18 | 部分满足 + 未满足。 |

最重要的判断是：

- 平台层已经有可用的基础能力。
- 应用层 `/chat` 的 plan 模式还不够“真”。
- 下一步重点不是再堆新概念，而是把已有 Planner / Executor 能力接到真实聊天链路里。

## 1.1 与 ReAct 模式合并后的判断

这份 PRD 的主目标仍然是 Plan 模式深化，但不能把 ReAct 当成已经完全不用管。

更准确的判断是：

| 模式 | 当前满足度 | 说明 |
| --- | --- | --- |
| ReAct | 已满足最小接入，未达到完整深化 | 能表达 turn、调用工具、记录 observation、进入 HITL 和生成最终汇总；但 `/chat` 中仍是单工具保守路径。 |
| Plan | 已有平台层基础，应用层仍需深化 | 能生成和执行基础 plan，但 `/chat` 还没有真正多步骤拆解、依赖执行和跨步骤汇总。 |

可以这样理解：

```text
ReAct 当前像一个“能先查一个工具再回答”的轻量 Agent。
Plan 当前像一个“有计划表和执行器，但在 /chat 里还只填了一行计划”的复杂任务骨架。
```

所以本 PRD 合并考虑后的范围是：

- Plan 是本次深化重点。
- ReAct 不能被误判为生产级完成。
- Plan 改造时要复用 ReAct 已有的通用能力，例如 `ToolExecutor`、`ToolObservation`、HITL metadata、final synthesizer。
- `/chat` 的最终聚合逻辑要兼容未来 ReAct 多工具和 Plan 多步骤，不要继续只依赖 latest observation。

ReAct 当前已经满足的部分：

- 有 `ReActRun / ReActTurn / ReActAction / ToolObservation` 合同。
- `ReActRuntime` 能处理 `tool_call / ask_user / final_answer / stop`。
- 能记录 tool observation、citations、HITL metadata 和 final synthesis。
- 平台层测试覆盖了单工具、多工具、RAG nested trace、HITL、retrying 和 max turns。
- `/chat` SSE 能输出 `stage=react_turn`。

ReAct 当前还没有完全满足的部分：

- `/chat` 使用的是 `_SingleToolReActActionSelector`，第一轮调用一个选定 RAG 工具，第二轮进入 final answer；还不是模型驱动的多轮多工具选择。
- ReAct retry 目前返回 `retrying`，但不会像 PlanExecutor 一样在同一次执行中自动 retry 到成功或失败。
- ReActRuntime 的 workflow 状态多为直接赋值，没有完全统一通过 `validate_transition()`。
- `/chat` 当前结果聚合偏向 latest observation；如果未来 ReAct 支持多工具，需要聚合所有 successful observations。
- HITL wait 已有，但 resume 后继续同一个 ReAct run 的策略仍是最小闭环。

因此，当前需求合并 ReAct 后的验收口径是：

> 本次优先把 Plan 模式补成真正的复杂任务执行链路；同时明确 ReAct 只完成了最小接入，后续如果要做 ReAct 深化，需要单独补模型驱动 action selection、多工具聚合、retry continuation 和状态机统一转移。

## 2. 当前代码边界

| 文件 | 现在负责什么 | 当前判断 |
| --- | --- | --- |
| `backend/platform/agent_runtime/contracts.py` | 定义 `PlanRun`、`PlanStep`、`ToolObservation` 等数据结构。 | 基础结构已具备，但 `PlanRun` 还缺少上下文摘要和 run 级 observations。 |
| `backend/platform/agent_runtime/plan/planner.py` | `MinimalPlanner` 生成计划，并校验工具、依赖和输入。 | 有最小计划生成能力，但语义偏规则化。 |
| `backend/platform/agent_runtime/validation.py` | 校验工具白名单、输入 schema 和 step 依赖 DAG。 | 依赖合法性校验已经满足。 |
| `backend/platform/agent_runtime/tool_executor.py` | 统一调用 RAG tool、scene tool、internal tool，并归一化输出。 | 工具执行边界清晰。 |
| `backend/platform/agent_runtime/plan/executor.py` | 按依赖执行 step，处理重试、失败、等待用户和最终汇总。 | 平台层执行能力已部分满足。 |
| `backend/application/runtime/chat_service_parts/agent_runtime.py` | 在 `/chat` 中执行 ReAct 或 Plan。 | 关键缺口：plan 模式目前固定生成一个 RAG step。 |
| `backend/application/runtime/chat_service_parts/events.py` | SSE 中输出 `stage=plan_step` 等进度字段。 | 能输出基础进度，但还不是逐 step 的完整事件流。 |
| `backend/application/runtime/graph_runtime_parts/agent_state.py` | 将 agent run 写入 checkpoint，并投影 HITL 状态。 | 状态记录已有，resume 后继续执行 plan 的闭环仍需增强。 |

## 3. 当前项目已经满足什么

### 3.1 已有 PlanRun / PlanStep / ToolObservation

项目已经能用三层结构表达一次计划任务：

```text
PlanRun
  表示一次完整计划任务
  -> 包含多个 PlanStep

PlanStep
  表示任务中的一个步骤
  -> 每个 step 调用一个工具
  -> 工具结果写入 observation

ToolObservation
  表示工具调用结果
  -> 成功、失败、引用、trace、错误、是否需要用户
```

这说明项目已经不是纯粹的一次性 RAG 调用，而是有了多步任务的基础表达方式。

### 3.2 已有依赖校验

`validate_plan_dependencies()` 已经能检查：

- step id 是否重复。
- step 依赖是否存在。
- step 是否依赖自己。
- 依赖关系是否有环。

这能避免计划里出现“第二步依赖不存在的第一步”或“第一步和第二步互相等待”的问题。

### 3.3 已有最小执行器

`PlanExecutor` 已经能做几件重要的事：

- 只执行依赖已经成功的 pending step。
- 工具成功后把 step 标记为 `succeeded`。
- 工具失败但可重试时进入 `retrying`。
- 重试耗尽或不可恢复错误时进入 `failed`。
- 工具要求用户补充信息时进入 `waiting_user`。
- 所有 step 成功后调用 final synthesizer 生成最终汇总。

这部分是后续深化的基础。

## 4. 当前主要缺口

### 4.1 `/chat` 的 plan 模式仍然是单步

当前 `/chat` 进入 plan 模式后，会人为构造一个固定的 `step-1`，并让它调用一个 RAG 工具。

这能证明 plan 模式接进来了，但还不能证明它真的会拆复杂任务。

理想情况应该是：

```text
用户：帮我对比 A 和 B，并给出落地建议

Planner 生成：
  step-1: 检索 A 的资料
  step-2: 检索 B 的资料
  step-3: 汇总比较结果

Executor 执行：
  先跑 step-1 和 step-2
  再跑依赖前两步的 step-3
```

### 4.2 PlanRun 缺少 run 级结果沉淀

当前每个 `PlanStep` 可以保存自己的 `observation`，但 `PlanRun` 没有显式的 `observations` 列表。

这会带来一个问题：

- 想看单个 step 的结果，可以看 step。
- 想看整个 run 的所有工具结果，需要从所有 step 里反查。

后续建议让 `PlanRun` 直接保存 run 级 observations，方便最终汇总、审计和 checkpoint 恢复。

### 4.3 缺少 blocked step 语义

现在如果某个 step 的依赖失败，Executor 会让整个 run 失败。

这对最小版本没问题，但复杂任务里最好能明确区分：

- `failed`：当前 step 自己执行失败。
- `blocked`：当前 step 没执行，是因为依赖它的前置步骤失败了。

举例：

```text
step-1: 查询订单，失败
step-2: 根据订单查询物流，依赖 step-1

step-1 应该是 failed
step-2 应该是 blocked
```

这样面试或排障时能清楚说明：不是 step-2 的工具坏了，而是它没有机会执行。

### 4.4 最终回答还没有充分消费整个 PlanRun

平台层 `PlanExecutor` 已经有 final synthesizer，但 `/chat` 最终回答上下文仍更偏向“最新一次 observation”。

后续应该改成：

- plan 模式最终回答优先使用整个 `PlanRun`。
- citations 从所有成功 step 的 observations 中聚合。
- documents 从所有成功 step 的 outputs 中聚合。
- 最终汇总基于 step summaries 和 observations，而不是只看最后一次工具结果。

## 5. 目标架构

目标架构可以理解成一个“先计划、再执行、再总结”的流程。

```text
POST /chat
  -> 找到当前 session 和 scene
  -> ModeSelector 判断走 react 还是 plan
  -> ToolExecutor 准备当前会话允许调用的工具
  -> Planner 生成 PlanRun 和 PlanStep
  -> PlanExecutor 按依赖执行步骤
  -> 每个工具结果写入 ToolObservation
  -> FinalSynthesizer 汇总所有成功步骤
  -> 写入 checkpoint / SSE / API response
```

分层边界保持不变：

| 层 | 负责什么 |
| --- | --- |
| `platform.agent_runtime` | 中立的 Planner、Executor、ToolExecutor、合同和校验。 |
| `application.runtime` | `/chat` 组装、SSE、checkpoint、resume、回答模型接入。 |
| `scenes` | 场景工具、场景策略、候选知识源和高风险规则。 |
| `platform.rag` | RAG 检索能力本身，作为工具被 Agent 调用。 |

重点是：RAG 是工具，不是整个 Agent Runtime。

## 6. PlanRun 设计

建议把 `PlanRun` 理解成“一张任务执行记录表”。

它应该能回答：

- 这个任务的用户目标是什么？
- 当前是 plan 模式还是其他模式？
- 有哪些步骤？
- 当前执行到哪一步？
- 每个工具返回了什么？
- 最终总结是什么？
- 当前 workflow 状态是什么？

建议补齐字段：

```python
class PlanRun(AgentRuntimeModel):
    plan_run_id: str
    session_id: str
    request_id: str
    mode: Literal["plan"] = "plan"
    user_goal: str
    context_summary: str = ""
    workflow_status: WorkflowRunState = "planning"
    steps: list[PlanStep] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    current_step_id: str | None = None
    current_tool_call: ToolExecutionMetadata | None = None
    final_answer: str | None = None
    result_summary: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

其中最重要的新增字段是：

| 字段 | 为什么需要 |
| --- | --- |
| `context_summary` | 保存这次计划的上下文摘要，例如场景、知识源、限制条件。 |
| `observations` | 保存整个 run 的工具结果，方便汇总、审计和恢复。 |

## 7. PlanStep 设计

建议把 `PlanStep` 理解成“计划中的一条待办事项”。

它应该能回答：

- 这一步要完成什么目标？
- 它要调用哪个工具？
- 工具输入是什么？
- 它依赖哪些前置步骤？
- 现在执行到什么状态？
- 工具返回了什么？
- 失败原因是什么？

当前字段大体够用，建议补充 `blocked` 状态：

```python
PlanStepStatus = Literal[
    "pending",
    "running",
    "waiting_user",
    "retrying",
    "succeeded",
    "failed",
    "blocked",
    "cancelled",
    "skipped",
]
```

状态解释：

| 状态 | 小白理解 | 后续步骤影响 |
| --- | --- | --- |
| `pending` | 还没轮到它执行 | 等依赖完成。 |
| `running` | 正在执行 | 暂时不执行依赖它的步骤。 |
| `succeeded` | 这一步成功了 | 依赖它的 step 可以继续。 |
| `retrying` | 这一步失败了，但还能再试 | 后续 step 继续等待。 |
| `waiting_user` | 需要用户补充或确认 | 整个 run 暂停。 |
| `failed` | 这一步失败且不能恢复 | 依赖它的 step 进入 blocked。 |
| `blocked` | 它自己没执行，是前置依赖失败了 | 不调用工具，只记录原因。 |
| `cancelled` | 用户拒绝或取消 | 后续不再执行。 |
| `skipped` | 非必要步骤被跳过 | 仅用于可选分支。 |

## 8. Planner 方案

Planner 的职责是“生成计划，不执行计划”。

它的输入包括：

- 用户目标。
- 当前 session。
- 当前 scene。
- 挂载的知识源。
- 当前允许调用的工具。
- scene policy。
- 可选的 proposed steps。

它的输出是：

- 一个 `PlanRun`。
- 1 到 3 个 `PlanStep`。
- 已校验的工具名、输入和依赖关系。

短期不需要复杂 LLM Planner，可以继续使用 `MinimalPlanner`，但要让它真的根据上下文生成 plan，而不是由 `/chat` 固定塞一个单步计划。

建议规则：

| 场景 | Planner 行为 |
| --- | --- |
| scene 显式给了 `plan_steps` | 使用 scene 提供的 steps，并严格校验。 |
| scene 给了 preferred tools | 按工具顺序生成 step，后一项依赖前一项。 |
| 用户目标明显包含多个子目标 | 尝试拆成多个 step。 |
| 挂载多个知识源 | 为不同知识源生成不同检索 step。 |
| 没有拆分信号 | 退化成单步 RAG plan。 |

## 9. Executor 方案

Executor 的职责是“按计划执行，不重新发明计划”。

### 9.1 如何选择下一个可执行 step

判断逻辑可以用这段话描述：

> 找到第一个状态是 pending，且所有依赖步骤都已经 succeeded 的 step。

伪流程：

```text
遍历 steps:
  如果 step 不是 pending，跳过
  如果依赖里有 failed / cancelled / blocked:
    当前 step 标记为 blocked
    跳过
  如果依赖里还有 pending / running / retrying / waiting_user:
    当前 step 继续等待
    跳过
  如果所有依赖都是 succeeded:
    返回这个 step，开始执行
```

本阶段仍建议串行执行。也就是说，即使有多个 step 都能执行，也先一次执行一个，避免引入并发调度复杂度。

### 9.2 失败、等待和重试如何影响后续步骤

| 当前 step 结果 | run 怎么变 | 后续 step 怎么办 |
| --- | --- | --- |
| 成功 | 继续 running，或全部完成后 succeeded | 依赖它的步骤可以执行。 |
| 可重试失败 | run 进入 retrying，再回 running | 后续步骤继续等待。 |
| 重试耗尽 | run 进入 failed | 依赖它的 pending step 标记 blocked。 |
| 需要用户 | run 进入 waiting_user | 整个 run 暂停，等待 resume。 |
| 用户拒绝 | run 进入 cancelled | 未完成步骤取消或阻塞。 |

## 10. 结果沉淀方案

工具返回后，不管成功还是失败，都应该先沉淀结果。

统一写入规则：

```text
step.observation = observation
step.output = observation.output
step.result_summary = observation.result_summary
step.error = observation.error
plan_run.observations.append(observation)
plan_run.current_tool_call = observation.execution
```

这样做的好处：

- 成功步骤能用于最终汇总。
- 失败步骤也能解释失败原因。
- 等待用户的步骤能保留问题和恢复点。
- checkpoint 里能看到整个任务发生过什么。

## 11. 最终汇总方案

最终回答应该来自“所有成功步骤的结果”，而不是只看用户原始问题或最后一个工具结果。

建议 final synthesizer 的输入包括：

```text
user_goal
context_summary
成功的 PlanStep 列表
PlanRun.observations
citations
execution_order
```

短期可以保持简单汇总：

- 汇总每个成功 step 的 `result_summary`。
- 合并所有 citations。
- 把全部成功 step 的 documents 作为回答证据。

后续如果接 LLM summarizer，也应该是“基于步骤结果汇总”，不是“重新直接回答用户问题”。

## 12. HITL 方案

HITL 可以理解成“执行中途需要人来决定一下”。

有两种常见情况：

| 类型 | 例子 | 应该怎么处理 |
| --- | --- | --- |
| 信息不足 | 用户没给订单号，工具无法查询 | 进入 `waiting_user`，让用户补充。 |
| 高风险操作 | 要执行退款、取消订单、写数据 | 执行工具前先让用户确认。 |

Plan 模式下的 HITL metadata 应该固定包含：

```json
{
  "mode": "plan",
  "plan_run_id": "plan-xxx",
  "current_step_id": "step-1",
  "user_prompt": "需要用户确认的问题",
  "source": "tool_observation|risk_policy"
}
```

resume 规则：

| 用户动作 | 行为 |
| --- | --- |
| `respond` | 把用户补充信息写回当前 step input，然后继续执行。 |
| `approve` | 执行原本等待审批的工具调用。 |
| `reject` | run 进入 `cancelled`，不能执行副作用工具。 |

## 13. `/chat` 接入方案

当前关键改造点：

```text
当前：
  plan 模式固定生成一个 RAG step

目标：
  plan 模式交给 Planner 生成 steps
  Executor 执行 steps
  ChatService 聚合整个 PlanRun 的结果
```

目标路径：

```text
if mode == "plan":
  plan_run = planner.create_plan(...)
  plan_run = plan_executor.execute(plan_run)
  agent_result = aggregate_plan_result(plan_run)
else:
  react_run = react_runtime.run(...)
```

`aggregate_plan_result(plan_run)` 应该输出：

- 所有成功 observations 中的 documents。
- 所有成功 observations 中的 citations。
- 完整 `plan_run`。
- 最终 answer context。
- 当前 step / current tool call。
- SSE 所需的 plan progress 信息。

## 14. SSE 方案

事件名保持兼容，不新增复杂协议：

```text
start -> history -> tool -> chunk -> done
start -> history -> tool -> waiting_user
resume -> done
error
```

plan 模式继续复用 `tool` 事件，但 payload 要更完整：

| 字段 | 含义 |
| --- | --- |
| `stage=plan_step` | 当前是 plan step 事件。 |
| `plan_run_id` | 当前计划 run。 |
| `step_id` | 当前步骤。 |
| `step_status` | 当前步骤状态。 |
| `tool_name` | 当前步骤调用的工具。 |
| `workflow_status` | 整个 run 的状态。 |
| `execution_order` | 已完成步骤顺序。 |
| `step_count` | 总步骤数。 |

本 PRD 不要求实现生产级逐 step 流式调度。后续可以先让一次 `tool` 事件携带完整 plan 快照，再逐步演进成 step event iterator。

## 15. 后续统一验证清单

本 PRD 明确“今日只记录后续统一验证范围，不拆散执行测试”。

候选命令：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agent_runtime_plan.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py -q -c backend\tests\pytest.ini
```

验证项：

| 验证项 | 要证明什么 |
| --- | --- |
| planner validation | 工具白名单、输入 schema、依赖 DAG 都能拦住非法计划。 |
| executor dependency | Executor 只执行依赖已完成的 step。 |
| blocked step | 依赖失败时，下游 step 被标记 blocked。 |
| tool observation | 工具结果同时写入 step 和 run。 |
| retry/fail | 可重试失败能 retry，重试耗尽能 failed。 |
| waiting_user | 信息不足或高风险步骤能进入 waiting_user。 |
| final summary | 最终回答消费全部成功 step 的结果。 |
| SSE plan_step | SSE 能表达 plan step 进度和等待用户状态。 |

## 16. 最终验收口径

这份设计文档给出的口径是：

- 当前 Planner / Executor 的代码入口和职责边界已经清楚。
- ReAct 已满足最小接入，但不等于完整深化完成。
- 本 PRD 当前主线是 Plan 深化，同时保留 ReAct 后续深化边界。
- `PlanRun` 表达一次完整计划任务。
- `PlanStep` 表达任务中的一个可执行步骤。
- `ToolObservation` 表达工具调用结果。
- Executor 只执行依赖已完成的 pending step。
- `failed`、`blocked`、`retrying`、`waiting_user` 对后续步骤的影响已经明确。
- 最终回答应该基于步骤结果汇总，而不是模型一次性直接回答。
- 今日只推进功能深化设计，验证结论和配套材料后续统一沉淀。
