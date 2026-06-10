# ReAct 代码阅读指南

这份文档只讲 `backend/platform/agent_runtime/react/` 这个包。目标是帮人工审核代码时快速知道：从哪里看、每个文件做什么、哪些地方可能需要重构。

一句话概括：这个包负责把一次 ReAct 对话交给 LangChain `create_agent` 跑完，然后把 LangChain 的结果整理成项目自己的 `ReActRun`。

## 先从哪里看

建议按这个顺序看：

1. `backend/platform/agent_runtime/chat_graph/graph/nodes/react_branch.py`
   - 这是 ChatGraph 进入 ReAct 的入口。
   - 如果当前模式是 `react`，它会创建 ReAct 依赖，然后调用 `ReActRuntime.run()`。

2. `backend/application/runtime/assembly/service_parts/agent_runtime.py`
   - 重点看 `_build_react_deps()`。
   - 这里把 scene、session、工具、模型、middleware、checkpoint 都准备好。
   - ReAct 包本身不应该知道具体 scene 怎么取工具，这些都在这里装配。

3. `runtime.py`
   - 重点看 `ReActRuntime.run()`。
   - 这是 ReAct 包自己的主入口。
   - 它会创建 LangChain tools，创建 LangChain agent，调用 agent，然后把结果转成 `ReActRun`。

4. `factory.py`
   - 这里真正调用 LangChain `create_agent`。
   - 也在这里挂 middleware，比如模型保护、人工审批、工具边界。

5. `projection.py`
   - 这里把 LangChain 输出整理成项目自己的数据结构。
   - 它会读 LangChain messages，生成 turns、observations、final answer。
   - 如果 LangChain 触发人工审批 interrupt，也是在这里转成 `waiting_user`。

6. 其他文件再按需要看：
   - `middleware.py`：模型调用和工具调用的保护层。
   - `tools.py`：把项目工具转成 LangChain tools。
   - `state.py`：LangChain agent 用的 state/context 类型。
   - `policy.py`：从 scene metadata 读取 ReAct 策略。
   - `config.py`：ReAct 运行需要的依赖包。

## 一次普通 ReAct 是怎么跑的

大概流程是：

```text
ChatGraph 进入 react_branch
  -> application 层准备 ReActDependencies
  -> ReActDependencies.build_runtime()
  -> ReActRuntime.run()
  -> 把 ToolExecutor 转成 LangChain tools
  -> ReActProviderFactory 创建 LangChain create_agent
  -> LangChain agent.invoke()
  -> projection.py 把结果转成 ReActRun
  -> application 层把 ReActRun 转回 ChatGraph state
```

可以简单理解为：

- ChatGraph 负责决定“要不要跑 ReAct”。
- application 层负责准备“用什么模型、什么工具、什么策略”。
- `react/` 包负责“真正跑 LangChain ReAct，并整理结果”。

## HITL 人工审批怎么理解

ReAct 包里没有自己写一个“人工审批节点”。

ReAct 的人工审批来自 LangChain 的 `HumanInTheLoopMiddleware`。当 LangChain 发现某个工具需要人工审批时，会产生 interrupt。然后：

1. `projection.py` 把这个 interrupt 转成 `ReActRun.workflow_status = "waiting_user"`。
2. ChatGraph 顶层把等待状态保存到 checkpoint。
3. `/chat/resume` 进来后，ChatGraph 顶层校验 `interrupt_id`。
4. 如果是 ReAct 的等待点，再把 resume command 交回 `ReActRuntime.run(..., resume_command=...)`。

所以边界是：

- `react/` 包：负责 LangChain interrupt 和 `ReActRun` 的转换。
- `chat_graph/hitl.py`：负责 checkpoint、resume 校验、幂等、副作用收口。

不要把这两层混在一起。

## 每个文件做什么

### `runtime.py`

核心类：`ReActRuntime`

它是 ReAct 包的主入口。主要做这些事：

- 准备 prompt。
- 把项目工具转成 LangChain tools。
- 创建 LangChain agent。
- 调用 `agent.invoke()`。
- 调用 `project_react_agent_output()`，把结果转成 `ReActRun`。

需要重点审核的地方：

```python
if initial_run is not None and resume_command is None:
    return initial_run
```

这段代码会直接返回已有 run，不再真正执行 LangChain。它可能是为了避免重复执行，但也可能让“子图没有真的跑”被隐藏。重构时建议确认它还需不需要。

### `config.py`

核心类：`ReActDependencies`

它只是一个依赖包，里面放 ReAct 运行需要的东西，比如：

- 工具执行器
- provider factory
- middleware bundle
- runtime context
- session/request/user goal
- system prompt
- max turns
- 已有的 `initial_run`
- 结果投影回调

这个文件最好保持简单，不要放业务判断。

### `factory.py`

核心类：`ReActProviderFactory`

它负责创建 LangChain `create_agent`。

这里最重要的是 middleware 顺序：

```text
LangChainModelGuardAdapter
HumanInTheLoopMiddleware
LangChainToolBoundaryAdapter
```

审核时要确认：高风险工具到底应该先经过人工审批，还是先经过项目工具策略校验。这个顺序会影响行为。

### `middleware.py`

核心类：

- `LangChainModelGuardAdapter`
- `LangChainToolBoundaryAdapter`

`LangChainModelGuardAdapter` 做模型调用保护：

- 调用项目的 model guard。
- 记录模型调用 trace。
- 如果模型失败，抛出统一错误。

`LangChainToolBoundaryAdapter` 做工具调用保护：

- 检查工具是否允许调用。
- 校验和修正工具参数。
- 捕获工具异常。
- 把工具结果整理成 `ToolObservation`。
- 记录工具调用 trace。

简单说：LangChain 想调工具时，必须先经过这里。

### `tools.py`

核心函数：`build_react_tools()`

它只是一个很薄的转换层：

```text
ToolExecutor -> LangChain BaseTool
```

真正转换逻辑在 `backend/platform/agent_runtime/tooling/langchain.py`。

### `projection.py`

核心函数：`project_react_agent_output()`

这是最重要的整理层。LangChain 跑完后，输出的是 messages 或 interrupt。项目不能直接依赖这些原始输出，所以要在这里转成稳定的 `ReActRun`。

它主要处理两种情况：

1. 普通完成
   - 读取 `AIMessage` 和 `ToolMessage`。
   - 工具调用变成 `ReActTurn`。
   - 工具结果变成 `ToolObservation`。
   - 最终回答变成 `final_answer`。

2. 人工审批等待
   - 读取 LangChain interrupt。
   - 转成 `workflow_status = "waiting_user"`。
   - 保存待审批工具、允许动作、interrupt 信息。

需要重点审核的地方：

- 现在只处理第一个 `action_request`。
- 人工审批等待点里固定用了 `current_turn_id = "turn-1"`。
- 一些字符串是硬编码的，比如 `"langchain_create_agent"`。
- 如果 LangChain 输出格式变化，这个文件最容易坏。

### `state.py`

核心类：

- `ReActInputState`
- `ReActState`
- `ReActContext`

这些是给 LangChain `create_agent` 用的类型。

目前 `ReActState` 只是继承 `ReActInputState`，没有新增字段。保留它可能是为了让 LangChain 的 state 名字更清楚；如果要精简，可以审核是否真的需要单独存在。

### `policy.py`

核心类：`ReActScenePolicy`

它从 scene metadata 里读取 ReAct 策略，比如：

- 优先使用哪些工具
- 是否允许多工具
- 最多跑几轮
- 没证据时怎么办
- 哪些工具是高风险工具
- 工具默认输入提示

这里不负责执行工具，只负责读策略。

### `__init__.py`

只导出公开对象。

这里应该保持轻量，不要放运行逻辑，避免循环导入。

## 哪些代码味道值得重点看

1. `ReActRuntime.run()` 直接返回 `initial_run`
   - 可能合理，也可能是旧逻辑残留。
   - 如果目标是“确保子图真的跑完”，这里要重点确认。

2. `projection.py` 只处理第一个 HITL action request
   - 如果未来允许一次模型输出多个待审批工具，这里不够用。

3. `projection.py` 里有不少硬编码字符串
   - 比如 `turn-1`、`langchain_create_agent`。
   - 如果这些值有跨文件约定，建议集中成常量或写清楚注释。

4. middleware 顺序需要确认
   - 当前是模型 guard -> LangChain HITL -> 工具边界。
   - 这会影响高风险工具审批和工具策略拒绝的先后。

5. `tools.py` 很薄
   - 保留它的好处是 ReAct 包边界清楚。
   - 如果团队想减少文件，也可以合并，但不建议为了少一层把工具转换逻辑散到 runtime 里。

6. application 层承担了很多 ReAct 投影后的解释逻辑
   - `_project_react_graph_result()` 和后面的 retrieval trace/citation 组装比较重。
   - 这部分不在 ReAct 包里，但人工审核时要知道：ReAct 包只产出 `ReActRun`，最终怎么变成 chat response 是 application 层负责。

## 重构时建议守住的边界

- `react/` 包只负责 LangChain ReAct 子图。
- scene 工具解析不要放进 `react/`。
- API schema 不要放进 `react/`。
- checkpoint resume 校验不要放进 `react/`。
- ReAct 输出统一通过 `ReActRun` 往外走，不要让上层直接依赖 LangChain messages。

## 建议先跑的测试

只改 ReAct 包时，优先跑：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_react_provider.py -q -c backend\tests\pytest.ini
```

如果改到了 ChatGraph ReAct 分支或 HITL resume，再补跑：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```
