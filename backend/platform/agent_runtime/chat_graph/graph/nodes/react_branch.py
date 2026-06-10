from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.agent_runtime.core.contracts import ReActRun
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_react_branch_node(dependencies: ChatGraphDependencies):
    """创建 ChatGraph 里的 ReAct 分支节点。"""

    prepared = dependencies.prepared

    def react_branch(state: RuntimeGraphState) -> dict[str, Any]:
        """执行一次 ReAct 子图，并返回要合并进 ChatGraph state 的字段。"""

        # 当前轮如果被路由到 plan，就什么都不做。LangGraph 会把空 dict
        # 视为“不更新状态”，这样 react_branch 和 plan_branch 可以共用后续节点。
        if str(state.get("agent_mode") or prepared.agent_mode) != "react":
            return {}

        # ReAct 的模型、工具、中间件、投影函数都由 application 层装配。
        # 如果没有提供装配函数，说明当前运行环境没有启用 ReAct 分支。
        if dependencies.build_react_deps is None:
            return {}

        # 这里拿到的是 ReAct 子图运行所需的完整依赖：
        # tool executor、provider factory、runtime context、project_result 等。
        provider_deps = dependencies.build_react_deps(prepared, state)
        run = provider_deps.build_runtime().run(
            session_id=provider_deps.session_id,
            request_id=provider_deps.request_id,
            user_goal=provider_deps.user_goal,
            react_run_id=provider_deps.react_run_id or f"react-{provider_deps.request_id}",
            # initial_run 用来承接 checkpoint 或 prepared 里已经存在的 ReActRun。
            # 当前 ReActRuntime 在没有 resume_command 时会直接返回它，避免重复执行工具。
            initial_run=provider_deps.initial_run
            or _react_run(prepared=prepared, state=state),
        )

        # ReActRun 是子图自己的运行记录；ChatGraph 还需要 documents、citations、
        # answer_mode 等顶层字段，所以通过 project_result 做一次状态投影。
        result = dict(provider_deps.project_result(run)) if provider_deps.project_result else {}
        return {
            **_chat_graph_result_fields(result),
            # checkpoint 里保存普通 dict，避免 Pydantic 对象直接混进 LangGraph state。
            "react_run": run.model_dump() if hasattr(run, "model_dump") else run,
        }

    return react_branch


def _react_run(
    *,
    prepared: Any,
    state: RuntimeGraphState,
) -> ReActRun | None:
    """从当前 graph state 或 prepared turn 里取出已有的 ReActRun。

    ChatGraph 恢复执行时，state 里的值通常来自 checkpoint，可能已经是 dict；
    这里统一转回 ReActRun，后面的 runtime 和投影代码就不用关心来源。
    """

    react_run = state.get("react_run") or getattr(prepared, "react_run", None)
    if react_run is None:
        return None
    return react_run if isinstance(react_run, ReActRun) else ReActRun.model_validate(react_run)


def _chat_graph_result_fields(result: dict[str, Any]) -> dict[str, Any]:
    """只允许 ReAct 投影结果更新 ChatGraph 明确认识的字段。

    这个白名单是边界保护：ReAct 子图可以产出自己的内部信息，但只有这里列出的
    字段会进入顶层 ChatGraph state，避免无关字段污染 checkpoint。
    """

    allowed = {
        "documents",
        "citations",
        "retrieval_trace",
        "knowledge_used",
        "final_decision",
        "answer_mode",
        "follow_up_question",
        "tool_event",
        "current_turn_id",
        "current_tool_call",
        "tool_observation",
        "agent_mode",
        "agent_mode_reason",
        "agent_mode_signals",
    }
    return {key: result[key] for key in allowed if key in result}

