from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.platform.agent_runtime.chat_graph.contracts import PreparedChatTurn
from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.workflow.langgraph.state import RuntimeGraphState


@dataclass(frozen=True)
class ChatGraphDependencies:
    """ChatGraph 节点运行所需的业务依赖。"""

    prepared: PreparedChatTurn
    answer_builder: Callable[[PreparedChatTurn], tuple[str, list[Any]]]
    build_agent_runtime_success_update: Callable[
        [RuntimeGraphState, str, list[Any], bool],
        dict[str, Any],
    ]
    select_agent_mode: Callable[[PreparedChatTurn], dict[str, Any]] | None = None
    build_react_graph_deps: Callable[
        [PreparedChatTurn, RuntimeGraphState],
        ReActGraphDependencies,
    ] | None = None
    build_plan_graph_deps: Callable[
        [PreparedChatTurn, RuntimeGraphState],
        PlanGraphDependencies,
    ] | None = None
    build_prepared_from_state: Callable[
        [PreparedChatTurn, RuntimeGraphState],
        PreparedChatTurn,
    ] | None = None
    build_hitl_wait_update: Callable[
        [PreparedChatTurn, RuntimeGraphState],
        dict[str, Any],
    ] | None = None
