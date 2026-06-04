from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.platform.agent_runtime.chat_graph.contracts import PreparedChatTurn
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


