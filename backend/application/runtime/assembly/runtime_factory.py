from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from backend.application.runtime.assembly.runtime_parts.agent_state import AgentRuntimeStateProjectionMixin
from backend.application.runtime.assembly.runtime_parts.answer_graph import AnswerGraphMixin
from backend.platform.agent_runtime.chat_graph.contracts import (
    AnswerBuilder,
    HistoryLoader,
    HitlResumeError,
    HitlResumeInput,
    HitlWaitInput,
    PreparedGraphTurn,
    RuntimeGraphResult,
)
from backend.application.runtime.assembly.runtime_parts.hitl import HitlRuntimeMixin
from backend.application.runtime.assembly.runtime_parts.state_store import RuntimeStateStoreMixin
from backend.platform.config.settings import AppSettings
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.lifecycle import GraphRunLifecycleRecorder


class ChatGraphRuntime(
    RuntimeStateStoreMixin,
    AnswerGraphMixin,
    HitlRuntimeMixin,
    AgentRuntimeStateProjectionMixin,
):
    """ChatService 使用的 LangGraph 入口，负责保存和读取会话运行状态。"""

    def __init__(
        self,
        *,
        checkpointer: SQLiteLangGraphCheckpointer,
        lifecycle: GraphRunLifecycleRecorder | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.lifecycle = lifecycle or GraphRunLifecycleRecorder()
        # 批准后可能执行写操作或外部 API，同一进程里先一次只处理一个 resume。
        self._resume_lock = Lock()

    @classmethod
    def from_settings(cls, app_settings: AppSettings) -> "ChatGraphRuntime":
        sqlite_path = Path(app_settings.data_dir) / "langgraph.db"
        return cls(checkpointer=SQLiteLangGraphCheckpointer(sqlite_path))

    def invoke(
        self,
        *,
        prepared: PreparedGraphTurn,
        answer_builder: AnswerBuilder,
        history_loader: HistoryLoader,
        select_agent_mode: Any | None = None,
        run_agent_runtime: Any | None = None,
        build_prepared_from_state: Any | None = None,
        build_hitl_wait_update: Any | None = None,
    ) -> RuntimeGraphResult:
        """同步执行 ChatGraph，返回完整的运行结果。

        流程：构建配置 → 创建 run 记录 → 编译图 → 执行图 → 更新 run 状态 → 返回结果。
        外部通过可选回调注入 Agent Runtime 相关逻辑（模式选择、Agent 执行、状态回填、HITL 等待）。
        """
        # 构建图运行配置（thread_id、metadata 等 LangGraph checkpoint 所需信息）
        config = self.build_config(prepared)

        # 创建 run 生命周期记录，用于跟踪本次图运行的状态变迁
        run = self.lifecycle.create_run(
            thread_id=prepared.session_id,
            request_id=prepared.request_id,
            metadata=dict(config["metadata"]),
        )
        # 标记 run 已开始，记录 agent 模式等初始状态
        self._mark_agent_run_started(run=run, prepared=prepared)

        # 编译 ChatGraph：注入依赖（prepared、answer_builder、Agent 回调等），生成可执行的 LangGraph 实例
        graph = self._compile_answer_graph(
            prepared=prepared,
            answer_builder=answer_builder,
            select_agent_mode=select_agent_mode,
            run_agent_runtime=run_agent_runtime,
            build_prepared_from_state=build_prepared_from_state,
            build_hitl_wait_update=build_hitl_wait_update,
        )

        try:
            # 构建输入状态并执行图：图内依次经过 prepare_turn → select_mode → route_mode
            # → react/plan 分支 → answer_mode → final_synthesis → persist_turn
            output = graph.invoke(
                self.build_input_state(
                    prepared=prepared,
                    history_loader=history_loader,
                    config=config,
                    run_id=run.run_id,
                ),
                config,
            )
        except Exception as exc:
            # 图执行异常，标记 run 失败并向上抛出
            self.lifecycle.mark_failed(run, exc)
            raise

        # 根据图输出的 status 更新 run 生命周期状态
        if output.get("status") == "waiting_user":
            # Agent 进入 HITL 等待，run 暂停，等待后续 resume 恢复
            self.lifecycle.mark_waiting_user(run)
        else:
            # 正常完成，标记 run 成功
            self.lifecycle.mark_succeeded(run)

        # 提取图输出，组装为统一的运行结果
        return RuntimeGraphResult(
            answer=str(output.get("answer") or ""),
            citations=list(output.get("citations") or []),
            state=output,
            config=config,
            run_id=run.run_id,
        )
