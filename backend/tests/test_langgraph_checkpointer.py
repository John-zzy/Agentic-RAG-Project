from __future__ import annotations

import sqlite3
from typing import Any, TypedDict

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph

from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.tests.test_support import make_test_runtime_dir


def _build_checkpointer(test_name: str) -> SQLiteLangGraphCheckpointer:
    runtime_dir = make_test_runtime_dir(test_name)
    return SQLiteLangGraphCheckpointer(runtime_dir / "langgraph.db")


def _checkpoint(checkpoint_id: str, answer: str) -> dict[str, Any]:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = checkpoint_id
    checkpoint["channel_values"] = {
        "messages": [f"user:{answer}"],
        "answer": answer,
    }
    checkpoint["channel_versions"] = {
        "messages": int(checkpoint_id),
        "answer": int(checkpoint_id),
    }
    return checkpoint


def test_sqlite_langgraph_checkpointer_writes_and_reads_latest_checkpoint() -> None:
    saver = _build_checkpointer("langgraph-checkpointer-latest")
    config = {
        "configurable": {
            "thread_id": "session-a",
            "checkpoint_ns": "chat_runtime",
        },
        "metadata": {
            "request_id": "req-1",
        },
    }

    first_config = saver.put(
        config,
        _checkpoint("0001", "first"),
        {"source": "unit"},
        {"messages": 1, "answer": 1},
    )
    second_config = saver.put(
        first_config,
        _checkpoint("0002", "second"),
        {"source": "unit", "request_id": "req-2"},
        {"messages": 2, "answer": 2},
    )

    latest = saver.get_tuple(config)
    explicit_first = saver.get_tuple(first_config)

    assert latest is not None
    assert latest.config["configurable"]["checkpoint_id"] == "0002"
    assert latest.checkpoint["channel_values"]["answer"] == "second"
    assert latest.metadata["request_id"] == "req-2"
    assert latest.parent_config == first_config
    assert second_config["configurable"]["checkpoint_id"] == "0002"
    assert explicit_first is not None
    assert explicit_first.checkpoint["channel_values"]["answer"] == "first"


def test_sqlite_langgraph_checkpointer_lists_with_filter_before_and_limit() -> None:
    saver = _build_checkpointer("langgraph-checkpointer-list")
    base_config = {
        "configurable": {
            "thread_id": "session-list",
            "checkpoint_ns": "chat_runtime",
        }
    }
    first_config = saver.put(
        base_config,
        _checkpoint("0001", "first"),
        {"request_id": "req-1", "branch": "keep"},
        {"messages": 1, "answer": 1},
    )
    saver.put(
        first_config,
        _checkpoint("0002", "second"),
        {"request_id": "req-2", "branch": "skip"},
        {"messages": 2, "answer": 2},
    )
    saver.put(
        {
            "configurable": {
                "thread_id": "other-session",
                "checkpoint_ns": "chat_runtime",
            }
        },
        _checkpoint("0003", "other"),
        {"request_id": "req-3", "branch": "keep"},
        {"messages": 3, "answer": 3},
    )

    listed = list(
        saver.list(
            base_config,
            before={"configurable": {"checkpoint_id": "0002"}},
            filter={"branch": "keep"},
            limit=1,
        )
    )

    assert len(listed) == 1
    assert listed[0].config["configurable"]["thread_id"] == "session-list"
    assert listed[0].config["configurable"]["checkpoint_id"] == "0001"
    assert listed[0].metadata["request_id"] == "req-1"


def test_sqlite_langgraph_checkpointer_persists_pending_writes() -> None:
    saver = _build_checkpointer("langgraph-checkpointer-writes")
    config = saver.put(
        {
            "configurable": {
                "thread_id": "session-writes",
                "checkpoint_ns": "chat_runtime",
            }
        },
        _checkpoint("0001", "answer"),
        {"request_id": "req-writes"},
        {"messages": 1, "answer": 1},
    )

    saver.put_writes(
        config,
        [
            ("messages", ["pending message"]),
            ("__error__", {"code": "model_error"}),
        ],
        task_id="answer-node",
        task_path="answer",
    )
    saver.put_writes(
        config,
        [("messages", ["ignored duplicate"])],
        task_id="answer-node",
        task_path="answer",
    )

    restored = saver.get_tuple(config)

    assert restored is not None
    assert restored.pending_writes == [
        ("answer-node", "__error__", {"code": "model_error"}),
        ("answer-node", "messages", ["pending message"]),
    ]


def test_sqlite_langgraph_checkpointer_deletes_thread_data() -> None:
    saver = _build_checkpointer("langgraph-checkpointer-delete")
    config = saver.put(
        {
            "configurable": {
                "thread_id": "session-delete",
                "checkpoint_ns": "chat_runtime",
            }
        },
        _checkpoint("0001", "answer"),
        {"request_id": "req-delete"},
        {"messages": 1, "answer": 1},
    )
    saver.put_writes(config, [("messages", ["pending"])], task_id="answer-node")

    saver.delete_thread("session-delete")

    assert saver.get_tuple(config) is None
    assert list(saver.list(config)) == []


def test_sqlite_langgraph_checkpointer_creates_required_tables_and_indexes() -> None:
    runtime_dir = make_test_runtime_dir("langgraph-checkpointer-schema")
    sqlite_path = runtime_dir / "langgraph.db"

    SQLiteLangGraphCheckpointer(sqlite_path)

    with sqlite3.connect(str(sqlite_path)) as conn:
        table_rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name LIKE 'langgraph_%'
            ORDER BY name
            """
        ).fetchall()
        index_rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
              AND name LIKE 'idx_langgraph_%'
            ORDER BY name
            """
        ).fetchall()

    assert [row[0] for row in table_rows] == [
        "langgraph_blobs",
        "langgraph_checkpoints",
        "langgraph_writes",
    ]
    assert [row[0] for row in index_rows] == [
        "idx_langgraph_blobs_channel_version",
        "idx_langgraph_checkpoints_thread_latest",
        "idx_langgraph_writes_checkpoint",
    ]


def test_sqlite_langgraph_checkpointer_works_with_minimal_state_graph() -> None:
    class State(TypedDict, total=False):
        question: str
        answer: str

    def answer_node(state: State) -> dict[str, str]:
        return {"answer": f"{state['question']}!"}

    saver = _build_checkpointer("langgraph-checkpointer-state-graph")
    builder = StateGraph(State)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)

    output = graph.invoke(
        {"question": "hello"},
        {
            "configurable": {
                "thread_id": "session-graph",
            },
            "metadata": {
                "request_id": "req-graph",
            },
        },
    )
    restored = saver.get_tuple({"configurable": {"thread_id": "session-graph"}})

    assert output["answer"] == "hello!"
    assert restored is not None
    assert restored.metadata["request_id"] == "req-graph"
    assert restored.checkpoint["channel_values"]["answer"] == "hello!"
