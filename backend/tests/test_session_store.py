from backend.memory.session_store import SQLiteSessionStore
from backend.tests.test_support import make_test_runtime_dir


def _build_store(test_name: str) -> SQLiteSessionStore:
    runtime_dir = make_test_runtime_dir(test_name)
    return SQLiteSessionStore(sqlite_path=runtime_dir / "sessions.db")


def test_session_store_persists_and_reads_turns() -> None:
    store = _build_store("session-store-persist")
    store.append_turn(
        session_id="session-a",
        request_id="req-1",
        user_message="你好",
        assistant_answer="你好，请问需要什么帮助？",
        retrieval_snippets=[{"citation_id": "P001", "namespace": "products", "snippet": "手机"}],
        timestamp="2026-04-23T00:00:00+00:00",
    )

    turns = store.get_recent_turns("session-a", limit=10)
    assert len(turns) == 1
    assert turns[0].request_id == "req-1"
    assert turns[0].retrieval_snippets[0]["citation_id"] == "P001"


def test_session_store_supports_session_resume() -> None:
    store = _build_store("session-store-resume")
    store.append_turn(
        session_id="session-resume",
        request_id="req-1",
        user_message="第一轮",
        assistant_answer="第一轮回复",
        retrieval_snippets=[],
        timestamp="2026-04-23T00:00:00+00:00",
    )
    store.append_turn(
        session_id="session-resume",
        request_id="req-2",
        user_message="第二轮",
        assistant_answer="第二轮回复",
        retrieval_snippets=[],
        timestamp="2026-04-23T00:01:00+00:00",
    )

    turns = store.get_recent_turns("session-resume", limit=10)
    assert [turn.request_id for turn in turns] == ["req-1", "req-2"]


def test_session_store_respects_recent_limit() -> None:
    store = _build_store("session-store-window")
    for index in range(1, 5):
        store.append_turn(
            session_id="session-window",
            request_id=f"req-{index}",
            user_message=f"u-{index}",
            assistant_answer=f"a-{index}",
            retrieval_snippets=[],
            timestamp=f"2026-04-23T00:0{index}:00+00:00",
        )

    turns = store.get_recent_turns("session-window", limit=2)
    assert [turn.request_id for turn in turns] == ["req-3", "req-4"]
