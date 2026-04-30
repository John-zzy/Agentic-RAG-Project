import sqlite3

import backend.memory.base.session_store as session_store_module
from backend.memory.base.session_store import SQLiteSessionStore
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


def test_session_store_creates_and_updates_session_metadata() -> None:
    store = _build_store("session-store-metadata")
    created = store.create_session(
        session_id="session-meta",
        now="2026-04-23T00:00:00+00:00",
    )
    touched = store.touch_session(
        session_id="session-meta",
        now="2026-04-23T00:05:00+00:00",
    )

    assert created.status == "active"
    assert created.created_at == "2026-04-23T00:00:00+00:00"
    assert touched is not None
    assert touched.last_active_at == "2026-04-23T00:05:00+00:00"
    assert touched.updated_at == "2026-04-23T00:05:00+00:00"


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


def test_session_store_marks_inactive_sessions_as_expired() -> None:
    store = _build_store("session-store-expire")
    store.create_session(
        session_id="session-expire",
        now="2026-04-23T00:00:00+00:00",
    )
    expired = store.cleanup_expired_sessions(
        now="2026-04-23T00:31:00+00:00",
        timeout_minutes=30,
        limit=10,
    )
    session = store.get_session("session-expire")

    assert expired == ["session-expire"]
    assert session is not None
    assert session.status == "expired"
    assert session.expired_at == "2026-04-23T00:31:00+00:00"


def test_session_store_migrates_legacy_database_to_data_dir(monkeypatch) -> None:
    runtime_dir = make_test_runtime_dir("session-store-migrate")
    legacy_path = runtime_dir / "memory" / "sessions.db"
    target_path = runtime_dir / "data" / "sessions.db"

    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(legacy_path))
    try:
        conn.execute("CREATE TABLE legacy_turns (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO legacy_turns DEFAULT VALUES")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(session_store_module, "LEGACY_SQLITE_PATH", legacy_path)
    monkeypatch.setattr(session_store_module, "SQLITE_PATH", target_path)
    monkeypatch.setattr(session_store_module.settings.session, "sqlite_path", target_path)

    store = SQLiteSessionStore()

    assert store._sqlite_path == target_path
    assert target_path.exists()
    assert not legacy_path.exists()
    assert store.count_turns("missing-session") == 0
