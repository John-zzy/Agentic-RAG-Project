from backend.memory.prompt_context import PromptContextBuilder
from backend.memory.session_store import SessionTurn


def _turn(index: int) -> SessionTurn:
    return SessionTurn(
        session_id="session-1",
        request_id=f"req-{index}",
        user_message=f"user-{index}",
        assistant_answer=f"assistant-{index}",
        retrieval_snippets=[{"citation_id": f"c-{index}", "namespace": "products", "snippet": "snippet"}],
        timestamp="2026-04-23T00:00:00+00:00",
    )


def test_trim_turns_within_window_returns_all() -> None:
    builder = PromptContextBuilder(window_size=5)
    turns = [_turn(1), _turn(2), _turn(3)]
    result = builder.trim_turns(turns)
    assert len(result) == 3
    assert result[0].user_message == "user-1"


def test_trim_turns_exceeding_window_keeps_most_recent() -> None:
    builder = PromptContextBuilder(window_size=2)
    turns = [_turn(1), _turn(2), _turn(3)]
    result = builder.trim_turns(turns)
    assert len(result) == 2
    assert "user-1" not in [t.user_message for t in result]
    assert "user-2" in [t.user_message for t in result]
    assert "user-3" in [t.user_message for t in result]


def test_trim_turns_empty_returns_empty() -> None:
    builder = PromptContextBuilder(window_size=3)
    result = builder.trim_turns([])
    assert result == []
