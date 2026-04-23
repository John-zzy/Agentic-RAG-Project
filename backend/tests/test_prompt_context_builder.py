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


def test_prompt_builder_handles_first_turn_without_history() -> None:
    builder = PromptContextBuilder(window_size=3)
    prompt = builder.build_prompt(
        user_message="推荐一款手机",
        history_turns=[],
        retrieval_snippets=[{"citation_id": "P001", "namespace": "products", "snippet": "续航很强"}],
    )

    assert "[History]" in prompt
    assert "(empty)" in prompt
    assert "[RetrievedKnowledge]" in prompt
    assert "products:P001" in prompt
    assert "[UserMessage]" in prompt


def test_prompt_builder_includes_multi_turn_history() -> None:
    builder = PromptContextBuilder(window_size=3)
    prompt = builder.build_prompt(
        user_message="还有别的推荐吗",
        history_turns=[_turn(1), _turn(2)],
        retrieval_snippets=[{"citation_id": "R010", "namespace": "reviews", "snippet": "音质不错"}],
    )

    assert "User: user-1" in prompt
    assert "Assistant: assistant-2" in prompt
    assert "reviews:R010" in prompt


def test_prompt_builder_trims_history_by_window_size() -> None:
    builder = PromptContextBuilder(window_size=2)
    prompt = builder.build_prompt(
        user_message="继续",
        history_turns=[_turn(1), _turn(2), _turn(3)],
        retrieval_snippets=[],
    )

    assert "user-1" not in prompt
    assert "user-2" in prompt
    assert "user-3" in prompt

