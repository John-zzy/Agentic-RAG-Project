from backend.platform.agent_runtime.mode_selector import (
    MinimalModeSelector,
    ModeSelectionContext,
)


def test_mode_selector_defaults_simple_request_to_react() -> None:
    selector = MinimalModeSelector()

    selection = selector.select(
        message="查询上传文档里的报销规则",
        complexity="simple",
        mounted_knowledge_sources=("documents",),
    )

    assert selection.mode == "react"
    assert selection.reason == "default_simple_react"
    assert selector.select_mode(
        ModeSelectionContext(
            user_message="查询上传文档里的报销规则",
            complexity="simple",
            mounted_knowledge_sources=("documents",),
        )
    ) == "react"


def test_mode_selector_uses_plan_for_explicit_or_complex_requests() -> None:
    selector = MinimalModeSelector()

    explicit = selector.select(
        message="请分步骤制定计划，然后检索资料并汇总结论",
        complexity="simple",
        mounted_knowledge_sources=("documents",),
    )
    complex_selection = selector.select(
        message="分析订单、商品和文档后给出处理建议",
        complexity="complex",
        mounted_knowledge_sources=("documents",),
    )

    assert explicit.mode == "plan"
    assert explicit.reason == "complex_or_explicit_plan"
    assert "计划" in explicit.signals["keyword_hits"]
    assert complex_selection.mode == "plan"
