from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from langchain_core.documents import Document

from backend.platform.rag.contracts import RetrievalPlan, RetrievalResult
from backend.platform.rag.orchestration.decisions import (
    RetrievalDecisionLogEntry,
    SufficiencyDecision,
)
from backend.platform.rag.pre_retrieval.query_rewrite import QueryRewrite


class AgenticRagOutcomeProjector:
    """把 typed graph state 投影回 Agentic Retrieval 对外 outcome。"""

    def __init__(
        self,
        *,
        outcome_factory: Callable[..., Any],
        round_factory: Callable[..., Any],
    ) -> None:
        self._outcome_factory = outcome_factory
        self._round_factory = round_factory

    def project(self, state: Mapping[str, Any]) -> Any:
        plan = self._coerce_plan(state.get("final_plan") or state.get("plan"))
        rounds = [self._round_from_snapshot(snapshot) for snapshot in list(state.get("rounds") or [])]
        final_decision = self._coerce_decision(
            state.get("final_decision"),
            fallback=rounds[-1].decision if rounds else None,
        )
        return self._outcome_factory(
            plan=plan,
            results=[self._coerce_result(result) for result in list(state.get("results") or [])],
            documents=[self._coerce_document(document) for document in list(state.get("documents") or [])],
            success=bool(state.get("success", False)),
            rounds=rounds,
            decision_log=[
                self._coerce_decision_log_entry(entry)
                for entry in list(state.get("decision_log") or [])
            ],
            final_plan=plan,
            final_decision=final_decision,
            exit_reason=str(state.get("exit_reason") or "ask_user"),
            follow_up_question=state.get("follow_up_question"),
        )

    def _round_from_snapshot(self, snapshot: Mapping[str, Any]) -> Any:
        rewrite = snapshot.get("rewrite")
        return self._round_factory(
            plan=self._coerce_plan(snapshot["plan"]),
            results=[self._coerce_result(result) for result in list(snapshot.get("results") or [])],
            documents=[
                self._coerce_document(document)
                for document in list(snapshot.get("documents") or [])
            ],
            result=self._coerce_result(snapshot["result"]),
            decision=self._coerce_decision(snapshot["decision"]),
            rewrite=QueryRewrite.model_validate(rewrite) if rewrite else None,
        )

    def _coerce_plan(self, value: Any) -> RetrievalPlan:
        if isinstance(value, RetrievalPlan):
            return value
        if isinstance(value, Mapping):
            return RetrievalPlan.model_validate(value)
        raise TypeError("Invalid retrieval plan payload.")

    def _coerce_result(self, value: Any) -> RetrievalResult:
        if isinstance(value, RetrievalResult):
            return value
        if isinstance(value, Mapping):
            return RetrievalResult.model_validate(value)
        raise TypeError("Invalid retrieval result payload.")

    def _coerce_document(self, value: Any) -> Document:
        if isinstance(value, Document):
            return value
        if isinstance(value, Mapping):
            return Document.model_validate(value)
        raise TypeError("Invalid retrieval document payload.")

    def _coerce_decision(
        self,
        value: Any,
        *,
        fallback: SufficiencyDecision | None = None,
    ) -> SufficiencyDecision:
        if isinstance(value, SufficiencyDecision):
            return value
        if isinstance(value, Mapping):
            return SufficiencyDecision.model_validate(value)
        if fallback is not None:
            return fallback
        raise TypeError("Invalid retrieval decision payload.")

    def _coerce_decision_log_entry(self, value: Any) -> RetrievalDecisionLogEntry:
        if isinstance(value, RetrievalDecisionLogEntry):
            return value
        if isinstance(value, Mapping):
            return RetrievalDecisionLogEntry.model_validate(value)
        raise TypeError("Invalid retrieval decision log payload.")
