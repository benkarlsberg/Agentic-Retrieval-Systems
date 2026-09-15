"""Hard budget tracking and termination checks."""

from __future__ import annotations

from ars.schema import Budget, StopReason


class BudgetTracker:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.steps = 0
        self.model_calls = 0
        self.retrieval_calls = 0
        self.docs = 0
        self.tokens = 0
        self.seen_doc_ids: list[str] = []

    def register_step(self) -> None:
        self.steps += 1

    def register_model_call(self, tokens: int = 0) -> None:
        self.model_calls += 1
        self.tokens += tokens

    def register_retrieval(self, doc_ids: list[str]) -> None:
        self.retrieval_calls += 1
        self.docs += len(doc_ids)
        self.seen_doc_ids.extend(doc_ids)

    @property
    def unique_docs(self) -> int:
        return len(set(self.seen_doc_ids))

    @property
    def duplicate_docs(self) -> int:
        return max(0, len(self.seen_doc_ids) - self.unique_docs)

    def check(self) -> StopReason | None:
        b = self.budget
        if self.steps >= b.max_steps:
            return StopReason.BUDGET_STEPS
        if self.model_calls >= b.max_model_calls:
            return StopReason.BUDGET_MODEL_CALLS
        if self.docs >= b.max_docs:
            return StopReason.BUDGET_DOCS
        if self.tokens >= b.max_tokens:
            return StopReason.BUDGET_TOKENS
        if self.retrieval_calls >= b.max_retrieval_calls and self.steps > 0:
            # Allow finishing after last allowed retrieval
            pass
        return None

    def can_retrieve(self) -> bool:
        return (
            self.retrieval_calls < self.budget.max_retrieval_calls
            and self.docs < self.budget.max_docs
            and self.check() is None
        )

    def snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "model_calls": self.model_calls,
            "retrieval_calls": self.retrieval_calls,
            "docs": self.docs,
            "tokens": self.tokens,
            "unique_docs": self.unique_docs,
            "duplicate_docs": self.duplicate_docs,
        }
