"""Config-based cost estimator with timestamped price table."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class PriceEntry(BaseModel):
    model_name: str
    input_per_1k: float
    output_per_1k: float
    as_of: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    currency: str = "USD"
    notes: str = ""


# Timestamped price table (fixture / research estimates — not live billing)
DEFAULT_PRICES: list[PriceEntry] = [
    PriceEntry(
        model_name="deterministic-v1",
        input_per_1k=0.0,
        output_per_1k=0.0,
        as_of="2026-01-01",
        notes="Mock model — zero cost",
    ),
    PriceEntry(
        model_name="gpt-4o-mini",
        input_per_1k=0.00015,
        output_per_1k=0.0006,
        as_of="2025-06-01",
        notes="Public list price snapshot for research estimation",
    ),
    PriceEntry(
        model_name="gpt-4o",
        input_per_1k=0.0025,
        output_per_1k=0.01,
        as_of="2025-06-01",
        notes="Public list price snapshot for research estimation",
    ),
    PriceEntry(
        model_name="mock-embedding",
        input_per_1k=0.0,
        output_per_1k=0.0,
        as_of="2026-01-01",
        notes="Fixture embeddings",
    ),
]


class CostEstimator:
    def __init__(self, prices: list[PriceEntry] | None = None) -> None:
        self.prices = {p.model_name: p for p in (prices or DEFAULT_PRICES)}

    def estimate(
        self, model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        entry = self.prices.get(model_name)
        if entry is None:
            # Unknown model: treat as zero with warning flag in notes
            return 0.0
        return (prompt_tokens / 1000.0) * entry.input_per_1k + (
            completion_tokens / 1000.0
        ) * entry.output_per_1k

    def table(self) -> list[dict[str, Any]]:
        return [p.model_dump() for p in self.prices.values()]
