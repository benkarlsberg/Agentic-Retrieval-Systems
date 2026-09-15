from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ars.models.base import ModelClient
from ars.models.cost import CostEstimator
from ars.retrieval.base import Retriever
from ars.schema import Example, ExampleResult, RunConfig


class ArchitectureRunner(ABC):
    name: str = "base"

    def __init__(
        self,
        config: RunConfig,
        retriever: Retriever,
        model: ModelClient,
        cost_estimator: CostEstimator | None = None,
        trace_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.model = model
        self.cost = cost_estimator or CostEstimator()
        self.trace_dir = trace_dir

    @abstractmethod
    def run_example(self, example: Example) -> ExampleResult:
        raise NotImplementedError

    def _trace_path(self, example_id: str) -> Path | None:
        if not self.trace_dir:
            return None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        return self.trace_dir / f"{self.config.experiment_id}_{example_id}.jsonl"
