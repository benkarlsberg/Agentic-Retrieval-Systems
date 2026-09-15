from __future__ import annotations

from pathlib import Path

from ars.architectures.base import ArchitectureRunner
from ars.architectures.multi_agent import MultiAgentRetrieval
from ars.architectures.rag import ConventionalRAG
from ars.architectures.single_agent import SingleAgentIterative
from ars.models.base import ModelClient
from ars.models.cost import CostEstimator
from ars.retrieval.base import Retriever
from ars.schema import ArchitectureName, RunConfig


def make_architecture(
    config: RunConfig,
    retriever: Retriever,
    model: ModelClient,
    trace_dir: Path | None = None,
    cost: CostEstimator | None = None,
) -> ArchitectureRunner:
    if config.architecture == ArchitectureName.RAG:
        return ConventionalRAG(
            config=config,
            retriever=retriever,
            model=model,
            cost_estimator=cost,
            trace_dir=trace_dir,
        )
    if config.architecture == ArchitectureName.SINGLE_AGENT:
        return SingleAgentIterative(
            config=config,
            retriever=retriever,
            model=model,
            cost_estimator=cost,
            trace_dir=trace_dir,
        )
    if config.architecture == ArchitectureName.MULTI_AGENT:
        return MultiAgentRetrieval(
            config=config,
            retriever=retriever,
            model=model,
            cost_estimator=cost,
            trace_dir=trace_dir,
        )
    raise ValueError(f"Unknown architecture: {config.architecture}")
