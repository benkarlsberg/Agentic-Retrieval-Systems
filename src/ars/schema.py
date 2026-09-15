"""Typed run schema shared across architectures, tracing, and evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ArchitectureName(str, Enum):
    RAG = "rag"
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"


class AgentRole(str, Enum):
    RETRIEVER = "retriever"
    PLANNER = "planner"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    SINGLE = "single"
    GENERATOR = "generator"


class StopReason(str, Enum):
    COMPLETED = "completed"
    BUDGET_STEPS = "budget_steps"
    BUDGET_MODEL_CALLS = "budget_model_calls"
    BUDGET_DOCS = "budget_docs"
    BUDGET_TOKENS = "budget_tokens"
    MAX_ITERATIONS = "max_iterations"
    ABSTAIN = "abstain"
    ERROR = "error"
    NO_MORE_RETRIEVAL = "no_more_retrieval"


class ExampleType(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    MULTI_PASSAGE = "multi_passage"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"


class Document(BaseModel):
    doc_id: str
    title: str = ""
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDoc(BaseModel):
    doc_id: str
    score: float
    title: str = ""
    text: str = ""
    rank: int = 0


class Citation(BaseModel):
    doc_id: str
    span: str | None = None
    confidence: float | None = None


class Budget(BaseModel):
    max_steps: int = 8
    max_model_calls: int = 10
    max_docs: int = 20
    max_tokens: int = 8000
    max_retrieval_calls: int = 5
    top_k: int = 5


class ModelSettings(BaseModel):
    provider: str = "mock"
    model_name: str = "deterministic-v1"
    temperature: float = 0.0
    max_tokens: int = 512
    extra: dict[str, Any] = Field(default_factory=dict)


class Example(BaseModel):
    example_id: str
    question: str
    gold_answer: str | None = None
    gold_doc_ids: list[str] = Field(default_factory=list)
    example_type: ExampleType = ExampleType.SINGLE_HOP
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_answerable: bool = True


class RunConfig(BaseModel):
    experiment_id: str
    architecture: ArchitectureName
    budget: Budget = Field(default_factory=Budget)
    model: ModelSettings = Field(default_factory=ModelSettings)
    retrieval_mode: str = "bm25"  # bm25 | dense | hybrid
    top_k: int = 5
    seed: int = 42
    comparison_mode: str = "unconstrained"  # equal_retrieval | equal_token | unconstrained
    tags: list[str] = Field(default_factory=list)
    ablation: str | None = None


class ExampleResult(BaseModel):
    experiment_id: str
    example_id: str
    architecture: ArchitectureName
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    gold_answer: str | None = None
    gold_doc_ids: list[str] = Field(default_factory=list)
    stop_reason: StopReason = StopReason.COMPLETED
    abstained: bool = False
    latency_ms: float = 0.0
    model_calls: int = 0
    retrieval_calls: int = 0
    docs_retrieved: int = 0
    unique_docs: int = 0
    duplicate_docs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    trace_path: str | None = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    architecture: ArchitectureName
    n_examples: int
    comparison_mode: str
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    fixture_offline: bool = True
    notes: str = ""
