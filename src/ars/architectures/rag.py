"""Conventional RAG: one retrieve, one generate."""

from __future__ import annotations

import time

from ars.agents.synthesizer import SynthesizerAgent
from ars.architectures.base import ArchitectureRunner
from ars.routing.budget import BudgetTracker
from ars.schema import ArchitectureName, Citation, Example, ExampleResult, StopReason
from ars.tracing.events import TraceEventType
from ars.tracing.tracer import Tracer


class ConventionalRAG(ArchitectureRunner):
    name = "rag"

    def run_example(self, example: Example) -> ExampleResult:
        t0 = time.perf_counter()
        budget = BudgetTracker(self.config.budget)
        trace_path = self._trace_path(example.example_id)
        tracer = Tracer(
            self.config.experiment_id,
            example.example_id,
            ArchitectureName.RAG.value,
            trace_path,
        )
        tracer.emit(
            TraceEventType.RUN_START,
            payload={"question": example.question, "architecture": "rag"},
        )

        stop = StopReason.COMPLETED
        answer = ""
        citations: list[Citation] = []
        error = None
        try:
            budget.register_step()
            top_k = self.config.top_k
            hits = self.retriever.retrieve(example.question, top_k=top_k)
            budget.register_retrieval([h.doc_id for h in hits])
            tracer.emit(
                TraceEventType.RETRIEVAL_QUERY,
                payload={"query": example.question, "top_k": top_k},
                agent_role="retriever",
            )
            tracer.emit(
                TraceEventType.RETRIEVAL_RESULT,
                payload={
                    "doc_ids": [h.doc_id for h in hits],
                    "scores": [h.score for h in hits],
                },
                agent_role="retriever",
            )
            context = "\n\n".join(
                f"[{h.doc_id}] title: {h.title}\n{h.text}" for h in hits
            )
            tracer.emit(
                TraceEventType.CONTEXT_UPDATE,
                payload={"context_chars": len(context), "n_docs": len(hits)},
            )
            synth = SynthesizerAgent(self.model, tracer)
            answer, citations, msg = synth.synthesize(example.question, context)
            toks = int(msg.metadata.get("tokens", 0))
            budget.register_model_call(toks)
            if answer.upper().startswith("ABSTAIN"):
                stop = StopReason.ABSTAIN
        except Exception as e:  # noqa: BLE001 — captured into result
            error = str(e)
            stop = StopReason.ERROR
            tracer.emit(TraceEventType.ERROR, error=error)

        tracer.emit(
            TraceEventType.TERMINATION,
            payload={"stop_reason": stop.value, "budget": budget.snapshot()},
        )
        tracer.emit(TraceEventType.RUN_END, payload={"answer": answer})

        latency = (time.perf_counter() - t0) * 1000
        cost = self.cost.estimate(
            self.model.name,
            budget.tokens // 2,
            budget.tokens - budget.tokens // 2,
        )
        return ExampleResult(
            experiment_id=self.config.experiment_id,
            example_id=example.example_id,
            architecture=ArchitectureName.RAG,
            question=example.question,
            answer=answer,
            citations=citations,
            gold_answer=example.gold_answer,
            gold_doc_ids=example.gold_doc_ids,
            stop_reason=stop,
            abstained=answer.upper().startswith("ABSTAIN"),
            latency_ms=latency,
            model_calls=budget.model_calls,
            retrieval_calls=budget.retrieval_calls,
            docs_retrieved=budget.docs,
            unique_docs=budget.unique_docs,
            duplicate_docs=budget.duplicate_docs,
            prompt_tokens=budget.tokens // 2,
            completion_tokens=budget.tokens - budget.tokens // 2,
            estimated_cost_usd=cost,
            error=error,
            trace_path=str(trace_path) if trace_path else None,
        )
