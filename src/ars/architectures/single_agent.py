"""Bounded single-agent iterative retrieval."""

from __future__ import annotations

import time

from ars.agents.single import SingleAgentController
from ars.agents.synthesizer import SynthesizerAgent
from ars.architectures.base import ArchitectureRunner
from ars.routing.actions import ActionType
from ars.routing.budget import BudgetTracker
from ars.schema import ArchitectureName, Citation, Example, ExampleResult, StopReason
from ars.tracing.events import TraceEventType
from ars.tracing.tracer import Tracer


class SingleAgentIterative(ArchitectureRunner):
    name = "single_agent"

    def run_example(self, example: Example) -> ExampleResult:
        t0 = time.perf_counter()
        budget = BudgetTracker(self.config.budget)
        trace_path = self._trace_path(example.example_id)
        tracer = Tracer(
            self.config.experiment_id,
            example.example_id,
            ArchitectureName.SINGLE_AGENT.value,
            trace_path,
        )
        tracer.emit(
            TraceEventType.RUN_START,
            payload={"question": example.question, "architecture": "single_agent"},
        )

        controller = SingleAgentController(self.model, tracer)
        synth = SynthesizerAgent(self.model, tracer)

        context_parts: list[str] = []
        seen: set[str] = set()
        stop = StopReason.COMPLETED
        answer = ""
        citations: list[Citation] = []
        error = None
        query = example.question

        try:
            while True:
                blocked = budget.check()
                if blocked:
                    stop = blocked
                    break

                budget.register_step()
                tracer.emit(
                    TraceEventType.STEP_START,
                    payload={"step": budget.steps},
                    advance_step=False,
                )
                tracer.emit(
                    TraceEventType.BUDGET_CHECK,
                    payload=budget.snapshot(),
                )

                ctx = "\n\n".join(context_parts) if context_parts else "(empty)"
                action = controller.decide(example.question, ctx, budget.steps - 1)
                # extract tokens from reason hack
                if action.reason and "tokens=" in action.reason:
                    try:
                        tok = int(action.reason.split("tokens=")[-1])
                        budget.register_model_call(tok)
                    except ValueError:
                        budget.register_model_call(50)
                else:
                    budget.register_model_call(50)

                if action.type == ActionType.ABSTAIN:
                    answer = "ABSTAIN: Agent chose to abstain."
                    stop = StopReason.ABSTAIN
                    break

                if action.type in (ActionType.RETRIEVE, ActionType.REFORMULATE):
                    if not budget.can_retrieve():
                        stop = StopReason.NO_MORE_RETRIEVAL
                        break
                    q = action.query or query
                    if action.type == ActionType.REFORMULATE and action.query:
                        query = action.query
                        q = query
                    hits = self.retriever.retrieve_with_ids(
                        q, top_k=self.config.top_k, exclude_ids=seen
                    )
                    budget.register_retrieval([h.doc_id for h in hits])
                    for h in hits:
                        if h.doc_id not in seen:
                            seen.add(h.doc_id)
                            context_parts.append(
                                f"[{h.doc_id}] title: {h.title}\n{h.text}"
                            )
                    tracer.emit(
                        TraceEventType.CONTEXT_UPDATE,
                        payload={
                            "n_docs": len(seen),
                            "query": q,
                            "new_ids": [h.doc_id for h in hits],
                        },
                    )
                    continue

                if action.type in (ActionType.STOP, ActionType.SYNTHESIZE):
                    break

            if stop == StopReason.COMPLETED or stop == StopReason.NO_MORE_RETRIEVAL:
                ctx = "\n\n".join(context_parts)
                answer, citations, msg = synth.synthesize(example.question, ctx)
                budget.register_model_call(int(msg.metadata.get("tokens", 0)))
                if answer.upper().startswith("ABSTAIN"):
                    stop = StopReason.ABSTAIN
                elif stop == StopReason.NO_MORE_RETRIEVAL:
                    pass
                else:
                    stop = StopReason.COMPLETED
        except Exception as e:  # noqa: BLE001
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
            self.model.name, budget.tokens // 2, budget.tokens - budget.tokens // 2
        )
        return ExampleResult(
            experiment_id=self.config.experiment_id,
            example_id=example.example_id,
            architecture=ArchitectureName.SINGLE_AGENT,
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
