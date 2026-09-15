"""Role-separated multi-agent retrieval: planner, retrievers, critic, synthesizer."""

from __future__ import annotations

import re
import time

from ars.agents.critic import CriticAgent
from ars.agents.planner import PlannerAgent
from ars.agents.retriever_agent import RetrieverAgent
from ars.agents.synthesizer import SynthesizerAgent
from ars.architectures.base import ArchitectureRunner
from ars.routing.budget import BudgetTracker
from ars.schema import AgentRole, ArchitectureName, Citation, Example, ExampleResult, StopReason
from ars.tracing.events import TraceEventType
from ars.tracing.tracer import Tracer


class MultiAgentRetrieval(ArchitectureRunner):
    name = "multi_agent"

    def run_example(self, example: Example) -> ExampleResult:
        t0 = time.perf_counter()
        budget = BudgetTracker(self.config.budget)
        trace_path = self._trace_path(example.example_id)
        tracer = Tracer(
            self.config.experiment_id,
            example.example_id,
            ArchitectureName.MULTI_AGENT.value,
            trace_path,
        )
        tracer.emit(
            TraceEventType.RUN_START,
            payload={"question": example.question, "architecture": "multi_agent"},
        )

        planner = PlannerAgent(self.model, tracer)
        fact_retriever = RetrieverAgent(
            self.model, self.retriever, specialty="fact", tracer=tracer
        )
        evidence_retriever = RetrieverAgent(
            self.model, self.retriever, specialty="evidence", tracer=tracer
        )
        critic = CriticAgent(self.model, tracer)
        synth = SynthesizerAgent(self.model, tracer)

        messages: list[dict] = []
        context_parts: list[str] = []
        seen: set[str] = set()
        stop = StopReason.COMPLETED
        answer = ""
        citations: list[Citation] = []
        error = None
        coordination_messages = 0

        try:
            # 1) Plan
            budget.register_step()
            plan_msg = planner.plan(example.question)
            budget.register_model_call(int(plan_msg.metadata.get("tokens", 50)))
            messages.append(plan_msg.model_dump())
            coordination_messages += 1
            tracer.emit(
                TraceEventType.AGENT_MESSAGE,
                payload={"from": "planner", "to": "retrievers"},
                agent_role=AgentRole.PLANNER.value,
            )

            # Extract simple subqueries from plan text / fall back to question
            subqueries = self._subqueries(plan_msg.content, example.question)

            # 2) Role-separated retrieval rounds
            max_rounds = max(1, min(3, self.config.budget.max_retrieval_calls))
            for round_i, sq in enumerate(subqueries[:max_rounds]):
                blocked = budget.check()
                if blocked:
                    stop = blocked
                    break
                if not budget.can_retrieve():
                    stop = StopReason.NO_MORE_RETRIEVAL
                    break
                budget.register_step()
                agent = fact_retriever if round_i % 2 == 0 else evidence_retriever
                hits = agent.retrieve(sq, top_k=self.config.top_k, exclude_ids=seen)
                budget.register_retrieval([h.doc_id for h in hits])
                msg = agent.as_message(hits)
                messages.append(msg.model_dump())
                coordination_messages += 1
                for h in hits:
                    if h.doc_id not in seen:
                        seen.add(h.doc_id)
                        context_parts.append(f"[{h.doc_id}] title: {h.title}\n{h.text}")

                # Critic after each round
                budget.register_step()
                ctx = "\n\n".join(context_parts)
                critique = critic.critique(example.question, ctx)
                budget.register_model_call(int(critique.metadata.get("tokens", 50)))
                messages.append(critique.model_dump())
                coordination_messages += 1

                verdict = critique.content.lower()
                if "action: abstain" in verdict or "verdict: abstain" in verdict:
                    answer = "ABSTAIN: Critic judged the question unanswerable."
                    stop = StopReason.ABSTAIN
                    break
                if "action: synthesize" in verdict or "verdict: sufficient" in verdict:
                    break
                # else continue retrieving

            if stop not in (StopReason.ABSTAIN, StopReason.ERROR) and not answer:
                ctx = "\n\n".join(context_parts)
                answer, citations, smsg = synth.synthesize(example.question, ctx)
                budget.register_model_call(int(smsg.metadata.get("tokens", 50)))
                messages.append(smsg.model_dump())
                coordination_messages += 1
                if answer.upper().startswith("ABSTAIN"):
                    stop = StopReason.ABSTAIN
                elif stop == StopReason.COMPLETED:
                    pass
        except Exception as e:  # noqa: BLE001
            error = str(e)
            stop = StopReason.ERROR
            tracer.emit(TraceEventType.ERROR, error=error)

        tracer.emit(
            TraceEventType.METRIC,
            payload={"coordination_messages": coordination_messages},
        )
        tracer.emit(
            TraceEventType.TERMINATION,
            payload={
                "stop_reason": stop.value,
                "budget": budget.snapshot(),
                "coordination_messages": coordination_messages,
            },
        )
        tracer.emit(TraceEventType.RUN_END, payload={"answer": answer})

        latency = (time.perf_counter() - t0) * 1000
        cost = self.cost.estimate(
            self.model.name, budget.tokens // 2, budget.tokens - budget.tokens // 2
        )
        result = ExampleResult(
            experiment_id=self.config.experiment_id,
            example_id=example.example_id,
            architecture=ArchitectureName.MULTI_AGENT,
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
        result.metrics["coordination_messages"] = float(coordination_messages)
        return result

    def _subqueries(self, plan: str, question: str) -> list[str]:
        found = re.findall(r"^\s*\d+\.\s*(.+)$", plan, re.M)
        if found:
            # Use plan lines as soft queries; always include original
            return [question] + [f"{question} {f}" for f in found[:2]]
        return [question, question + " evidence", question + " related facts"]
