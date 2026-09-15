"""Deterministic/heuristic model for offline fixture evaluation."""

from __future__ import annotations

import re
import time
from typing import Any

from ars.models.base import GenerationRequest, GenerationResponse, ModelClient


class DeterministicModel(ModelClient):
    """Keyword/heuristic answers from retrieved context — offline-safe."""

    name = "deterministic-v1"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.call_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_count += 1
        t0 = time.perf_counter()
        text = self._respond(request.prompt, request.system, request.metadata)
        latency = (time.perf_counter() - t0) * 1000
        return GenerationResponse(
            text=text,
            model_name=self.name,
            prompt_tokens=self.estimate_tokens(request.prompt),
            completion_tokens=self.estimate_tokens(text),
            latency_ms=latency,
            finish_reason="stop",
        )

    def _respond(
        self, prompt: str, system: str | None, metadata: dict[str, Any]
    ) -> str:
        role = (metadata.get("role") or "").lower()
        task = (metadata.get("task") or "").lower()

        if task == "plan" or role == "planner":
            return self._plan(prompt)
        if task == "critique" or role == "critic":
            return self._critique(prompt)
        if task == "route" or "retrieve" in task or "decide" in task:
            return self._decide_action(prompt)
        if task == "reformulate":
            return self._reformulate(prompt)
        return self._answer(prompt)

    def _extract_question(self, prompt: str) -> str:
        m = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"Current question:\s*(.+?)(?:\n|$)", prompt, re.I)
        if m:
            return m.group(1).strip()
        return prompt.strip().split("\n")[0][:200]

    def _extract_context(self, prompt: str) -> str:
        m = re.search(r"Context:\s*(.*?)(?:\n(?:Question|Answer|Instruction)|$)", prompt, re.S | re.I)
        if m:
            return m.group(1).strip()
        return prompt

    def _plan(self, prompt: str) -> str:
        q = self._extract_question(prompt).lower()
        # Multi-hop cues
        if any(w in q for w in ("who founded", "capital of the country", "born in", "spouse of")):
            return (
                "SUBQUESTIONS:\n"
                "1. Identify the intermediate entity mentioned in the question.\n"
                "2. Retrieve the attribute asked about that entity.\n"
                "STRATEGY: sequential multi-hop retrieval"
            )
        if any(w in q for w in ("compare", "both", "and also")):
            return (
                "SUBQUESTIONS:\n"
                "1. Evidence for first claim.\n"
                "2. Evidence for second claim.\n"
                "STRATEGY: parallel multi-passage"
            )
        return (
            "SUBQUESTIONS:\n"
            "1. Direct fact lookup for the question.\n"
            "STRATEGY: single-hop"
        )

    def _critique(self, prompt: str) -> str:
        ctx = self._extract_context(prompt).lower()
        q = self._extract_question(prompt).lower()
        if "insufficient" in prompt.lower() or len(ctx) < 40:
            return "VERDICT: insufficient\nREASON: Context too thin; need another retrieval.\nACTION: retrieve"
        # Unanswerable heuristics
        if any(w in q for w in ("unicorn ceo of nasa", "invisible city of atlantis capital")):
            return "VERDICT: abstain\nREASON: No supporting evidence for an answerable claim.\nACTION: abstain"
        if "doc_" in ctx or "title:" in ctx or len(ctx) > 80:
            return "VERDICT: sufficient\nREASON: Context supports a grounded answer.\nACTION: synthesize"
        return "VERDICT: insufficient\nREASON: Missing key evidence.\nACTION: retrieve"

    def _decide_action(self, prompt: str) -> str:
        # Used by single-agent loop
        step = 0
        m = re.search(r"Step:\s*(\d+)", prompt)
        if m:
            step = int(m.group(1))
        ctx = self._extract_context(prompt)
        q = self._extract_question(prompt).lower()
        has_ctx = len(ctx.strip()) > 50 and ("title:" in ctx.lower() or "doc" in ctx.lower())

        if any(w in q for w in ("unicorn ceo of nasa", "invisible city")):
            if step >= 1:
                return "ACTION: stop\nREASON: unanswerable after retrieval attempt"
            return "ACTION: retrieve\nQUERY: " + q

        if not has_ctx:
            return "ACTION: retrieve\nQUERY: " + self._extract_question(prompt)

        if step == 0 and any(w in q for w in ("who founded", "capital of the country", "born")):
            return "ACTION: reformulate\nQUERY: " + self._next_hop_query(q)

        if step >= 1 and has_ctx:
            return "ACTION: stop\nREASON: enough context to answer"
        if has_ctx:
            return "ACTION: stop\nREASON: context present"
        return "ACTION: retrieve\nQUERY: " + self._extract_question(prompt)

    def _next_hop_query(self, q: str) -> str:
        if "who founded" in q and "company" in q:
            return "founder of the company that created the programming language"
        if "capital of the country" in q:
            return "country where the river originates capital"
        return q + " intermediate entity"

    def _reformulate(self, prompt: str) -> str:
        q = self._extract_question(prompt)
        return f"REFORMULATED: {self._next_hop_query(q.lower())}"

    def _answer(self, prompt: str) -> str:
        q = self._extract_question(prompt).lower()
        ctx = self._extract_context(prompt)

        if any(w in q for w in ("unicorn ceo of nasa", "invisible city")):
            return "ABSTAIN: The retrieved context does not support an answer."

        # Extract answer-like spans from context via simple patterns
        answer = self._heuristic_extract(q, ctx)
        cites = self._find_doc_ids(ctx)
        cite_str = ",".join(cites[:3]) if cites else ""
        if answer:
            suffix = f"\nCITATIONS: {cite_str}" if cite_str else ""
            return f"{answer}{suffix}"
        # Fall back: first sentence-like chunk
        sentences = re.split(r"(?<=[.!?])\s+", ctx.strip())
        for s in sentences:
            if len(s) > 20 and not s.lower().startswith("question"):
                suffix = f"\nCITATIONS: {cite_str}" if cite_str else ""
                return f"{s.strip()}{suffix}"
        return "ABSTAIN: Unable to find supporting evidence."

    def _heuristic_extract(self, q: str, ctx: str) -> str | None:
        ctx_l = ctx.lower()
        # Question-gated fixture facts (offline deterministic).
        # Require strong overlap with the QUESTION, then context support.
        patterns: list[tuple[list[str], str]] = [
            (["who created python", "created python", "python"], "Guido van Rossum"),
            (["capital of the country", "creator of python was born", "netherlands capital"], "Amsterdam"),
            (["height of mount everest", "mount everest"], "8,849 meters"),
            (["water boil", "temperature celsius", "boil"], "100"),
            (["dna stand", "does dna"], "Deoxyribonucleic acid"),
            (["moons of mars", "mars"], "Phobos and Deimos"),
            (["photosynthesis", "gas"], "oxygen"),
            (["relativity", "published", "einstein"], "1905"),
            (["founded amazon", "who founded amazon", "amazon"], "Jeff Bezos"),
            (["javascript", "who created"], "Brendan Eich"),
            (["brendan eich", "mozilla", "company"], "members of Netscape"),
            (["waterloo", "napoleon"], "1815"),
            (["compare", "python", "javascript"], "Guido van Rossum and Brendan Eich"),
            (["moons of mars", "photosynthesis"], "Phobos and Deimos; oxygen"),
            (["speed of light"], "299,792,458 meters per second"),
            (["alan turing", "born"], "1912"),
        ]
        best: tuple[int, str] | None = None
        for keys, ans in patterns:
            q_hits = sum(1 for k in keys if k in q)
            if q_hits == 0:
                continue
            # Prefer patterns with more question hits; require answer or key evidence in context
            ctx_ok = ans.lower() in ctx_l or any(k in ctx_l for k in keys if len(k) > 4)
            if not ctx_ok and ans.lower().split(";")[0].strip() not in ctx_l:
                # still allow if distinctive answer tokens appear
                ans_toks = [t for t in re.findall(r"[a-z0-9]+", ans.lower()) if len(t) > 3]
                ctx_ok = any(t in ctx_l for t in ans_toks)
            if not ctx_ok:
                continue
            score = q_hits * 10 + (5 if ans.lower() in ctx_l else 0)
            if best is None or score > best[0]:
                best = (score, ans)
        if best:
            return best[1]
        # Conservative regex fallbacks gated on question intent
        if "born" in q:
            m = re.search(r"born in (\d{4})", ctx_l)
            if m:
                return m.group(1)
        if "capital" in q:
            m = re.search(r"capital (?:city )?is ([A-Z][a-zA-Z\s]+)", ctx)
            if m:
                return m.group(1).strip().rstrip(".")
        if "found" in q:
            m = re.search(r"founded by ([A-Z][a-zA-Z\s]+)", ctx)
            if m:
                return m.group(1).strip().rstrip(".")
        if "creat" in q and "python" in q:
            if "guido" in ctx_l:
                return "Guido van Rossum"
        if "creat" in q and "javascript" in q:
            if "brendan" in ctx_l:
                return "Brendan Eich"
        return None

    def _find_doc_ids(self, ctx: str) -> list[str]:
        return re.findall(r"doc_[a-z0-9_]+", ctx, re.I)
