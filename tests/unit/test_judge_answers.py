"""LLM-judge script: reply parsing, retries, blinding and caching (no network)."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "judge_answers", Path(__file__).resolve().parents[2] / "scripts" / "judge_answers.py"
)
judge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(judge)

EXAMPLES = {
    "q1": {"example_id": "q1", "question": "Who wrote X?", "gold_answer": "Jane Doe",
           "aliases": [], "is_answerable": True, "example_type": "multi_hop"},
    "q2": {"example_id": "q2", "question": "Capital of Atlantis?", "gold_answer": None,
           "aliases": [], "is_answerable": False, "example_type": "unanswerable"},
}


def _reply(verdict="correct", model="gpt-4o-mini-2024-07-18"):
    return {"text": json.dumps({"reason": "r", "verdict": verdict}),
            "response_model": model, "prompt_tokens": 100, "completion_tokens": 10}


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    for arch in judge.ARCHS:
        rows = [{"example_id": "q1", "answer": f"It was Jane Doe ({arch})."},
                {"example_id": "q2", "answer": "ABSTAIN."}]
        (run / f"results_{arch}.json").write_text(json.dumps(rows), encoding="utf-8")
    return run


def test_parse_verdict_accepts_schema_and_fences():
    assert judge.parse_verdict('{"verdict": "Correct", "reason": "ok"}') == {
        "verdict": "correct", "reason": "ok"}
    fenced = '```json\n{"reason": "x", "verdict": "incorrect"}\n```'
    assert judge.parse_verdict(fenced)["verdict"] == "incorrect"


@pytest.mark.parametrize("text", [
    "correct", '{"verdict": "partially correct", "reason": ""}', '["correct"]',
    '{"verdict": "correct", "reason": 3}', "",
])
def test_parse_verdict_rejects_malformed(text):
    with pytest.raises(ValueError):
        judge.parse_verdict(text)


def test_judge_one_retries_malformed_reply_and_counts_all_tokens():
    replies = iter([{"text": "not json", "response_model": "m", "prompt_tokens": 5,
                     "completion_tokens": 1}, _reply("incorrect")])
    job = {"question": "Q", "gold": ["A"], "answer": "B", "key": "k"}
    out = judge.judge_one(job, lambda _m: next(replies), sleep=lambda _s: None)
    assert out["verdict"] == "incorrect" and out["attempts"] == 2
    assert out["prompt_tokens"] == 105 and out["response_model"] == "gpt-4o-mini-2024-07-18"


def test_messages_are_blind_and_use_the_right_rubric():
    ans = judge.build_messages("Who wrote X?", ["Jane Doe"], "Jane Doe.")
    una = judge.build_messages("Capital of Atlantis?", None, "ABSTAIN.")
    assert ans[0]["content"] == judge.SYSTEM_PROMPT_ANSWERABLE
    assert una[0]["content"] == judge.SYSTEM_PROMPT_UNANSWERABLE
    text = json.dumps(ans + una).lower()
    for arch in ("rag", "single_agent", "multi_agent", "architecture"):
        assert arch not in text


def test_run_judge_caches_results_and_records_failures(tmp_path):
    run = _make_run(tmp_path)
    calls = []

    def fake(messages):
        calls.append(messages)
        return _reply()

    judge.run_judge([run], fake, workers=2, examples=EXAMPLES)
    assert len(calls) == 6
    cache = json.loads((run / "judge_rag.json").read_text(encoding="utf-8"))
    assert cache["meta"]["n_items"] == 2 and cache["meta"]["n_failed"] == 0
    assert cache["meta"]["response_models"] == ["gpt-4o-mini-2024-07-18"]
    assert cache["items"]["q1"]["prompt_version"] == judge.PROMPT_VERSION

    judge.run_judge([run], fake, workers=2, examples=EXAMPLES)  # fully cached
    assert len(calls) == 6

    # A changed answer invalidates only that item; a failing call is recorded, not dropped.
    rows = json.loads((run / "results_rag.json").read_text(encoding="utf-8"))
    rows[0]["answer"] = "Someone else."
    (run / "results_rag.json").write_text(json.dumps(rows), encoding="utf-8")

    def broken(_messages):
        raise RuntimeError("503")

    res = judge.run_judge([run], broken, workers=1, examples=EXAMPLES, max_attempts=2,
                          sleep=lambda _s: None)
    assert res == {"total": 6, "todo": 1, "failed": 1}
    cache = json.loads((run / "judge_rag.json").read_text(encoding="utf-8"))
    assert cache["items"]["q1"]["verdict"] is None
    assert "503" in cache["items"]["q1"]["error"]
