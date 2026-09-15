from pathlib import Path

from ars.tracing.events import TraceEventType, hash_text
from ars.tracing.tracer import Tracer


def test_tracer_jsonl(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    tr = Tracer("exp", "ex1", "rag", p)
    tr.emit(TraceEventType.RUN_START, payload={"ok": True}, prompt="hello")
    assert p.exists()
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert hash_text("hello")
