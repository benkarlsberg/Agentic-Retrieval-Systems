from __future__ import annotations

import html
import json
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ARS Trace Viewer</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f1115; color: #e6e6e6; }}
h1 {{ font-size: 1.2rem; }}
.meta {{ color: #9aa; margin-bottom: 1rem; }}
.event {{ border: 1px solid #333; border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0; background: #171a21; }}
.etype {{ color: #7cb7ff; font-weight: 600; }}
.role {{ color: #f0a; }}
pre {{ white-space: pre-wrap; word-break: break-word; font-size: 0.85rem; }}
</style></head><body>
<h1>ARS Trace Viewer</h1>
<div class="meta">Offline / fixture traces — not live production logs.</div>
<div class="meta">{meta}</div>
{body}
</body></html>
"""


def write_trace_html(trace_path: Path, out_path: Path | None = None) -> Path:
    events = []
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    out = out_path or trace_path.with_suffix(".html")
    parts = []
    for e in events:
        payload = html.escape(json.dumps(e.get("payload", {}), indent=2))
        parts.append(
            f'<div class="event"><div><span class="etype">{html.escape(e.get("event_type",""))}</span> '
            f'step={e.get("step")} <span class="role">{html.escape(str(e.get("agent_role")))}</span></div>'
            f'<pre>{payload}</pre></div>'
        )
    meta = f"file={html.escape(str(trace_path))} events={len(events)}"
    out.write_text(HTML_TEMPLATE.format(meta=meta, body="\n".join(parts)), encoding="utf-8")
    return out


def write_trace_json_bundle(trace_paths: list[Path], out_path: Path) -> Path:
    bundle = []
    for p in trace_paths:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            bundle.append({"path": str(p), "events": [json.loads(line) for line in f if line.strip()]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return out_path
