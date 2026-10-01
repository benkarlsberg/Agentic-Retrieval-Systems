"""RAG top-k control: the configured top_k reaches RAG retrieval and nothing else changes."""

import json
from pathlib import Path

import yaml

from ars.experiment import run_from_config

REPO = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, top_k: int) -> list[dict]:
    raw = yaml.safe_load(
        (REPO / "configs" / "experiments" / "offline_fixture_compare.yaml").read_text(encoding="utf-8")
    )
    raw.update(experiment_id=f"pytest_rag_k{top_k}", architectures=["rag"], top_k=top_k,
               max_examples=3)
    cfg = tmp_path / f"k{top_k}.yaml"
    cfg.write_text(yaml.dump(raw), encoding="utf-8")
    out = tmp_path / f"run_k{top_k}"
    run_from_config(cfg, output_dir=out)
    assert not (out / "results_multi_agent.json").exists()
    return json.loads((out / "results_rag.json").read_text(encoding="utf-8"))


def _retrieval(r: dict) -> tuple[int, list[str]]:
    """(requested top_k, returned doc ids) from the RAG trace."""
    top_k, ids = -1, []
    for line in Path(r["trace_path"]).read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev["event_type"] == "retrieval_query":
            top_k = ev["payload"]["top_k"]
        elif ev["event_type"] == "retrieval_result":
            ids = ev["payload"]["doc_ids"]
    return top_k, ids


def test_rag_top_k_controls_passages_read(tmp_path: Path):
    k2, k4 = _run(tmp_path, 2), _run(tmp_path, 4)
    assert [r["example_id"] for r in k2] == [r["example_id"] for r in k4]
    for a, b in zip(k2, k4, strict=True):
        assert a["retrieval_calls"] == b["retrieval_calls"] == 1
        assert a["model_calls"] == b["model_calls"] == 1
        assert a["docs_retrieved"] == a["unique_docs"] == 2
        assert b["docs_retrieved"] == b["unique_docs"] == 4
        (ka, ids_a), (kb, ids_b) = _retrieval(a), _retrieval(b)
        assert (ka, kb) == (2, 4)
        # A larger k returns a superset: the top-2 passages are the first two of the top-4.
        assert ids_b[:2] == ids_a


def test_live_top_k_control_configs_differ_only_in_top_k():
    base = yaml.safe_load(
        (REPO / "configs/experiments/live_hotpot150_unconstrained.yaml").read_text(encoding="utf-8"))
    for k in (13, 15):
        cfg = yaml.safe_load(
            (REPO / f"configs/experiments/live_hotpot150_rag_top{k}.yaml").read_text(encoding="utf-8"))
        assert cfg["top_k"] == k and cfg["budget"]["top_k"] == k
        assert cfg["architectures"] == ["rag"]
        for key in ("retrieval_mode", "seed", "corpus", "dataset", "model", "comparison_mode"):
            assert cfg[key] == base[key], key
        strip = lambda b: {x: y for x, y in b.items() if x != "top_k"}  # noqa: E731
        assert strip(cfg["budget"]) == strip(base["budget"])
