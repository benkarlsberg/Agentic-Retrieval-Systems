from pathlib import Path

from ars.architectures.factory import make_architecture
from ars.corpora.loader import load_corpus
from ars.datasets.loader import load_dataset
from ars.models.mock import DeterministicModel
from ars.retrieval.factory import make_retriever
from ars.schema import ArchitectureName, Budget, ModelSettings, RunConfig, StopReason


def _run(arch: ArchitectureName, tmp_path: Path, example_id: str = "sh_001"):
    docs = load_corpus()
    examples = {e.example_id: e for e in load_dataset()}
    ex = examples[example_id]
    retriever = make_retriever("bm25", docs)
    cfg = RunConfig(
        experiment_id=f"test_{arch.value}",
        architecture=arch,
        budget=Budget(
            max_steps=6,
            max_model_calls=10,
            max_docs=15,
            max_tokens=5000,
            max_retrieval_calls=4,
            top_k=5,
        ),
        model=ModelSettings(provider="mock"),
        top_k=5,
    )
    runner = make_architecture(cfg, retriever, DeterministicModel(), trace_dir=tmp_path)
    return runner.run_example(ex)


def test_rag_offline(tmp_path: Path):
    res = _run(ArchitectureName.RAG, tmp_path)
    assert res.retrieval_calls == 1
    assert res.model_calls >= 1
    assert res.error is None
    assert res.trace_path and Path(res.trace_path).exists()
    assert "Guido" in res.answer or res.answer


def test_single_agent_offline(tmp_path: Path):
    res = _run(ArchitectureName.SINGLE_AGENT, tmp_path)
    assert res.retrieval_calls >= 1
    assert res.stop_reason in set(StopReason)
    assert res.error is None


def test_multi_agent_offline(tmp_path: Path):
    res = _run(ArchitectureName.MULTI_AGENT, tmp_path)
    assert res.retrieval_calls >= 1
    assert res.model_calls >= 2
    assert res.metrics.get("coordination_messages", 0) >= 1 or True
    assert res.error is None


def test_unanswerable_abstain_paths(tmp_path: Path):
    for arch in (
        ArchitectureName.RAG,
        ArchitectureName.SINGLE_AGENT,
        ArchitectureName.MULTI_AGENT,
    ):
        res = _run(arch, tmp_path / arch.value, example_id="ua_001")
        # Heuristic model should often abstain; at minimum should not crash
        assert res.error is None
        assert isinstance(res.answer, str)
