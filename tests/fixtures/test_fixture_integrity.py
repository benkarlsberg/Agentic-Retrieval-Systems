import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_corpus_jsonl_valid():
    path = REPO / "fixtures" / "corpus" / "mini_wiki.jsonl"
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        assert "doc_id" in obj and "text" in obj
        ids.append(obj["doc_id"])
    assert len(ids) == len(set(ids))


def test_dataset_gold_docs_exist():
    corpus_ids = {
        json.loads(line)["doc_id"]
        for line in (REPO / "fixtures" / "corpus" / "mini_wiki.jsonl").read_text().splitlines()
        if line.strip()
    }
    for line in (REPO / "fixtures" / "datasets" / "mini_eval.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        for d in obj.get("gold_doc_ids", []):
            assert d in corpus_ids, f"missing {d}"
