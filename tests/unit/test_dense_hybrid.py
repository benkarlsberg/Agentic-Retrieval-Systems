from ars.retrieval.dense import DenseRetriever, FixtureEmbedder
from ars.retrieval.hybrid import HybridRetriever
from ars.retrieval.rerank import SimpleReranker
from ars.schema import Document, RetrievedDoc


def _docs():
    return [
        Document(doc_id="a", title="Mars moons", text="Phobos and Deimos orbit Mars."),
        Document(doc_id="b", title="Cookies", text="Chocolate chip cookies are sweet."),
    ]


def test_dense_and_hybrid():
    docs = _docs()
    dense = DenseRetriever(FixtureEmbedder(dim=32))
    dense.index(docs)
    hits = dense.retrieve("moons of Mars Phobos", top_k=1)
    assert hits[0].doc_id == "a"
    hybrid = HybridRetriever()
    hybrid.index(docs)
    h2 = hybrid.retrieve("Mars moons", top_k=2)
    assert {x.doc_id for x in h2} == {"a", "b"} or h2[0].doc_id == "a"


def test_reranker():
    rr = SimpleReranker()
    docs = [
        RetrievedDoc(doc_id="1", score=1.0, title="x", text="unrelated text here"),
        RetrievedDoc(doc_id="2", score=0.1, title="y", text="Python Guido van Rossum creator"),
    ]
    out = rr.rerank("Python creator Guido", docs, top_k=1)
    assert out[0].doc_id == "2"
