from ars.retrieval.bm25 import BM25Retriever
from ars.schema import Document


def test_bm25_ranks_relevant_first():
    docs = [
        Document(doc_id="d1", title="Cats", text="Cats are animals that meow."),
        Document(doc_id="d2", title="Python", text="Python was created by Guido van Rossum."),
        Document(doc_id="d3", title="Baking", text="Baking soda is used in cookies."),
    ]
    r = BM25Retriever()
    r.index(docs)
    hits = r.retrieve("Who created Python Guido", top_k=2)
    assert hits[0].doc_id == "d2"
