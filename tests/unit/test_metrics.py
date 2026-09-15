from ars.evaluation.metrics import (
    abstention_correctness,
    citation_precision_recall,
    exact_match,
    normalize_answer,
    recall_at_k,
    token_f1,
)
from ars.evaluation.stats import mean_ci, paired_bootstrap_ci


def test_normalize_and_em():
    assert normalize_answer("The Amsterdam!") == "amsterdam"
    assert exact_match("Amsterdam", "Amsterdam") == 1.0
    assert exact_match("Amsterdam, Netherlands", "Amsterdam", ["Amsterdam"]) == 1.0


def test_token_f1():
    assert token_f1("Guido van Rossum", "Guido van Rossum") == 1.0
    assert 0.0 < token_f1("Guido Rossum", "Guido van Rossum") < 1.0


def test_retrieval_metrics():
    assert recall_at_k(["a", "b", "c"], ["b", "d"], k=2) == 0.5
    assert citation_precision_recall(["a", "b"], ["b"])[0] == 0.5


def test_abstention():
    assert abstention_correctness(True, False) == 1.0
    assert abstention_correctness(False, False) == 0.0
    assert abstention_correctness(False, True) == 1.0


def test_bootstrap():
    ci = mean_ci([0.1, 0.2, 0.3], n_bootstrap=200)
    assert ci.low <= ci.mean <= ci.high
    pci = paired_bootstrap_ci([0.5, 0.6], [0.4, 0.5], n_bootstrap=200)
    assert pci.mean > 0
