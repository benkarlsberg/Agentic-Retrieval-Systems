from ars.models.base import GenerationRequest
from ars.models.cost import CostEstimator
from ars.models.mock import DeterministicModel


def test_mock_answers_from_context():
    m = DeterministicModel()
    resp = m.generate(
        GenerationRequest(
            prompt="Question: Who created Python?\nContext:\n[doc_python] title: Python\nPython was created by Guido van Rossum.\n",
            metadata={"task": "answer"},
        )
    )
    assert "Guido" in resp.text


def test_mock_abstain():
    m = DeterministicModel()
    resp = m.generate(
        GenerationRequest(
            prompt="Question: Who is the unicorn CEO of NASA?\nContext:\n(empty)\n",
            metadata={"task": "answer"},
        )
    )
    assert "ABSTAIN" in resp.text.upper()


def test_cost_zero_for_mock():
    c = CostEstimator()
    assert c.estimate("deterministic-v1", 1000, 1000) == 0.0
