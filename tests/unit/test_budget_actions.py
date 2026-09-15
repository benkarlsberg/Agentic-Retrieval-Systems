from ars.routing.actions import ActionType, parse_action
from ars.routing.budget import BudgetTracker
from ars.schema import Budget, StopReason


def test_parse_action():
    a = parse_action("ACTION: retrieve\nQUERY: python creator\nREASON: need facts")
    assert a.type == ActionType.RETRIEVE
    assert a.query and "python" in a.query.lower()


def test_budget_stops():
    b = BudgetTracker(Budget(max_steps=2, max_model_calls=10, max_docs=100, max_tokens=99999, max_retrieval_calls=5))
    b.register_step()
    assert b.check() is None
    b.register_step()
    assert b.check() == StopReason.BUDGET_STEPS
