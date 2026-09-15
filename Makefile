.PHONY: install test lint typecheck ci experiment report clean

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src tests scripts

typecheck:
	$(PYTHON) -m mypy src/ars

ci: lint typecheck test

experiment:
	$(PYTHON) scripts/run_experiment.py --config configs/experiments/offline_fixture_compare.yaml

report:
	$(PYTHON) scripts/generate_report.py --run-dir results/runs/latest

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
