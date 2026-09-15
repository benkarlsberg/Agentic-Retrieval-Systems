from pathlib import Path

import pytest

from ars.corpora.loader import load_corpus
from ars.datasets.adapters import HotpotQAAdapter
from ars.datasets.loader import load_dataset
from ars.schema import ExampleType


def test_fixtures_load():
    docs = load_corpus()
    examples = load_dataset()
    assert len(docs) >= 20
    assert len(examples) >= 10
    types = {e.example_type for e in examples}
    assert ExampleType.SINGLE_HOP in types
    assert ExampleType.MULTI_HOP in types
    assert ExampleType.UNANSWERABLE in types
    assert ExampleType.AMBIGUOUS in types
    assert ExampleType.MULTI_PASSAGE in types


def test_adapter_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        HotpotQAAdapter().load(tmp_path / "missing.json")
