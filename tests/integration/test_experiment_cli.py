import json
from pathlib import Path

from ars.experiment import run_from_config

REPO = Path(__file__).resolve().parents[2]


def test_offline_experiment_smoke(tmp_path: Path):
    cfg = REPO / "configs" / "experiments" / "offline_fixture_compare.yaml"
    # Limit via a tiny copied config
    import yaml

    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    raw["experiment_id"] = "pytest_smoke"
    raw["max_examples"] = 3
    raw["architectures"] = ["rag", "single_agent", "multi_agent"]
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(raw), encoding="utf-8")
    out = tmp_path / "run"
    meta = run_from_config(p, output_dir=out)
    assert meta["fixture_offline"] is True
    assert (out / "comparison.json").exists()
    assert (out / "results_rag.json").exists()
    data = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert len(data) == 3
