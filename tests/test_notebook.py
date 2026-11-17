import numpy as np

from exlightnet.io import write_json
from exlightnet.metrics import score_predictions
from exlightnet.notebook import show_task


def test_missing_seed42_is_explicit_and_fallback_is_labelled(tmp_path, capsys):
    row = score_predictions(np.array([0, 1]), np.array([[.9, .1], [.1, .9]]), ["normal", "attack"])
    row.update(task="binary", model="exlightnet", seed=43)
    write_json(tmp_path / "results/full/results.json", [row])
    show_task(tmp_path, "full", "binary")
    output = capsys.readouterr().out.lower()
    assert "seed 42 unavailable" in output
    assert "seed 43" in output
