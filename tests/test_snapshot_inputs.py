"""jenga.model.snapshot_inputs: the prompt and norm stats a snapshot is served and evaluated with (CPU, no weights)."""
import json

import pytest

from jenga.model import norm_stats_digest, snapshot_inputs


def norm_stats(directory, value=1.0):
    (directory / "norm_stats").mkdir(parents=True)
    (directory / "norm_stats/norm_stats.json").write_text(json.dumps({"norm_stats": {"state": {"mean": [value]}}}))
    return directory


def test_release_layout_reads_norm_stats_and_run_json_beside_params(tmp_path):
    model = norm_stats(tmp_path / "model")
    (model / "params").mkdir()
    run = {"args": {"prep_dir": "/elsewhere/runs/prep"}, "prompt": "official",
           "norm_stats_sha256": norm_stats_digest(model / "norm_stats")}
    (model / "run.json").write_text(json.dumps(run))
    (tmp_path / "run.json").write_text(json.dumps({"prompt": "train"}))  # a parent run.json is not consulted
    assert snapshot_inputs(model) == ("official", model)


def test_release_layout_without_run_json_defaults_to_official(tmp_path):
    model = norm_stats(tmp_path / "model")
    assert snapshot_inputs(model) == ("official", model)


def test_release_layout_rejects_norm_stats_that_differ_from_training(tmp_path):
    model = norm_stats(tmp_path / "model")
    (model / "run.json").write_text(json.dumps({"norm_stats_sha256": "sha256:" + "0" * 64}))
    with pytest.raises(ValueError, match="differs"):
        snapshot_inputs(model)


def test_run_layout_follows_prep_dir_of_run_json(tmp_path):
    prep = norm_stats(tmp_path / "prep")
    run_dir = tmp_path / "runs/v3"
    (run_dir / "params_28000/params").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "args": {"prep_dir": str(prep)}, "prompt": "official", "norm_stats_sha256": norm_stats_digest(prep / "norm_stats")}))
    assert snapshot_inputs(run_dir / "params_28000") == ("official", prep)
    (run_dir / "run.json").write_text(json.dumps({"args": {"prep_dir": str(prep)}, "norm_stats_sha256": "sha256:0"}))
    with pytest.raises(ValueError, match="differs"):
        snapshot_inputs(run_dir / "params_28000")


def test_no_layout_fails_instead_of_falling_back(tmp_path):
    (tmp_path / "snapshot/params").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no norm stats"):
        snapshot_inputs(tmp_path / "snapshot")
    run_dir = tmp_path / "runs/x"
    (run_dir / "params_10").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"args": {"prep_dir": str(tmp_path / "gone")}}))
    if not (snapshot_inputs.__globals__["ROOT"] / "runs/prep/norm_stats/norm_stats.json").exists():
        with pytest.raises(FileNotFoundError, match="no norm_stats"):
            snapshot_inputs(run_dir / "params_10")
