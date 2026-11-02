"""Identified Nero kinematics: FK and base -> world against the recorded data, IK round trip."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jenga.kinematics import ARMS, DEFAULT_CONFIG, load_arms, matrix_to_rotvec, quat_xyzw_to_matrix
from jenga.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
DATA = DATA_ROOT  # env JENGA_DATA_ROOT
SPLIT = ROOT / "runs/prep/split.json"
needs_config = pytest.mark.skipif(not DEFAULT_CONFIG.exists(), reason="configs/nero_kinematics.json missing")
needs_data = pytest.mark.skipif(not (DATA / "MANIFEST.jsonl").exists() or not SPLIT.exists(),
                                reason="needs the local TOWER release and runs/prep")


@needs_config
def test_ik_round_trip():
    """Converges to the pose (which also exercises the Jacobian) and keeps the redundancy near the reference."""
    rng = np.random.default_rng(1)
    for arm in load_arms().values():
        q = rng.uniform(arm.q_lo, arm.q_hi, (64, 7))
        R, p = arm.fk_world(q)
        q2, ep, er = arm.ik_world(R, p, q + rng.normal(0, 0.05, q.shape), q)
        assert ep.max() < 1e-6 and er.max() < 1e-5
        assert np.abs(q2 - q).max() < 0.05


@needs_config
@needs_data
def test_fk_and_base_against_dataset():
    split = json.loads(SPLIT.read_text())
    rel = next(t["val"][0] for k, t in split["tasks"].items() if k.startswith("task1-3"))  # both arms pick
    arms = load_arms()
    with h5py.File(DATA / rel, "r") as f:
        q = f["observations/qpos"][::40].astype(float)
        eef = f["observations/eef"][::40].astype(float)
        names = json.loads(f["sim"].attrs["phase_names"])
        ph = f["sim/task_phase"][:]
        ab = f["sim/active_block"][:]
        tm = {m["block_index"]: m["arm"] for m in json.loads(f["sim"].attrs["target_mapping"])}
        lift = np.flatnonzero((ph == names.index("lift")) & np.isin(ab, list(tm)))
        poses = f["sim/block_poses"][lift]
        qlift = f["observations/qpos"][lift].astype(float)
    for a, name in enumerate(ARMS):
        R, p = arms[name].chain.fk(q[:, 8 * a:8 * a + 7])
        assert np.abs(p - eef[:, 7 * a:7 * a + 3]).max() < 5e-5
        Rm = quat_xyzw_to_matrix(eef[:, 7 * a + 3:7 * a + 7])
        assert np.degrees(np.linalg.norm(matrix_to_rotvec(Rm @ np.swapaxes(R, 1, 2)), axis=1)).max() < 0.05
    # held block centre = base * (eef + R tool) during 'lift'
    errs = []
    for k, i in enumerate(lift):
        a = tm[int(ab[i])]
        arm = arms[ARMS[a]]
        R, p = arm.fk_world(qlift[k, 8 * a:8 * a + 7][None])
        errs.append(np.linalg.norm(arm.tool_world(R, p)[0] - poses[k, ab[i], :3]))
    assert len({tm[int(ab[i])] for i in lift}) == 2 and np.median(errs) < 0.003
