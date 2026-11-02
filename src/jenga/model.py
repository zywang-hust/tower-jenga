"""The one place that builds the OpenPI model config, so training, evaluation, probes and serving agree."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jenga.dataset import MAX_TOKEN_LEN
from jenga.hdf5_dataset import MODEL_CAMERAS


ROOT = Path(__file__).resolve().parents[2]


def norm_stats_digest(norm_stats_dir: Path) -> str:
    return "sha256:" + hashlib.sha256((Path(norm_stats_dir) / "norm_stats.json").read_bytes()).hexdigest()


def snapshot_inputs(params_dir: Path) -> tuple[str, Path]:
    """(prompt, prep dir with norm_stats/ and, in a run, split.json) the snapshot was trained with, so evaluation,
    probes and serving feed the inputs it was trained on. Two layouts:
      release  <params_dir>/{params/, norm_stats/norm_stats.json, run.json (optional)}: prep dir is <params_dir>
      run      <run>/{run.json, params_<step>/params/}: prep dir from run.json args.prep_dir (default runs/prep)
    Prompt from run.json, default 'official'. Fails if neither layout is found, or if the norm stats differ from
    the ones run.json recorded at training time."""
    params_dir = Path(params_dir)
    if (params_dir / "norm_stats" / "norm_stats.json").exists():  # release layout: everything beside params/
        run_json = params_dir / "run.json"
        run = json.loads(run_json.read_text()) if run_json.exists() else {}
        trained = run.get("args", {})
        prep = params_dir
    else:
        run_json = params_dir.resolve().parent / "run.json"
        if not run_json.exists():
            raise FileNotFoundError(f"no norm stats for {params_dir}: neither {params_dir / 'norm_stats/norm_stats.json'} "
                                    f"(release layout) nor {run_json} (run layout)")
        run = json.loads(run_json.read_text())
        trained = run.get("args", {})
        prep = Path(trained.get("prep_dir", ROOT / "runs/prep"))
        prep = prep if prep.is_absolute() else ROOT / prep
        if not (prep / "norm_stats").exists():  # runs before the runs/ layout change recorded runs/r1
            prep = ROOT / "runs/prep"
        if not (prep / "norm_stats" / "norm_stats.json").exists():
            raise FileNotFoundError(f"{run_json} names prep dir {trained.get('prep_dir')}, which has no norm_stats/")
    recorded = run.get("norm_stats_sha256")
    if recorded is not None and recorded != norm_stats_digest(prep / "norm_stats"):
        raise ValueError(f"{prep / 'norm_stats'} differs from the norm stats {run_json} was trained with")
    return run.get("prompt", trained.get("prompt", "official")), prep


def model_config(*, dtype: str = "bfloat16"):
    """pi05 with the four TOWER cameras; the external views keep their geometry (fixed cameras: object position
    cues), only colour is jittered in training (docs/recipe.md §4)."""
    from openpi.models.pi0_config import Pi0Config

    return Pi0Config(pi05=True, image_keys=MODEL_CAMERAS, action_dim=32, action_horizon=32,
                     max_token_len=MAX_TOKEN_LEN, image_geometry_augment=False, dtype=dtype)


def load_policy(params_dir: Path, device=None, *, weights: str = "bfloat16", compute: str = "bfloat16"):
    """Snapshot dir with params/ -> eval-mode model.

    weights: dtype the params are cast to (float32 needs an fp32 source: a training checkpoint step dir, whose
    params/ item is the EMA; exported params_<step> are already bf16). compute: Pi0Config.dtype, the dtype of
    Gemma embeddings and SigLIP matmuls; the small projections run in the weights' dtype."""
    import jax
    import jax.numpy as jnp
    import openpi.models.model as _model

    params = _model.restore_params(Path(params_dir) / "params", restore_type=jax.Array, dtype=jnp.dtype(weights))
    if device is not None:
        params = jax.device_put(params, device)
    if compute == "float32":
        import openpi.models.siglip as siglip

        siglip.ATTENTION_IMPL = None  # the fused cuDNN attention is half precision only
    policy = model_config(dtype=compute).load(params)
    policy.eval()
    return policy
