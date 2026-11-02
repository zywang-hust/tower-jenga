# Jenga / TOWER: contributor and agent notes

The code of a π0.5 (OpenPI, JAX) policy for the TOWER bimanual Jenga challenge: data preparation, training, the
`tower-infer-v1` policy server and a local closed-loop simulator. Setup, results and the full recipe are in README.md
and docs/.

## Rules

- Changes to OpenPI go into `patches/openpi-tower.patch` (applied by `scripts/bootstrap.sh` to the pinned
  commit), each with a targeted regression test in `tests/`.
- Do not edit files that a running job imports: DataLoader workers re-import `src/jenga` when they start, and the
  closed-loop drivers and Isaac workers import `sim/`.
- Weights, data, logs, generated environments, `runs/` and `external/` are never committed.
- Serving behaviour is pinned by the artifact identity the server prints (weights, norm stats and serving config)
  and by `PLUGIN_VERSION` (hash of `server.py`, `policy.py`, `transforms.py`, `stitching.py`). Any change to those
  files changes what the policy does online.

## Layout

- `src/jenga/` library code; `scripts/` thin entry points; `sim/` the local closed-loop simulator (environments
  in README); `tests/` CPU regression tests.
- `src/jenga/paths.py` holds every location outside the repository, each with an environment override:
  `JENGA_DATA_ROOT` (TOWER dataset, default `data/tower`), `JENGA_MODELS` (`pi05_base`, default `models/`),
  `JENGA_CACHE`, `CUDA_HOME`, `OPENPI_DATA_HOME`. Shell entry points source `$JENGA_ENV` first when it is set
  (site setup); `sim/isaac.sh` needs `ISAAC_PYTHON`.
- `configs/`: `track_local.json` is a placeholder track, `track_20261013-v1-task45-30hz.json` the official track
  used for the final evaluation; the `layouts` sections of `nero_kinematics.json` and `cameras.json` are generated
  by `scripts/fit_kinematics.py`, the camera section of `cameras.json` is static. `start_qpos.json` and
  `orient_reference.npz` are serving inputs.
- `external/` (gitignored): pinned OpenPI and the official starter (`scripts/fetch_starter.py`), both installed by
  `scripts/bootstrap.sh`.
- `runs/` (gitignored) layout:
  - `prep/` (split and norm stats), `<tag>/` (`run.json`, `train.jsonl`, `params_<step>` bf16 EMA exports,
    `synthetic_files.txt`), `checkpoints/tower/<tag>/<step>` (full train states, needed for `--init-from`)
  - `eval/<tag>/`, `probe/<tag>/` offline results, tag `<run>-<step>`
  - `simdev/`: simulator assets, refinement nets, rollouts and closed-loop results (`fleet/<tag>_v2/`)
  - `serve_records/`: diagnostic records of the policy server (local servers write elsewhere with `--records`)

## Commands

```bash
scripts/bootstrap.sh
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/ruff check --no-cache . --exclude external
```

Training, evaluation, serving, closed loop and data generation commands are in README.md.

## Correctness boundaries

- The competition protocol is `tower-infer-v1`; the old GitHub starter describes a different protocol.
- Physical state and actions are 16-D: per arm 7 joints and a gripper. The model pads to 32-D.
- Online actions are absolute joint radians plus the gripper opening width in metres (fully open 0.1). Online
  code must not depend on hidden simulator fields.
- All four camera images go to the model.
- The model config is built only in `jenga.model.model_config`; weights load only through `jenga.model.load_policy`;
  prompt and norm stats come only from `jenga.model.snapshot_inputs`.
- Split by episode, never by frame. Offline action error is not closed-loop task success.
- Policy state and RNG are isolated per session.
- Validation episodes never enter training data, including DART and DAgger data: synthetic data is generated from
  training demos only; `val_all.txt` is used only for closed-loop evaluation.
- Server records are for diagnosis only, never for training or tuning.
- The Isaac physics worker runs CPU physics only and must not see a GPU: drivers clear `CUDA_VISIBLE_DEVICES` and
  mask the Vulkan ICD when they start it (`sim/physics_client.py`). With `sim/isaac.sh` by hand, `GPU=` restricts
  CUDA only; Vulkan still sees every GPU.
- Compare closed-loop results only under the same renderer version (directory suffix `_v2`): swapping refine v1
  for v2 alone doubled the LHPR of the same snapshot.
- `sim/score.py` LHPR/TSR/ODR approximate the official metrics and are only for comparing runs. Closed-loop
  episodes are 210 s and must not be shortened.
- Concurrent environments plus variants per GPU must stay within the server's 64 sessions (48 environments with
  the 9 test variants).
