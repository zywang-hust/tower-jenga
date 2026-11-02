# Reproducing training

This document describes how the released checkpoint was trained. Neither the synthetic data nor any intermediate
checkpoint is released, so reproduction means regenerating the data. Section 1 lists what you need to obtain.
Sections 2 to 5 follow the order of the pipeline. The reasons behind the choices are in [method.md](method.md).

The commands are reconstructed from what the training and generation runs recorded: training arguments in
`run.json`, and seeds, perturbation kind, replanning interval and per-rollout noise in each generation batch's
`summary.json` and result files. The full command lines of the generation batches were not logged. The options
given for them below reproduce the recorded settings and episode counts, but we have not rerun them end to end.

## 1. What you need

| Item | Source | Used for |
|---|---|---|
| Tower-SimData v1.0.0 (531 episodes, about 1.75 M frames; our copy's `MANIFEST.jsonl` matches the published one byte for byte) | [Hugging Face](https://huggingface.co/datasets/tower-benchmark/TOWER-SimData), CC BY 4.0 | split, normalization statistics, initial layouts and goal mappings for generation, kinematics fit, refinement-network training pairs |
| TOWER-Assets (station USDZ, block and arm assets) | [Hugging Face](https://huggingface.co/datasets/tower-benchmark/TOWER-Assets), CC BY 4.0 | local physics and rendering |
| π0.5 base checkpoint | `gs://openpi-assets/checkpoints/pi05_base` (OpenPI). `scripts/download_pi05.py` verifies a mirrored copy against the official checksums | initialization |
| OpenPI at commit `215abfb` with `patches/openpi-tower.patch` | installed by `scripts/bootstrap.sh` | model and training step |
| Isaac Sim 5.1 in its own Python 3.11 environment | NVIDIA | physics for expert rollouts and closed-loop evaluation (CPU PhysX) |
| A renderer environment (Python 3.12 with PyTorch, nvdiffrast) | see the docstrings of `sim/render.py` and `sim/expert_write.py` | rasterizer, refinement network, data writing |
| GPUs | we used NVIDIA H800 80 GB | training needs 8 GPUs at batch 64 as configured; rendering uses one process per GPU |
| CPU cores | we used one node with 115 usable cores | expert rollouts run on CPU PhysX |
| Disk | | the full-frame synthetic set is about 2 TiB (d2 alone is 1.8 TiB; about 500 MB per episode); a full training state is about 42 GiB |

Set `JENGA_DATA_ROOT` to the Tower-SimData directory (the one containing `MANIFEST.jsonl`) and `JENGA_MODELS` to a
directory that contains `pi05_base/`. `ISAAC_PYTHON` must point to the Python of the Isaac Sim environment; Isaac
processes are started through `sim/isaac.sh`. The station USDZ goes where `sim/render.py` expects it (`STATION`).

## 2. Split and normalization statistics

```bash
source .venv/bin/activate
python scripts/audit.py   "$JENGA_DATA_ROOT"     # data checks, writes runs/audit/
python scripts/prepare.py "$JENGA_DATA_ROOT"     # runs/prep/split.json, runs/prep/norm_stats/
```

`prepare.py` holds out 10% of the episodes in each variant (seed 42), which gives 475 training and 56 validation
episodes. The normalization statistics are computed on the training episodes only, in model space (joint deltas
over every 32-step window, gripper closure). The released weights ship with these statistics, and serving checks
their SHA-256 against the value recorded at training time.

## 3. Renderer refinement network

Synthetic episodes are rendered by the rasterizer in `sim/render.py`. A residual U-Net (`sim/refine.py`) then
corrects the raster toward the official ray-traced appearance. It is trained on pairs of a raster image and the
recorded official JPEG of the same dataset state, using training-split frames only:

```bash
CUDA_VISIBLE_DEVICES=0 <render-venv>/bin/python sim/train_refine.py <out_dir> \
    --steps 30000 --crop 0 --widths 32,64,128,256,256 --lap <w> --grip <w>
```

The network we used (v2) has five levels with widths 32, 64, 128, 256 and 256. It was trained on full 640×480
frames with an L1 loss plus a Laplacian-pyramid L1 term, with the wrist cameras' gripper region weighted ×2, for
30,000 steps. The exact values of the two loss weights were not recorded with the checkpoint. On fixed validation
states it reaches a full-frame PSNR of 34.8 to 36.6 dB per camera, against 17.3 to 24.5 dB for the bare raster.
The more relevant measure is how the policy responds: [method.md](method.md#3-local-simulator) gives the median
action difference against recorded frames (0.20 mrad, with 0.19 mrad from seed noise alone).
`sim/train_refine.py --eval-only <refine.pt>` reports the image metrics, and `sim/policy_gap.py` reports the
policy-gap measurement.

## 4. Synthetic data

### 4.1 Expert rollouts

The privileged expert runs in closed loop against the local physics, starting from the layouts of training
episodes. `sim/expert_pool.py` keeps a pool of persistent Isaac workers and accepts the options of
`sim/expert_batch.py`. Each batch writes one result per rollout and a `summary.json`, and finishes with
`EXPERT_BATCH_DONE`.

```bash
# template: one batch directory per seed
.venv/bin/python sim/expert_pool.py <work>/<batch> --workers 40 \
    --perturb d2 --seed <seed> --k <k> --episodes 999 --repeats <r> --noise-max <rad> [--variants ...] \
    [--start <i>] [--noise-place <rad>] [--noise-insert <rad>]
```

`--perturb d2` draws a rigid layout offset of 0 to 25 mm in a random direction and up to ±5° of yaw. On tasks 1 to
4 it also jitters each layer independently by up to 8 mm and 5°, and on task5-2 it enables the fidelity physics
by itself. For task5-3 the fidelity physics must be set for the whole batch with `JENGA_PHYS_FIDELITY=1` (the
worker logs of our task5-3 batches s5 and s10 confirm it was set). Without it, the destination tower is frozen,
and we removed the frozen-tower task5-3 rollouts of s0 and s1 from the data. `--noise-max`
injects DART-style joint noise during transit segments, with a per-episode standard deviation drawn uniformly from
zero to that value. `--noise-place` and `--noise-insert` extend the noise into the placement and pre-insertion
segments respectively.

The batches behind the released checkpoint:

| Batch | Seed | Variants | `--repeats` | `--k` | Noise (rad) | Rollouts | Kept in the final data |
|---|---|---|---|---|---|---|---|
| d2 s0 | 200 | all | 2 | 8 | `--noise-max 0.012` | 950 | 839 (task5-3 removed) |
| d2 s1 | 201 | all | 4 | 8 | `--noise-max 0.012` | 1,900 | 1,672 (task5-3 removed) |
| d2 s5 | 205 | task5-3, `JENGA_PHYS_FIDELITY=1` | 6 | 8 | `--noise-max 0.012` | 270 | 253 |
| d2 s6 | 206 | task1 to task3, `--start 20 --episodes 30` | 2 | 4 | `--noise-max 0.009 --noise-place 0.006` | 350 | 328 |
| d2 s7 | 207 | task1 to task3 | 3 | 4 | `--noise-max 0.009 --noise-place 0.006` | 885 | 837 |
| d2 dg1 | | task1 to task4 | | 4 | none | 78 | 66 |
| v3 s8 | 208 | task5-2 | 6 | 8 | `--noise-max 0.009 --noise-insert 0.003` | 270 | 243 |
| v3 s9 | 209 | task5-2 | 6 | 8 | `--noise-max 0.009 --noise-insert 0.006` | 270 | 116 |
| v3 s10 | 210 | task5-3, `JENGA_PHYS_FIDELITY=1` | 8 | 8 | `--noise-max 0.012` | 360 | 345 |
| v3 dg52, dg53 | | task5-2, task5-3 | | 8 | none | 12, 24 | 9, 4 |
| **Total** | | | | | | | **4,712** (d2 3,995; v3 717) |

The takeover batches use `--perturb policy --fleet <rollout dir>`. Each one starts from a logged closed-loop
rollout of an earlier policy on training episodes, at the first tick where the block of the first failed operation
comes within 60 mm of its goal, and lets the expert finish from there. dg1 used rollouts of d1 at step 6,000; dg52
and dg53 used rollouts of d2 at step 16,000. Only rollouts in which every operation succeeded are kept. The noise
column gives the upper bound of the recorded per-rollout noise. In s6 and s7 the placement noise is 6 mrad, as
logged in [distill.md](distill.md) §2.

### 4.2 Rendering into the dataset format

Each kept rollout is rendered at every tick with the four cameras and written in the Tower-SimData HDF5 layout
under `<synthetic root>/data/<variant>/`. Labels (`leader/qpos`) are the expert's commands, and observations are
the resulting states and images. Each process handles one shard on one GPU:

```bash
CUDA_VISIBLE_DEVICES=<g> <render-venv>/bin/python sim/expert_write.py <work>/<batch> <synthetic root> \
    --refine <refine.pt> --shard <g>/<N> --workers 3
```

Finished files are skipped, so a rerun resumes. Each writer process logged 130 to 150 frames per second. We did
not total the rendering cost. From job durations, the main d2 write took about 1.5 h on 5 GPUs and the v3 writes
about 40 min on 4 GPUs; the placement-noise and takeover writes are not included in these figures. Writing every fourth frame (`--frame-stride 4`) cuts the size of each file by a
factor of four, and the loader then weights each sample by its stride. We added this option after v3, so the
released checkpoint was trained on full-frame data.

The data loader rejects any synthetic file whose name contains a validation episode. All generation draws only
from the training split.

## 5. Training

Training uses `scripts/train.py`. All visible GPUs form one FSDP mesh, and the batch is split across them. With
`--synthetic-frac 1.0`, every sample is drawn from the synthetic root. The variant shares of
[method.md](method.md#5-model-and-training) still apply. The official demonstrations contribute the split and the
normalization statistics, and their frames are not sampled.

Stage d2, from π0.5 base:

```bash
python scripts/train.py --tag d2 --steps 20000 --batch 64 --lr 2.5e-5 --warmup 1000 --ema 0.999 \
    --workers 16 --save-interval 2000 --export-interval 2000 \
    --synthetic-root <synthetic root> --synthetic-frac 1.0
```

Stage v3 continues from the full d2 training state (parameters, EMA, optimizer moments, step). The synthetic root
now also contains the v3 task5 episodes, for 4,712 files in total.

```bash
python scripts/train.py --tag v3 --init-from runs/checkpoints/tower/d2 --steps 28000 --batch 64 --lr 2.5e-5 \
    --workers 12 --save-interval 4000 --export-interval 2000 \
    --synthetic-root <synthetic root> --synthetic-frac 1.0
```

The synthetic root keeps growing while data is generated, so `train.py` pins the file set in
`runs/<tag>/synthetic_files.txt` on first start. The exact sets used for the release are in
`configs/synthetic/d2_files.txt` (3,995 files) and `configs/synthetic/v3_files.txt` (4,712), relative to the
synthetic root. Copy one to `runs/<tag>/synthetic_files.txt` before the first start to train on the same set.

`--steps` is the absolute step count. With `--init-from`, the sampling seed is offset by the restored step, so v3
does not replay d2's sample order (its `run.json` records seed 20042 = 42 + 20000). The released weights are the
bf16 EMA export at step 28,000 (`runs/v3/params_28000`).

| Setting | Value |
|---|---|
| Initialization | π0.5 base, full fine-tune, flow-matching loss only |
| Lineage | π0.5 base → d2 (20,000 steps) → v3 (continued to 28,000) |
| Batch | 64 (8 per GPU) |
| Optimizer | AdamW, β2 0.95, weight decay 1e-10, gradient clipping 1.0 |
| Learning rate | 2.5e-5, 1,000 warm-up steps, then constant (no decay in either stage) |
| EMA | 0.999; the EMA weights are exported and served |
| Action horizon | 32 steps at 30 Hz; joint deltas and gripper closure, quantile-normalized |
| Inputs | four cameras at 224×224 (letterboxed), discrete state tokens, official instruction; token budget 128 |
| Augmentation | lighting (p = 0.5, gain 0.3 to 2.0, 3000 to 8500 K) and colour jitter; no geometric augmentation of the external views |
| Samples drawn | d2: 1.28 M; v3: 0.51 M more |
| Hardware | 8 × NVIDIA H800 80 GB, one node |
| Step time | about 0.36 s (median of the logged 10-step intervals) |
| Wall time | d2: about 2 h 10 min; v3: about 58 min (from the run timestamps) |
| GPU-hours | about 17.3 (d2) + 7.7 (v3) = 25 H800 GPU-hours |

Each run directory contains `run.json` (arguments, data counts, the SHA-256 of the normalization statistics) and
`train.jsonl` (loss, gradient norm, loader wait and wall time every 10 steps). Exports are bf16 EMA parameters
(about 4.9 GB). Full training states, about 42 GiB each, are written only with `--save-interval`. They are needed
for `--init-from`.

## 6. Checkpoint selection

We selected the checkpoint by local closed-loop evaluation, not by offline error ([method.md](method.md#5-model-and-training)).
Snapshots of d1, d2 and the annealed variant d2cd were compared on 51 held-out episodes with layout offsets
([distill.md](distill.md) §3); d2 at step 20,000 scored best there (0.928, against 0.870 to 0.887 for d2cd). The
final selection run evaluated held-out episodes with randomized layout offsets: task1 to task4 ran 46 episodes × 4
rounds, task5 ran 10 episodes × 4 rounds in fidelity physics, and `scripts/final_table.py` summarizes the result.
Only v3 at step 28,000 completed this run. The runs of d2-20000 and v3-24000 were stopped early, so the final
choice was not a full head-to-head comparison. v3 at step 28,000 scored a mean LHPR over variants of 0.874. Tasks 1 to 4 did not degrade relative to d2
(per-round means of 1.000, 0.998, 0.989 and 0.986), and task5-2 improved from 5 of 10 to 7 of 10 on the same
rounds ([distill.md](distill.md) §5.1). [results.md](results.md#2-local-closed-loop-evaluation) gives the full
local table.

## 7. Variance and what was not controlled

- Two closed-loop runs of identical code agree on about 43 of 51 episodes. We did not repeat the training runs, so
  the seed-to-seed variance of training is unknown.
- Whether the placement-noise data contributed was never ablated.
- The variant mix and the lighting augmentation were not tested in isolation.
- Task5 data and local task5 evaluation both use our approximation of the official task5 physics. The official
  collision overlay for task5-3 is unpublished, and the official tower behaves differently during insertion. See
  [results.md](results.md#43-task5-insertion).
