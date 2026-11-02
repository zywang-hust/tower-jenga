# Method

This document explains how the released policy was built and why. Each design choice is listed with the evidence
we had for it. Unless marked otherwise, numbers are our own measurements. Local closed-loop numbers come from our
simulator reconstruction (Section 3), which only approximates the official one. Official results are reported
separately in [results.md](results.md). The Chinese working notes ([research.md](research.md),
[distill.md](distill.md), [recipe.md](recipe.md)) hold the full experimental record, and section references to them
are given where a claim is condensed from there.

## 1. Setting

The TOWER Jenga challenge consists of ten simulated bimanual tasks: five task families, each with a two-layer and
a three-layer variant. Task1 to task3 move blocks between towers (tower transfer, cross-tower transfer, vertical
stacking). Task4 pushes blocks off a full tower, and task5 inserts blocks into slots in a tower. The robot has two
7-joint arms (left and back) with parallel grippers. Each request to the policy carries four 640×480 RGB images
(two external cameras, one camera on each wrist), the joint positions, the gripper widths, the instruction and the
task variant. The policy returns absolute joint targets and gripper widths. We serve at 30 Hz. An episode is
limited to 210 s of simulated time, and the simulation pauses while the policy computes.

Scoring weights the families 1:1:1:3:3, so tasks 4 and 5 carry two thirds of the total.

The official release, Tower-SimData v1.0.0, holds 531 successful demonstrations (1,748,580 frames at 30 Hz). We
split them by episode within each variant (10% held out, seed 42): 475 episodes for training and 56 for
validation. The validation episodes were never used to generate training data.

## 2. Why we distilled from an expert

**The demonstrations do not vary the layout.** Across the 531 demonstrations, the initial translation of the
movable blocks has a p95 of 3.5 to 3.9 mm (task4: 0.6 to 0.7 mm) and a maximum of 5.3 mm. The initial yaw is
always zero, and each variant uses a single block order ([research.md](research.md) §1.1). The official evaluation,
in contrast, shifts the movable assembly by up to 15 mm and 10°, jitters individual layers by up to 8 mm and 5°,
and changes the lighting.

**A policy fine-tuned on the demonstrations replays them.** Our first policies (r1 and r2, π0.5 fine-tuned on the
demonstrations for 25,000 and 40,000 steps) learned what amounts to timed replay. Shifting the joint state by
δ moved the output by δ (recovery rate 0.002 to 0.004 for r1). Translating the external images by 8 to 16 px
changed actions by 0.10 to 0.12 mrad, which is within sampling noise ([research.md](research.md) §1.3). In our
simulator r2-40000 reached a local LHPR of 0.292 on the demonstrated layouts and 0.113 when the layout was offset
by 10 to 30 mm ([research.md](research.md) §5.5).

**The task information is in the variant and the joints.** A diagnostic expert was given only the true start pose
and the mean goals of the variant, and tracked the blocks by dead reckoning. It completed 49 of 51 validation
episodes with a median placement error of 3.5 mm ([research.md](research.md) §6.0). Little scene understanding is
required beyond knowing where the blocks start. What the learned policy lacked was millimetre-level execution
across varied layouts.

We therefore generated demonstrations that do vary the layout. A scripted expert solved the tasks from perturbed
start states, and we trained the policy on its trajectories, rendered into the dataset format.

## 3. Local simulator

Generating data and ranking checkpoints both require a simulator close to the official one. The official
evaluator renders with Isaac Sim's RTX ray tracing, which does not run on our H800 GPUs because they have no RT
cores ([research.md](research.md) §4.2). We therefore separated physics from rendering.

**Physics.** We used Isaac Sim 5.1 PhysX on CPU, with the official station USDZ (its SHA-256 matches the dataset
provenance). All parameters come from the episode HDF5 attributes: 120 Hz stepping, command delays of 5 ms (left
arm) and 10 ms (back arm), and the drive gains. The gripper follows the commanded width directly, as the
evaluation controller does. Replaying the recorded commands open loop on tasks 1 to 3 reproduces the joint
trajectories within a p95 of 0.3 to 0.5 mrad and the final block poses within 0.5 to 4.0 mm. All 12 replayed
episodes end in the same success state as the recording ([research.md](research.md) §4.3). Task4 diverges by
7 to 31 cm once a pushed block tumbles off its support. Neither GPU PhysX nor different friction changed this, and
we treat the tumbling as chaotic.

For task5 we added a fidelity mode (`JENGA_PHYS_FIDELITY=1`). The task5 side blocks are 32 mm tall rather than
30 mm, according to the HDF5 attributes. Task5-3 uses a collision overlay that turns every block into a full
cuboid; the overlay is not published, so we approximated it with box colliders and a contact rest offset of
0.05 mm fitted to the demonstrations. With these changes the tower settles within 0.32 mm of the official
first-frame heights ([research.md](research.md) §4.3.1).

**Rendering.** An nvdiffrast rasterizer draws the scene from the FK of the arms and the block poses. A residual
U-Net, trained on pairs of raster images and recorded official JPEGs of the same state, then maps it toward the
official appearance (refinement network v2: five levels, full 640×480 frames, Laplacian-pyramid L1 loss,
gripper region weighted ×2, 30,000 steps). Rendering quality was judged by its effect on the policy rather than by
image metrics. On 357 fixed validation states we fed a trained policy either the recorded frame or our render and
compared the actions. The median difference was 0.39 mrad for the bare raster and 0.20 mrad with refinement v2,
against 0.19 mrad for a change of sampling seed ([research.md](research.md) §4.4.1). A Mitsuba path tracer with
matched light transport, exposure and white balance only reached the raster's level (0.34 mrad), so we stopped
investing in render fidelity.

Earlier we hypothesized that the residual appearance gap biases where the policy grasps. A direct test rejected
this. We fed the same demonstration states once with official RTX frames and once with our renders. After FK,
the gripper positions differed by a median of 0.8 mm, with a signed bias along the block axis of
0.01 ± 0.10 mm, while a change of sampling seed moved them by 2.2 to 2.8 mm ([distill.md](distill.md) §5.2).

## 4. The expert and the synthetic data

**Privileged expert** (`sim/expert.py`). The expert reads the true block poses and the goal mapping of the
demonstration. It is stateless: each call determines the phase from the geometry, plans the next segment from the
measured state, and returns 30 Hz joint commands in the dataset format. Grasp poses relative to the block,
approach heights and release heights were taken from the demonstrations. Placement corrects for the measured pose
of the held block, IK stays on the demonstrated elbow branch, and a static sag model compensates the arms' droop
under gravity. On freshly drawn validation layouts it succeeded in 70 of 70 nominal episodes, 71 of 72 episodes
with 10 to 30 mm start offsets, 50 of 56 with forced off-centre grasps, and 56 of 56 with a mid-episode push. The
median placement error was 1.8 to 3.1 mm ([research.md](research.md) §6.0). Because it reads simulator state, the
expert serves only as a teacher.

Task5 required extra work on the expert. With the 32 mm side blocks and cuboid colliders of the fidelity physics
(Section 3), the expert first failed task5-3 on a dynamic tower in 8 of 8 episodes. After its slide height, seating
and slot tracking were adapted, it completed 16 of 16 training episodes ([research.md](research.md) §4.3.1). It also
advances the back arm's joints during the slide to cancel the arm's velocity-dependent tracking lag
([research.md](research.md) §6.1). In the generation batches for the final data, it fully solved 253 of 270 and
345 of 360 task5-3 rollouts from perturbed layouts.

**Generation** (`sim/expert_batch.py`, `sim/expert_pool.py`, `sim/expert_write.py`). Each training episode
supplies an initial layout and goal mapping. The expert runs from that layout with perturbations, entirely on CPU.
Rollouts in which any operation fails are discarded. The remaining ones are rendered every tick with refinement v2
and written in the dataset format. The label at tick t is the command the expert sent at tick t, and the
observation is the state that command produced. The final training set:

| Set | Episodes | Variants | Perturbation |
|---|---|---|---|
| d2 base | 2,764 | all ten | layout offset 0 to 25 mm and ±5°; layer jitter 8 mm and 5° on tasks 1 to 4; DART-style joint noise during transit, std drawn per episode from U(0, 12 mrad); task5 in fidelity physics (dynamic tower); task5-3 comes from a separate batch |
| d2 placement noise | 1,165 | task1 to task3 | as above, plus 6 mrad noise while lowering onto the goal; the expert pulls the block back |
| d2 takeover | 66 | task1 to task4 | the expert takes over from failure states of an earlier policy (d1, step 6,000) on training episodes |
| v3 task5 | 717 | task5-2, task5-3 | task5-2 with 3 mrad (243) or 6 mrad (116) noise before insertion; task5-3 base (345); takeovers from failure states of d2 at step 16,000 (9 and 4) |
| **Total** | **4,712** | | 11.58 M frames |

The noise in the transit segments stops before contact, because 10 mrad during a placement already fails most
pick-and-place episodes. The expert's commands are absolute targets, so its clean labels pull the noisy arm back.
That is what teaches correction.

Two observations shaped this mix. First, at d1 (an earlier and smaller version of the data) the remaining
failures were placements 10 to 13 mm off. They disappeared as training went on, so we cannot attribute any gain to
the placement-noise set; we never ran an ablation ([distill.md](distill.md) §3). Second, the d2 task5 failures all
happened at slot entry. The expert's trajectories hold an almost constant pose during insertion, so the data
contained no example of a misaligned block being corrected. The v3 task5 set adds such corrections by injecting
noise before insertion ([distill.md](distill.md) §5).

## 5. Model and training

**Base model and loss.** π0.5 base (OpenPI JAX checkpoint, 3.35 B parameters), fine-tuned in full with the
flow-matching loss only. OpenPI supports only the flow head for π0.5, and a prior report that we relied on found LoRA
23 to 44 points below full fine-tuning ([recipe.md](recipe.md) §7).

**Inputs.** All four cameras are used. Upstream OpenPI keeps three, and `patches/openpi-tower.patch` adds the
fourth. Images are letterboxed to 224×224, exactly as in serving. The 16-dimensional state is quantile-normalized
and enters the prompt as 256-bin tokens, which is π0.5's discrete state interface. One bin spans 2.6 to 7.2 mrad on
the arm joints, coarser than the action error, so the policy cannot read absolute joint positions precisely from
the state ([research.md](research.md) §2.1). The prompt is the official instruction of the task family. Switching
to it from the dataset's own instruction cost nothing measurable ([research.md](research.md) §1.4).

**Action representation.** All 32 predicted joint targets are expressed relative to the current measured joint
state. We chose deltas because the normalization statistics shipped with π0.5 base place the pretraining actions
within ±0.5 to 1 rad, which indicates that π0.5 was pretrained on delta actions ([recipe.md](recipe.md) §3). The
gripper is expressed as closure `1 − w/0.1`, following π's convention that 0 is open and 1 is closed. Actions are
quantile-normalized to [−1, 1] without clipping and padded from 16 to 32 dimensions so that the base weights load
unchanged. One consequence of deltas matters at test time. Nothing in the output pulls a redundant arm
configuration back toward the demonstrated one, so a start pose that differs from the training start persists
through the whole episode (Section 7.1).

**Variant mix.** The sampling probability of a variant is proportional to the square root of the product of its
frame share and its score share. This is a compromise. Sampling by frames under-weights the high-scoring tasks 4
and 5, while sampling by score alone would repeat the 50 task5-2 episodes excessively. Tasks 4 and 5 receive 55.5%
of the samples, against 43.7% if sampling were by frames ([recipe.md](recipe.md) §5). We did not test this choice
directly.

**Augmentation.** Lighting augmentation is applied with probability 0.5: a log-uniform gain in [0.3, 2.0] and a
colour temperature in 3000 to 8500 K, shared across the four cameras. The demonstrations contain no lighting
variation, and the official lighting case spans 0.25 to 2.0× and 3000 to 8500 K. Geometric augmentation of the
external views (OpenPI's default 95% crop and ±5° rotation) is disabled. The cameras are calibrated and fixed in
TOWER, and those augmentations train the policy to ignore exactly the image shifts that encode a block offset. In
an ablation that kept them, sensitivity to the external views stayed below 0.13 mrad and offline error was 2.5%
higher ([research.md](research.md) §2.3).

**Optimization.** Batch 64, AdamW (β2 0.95), gradient clipping at 1.0, learning rate 2.5e-5 after 1,000 warm-up
steps and constant thereafter, EMA 0.999. With a constant learning rate and EMA, every exported snapshot is usable
as is. A cosine anneal to zero (d2cd) did not improve the local closed-loop score: 0.870 to 0.887 against 0.928
without it ([distill.md](distill.md) §3).

**Selecting the step.** Offline action error is not useful for choosing the number of steps. On d1 the error
reached its noise floor (about 5 mrad) by step 6,000, while the local closed-loop score continued to rise from
0.594 to 0.895 by step 16,000 ([distill.md](distill.md) §3). We selected checkpoints only by local closed-loop
evaluation. Two runs of identical code agree on about 43 of 51 episodes, so we treat differences below about 0.1
as noise unless the per-variant results agree.

## 6. Chunking and post-processing

**Horizon and execution length.** The model predicts H = 32 steps (1.07 s at 30 Hz) with 10 Euler steps of the
flow head, and the server returns the first K = 8. On task5 we also tried replanning every 4 steps. It made no
difference (task5-2 0.6 with both K = 4 and K = 8 at d2 step 16,000), which suggests that the insertion failures
do not come from the replanning rate ([distill.md](distill.md) §5).

**Smoothing and fade.** Flow sampling leaves high-frequency jitter inside each chunk: the second difference is
about 4e-4 rad, 11 to 28 times that of the demonstrations, and new chunks jump at their boundaries. The server
smooths the joint channels of each chunk with a second-order Savitzky–Golay filter of window 9 (the gripper is not
smoothed) and blends the first two steps linearly into the previous chunk's prediction for the same ticks
(`jenga.stitching.ChunkFader`). On the commands actually executed on held-out demonstrations (r1, step 25,000;
[research.md](research.md) §3):

| Rule | MAE (mrad) | Boundary jump p95 (mrad) | Second difference (1e-4 rad) |
|---|---|---|---|
| None | 0.819 | 19.86 | 5.57 |
| Fade 2 | 0.889 | 13.75 | 3.80 |
| SG9 | 0.800 | 19.79 | 3.35 |
| SG9 + fade 2 (used) | 0.870 | 13.78 | 1.66 |
| SG9 + fade 4 | 0.947 | 13.40 | 1.30 |
| Demonstrations | 0 | 13.33 | 0.27 |

A two-step fade brings boundary jumps down to the level of the demonstrations; a longer fade only adds error. The
measurement is open loop, and the fade delays the effect of a new observation by about two ticks. We did not
measure that delay's effect on contact in closed loop.

**Execution details.** Each request's flow noise is drawn from the challenge's `sample_seed`, so outputs do not
depend on arrival order or batch composition. Images go through a JPEG round trip at quality 95, because the
training images were stored that way. The server clips actions to the track's bounds. Every session keeps its own
fade state, which is reset on `reset`.

## 7. Serving-time measures

### 7.1 Start-state homing

In the official track every trial begins from the same arm configuration. For nine variants it equals the training
start pose (0 mrad apart). For task1-2 it differs by up to 165 mrad per joint, and no training episode starts
within that distance ([distill.md](distill.md) §5.2). Because the policy outputs deltas, a start pose it has not
seen persists in the redundant degrees of freedom. Starting our simulator from the official pose, the back arm
stayed in a different redundant configuration for the whole episode, and the second block was grasped 3 to 9 mm
off and up to 6.4 mm low. None of 18 task1-2 rollouts succeeded fully (LHPR 0.236, median placement error 31 mm).
Replacing the start pose of only one arm gave a full-success rate of 0.83.

The server therefore homes (`jenga.server.homing_plan`, `configs/start_qpos.json`). If any joint is more than
20 mrad from the variant's training start pose, the server sends a cosine ramp to that pose. The ramp lasts at
least 1 s and keeps the mean speed at or below 0.25 rad/s. A 0.5 s hold follows, and then the policy takes over.
For task1-2 this takes 45 ticks (1.5 s) of the 210 s budget; other variants never trigger it. From the official
start pose, homed rollouts succeeded in 17 of 18 (layout offset) and 18 of 18 (7.8° yaw) cases, with median
placement errors of 3.0 and 4.0 mm against 3.4 mm from the training start. Results in the official evaluation are
in [results.md](results.md#41-task1-2-start-pose).

A data-side fix, training from randomized start poses, would be cleaner but needed new data and a new training
run. Homing is exact for the one start pose the track uses.

### 7.2 External camera orientation

In the training data both external cameras are rotated by 180° relative to the raw render. The official evaluator
sends them rotated in some trials and not in others, and the request headers are identical in both cases. A policy
fed upside-down external views is out of distribution.

The server decides once per trial, from the first frame (`jenga.orient`, `configs/orient_reference.npz`). The
reference is the median grey first frame of the training split, together with a mask of pixels that are static
across all training episodes (walls, curtains, table edges, robot bases). Each 20×20 block of the mask votes with
the Spearman rank correlation of the frame against the reference, as given minus rotated. The frame is turned if
the median vote exceeds a margin of 0.10. Because the rule ignores the blocks, it also works on empty scenes and
under strong lighting changes, and rank correlation is invariant to monotone changes of brightness. The earlier
rule matched a block template and could not decide on empty scenes, where both orientations score zero. On the
first frames of the 56 validation episodes, each given as delivered and rotated (112 decisions), the new rule made
no errors, with a minimum margin of 0.874 ([distill.md](distill.md) §5.2). In the final official evaluation the
server turned all 40 task4 and task5 trials and none of the 60 task1 to task3 trials.

### 7.3 Latency

On H800 the model forward takes a median of 48 ms and 99th percentile of 52 ms, and the adapter (decoding,
preprocessing, post-processing) adds about 10 ms. These figures come from the server's own timing over about
76,000 requests of the official evaluation. The evaluator's requests arrived about 3.0 s apart, so inference was
not the bottleneck. The simulation pauses during inference in any case.

## 8. What we tried and dropped

| Approach | Outcome | Evidence |
|---|---|---|
| Test-time frame transform: estimate the block offset from the external cameras and warp state and actions into the canonical layout | Local LHPR on offset layouts fell from 0.113 to 0.073. The warped state no longer matched the images, and the policy reacted more to that mismatch than to the offset itself | [research.md](research.md) §2.4 |
| DART-style rendered perturbations of the demonstrations (r3b) | A +0.24 local gain, but a control trained on the same renders without perturbations (r3d) matched it. The gain came from familiarity with our renders, not from the corrections | [research.md](research.md) §5.2, §7 |
| Integer-bin state noise on the demonstrations (r2) | Taught small corrections offline (recovery 0.76 at 4 mrad), but did not fix layout generalization | [research.md](research.md) §1.4 |
| Continuous state bypass alongside the discrete tokens | No gain over discrete tokens with noise | [research.md](research.md) §2.2 |
| Physically based rendering (Mitsuba 3) | Same policy gap as the bare raster | [research.md](research.md) §4.4.2 |
| Another base model (LingBot-VLA 2.0, 6B) | Offline error twice that of r2 at a similar step count, and inference 701 ms | [research.md](research.md) §9 |
| Cosine anneal at the end of training (d2cd) | No local gain | [distill.md](distill.md) §3 |
| K = 4 replanning on task5 | No change | [distill.md](distill.md) §5 |
