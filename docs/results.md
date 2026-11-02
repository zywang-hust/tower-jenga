# Results

This page reports the official scores of the released checkpoint (`v3-28000`), our local closed-loop measurements of
the same checkpoint, and what we know about its failures. Official numbers come from the organizers. Every other
number is our own measurement and is marked as local. Where a statement is a hypothesis rather than a measurement,
it says so.

The official evaluation records were used for diagnosis only. This page quotes aggregate statistics derived from
them; it does not reproduce the records or their images, and none of them entered training.

## 1. Official results

Track `20261013-v1-task45-30hz`: 30 Hz control, 210 s per episode, 10 episodes per variant, 100 in total. The
checkpoint was served by `scripts/serve_policy.py` with start-state homing and the per-trial camera orientation
check ([method.md](method.md) §7). According to our server log, all ten task1-2 trials were homed, the external
cameras of all 40 task4 and task5 trials were judged rotated, and those of all 60 task1 to task3 trials were used
as delivered.

| Variant | TSR | LHPR | ODR | Task score | Weight | Points |
|---|---|---|---|---|---|---|
| task1-2 | 100.0% | 100.0% | 0.000 | 100.00 | ×1 | 5.56 |
| task1-3 | 100.0% | 100.0% | 0.000 | 100.00 | ×1 | 5.56 |
| task2-2 | 90.0% | 92.5% | 0.026 | 91.43 | ×1 | 5.08 |
| task2-3 | 100.0% | 100.0% | 0.000 | 100.00 | ×1 | 5.55 |
| task3-2 | 100.0% | 100.0% | 0.000 | 100.00 | ×1 | 5.55 |
| task3-3 | 100.0% | 100.0% | 0.117 | 98.96 | ×1 | 5.50 |
| task4-2 | 50.0% | 83.3% | 0.509 | 69.79 | ×3 | 11.63 |
| task4-3 | 30.0% | 71.1% | 1.197 | 54.28 | ×3 | 9.05 |
| task5-2 | 0.0% | 50.0% | 2.300 | 31.01 | ×3 | 5.17 |
| task5-3 | 0.0% | 35.0% | 1.917 | 21.80 | ×3 | 3.63 |
| **Sum of points** | | | | | | **62.28** |

All columns except the sum are official. The sum is our own arithmetic.

Definitions, following the leaderboard's published scoring code (summarized in [evaluation.md](evaluation.md) §1.4):

- TSR (task success rate) is the fraction of episodes in which every planned operation succeeded.
- LHPR is the fraction of planned grasp and place operations that were completed.
- ODR counts disturbed non-target blocks per counted step. A step is counted for each grasp, each place and once
  more for an unsuccessful episode, so the ratio is not bounded by 1.
- The task score is S = 60·LHPR + 30·TSR + 10·P/(1 + ODR), with P = (2·LHPR + TSR)/3.
- Points are S × weight / 18. Tasks 4 and 5 carry weight 3, the others 1, so tasks 4 and 5 account for 66.67 of
  the 100 available points.

Tasks 1 to 3 (pick-and-place, stacking) yield 32.80 of 33.33 available points; tasks 4 and 5 yield 29.48 of 66.67.

## 2. Local closed-loop evaluation

The local numbers come from our reconstruction of the TOWER simulator ([method.md](method.md) §3): Isaac Sim 5.1
with CPU PhysX at 120 Hz, the released station and blocks, rendering by a rasterizer plus a learned refinement
network. Policies are served exactly as in the official evaluation, through the same server and the same
post-processing.

Protocol of the selection run (`scripts/fleet.sh`, `scripts/final_table.py`):

- Episodes are held-out validation episodes of the official demonstrations; none of them, nor their layouts,
  entered training. Tasks 1 to 4 use 46 episodes, task5 uses 10.
- Each episode is run four times with a different rigid layout offset per round: two rounds at 10 to 30 mm and up
  to 3° of yaw, two rounds at up to 20 mm and up to about 10° of yaw. The simulator seed is 0.
- Task5 runs use `JENGA_PHYS_FIDELITY=1`, which applies the 32 mm side blocks and cuboid colliders of the official
  task5 stage ([method.md](method.md) §3).
- 224 rollouts in total: 24 per variant for tasks 1 to 3, 20 per variant for tasks 4 and 5.

| Variant | n | Local LHPR | Local TSR |
|---|---|---|---|
| task1-2 | 24 | 0.99 | 95.8% |
| task1-3 | 24 | 1.00 | 100.0% |
| task2-2 | 24 | 1.00 | 100.0% |
| task2-3 | 24 | 1.00 | 100.0% |
| task3-2 | 24 | 1.00 | 100.0% |
| task3-3 | 24 | 1.00 | 100.0% |
| task4-2 | 20 | 0.975 | 85.0% |
| task4-3 | 20 | 0.972 | 80.0% |
| task5-2 | 20 | 0.75 | 75.0% |
| task5-3 | 20 | 0.05 | 0.0% |
| **Mean over variants** | | **0.874** | |

The local scorer (`sim/score.py`) approximates the official one. Its disturbance measure is a different quantity,
bounded by 1, so we do not report local ODR or local task scores.

Further local measurements ([distill.md](distill.md) §3 to §5.1):

- Tasks 1 to 4 scored 1.000, 0.998, 0.989 and 0.986 LHPR in the four offset rounds.
- On task5-2, in the two rounds where both were run, the parent checkpoint `d2-20000` succeeded in 5 of 10 rollouts
  and `v3-28000` in 7 of 10.
- In 8 of the 40 task5-3 rollouts the policy never approached the first block (closest distance about 165 mm). In
  several others the second block came within 10 to 40 mm of its goal, which suggests the operations were attempted
  out of order. This is a hypothesis; we did not analyse it further.

**Robustness.** The official evaluation draws layouts and physical parameters that the demonstrations do not cover.
We tested the robustness cases listed in the official release on the parent checkpoint `d2-20000` (not on
`v3-28000`), with two validation episodes per variant of tasks 1 to 4, 16 episodes per case. The table gives local
LHPR ([distill.md](distill.md) §4).

| Case | Local LHPR, `d2-20000` |
|---|---|
| Translation up to 15 mm, yaw 5° to 10° (training covered ±5°) | 1.00 |
| Irregular layers, 8 mm and 5° per layer | 1.00 |
| Both combined | 0.974 |
| Block mass ×0.4 | 1.00 |
| Block mass ×2.0 | 0.536 |
| Friction 0.15 | 1.00 |
| Friction 1.3 | 0.955 |

With mass ×2.0, task1-3 and task2-2 scored 0. The expert's grasp and transport motion fails under the same condition (task1-2 0.5,
task1-3 0, task2-2 0, task2-3 0.33; [research.md](research.md) §8.1): a doubled block slides out of fingers limited
to 3 N each during transport. More data would not help, and we do not know whether the official controller uses
the same finger force. Changes of light direction were not tested, because our renderer cannot move the light.

## 3. Official versus local

On tasks 1 to 3 the two agree. On tasks 4 and 5-2 the local numbers are far higher than the official ones, and on
task5-3 the local number is lower:

| Variant | Official LHPR | Local LHPR | Official TSR | Local TSR |
|---|---|---|---|---|
| task4-2 | 83.3% | 97.5% | 50.0% | 85.0% |
| task4-3 | 71.1% | 97.2% | 30.0% | 80.0% |
| task5-2 | 50.0% | 75.0% | 0.0% | 75.0% |
| task5-3 | 35.0% | 5.0% | 0.0% | 0.0% |

The local simulator is therefore optimistic for tasks 4 and 5-2 and pessimistic for task5-3. These are exactly the
tasks where our reconstruction is known to be least faithful (sections 4.2 and 4.3). Before the final submission we
selected checkpoints only on local numbers. We have no official measurement of any other checkpoint and cannot say
whether a different choice would have scored better.

## 4. Failure analysis

### 4.1 task1-2 start pose

In the official track every trial starts from the same arm configuration. For nine variants it is identical to the
training start pose. For task1-2 it is up to 165 mrad away per joint, and no training episode starts that far away
([distill.md](distill.md) §5.2).

Diagnosis and fix (local measurements):

| Start of the rollout | Rollouts | Full success | LHPR | Median placement error |
|---|---|---|---|---|
| Training start pose | | | | 3.4 mm |
| Official start pose, no homing | 18 | 0 of 18 | 0.236 | 31 mm |
| Official start pose for one arm only (either arm), no homing | | 0.83 | | |
| Official start pose, homed, offset (−3.0 mm, 3.4 mm, 0.2°) | 18 | 17 of 18 | | 3.0 mm |
| Official start pose, homed, offset (1.3 mm, 1.6 mm, 7.8°) | 18 | 18 of 18 | | 4.0 mm |

Empty cells were not recorded. Because the policy predicts joint deltas from the measured state, an unfamiliar
start configuration persists through the whole episode. In the unhomed rollouts the back arm kept a different
redundant configuration throughout, and the second block was grasped 3 to 9 mm off and up to 6.4 mm low. With
homing ([method.md](method.md) §7.1) all ten official task1-2 trials scored, giving TSR and LHPR of 100%.

We also ruled out the rendering gap as a cause ([distill.md](distill.md) §5.2). Feeding the policy official frames
instead of our renders of the same demonstration state moved the predicted gripper position by 0.8 mm (median),
against 2.2 to 2.8 mm between sampling seeds. Homing is specific to the one start pose the track uses. A policy
trained from randomized start poses would not need it.

### 4.2 task4 push-down

Officially, task4-2 reached 50% TSR and task4-3 30%, with ODR 0.509 and 1.197. Locally the same checkpoint reached
85% and 80%. We have not identified the mechanism. What we know:

- **Our physics cannot reproduce pushes.** In open-loop replay of the official demonstrations, block positions
  after a push diverge by 7 to 31 cm once a pushed block tumbles off its support, while tasks 1 to 3 replay to
  within 0.5 to 4.0 mm ([research.md](research.md) §4.3). Neither GPU PhysX nor other friction values helped. The
  training data for task4 were generated in this simulator, so the expert's labels and our evaluation share its
  errors. Local task4 numbers, and in particular local disturbance, say little about official behaviour.
- **The demonstrations barely vary.** Initial block positions in the official task4
  demonstrations vary by only 0.6 to 0.7 mm (p95), against 3.5 to 3.9 mm in the other tasks
  ([research.md](research.md) §1.1).
- **Expert takeover rarely rescued task4.** When the privileged expert took over failed rollouts of an earlier
  policy (batch dg1), it completed 0 of 5 task4-2 and 1 of 7 task4-3 episodes, against 65 of 66 episodes of
  tasks 1 to 3. Recovery data for task4 are therefore nearly absent from the training set.

Our hypothesis, not tested, is that tumbling blocks in the official physics end in places our training data never
show, after which the policy disturbs the rest of the tower. The official ODR above 1 on task4-3 is consistent with
this but does not prove it.

### 4.3 task5 insertion

No task5 episode succeeded officially. LHPR was 50% on task5-2 and 35% on task5-3, and ODR was the highest of all
variants (2.300 and 1.917). Insertion is the tightest operation in the challenge: a 30 mm block enters a 32 mm slot
between two side blocks, leaving about 1 mm of clearance vertically and on each side ([distill.md](distill.md) §5).

Local analysis of the policy before the v3 data ([distill.md](distill.md) §5):

- Grasps succeed; failures happen during the slide into the slot.
- On task5-2 the held block entered 2 to 4 mm too high, with a median pitch of 4.5°.
- On task5-3 it was consistently 19 to 23 mm off laterally (5 of 5 in the same direction) and knocked the tower
  apart while sliding. The cause is unknown.
- The expert's trajectories held the block in an almost constant pose during insertion, so the data had no
  examples of correcting a misaligned block. Executing 4 instead of 8 steps per chunk did not help.

The v3 data added insertions with noise injected before alignment. The expert itself tolerates little noise there:
with 0, 3 and 6 mrad it succeeded in 12, 11 and 8 of 12 task5-2 episodes and 12, 3 and 0 of 12 task5-3 episodes. v3
raised local task5-2 from 5 of 10 to 7 of 10 in matched rounds and left task5-3 at 0.05 LHPR.

The official physics of task5-3 differs from ours. The official task5-3 station overlays an unpublished
collision layer with cuboid colliders. We approximated it with 32 mm side blocks and cuboid colliders
([research.md](research.md) §4.3.1). The policy's official LHPR on task5-3 (35%) is far above its local one (5%),
so our approximation is harder than the real task, at least for this policy. How much this difference contributes
to the official result is unknown. It concerns the last millimetres of the slide, where the policy's local errors
were already larger than the clearance.

### 4.4 Other observations

- **task2-2.** One of ten episodes failed (official TSR 90%, LHPR 92.5%, ODR 0.026). We did not analyse this
  episode.
- **task3-3.** All episodes succeeded, but ODR was 0.117: the policy disturbed non-target blocks in some episodes
  while still completing every operation. We did not analyse which blocks.

## 5. What was not evaluated

- The robustness cases (light direction, mass, friction, irregular layers) were not measured on `v3-28000`, only on
  its parent `d2-20000` and only locally (section 2).
- The local evaluation uses one simulator seed. Re-running the same code changes the outcome of about 8 of 51
  episodes ([distill.md](distill.md) §3), so local differences below about 0.1 LHPR on small sets are not
  meaningful.
- We have no measurement on real hardware.
