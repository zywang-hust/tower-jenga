# 训练配方（学习策略线）

> **Working log (Chinese).** This note was written during the challenge and is kept as it was; only broken links
> and internal paths were edited since. Statements about the "current approach" reflect the date they were written:
> the released policy is the distilled π0.5 described in [method.md](method.md), with results in
> [results.md](results.md). Paths under `runs/` refer to experiment artifacts that are not part of this release.
>
> **工作日志。** 本文写于比赛期间，保留原样，发布时只修改了失效链接和内部路径。文中的"当前方案"以写作当时为准；
> 发布的策略是 [method.md](method.md) 所述的蒸馏 π0.5，结果见 [results.md](results.md)。`runs/` 下的路径指实验产物，
> 不随本仓库发布。

本文记学习策略线（r1、r2、r3a/b/d）的数据事实、动作表示、配比、训练参数、存档和各 run 的决策。这条线后来没有继续：发布的策略是从特权专家蒸馏的 π0.5（[distill.md](distill.md)）。比赛接口、服务和离线评测在 [evaluation.md](evaluation.md)，速度在 [throughput.md](throughput.md)，诊断和闭环结果在 [research.md](research.md)。

π0.5 JAX（openpi @215abfb）从 `pi05_base` 全参微调，只用 flow loss。

标记：[P] 有论文或官方代码依据，[M] 是本机落盘的实测，[H] 是假设。R1 同时并入了几项 [H]，无法逐项归因。

## 1. 数据事实

Tower-SimData v1.0.0（`$JENGA_DATA_ROOT`）是完整数据：531 条 episode，1,748,580 帧，10 个变体。审计只读大小与 `MANIFEST.jsonl` 一致的文件，531 条全部通过（`episodes_skipped_incomplete = 0`）。审计脚本 `scripts/audit.py`，汇总在 `runs/audit/r1_summary.json`，逐 episode 结果在 `runs/audit/r1_episodes.jsonl`。

| 变体 | episode | 帧 | 平均时长 s | 开头静止中位 s | 结尾静止中位 s | instruction 开头 |
|---|---|---|---|---|---|---|
| task1-2 | 56 | 196,031 | 117 | 0 | 0.5 | Transfer 4 upper side blocks… |
| task1-3 | 55 | 294,441 | 178 | 0 | 0.5 | Transfer 6 upper side blocks… |
| task2-2 | 55 | 100,086 | 61 | 0 | 0.5 | Transfer 2 centered blocks… |
| task2-3 | 55 | 155,176 | 94 | 0.63 | 0.5 | Transfer 3 centered blocks… |
| task3-2 | 55 | 97,124 | 59 | 0 | 0.5 | Transfer 2 aligned single blocks… |
| task3-3 | 55 | 140,900 | 85 | 0 | 0.5 | Transfer 3 aligned single blocks… |
| task4-2 | 50 | 177,203 | 118 | 3.4 | 4.9 | Push 6 upper blocks… |
| task4-3 | 50 | 258,895 | 173 | 3.4 | 4.9 | Push 9 upper blocks… |
| task5-2 | 50 | 88,549 | 59 | 0 | 0.5 | Insert 1 blocks… |
| task5-3 | 50 | 240,175 | 160 | 0 | 0.57 | Insert 2 blocks… |

平均时长 = 帧数 ÷ episode 数 ÷ 30 Hz。leader 关节每帧变化小于 1e-4 rad 记为静止。

影响配方的审计结论：

- 夹爪单位是米，全开 0.1 m。`leader/qpos` 夹爪列落在 [0, 0.1]，与 `sim/gripper_requested_width_m` 逐值相等。观测夹爪最小 −1.32 mm，这是仿真偏移，不截断。常数为 `jenga.transforms.GRIPPER_FULL_WIDTH_M`。
- 升降轴从不运动（`lifter_control_mode="stationary"`，高度恒为 440 mm），16 维关节标签与在线运动学一致。
- `sim/action_valid` 在每条 episode 里都是连续前缀，只有末行无效。尾部保持标签依赖这一点，`TowerFrames(tail_hold=True)` 遇到中间空洞直接报错。
- 531 条演示都以成功结束，没有成功之后的片段，没有倒塌（`collapse_any_frac = 0`），也没有纠正片段（`sim/recovery_attempt` 全为 0）。
- 没有光照变化：外部相机亮度 p5–p95 为 116–126，R/B 为 1.10–1.12。
- 积木初始布局几乎不变，偏航全为 0。分布细节和它对空间泛化的影响见 [research.md](research.md) §1.1。
- X-2 与 X-3 的 instruction 已写明块数，prompt 不追加层数。
- 结尾静止约 0.5 s，Task4 约 4.9 s。保持标签主要作用在每条 episode 最后 31 个起点上。
- 32 步关节 delta 的绝对值最大 0.85 rad。

## 2. 样本与 split

`src/jenga/hdf5_dataset.py::TowerFrames` 从 HDF5 读样本，`src/jenga/dataset.py::TowerDataset` 转成模型输入，不做 LeRobot 转换。

| 字段 | 来源 |
|---|---|
| `state[16]` | `observations/qpos`：左臂 7 关节 + 夹爪，后臂 7 关节 + 夹爪 |
| `actions[32,16]` | `leader/qpos[t:t+32]` 绝对指令；越过最后一条有效命令的行重复该命令 |
| 4 路图像 | `observations/images/*` 的 JPEG，只读当前帧 |
| prompt | 官网每个任务族一句（`jenga.instructions`）；HDF5 属性 `instruction` 只用于 `evaluate.py --prompt train` 对照 |

`sim/block_poses`、`sim/task_phase` 等仿真内部量不在线上观测里，模型不用。窗口不跨 episode。HDF5 句柄在每个 worker 里懒打开，LRU 保留 64 个。

`scripts/prepare.py` 在每个变体内按 episode 留出 10%（seed 42）：训练 475 条（1,564,814 个样本起点），验证 56 条（`runs/prep/split.json`、`runs/prep/prepare_report.json`）。norm stats 只用训练 episode 计算，存于 `runs/prep/norm_stats/`，训练、评测和服务共用。验证 episode 不进任何训练数据，DART 和 DAgger 生成的数据也一样（§5）。

## 3. 动作、状态与归一化

| 项 | 设定 | 理由 |
|---|---|---|
| 关节动作 | delta：未来 32 步绝对指令都减去同一个当前实测 state | pi05_base 预训练 norm stats 里 actions 的 q01/q99 在 ±0.5–1 rad，说明预训练动作就是 delta；Dream Machines 测得 delta 与 absolute 差别不大（73% 对 76%，n=40）[P] |
| 夹爪 | 动作和 state 都转成闭合度 `c = 1 − w/0.1`，回发前换回米 | 对齐 π 预训练 0 开 1 闭的约定 [H] |
| 归一化 | 训练 split 分位数归一化，`(x − q01)/(q99 − q01)` 映射到 [−1, 1]，不 clip | π0.5 默认（openpi `config.py` L187）[P] |
| 末端窗口 | 每个有效行都可作起点，越界部分重复最后一条有效命令，不加 mask | 否则末尾 31 帧从不作为输入出现，策略没见过完成状态 [H] |
| 维度 | 16 维 pad 到 32，32 维都参与 flow loss | `action_dim=32` 才能加载 π0.5 权重 |
| 时域 | 预测 H=32（1.07 s），执行 K=8，30 Hz，Euler 10 步 | Comet 表 3：32 步优于 8/16/50 [P]；K 的取舍见 §9 |

变换在 `src/jenga/transforms.py`，往返单测在 `tests/test_transforms.py` 和 `tests/test_dataset.py`。训练 split 上关节 delta 的 q01/q99 约在 ±0.04 到 ±0.34 rad，夹爪闭合度 [0, 0.9998]（`runs/prep/prepare_report.json`）。推理端逆序执行：反归一化，取前 16 维，关节加实测 state，夹爪换回米，检查有限值和上下限。

state 以离散文本 token 进模型（`discrete_state_input=True`）：归一化后的 16 维 state 按 256 档拼进 prompt。档宽和它带来的限制见 [research.md](research.md) §2.1。

## 4. 相机与 token 预算

四路相机全部进模型，顺序固定。原版 OpenPI 只保留三路，`patches/openpi-tower.patch` 让 `Pi0Config` 显式声明 `image_keys`。

| TOWER 相机 | 模型键 |
|---|---|
| left_back_camera | base_0_rgb |
| left_wrist_camera | left_wrist_0_rgb |
| back_wrist_camera | right_wrist_0_rgb |
| right_front_camera | second_base_rgb |

解码后用 `resize_with_pad` letterbox 到 224×224，与服务端逐像素一致。训练增强沿用上游 `train=True`：非腕部视角 95% crop 与 ±5° 旋转，所有视角 color jitter（openpi @215abfb `models/model.py` L173-181）。从 r2 起外部视角不做 crop 和旋转（`jenga.model.model_config` 的 `image_geometry_augment=False`），理由见 [research.md](research.md) §2.3。

`max_token_len=128`（`jenga.dataset.MAX_TOKEN_LEN`），训练、评测和服务共用。10 条训练 instruction 配任意 state，最长 101 token（`src/jenga/dataset.py` 注释）。tokenizer 超长时只警告并截断，所以 `TowerDataset` 在 token 填满预算时直接报错。改预算不改变模型输出，提速见 [throughput.md](throughput.md) §5。

模型配置只在 `jenga.model.model_config` 一处构造，训练、评测、探针和服务共用；`jenga.model.load_policy` 按参数里有没有 `state_proj` 识别连续状态旁路。

## 5. 数据配比

变体 v 的采样概率 ∝ √(帧占比 × 分数占比)，分数权重按官方计分 Task1–5 = 1:1:1:3:3（[evaluation.md](evaluation.md) §1.4）。实现为 `jenga.recipe.variant_mix` / `sample_weights`，喂给 `WeightedRandomSampler`（有放回），变体内部按样本均匀抽取。按完整数据帧数计算：

| 变体 | 帧占比 % | 分数占比 % | 采样 % | 相对按帧采样的倍数 |
|---|---|---|---|---|
| 1-2 / 1-3 | 11.2 / 16.8 | 5.6 / 5.6 | 8.3 / 10.1 | 0.74 / 0.60 |
| 2-2 / 2-3 | 5.7 / 8.9 | 5.6 / 5.6 | 5.9 / 7.4 | 1.03 / 0.83 |
| 3-2 / 3-3 | 5.6 / 8.1 | 5.6 / 5.6 | 5.8 / 7.0 | 1.05 / 0.87 |
| 4-2 / 4-3 | 10.1 / 14.8 | 16.7 / 16.7 | 13.6 / 16.5 | 1.34 / 1.11 |
| 5-2 / 5-3 | 5.1 / 13.7 | 16.7 / 16.7 | 9.6 / 15.8 | 1.90 / 1.15 |

Task4/5 合计占 55.5%，按帧采样是 43.7%，按变体均匀是 40%。几何平均是折中：Task5-2 只有 50 条，完全按分数加权会过度重复，按帧采样又让高分任务吃亏。`sample_weights` 用训练 split 自身的样本数，实际占比与上表略有出入。这一项是 [H]：Comet 表 4 显示按技能段重加权没有可测改善，Re-Mix 支持学习配比，两者都没直接检验按分数加权。

下面两个可选项默认都关，关时权重与上表逐位相同（`tests/test_dataset.py`）。

### 接触段加权

`--contact-weight F`：起点离抓取或松手事件 1.5 s（45 帧）以内的样本，抽中概率是同变体其他样本的 F 倍，每个变体的总份额不变（`jenga.recipe.emphasize`）。事件从演示 HDF5 推出（`jenga.hdf5_dataset.contact_events`）：抓取是 `sim/active_block` 变成某块，或 `leader/qpos` 夹爪指令宽度变小；松手是 active block 清空、`sim/block_released` 置位，或指令宽度变大。Task4 的"抓取"是推块开始接触。训练 split 上 29.6% 的样本在接触段内（Task5-3 21%，其余变体 29–33%），F=3 时占抽样的 55.1% [M]。加它的依据是 refine v1 下测到的放置偏差和抓深偏差，这两项没有在 v2 下重测（[research.md](research.md) §5.4）。离线用 `scripts/evaluate.py --frames contact` 量接触段误差。

### 合成 episode

`--synthetic-root R --synthetic-frac p`：读 `R/data/<变体目录>/*.hdf5`，字段与演示相同。各变体份额仍按上表；有合成副本的变体，其份额中的 p 从合成样本里抽，变体内均匀。合成数据是逐步生成的，按变体分配能保证还没生成的变体不被挤掉份额；全部生成后，这等价于"合成集单独按同一配比、占总抽样 p"。合成文件没有 `sim/active_block` 等事件字段时，接触段取自属性 `source` 指向的原演示（帧数必须相同）。文件名含验证 episode 名时直接报错。启动时的文件清单写入 `runs/<run>/synthetic_files.txt`，`--resume` 沿用它。

### 合成数据的来源

合成数据都只从训练 split 生成。生成方法、覆盖和已知缺陷见 [research.md](research.md) §5.5（专家偏移布局）、§7（DART）和 §6.2（DAgger），这里只记训练侧要知道的格式。

| 数据 | 渲染 | 标签 | 训练侧状态 |
|---|---|---|---|
| 专家偏移布局 `expert_offset_v1`（`sim/expert_batch.py --perturb offset` 加 `sim/expert_write.py`） | refine v2 | 专家下发的指令写成 `leader/qpos`，观测是执行后的状态 | reader 直接可读；写到 855 个文件（无 task5-3）后停止，没有训练用过 |
| DART `dart_v1`（`sim/perturb.py`，已删除） | refine v1（`perturb.py` 里固定） | 演示的 `leader/qpos`，教的是"偏了就回到演示" | r3b、r3d 用了其中 288 个文件 |
| DAgger（`sim/expert_dagger.py`） | refine v2 | 专家从该状态规划的 32 行指令，30 Hz，第 0 行对应 tick t，存为 `dagger/actions [N,32,16]`；故意不写 `leader/qpos`，没打补丁的 reader 会直接失败 | reader 还不支持 `dagger/actions`，也不支持多个 synthetic root |

## 6. 光照增强

每个样本以 p=0.5 触发，四路相机共用一组参数：亮度增益在 [0.3, 2.0] 内对数均匀，色温在 3000–8500 K 内均匀（黑体 RGB 近似，6500 K 为中性），最后 clip 到 [0, 255]。实现在 `jenga.dataset.lighting`，从 r2 起训练时始终开启。

演示里没有光照变化（§1），而鲁棒 release 的 Lighting case 是 0.25–2.0×、3000–8500 K、方位 ±60°，Combined case 是 0.4–1.6×。上游 `ColorJitter(brightness=0.3)` 到不了 0.25× 和 2×，也不改色温。Dream Machines §2.4 发现同分布下增强没有收益，这里针对的是分布外，记为 [H]。阴影方向没有覆盖。

## 7. 训练入口与参数

入口是 `scripts/train.py`。参数写入 `runs/<run>/run.json`，同时记下 `base_seed`、`init_step` 和 norm stats 的 `norm_stats_sha256`；每 10 步的日志写入 `runs/<run>/train.jsonl`。所有可见 GPU 组成一个 FSDP mesh（半节点 4 卡、整节点 8 卡），每卡 batch 为 `--batch` 除以卡数。8 卡实测见 [throughput.md](throughput.md) §4。

基础参数是 R1 定下的，后续 run 只改 §8 列出的项：

| 项 | 取值 | 依据 |
|---|---|---|
| 初始化 | `pi05_base` 全参，只用 flow loss | openpi README L8 只支持 flow head；Dream Machines §2.2 中 LoRA 比全参低 23–44 pp [P] |
| batch / 步数 | 64；R1 按 55,000 步启动，日志到 25,060 步时停止，快照取 25,000（1.6M 样本，约 1 个 epoch） | Dream Machines 约 2 epoch 后进入平台，BEHAVIOR 第一名约 2 epoch 仍未收敛 [P] |
| 学习率 | R1 warmup 1,000 步后恒定 2.5e-5；续训用余弦冷却（§8） | `pi05_libero` 设 `decay_lr=peak`；Dream Machines 测得 2.5e-5 到 1e-4 不敏感，1e-4 不稳定 [P] |
| 优化器 | AdamW（b2=0.95，wd=1e-10），梯度 clip 1.0 | openpi `training/optimizer.py` [P] |
| EMA | 0.999 | `pi05_libero` 的设定，openpi 默认 0.99 [P] |
| 数据 | 训练 split 475 条，§5 配比，保持标签，光照增强 | §2–§6 |
| 并行 | 单进程 JAX，`enable_fast_path(per_device_batch=16)`，loader 16 worker（r3 用 12） | [throughput.md](throughput.md) §4 |

恒定学习率配合 EMA，任何一个快照都能直接用，不必事先定死总步数。openpi 的 `CosineDecaySchedule` 默认 `decay_steps=30_000`，不随总步数变化；`train.py` 令 `decay_lr=peak_lr`，效果就是 warmup 后恒定。R1 在 25,000 步主动停下：17.5k 之后 mae8 每 2.5k 步只降 0.01–0.04 mrad，而开环误差再降也解决不了线上的位姿外推。

单 episode 过拟合检查（task5-2 的一条 episode，产物已清理）：raw 权重前 8 步关节 MAE 1.1 mrad，原地保持基线 9.7 mrad，夹爪 1.3 mm，说明 delta、闭合度、归一化和逆变换这条链路是通的。

`train.py` 默认设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.94`（环境变量显式给出时以环境变量为准），启动前会确认 GPU 空闲。

### 存档与命名

- `--save-interval N`（N > 0）时，每 N 步和结束时把完整训练状态（fp32 参数、EMA、Adam moment）存到 `runs/checkpoints/tower/<run>/<step>`，每份约 42 GiB。`--export-interval` 每隔指定步数把 bf16 EMA 参数导出到 `runs/<run>/params_<step>`，约 4.9 GiB，服务直接加载。
- 目录名就是其中 `state.step` 的值。快照 tag 写作 `<run>-<step>`。
- 导出时如果 `params_<step>` 已存在：它正是 `--resume` 的起点（同一状态）就跳过；否则旧目录改名为 `params_<step>.superseded_<时间戳>`，它属于被放弃的那条轨迹。
- 评测、探针和服务（`evaluate.py`、`probe.py`、`serve_policy.py`）默认按快照所在 run 的 `run.json` 取 prompt 和 prep 目录（`jenga.model.snapshot_inputs`）。记录的 prep 目录不存在时退回 `runs/prep`；norm stats 的 sha256 与 `norm_stats_sha256` 不符时报错。`evaluate.py`、`probe.py` 可用 `--prompt`、`--prep-dir` 覆盖，服务端不提供覆盖。
- 现存产物：`runs/r2/params_40000` 和完整状态 `runs/checkpoints/tower/r2/40000`；`runs/r3a/params_44000`、`runs/r3b/params_44000`。r3a 和 r3b 用 `--save-interval 0`，没有完整状态，所以 `--init-from` 目前只能接 r2。

### 续训

- `--init-from <checkpoint dir>`：从另一个 run 的最新完整状态出发（参数、EMA、Adam moment 和 step）。`--steps` 是绝对步数，必须大于源 step。采样种子加上源 step，不会重放源 run 的样本顺序。
- `--final-lr`：配合 `--init-from`，从源 step 处的 `--lr` 余弦降到 `--final-lr`。
- `--resume`：从本 run 自己的最新状态继续，sampler 接着已训练的步数抽样。原始参数要原样重复，`--seed` 也给原值（即 `run.json` 的 `base_seed`，不是加过源 step 的值）。与 `--init-from` 同时给出时，源 step 取 `run.json` 的 `init_step`，不看源目录现在的最新 step，所以源 run 之后被清理或多存了 step，也能重建同一条冷却曲线和同一个种子。
- 跨 mesh 恢复：r2 的状态是 8 卡存的，恢复到 4 卡时 openpi 的 `restore_state` 沿用保存时的 sharding 会失败。`train.py` 改为按当前 mesh 构造目标 sharding，显式传 orbax restore args。
- 状态噪声只有整数档一种：`--state-noise-bins K`，半数样本扰动，含义见 [research.md](research.md) §2.2。
- 官方指令、光照增强、关闭外部视角几何增强从 r2 起固定，写在 `train.py` 和 `jenga.model.model_config` 里，不是参数。连续状态旁路（试点 E）没有训练入口。

## 8. 各 run 的决策

### R2

R2 从 R1 第 25000 步续训，在一个 8 卡 pod 上训完（`runs/logs/r2.log` 以 `TRAIN_DONE r2` 结束）。试点和对照的细节在 [research.md](research.md) §1.4 和 §2。

| 决策 | 参数 | 依据 |
|---|---|---|
| 从 R1 25000 续训，余弦冷却到 2.5e-6 | `--init-from runs/checkpoints/tower/r1 --final-lr 2.5e-6` | 17.5k 后 R1 进入平台（§7） |
| 官方指令（现为固定行为） | 当时的 `--prompt official` | 试点 A 2000 步后 mae8 与 R1 持平，文本取自官网公布的指令 |
| 光照增强（现为固定行为） | 当时的 `--lighting` | 对应 Lighting case（§6） |
| 关闭外部几何增强（现为固定行为） | 当时的 `--no-geometry-aug` | 相机固定；对照 r2g 保留增强，恢复率相同，40000 步 mae8 高 2.5% |
| 整数档 DART，不用连续状态旁路 | `--state-noise-bins 2` | 试点 D 的恢复率在三个尺度上都高于 B，mae8 更低；E 加旁路没有增益 |

测试期参考系变换只在推理端，不影响训练；它在闭环里有害，已从服务去掉（[research.md](research.md) §2.4）。

```bash
python scripts/train.py --tag r2 --init-from runs/checkpoints/tower/r1 --steps 40000 --lr 2.5e-5 \
    --final-lr 2.5e-6 --state-noise-bins 2 --save-interval 2500 --export-interval 2500
```

R1 的产物已清理，这条命令不能原样重跑，只作记录（命令按现在的参数写，当时还带 `--prompt official --lighting --no-geometry-aug`）。最终快照 `runs/r2/params_40000`：mae8 0.673 mrad（R1 0.819），4 mrad 恢复率 0.763。EMA 比 raw 权重低 6%（mae8 0.673 对 0.713，`runs/eval/{r2,raw}-40000/`），继续用 EMA。

### R3a / R3b

两者都从 r2 的 40000 步完整状态续训到 44000，用 pod 的 GPU 0-3：batch 64，lr 1e-5 余弦降到 2.5e-6，0.725 s/step，每 2000 步导出，不存完整状态。

```bash
python scripts/train.py --tag r3a --init-from runs/checkpoints/tower/r2 --steps 44000 --lr 1e-5 \
    --final-lr 2.5e-6 --state-noise-bins 2 --workers 12 --save-interval 0 --export-interval 2000 \
    --contact-weight 3
# r3b：--tag r3b，其余参数同 r3a，再加
#   --synthetic-root <dart_v1 目录> --synthetic-frac 0.4
```

对照 r3d：参数同 r3b，`runs/r3d/synthetic_files.txt` 复制自 r3b 以固定同一批 288 个文件，合成帧只抽两臂都没被扰动的帧（约 26%，`perturb/offset` 全零），各变体的合成数据份额按比例补回。它得到同样的渲染画面曝光，没有纠偏样本。

r3b、r3d 都不能原样重跑：`dart_v1` 已删除，r3d 的采样开关也已从 `train.py` 删除。

数字只在一处：离线表在 [evaluation.md](evaluation.md) §3.2，闭环表在 [research.md](research.md) §5.2。

| run | 相对 r2 的改动 | 目的 | 结论 |
|---|---|---|---|
| r3a | `--contact-weight 3`：接触段样本占抽样从 29.6% 升到 55.1% | 针对抓取和放置精度 | 接触段误差略好，整体 mae8 持平；闭环增益在 1 个 SE 内，和多训 4000 步分不开（无接触加权的对照 r3c 在导出前就停了） |
| r3b | r3a 加 40% DART（`dart_v1`，启动时冻结 288 个文件，`runs/r3b/synthetic_files.txt`） | 让策略见到"手偏了"的画面 | 离线全变体退化，大偏移恢复明显更好，外部视角依赖变成 7 倍；本地闭环大幅提高，但见 r3d |
| r3d | r3b 的合成帧限定为未扰动帧 | 对照：只有本地渲染画面，没有纠偏样本 | 闭环与 r3b 相同，离线与 r3a 相同：r3b 的闭环收益全是画面熟悉度，离线退化来自纠偏标签，DART 纠偏无可测效果 |

r3a、r3b、r3d 都不提交。

## 9. 优先级（学习策略线，已停）

这条线停下后，下面三项都没有做完。重启学习线时按这个顺序：

1. 偏移布局数据。r2-40000 在布局偏移 10–30 mm 时本地 LHPR 从 0.292 降到 0.113（[research.md](research.md) §5.5）。特权专家在训练 split 的偏移布局上 rollout，渲染成示教格式（§5 的专家偏移布局数据），用 `--synthetic-root` 混入续训。判据是名义集与偏移集的闭环差距缩小。
2. DAgger 蒸馏特权专家。专家用真实块位姿和演示的 `target_mapping` 目标，从任意状态给出动作；受限信息诊断表明任务信息几乎都能从变体和关节推出来（[research.md](research.md) §6.0），学到的策略差在执行精度上。流程、验证数字和现状见 [research.md](research.md) §6.2。训练从 r2 和 r3a 中较好的一个出发，前提是 reader 支持 `dagger/actions` 和多个 synthetic root（§5）。
3. 分析"放到又丢"。漏斗（`sim/funnel.py`）显示主要损失在这里：r2 第一个操作里，到过目标 10 mm 以内却没放成的比最终放成的还多（[research.md](research.md) §5.3）。refine v1 下的放置偏差和抓深结论没有在 v2 下重测（[research.md](research.md) §5.4）。

渲染版 DART 不再训练：纠偏没有可测效果（§8），渲染保真也已到种子噪声水平（[research.md](research.md) §4.4.1）。

不需要训练、可以随时做的：

- 执行步长 K：同一 checkpoint 比较 Task4/5 用 K=4 和全部用 K=8，推理次数翻倍。RTC §3 指出执行步长太长削弱反应，太短会导致 mode jumping。本地闭环可以比较（`serve_policy.py --chunk`），但结论受渲染差距限制。

有离线或闭环信号才做，而且只做一项：

- K=4 时，在 t>0.3 的去噪步用上一个 chunk 的剩余动作做 hard-prefix inpainting，不用训练（BEHAVIOR 第一名 §7.1，RTC §4.2）。
- 按时间步归一化（BEHAVIOR 第一名 §4.5.2）：需要改 Normalize 并重训，出现明显前缀误差才做。
- 绝对动作对照：离线指标显示 delta 有关节漂移才做，需要重训。

不做：

| 项 | 理由 |
|---|---|
| 提交特权控制器 | 它读真实积木位姿，线上观测里没有；只作 DAgger 老师 |
| RFT / 在线 RL | 本地闭环每卡 2.73 条/分钟（[throughput.md](throughput.md) §8），撑不起 RL 的样本量；专家已能从多数策略状态完成任务，DAgger 蒸馏更直接 |
| LoRA | Dream Machines 测得比全参低 23–44 pp |
| R1 用余弦或 WSD 衰减 | 恒定 LR 加 EMA 让任一快照可用 |
| 只采样 32 步全有效窗口 | 丢掉每条末尾 31 个起点 |
| task → episode → 窗口逐级均匀采样 | Task4/5 份额降到 40%，与 1:1:1:3:3 方向相反 |
| prompt 追加层数 | instruction 已含块数 |
| 16 维 loss、尾维置零 | 改变预训练噪声分布 |
| 复用 ALOHA 夹爪变换或 `trossen` 统计 | TOWER 是两臂各 7 关节，夹爪单位已由审计确定 |
| K=8 下的 RTC / inpainting | 推理时仿真暂停，没有延迟可补 |
| 相关噪声、多样本 flow、FAST 辅助、KI 复现 | 成本高，冻结前无法验证 |
| 阶段预测加投票 | 需要阶段标签，第一名自己也缺消融 |
| 提高分辨率 | Comet 实际仍 resize 到 224 |
| 动作加速（1.3× 样条压缩） | 改变接触动力学且无法验证，只在超时证据确凿时备选 |
| b128 | 原生 OOM，梯度累积编译过长（[throughput.md](throughput.md) §4） |
| 用评测记录（`runs/serve_records/`）训练或调参 | 等于拟合冻结的评测场景；记录只用于失败诊断 |

超时风险：3 层变体的演示平均时长是 Task1-3 178 s、Task4-3 173 s、Task5-3 160 s，上限 210 s，余量只有 15–24%。策略执行通常比演示慢，超时可能是 3 层变体的主要失分来源。LHPR 有部分分，超时不等于零分。

## 10. 提交与冻结

冻结与截止日期见 [evaluation.md](evaluation.md) §1.3。若提交学习策略，打包 checkpoint、norm stats 和配置。

学习线在本地回答不了的问题：隐藏的测试布局和鲁棒 case；渲染与官方 RTX 之间的残差；配比与光照增强在官方场景的实际效果；在线 episode 成功后是否继续运行（决定保持标签是否合理）；多 checkpoint bundle 的身份 hash 能否通过资格检查。

## 参考

- openpi @215abfb：`training/config.py` L187、L488-518、L744-762；`training/optimizer.py` L16-30；`models/model.py` L169-184；`models/tokenizer.py` L22-28；`training/checkpoints.py` L146-158
- 代码：`src/jenga/{transforms,recipe,dataset,hdf5_dataset,model,instructions}.py`，`scripts/{audit,prepare,train}.py`，`sim/{perturb,expert,expert_write,expert_dagger}.py`
