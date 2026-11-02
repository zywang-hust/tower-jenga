# 评测：比赛接口、规则、服务、离线与本地闭环

> **Working log (Chinese).** This note was written during the challenge and is kept as it was; only broken links
> and internal paths were edited since. Statements about the "current approach" reflect the date they were written:
> the released policy is the distilled π0.5 described in [method.md](method.md), with results in
> [results.md](results.md). Paths under `runs/` refer to experiment artifacts that are not part of this release.
>
> **工作日志。** 本文写于比赛期间，保留原样，发布时只修改了失效链接和内部路径。文中的"当前方案"以写作当时为准；
> 发布的策略是 [method.md](method.md) 所述的蒸馏 π0.5，结果见 [results.md](results.md)。`runs/` 下的路径指实验产物，
> 不随本仓库发布。

本文是比赛接口、计分、规则原文和合规对照的唯一出处，也写服务实现、离线评测和本地闭环的协议。方案和闭环结果在 [research.md](research.md)，训练配方在 [recipe.md](recipe.md)，速度在 [throughput.md](throughput.md)。

协议以选定 release 的 track、官网当前文档和 starter 包为准；旧 GitHub README 的 `tower-openpi-v1` 与官网不一致，不要沿用。注册、资格检查、向组织方提交和任何公网服务都要用户明确授权，本仓库不启动公网 endpoint。

## 1. 比赛接口 [P]

来源：官网 [连接](https://challenge.tower-benchmark.com/docs/websocket/)、[观察](https://challenge.tower-benchmark.com/docs/observations/)、[动作](https://challenge.tower-benchmark.com/docs/actions/)、[编码](https://challenge.tower-benchmark.com/docs/encoding/)、[reset/seed](https://challenge.tower-benchmark.com/docs/reset-seeds/)、[资格检查](https://challenge.tower-benchmark.com/docs/qualification/)、[参考 server](https://challenge.tower-benchmark.com/docs/python-example/)、[指标](https://challenge.tower-benchmark.com/docs/results/)；HF [排行榜 README](https://huggingface.co/spaces/tower-benchmark/TOWER-Leaderboard/blob/main/README.md) 与 [排名代码](https://huggingface.co/spaces/tower-benchmark/TOWER-Leaderboard/blob/main/leaderboard.js)。

### 1.1 协议与消息

协议名 `tower-infer-v1`。worker 建连后 server 要在 10 秒内发出二进制 metadata，然后是 `reset → reset_ack` 和多轮 `infer → infer_result`。每个响应原样带回 `protocol/session_id/trial_id/request_id/policy_id`。

| 项 | 规格 |
|---|---|
| metadata | `ready`、`policy_id`、`artifact_identity=sha256:<64位小写hex>`、plugin/version、seed 方案、observation/action spec。资格检查锁定 policy 和 checkpoint 身份，之后不能换权重；hash 必须真算 |
| observation | `joint_positions float32[2,7]`、`gripper_widths_m float32[2]`、四路 RGB（按名称读 `left_back_camera/right_front_camera/left_wrist_camera/back_wrist_camera`）、`instruction`、`task_variant`、仿真时间、`step_index`（已执行的控制动作数，见 starter `contract.md`）。每次请求都要用到 instruction |
| action | 有限值 little-endian `float32[H,16]`，`1≤H≤16`：左臂 7 关节、左夹爪、后臂 7 关节、后夹爪。关节为绝对弧度，夹爪为开口宽度（米，全开 0.1）。旧版 18 维的末两维是不执行的 lifter，只截掉两维不算适配完成 |
| 编码 | binary MessagePack，NumPy extension 42，原始连续字节；图像 `\|u1`，动作 `<f4`；单条 ≤64 MiB；不用 JPEG 或 base64；关闭 WebSocket 压缩 |

### 1.2 Reset、种子、并发与时限

- reset 清空单 episode 的时间缓存、动作 buffer、循环状态和采样计数，权重保留；同一连接可多次 reset，seed 重复也要清状态。`sample_index` 从 0 开始，每次 infer 加 1，与 chunk 行数无关。
- sample_seed 取 SHA256(`b"tower-seed-stream-v1\0" + uint32_be(base_seed) + uint64_be(sample_index)`) 前 8 字节的大端整数，随机策略必须用它生成噪声。固定 seed 不等于固定场景，也不能借 seed 推断隐藏场景。
- 每队最多 6 个排队或运行中的 evaluation，同队逐个执行；一次 evaluation 最多 8 个 worker，资格检查可能再开一个 session，容量按 8+1 设计。可变状态按 session 隔离，不能有全局动作 buffer 或全局 RNG。排队时间算进推理 deadline。
- 10 Hz 用 `dt_s=0.1`，30 Hz 用精确的 `1/30`，两档各做资格检查。推理时仿真暂停，但 episode 的 wall-clock timeout 照样生效。
- Robustness release 每变体 10 个场景，10 个变体共 100 个 episode，每个仿真预算 210 秒；位姿类 case 的扰动范围见 [research.md](research.md) §2.4，光照 case 见 [recipe.md](recipe.md) §6。相机尺寸、动作上下限和各项超时从 track 读：正式 track 是 `configs/track_20261013-v1-task45-30hz.json`，`configs/track_local.json` 是早期的本地替身，值是占位的。
- 断线恢复：worker 在 60 秒内最多重连 5 次，退避 1/2/4/8/16 秒，重连后重发相同请求。声明 `tower-reconnect-v1` 要求 session 持久化、按 request ID 缓存响应、串行化和幂等；同 ID 不同字节的请求要拒绝，进程重启丢了状态要报错。没跑完的 episode 可用 Retry failed episodes 重跑。

### 1.3 部署与提交

- 链路：组织方 worker → 公网 `wss://域名:443` → TLS 反向代理（可信 CA，支持 WebSocket 升级，读超时够长；参考 server 设 `ping_timeout=None`）→ 本机 `127.0.0.1:9001`。现行 server 不校验 API key。
- 每请求原始图像 `B = 3·ΣH_i·W_i` 字节；C 个并发、频率 f、每次返回 E 步时，1 倍仿真速度下行带宽约 `8·B·C·f/E` bit/s。
- 冻结与截止：10/30（AoE）冻结，10/31 AoE 截止，只作提交缓冲。组织方评测需要用户授权。

### 1.4 计分

官网 LHPR = 成功的 G+P 操作数 / 计划的 G+P 总数，前面失败不影响后面计分。ODR 按操作统计，非目标块中心位移超过 10 mm 算扰动。本地近似见 §4.3。

HF 排行榜：L=LHPR，T=TSR，D=ODR，`P=(2L+T)/3`，每变体 `S=60L+30T+10P/(1+D)`。五个任务族权重 1:1:1:3:3，族内两变体取平均，Task4/5 合计占 2/3。缺项记零分，每队取单次 submission 的最高总分。官网允许同一 endpoint 按 `task_variant` 路由到不同 checkpoint。

### 1.5 规则原文与合规对照 [P]

用户 10-27 转录的官方规则原文：

> - The best evaluated submission per team is ranked; see the Docs for submission limits.
> - The endpoint must serve the same model for the whole evaluation window. Do not change weights or behavior between episodes.
> - The policy may use only the provided observations. Human intervention, scripted access to simulator state, or task-specific hard-coding from test layouts disqualifies the run.
> - Top teams must share inference code or a Docker image for verification.
>
> Every task is evaluated separately on fixed initial layouts with fixed inference seeds. The simulation pauses during inference; the policy runs at 10 Hz and 16 actions are executed per request. An episode ends on success, the per-task time limit, or a policy error. Metrics: Task Success Rate (fully successful / valid episodes), Grasp Success Rate, Place Success Rate (target blocks placed stably / planned), Progress Rate (longest completed in-order prefix of task points / all task points), Disturbed Block Count (distinct non-target blocks disturbed per episode). Ranking (draft): mean TSR over the six tasks, ties by mean Progress Rate, then lower Disturbed Block Count, then earlier submission. A policy error (malformed reply, disconnect, timeout) counts as a failed episode; infrastructure failures on their side are re-run. Evaluation runs inside the availability window on layouts and seeds hidden from participants; results are verified before publication.

这段文字比 §1.1–1.4 的官网页面旧（"六个任务"、"10 Hz 固定 16 步"、"Ranking (draft)"）。控制频率和计分以官网当前页面为准；四条规则条款没有被后来的页面改写。

组织方在 Discord 上说（kevinzhang5539，2026-10-21）："we encourage everyone to use a learned policy to echo with the theme of the workshop / conference"。不含学习成分的脚本化方案不满足挑战赛对学习策略的要求，所以工作转向把特权专家蒸馏进 π0.5（[distill.md](distill.md)）。特权专家读仿真里的真实积木位姿，违反"只用提供的观测"和"不得脚本访问仿真状态"两条，只作老师；提交的是蒸馏出的策略。

待组织方确认：

| 问题 | 现状 |
|---|---|
| 分榜还是共榜 | 官网 Results 按 release 和控制频率分榜，HF README 写共榜 |
| 提交次数上限 | 只查到同时排队或运行最多 6 个 |
| 多 checkpoint bundle 的身份 hash | 按变体路由时 `artifact_identity` 要覆盖整个 bundle，是否接受未确认 |

## 2. 服务

服务讲 tower-infer-v1 协议，就绪时打印 `SERVER_READY`。会话、幂等重放和过期请求沿用官方参考 server。实现是 `src/jenga/policy.py`（adapter）加 `src/jenga/server.py`，入口 `scripts/serve_policy.py`。

| 参数 | 默认 | 说明 |
|---|---|---|
| 控制频率 | 30 Hz | track 的 `action_dt_s` 不是 1/30 时启动报错 |
| 预测 / 执行 | 32 / 8 | 每次回传前 K 行；Task4/5 的 K=4 备选见 [recipe.md](recipe.md) §9 |
| 采样 | Euler 10 步 | 噪声由该请求的 `sample_seed` 生成，输出与到达顺序、同批成员无关 |
| 后处理 | SG9 平滑 + 2 步淡入 | 依据见 [research.md](research.md) §3 |
| 副本与合批 | `--devices` 每卡一份，3 ms 窗口，桶 1/2/4 | 时延见 [throughput.md](throughput.md) §7 |
| prompt 与 norm stats | 取快照所在 run 的 `run.json` | 命名见 [recipe.md](recipe.md) §7 |

- `artifact_identity` 是 `params/` 与 norm stats 目录下全部文件（相对路径加字节）和服务配置（k、fade、smooth、prompt、jpeg_quality）的 sha256；快照目录里别的文件不影响。`plugin_version` 为 `1+<sha256 前 12 位>`，覆盖 `server.py` 中 `ADAPTER_SOURCES` 列出的源码。
- 请求在模型计算前校验，错误码为 `invalid_observation`、`invalid_instruction`、`invalid_step_index`、`invalid_task_variant`、`missing_field`。
- 每会话一个 `jenga.stitching.ChunkFader`，reset 时清空。两次请求的 `step_index` 差就是上一 chunk 已执行的行数，只有 1 ≤ 行数 ≤ k 且小于上一 chunk 长度时才淡入；重发、回退或跳跃时直接用新 chunk（`tests/test_server.py`）。动作按 track 上下限裁剪。
- 不做测试期参考系变换：布局偏移 10–30 mm 的本地闭环里它让 LHPR 从 0.113 降到 0.073（[research.md](research.md) §2.4）。
- starter 用 `python scripts/fetch_starter.py` 下到 `external/tower-starter`，只下固定官网 URL、记 SHA256、不执行包内代码。starter pin 了 NumPy 2.x，OpenPI 要 `numpy<2`，不要装进训练 venv。

### 2.1 训推一致性

| 项 | 训练 | 线上 | 处理 |
|---|---|---|---|
| 相机、resize、state 布局、闭合度、归一化、token | `TowerDataset` | adapter | `tests/test_policy.py`：10 个变体 20 帧经官方 msgpack 编解码后逐位相同，动作反解码还原 demo 绝对命令 |
| 指令 | 数据集 instruction 或官网每族一句 | 官网文本，2 块和 3 块变体相同 | 按 `task_variant` 选训练时见过的那句。R1 20k 从训练指令换成官方指令，mae8 从 0.86 升到 0.97 mrad（`runs/eval/r1-20000{,_official}/`），所以按快照的 `run.json` 选 |
| 图像 | JPEG q95 解码图 | 原始 RGB | adapter 做 q95 编解码 |
| 相机朝向 | 两路外部相机相对原始渲染转了 180°（HDF5 `camera_transforms`） | 原始渲染 | 服务端把两路外部相机转 180°（`to_dataset_orientation`）。官方不同 trial 的相机方向并不一致，发布策略按 trial 首帧判定，见 [method.md](method.md) §7.2 |

### 2.2 协议回放 [M]

`scripts/replay_client.py` 模拟官方 worker，用留出 episode 的数据集帧做观察（开环），检查 metadata、身份字段回传、`sample_index` 与 `sample_seed`、`dt_s`、动作形状与上下限、断线后同字节重放返回缓存。`runs/replay/run_replay.sh <tag> <params_dir> [gpus]` 起服务、回放、停服务，成功打印 `REPLAY_SCRIPT_DONE`，结果在 `runs/replay/<tag>.json`。

r2-40000，56 条留出 episode 从第 0 帧放到结束，9 个并发会话，2 卡服务端，变换和记录器都开（`runs/replay/r2-40000_full.json`）：协议检查 0 失败；mae 0.666 mrad；边界跳变 p95 15.8 mrad（演示 15.7）；往返 p50 154 ms、p99 256 ms；开局估计 p50 80 ms、最长 136 ms。

- 变换开关对比（r2-30000，`runs/replay/r2-30000_{ft,noft}_full.json`）：mae 都是 0.888 mrad，配对差均值 +0.0002 mrad，和非确定性同量级。训练分布内的开局偏移不超过 5.6 mm、2.1°（`runs/serve_ft/dataset_offsets.jsonl`），变换在分布内既不帮也不伤。
- 回放必须从第 0 帧开始。从中段开始时积木已被挪过，开局布局估计给出最大 26 mm、10° 的假偏移，变换开时 mae 从 0.92 升到 1.25 mrad。
- 资格检查用合成黑图，通过不代表任务能成功。

记录器 `jenga.record` 把每个 reset 和 infer（四路 q95 图像、状态、step、动作、时延）写到 `runs/serve_records/<变体>/<trial>.rec`，用 `read_records` 读。后台线程写盘，跟不上就丢弃计数，不阻塞请求；开记录器后往返 p50/p99 从 146/233 ms 到 150/241 ms。r2-40000 全量回放 22878 个请求 0 丢弃，约 240 KB/请求（`runs/replay/r2-40000_full_records.json`）。记录只用来分析失败，不进训练和调参。

## 3. 离线评测（学习策略）

### 3.1 协议

问题：一个参数快照在留出演示上，执行前缀的动作离演示多远（物理单位、分变体），每 K 帧拼接时边界跳多大。实现 `scripts/evaluate.py`。

数据是 `runs/prep/split.json` 的 56 条验证 episode（Task1–3 每变体 6 条，Task4/5 每变体 5 条）。默认每条取居中一段连续帧，每隔 K=8 帧用演示里的真实观察预测一次，最多 120 个 chunk。观察不来自策略自己的执行，所以是开环。第 t 帧噪声由 `np.random.default_rng(t)` 生成，快照之间配对比较。Euler 10 步、bf16，每批 16。预测经逆归一化、加实测 state、夹爪换回米后，和演示的绝对命令比。

| 指标 | 定义 |
|---|---|
| `mae4_rad` / `mae8_rad` | 前 4 / 8 步关节绝对误差均值，即 K=4 / 8 时实际执行的部分 |
| `hold8_rad` | 原地保持当前 state 的同一误差，下限基线 |
| `grip8_m` | 前 8 步夹爪宽度误差 |
| `jump_p95_rad` / `demo_step_p95_rad` | 相邻 chunk 的 `\|a_{c+1}[0] − a_c[K−1]\|`（关节取最大）的 p95，不含后处理 / 演示自身同一量 |
| contact mae8、grasp / release | `--frames contact`：全 episode 内离抓取或松手 1.5 s 以内的帧；tag 加 `_contact`，不跑 `stitch.py` |
| recovery 4/8/16 | `scripts/probe.py`：关节 state 偏 4/8/16 mrad 后第 k 步拉回的比例，0 = 不纠正，1 = 完全拉回 |
| 外部 16 px 平移敏感度 | `scripts/probe.py`：两路外部相机平移 16 px（224 px 输入）后的动作变化 |

`scripts/eval_snapshot.sh <tag> <params_dir> <gpus>` 跑完分片评测、汇总（变体内平均，再对 10 个变体等权）、拼接比较（`scripts/stitch.py`）和探针（`scripts/probe.py`，定义见 [research.md](research.md) §1.2），成功打印 `SNAPSHOT_DONE <tag>`，结果在 `runs/eval/<tag>/` 和 `runs/probe/<tag>/`。tag 为 `<run>-<step>`，不同设置加后缀。`scripts/compare.py <tag>...` 并排打印。

### 3.2 结果 [M]

宏平均，单位 mrad，recovery 越高越好（`runs/eval/<tag>`、`runs/probe/<tag>`）。各 run 的训练差异见 [recipe.md](recipe.md) §8，解读见 [research.md](research.md) §1.4。

| 快照 | mae8 | contact mae8 | grasp / release | recovery 4/8/16 | 换画面增益 | 外部 16 px 敏感度 |
|---|---|---|---|---|---|---|
| R1 25000 | 0.819 | | | 0.002 / 0.004 / 0.004 | 0.330 | 0.118 |
| r2-40000 | 0.673 | 0.567 | 0.534 / 0.612 | 0.763 / 0.639 / 0.394 | 0.387 | 0.171 |
| r3a-44000 | 0.677 | 0.549 | 0.519 / 0.591 | 0.782 / 0.657 / 0.405 | | 0.182 |
| r3b-44000 | 0.779 | 0.623 | 0.600 / 0.658 | 0.760 / 0.687 / 0.572 | | 1.258 |
| r3d-44000 | 0.679 | | | 0.779 / 0.655 / 0.406 | | 0.180 |

- 参照尺度（R1 时的宏平均）：hold8 约 11.6 mrad，演示步长 p95 13.3 mrad；不做后处理时跳变 p95 是演示步长的 1.5 倍。
- 这是开环误差，说明不了闭环成功率。

## 4. 本地闭环评测

### 4.1 仿真器

官方用 Isaac Sim 5.1.0，相机走 RaytracedLighting（HDF5 `sim` attrs 的 `provenance_json`、`config_json`）。HF `tower-benchmark/TOWER-Assets` 的工位 USDZ 与数据的 `runtime_stage_sha256` 一致；Task1、Task2-2、Task3 的布局运行时生成，生成器、控制器和评分都没公开。

本地仿真器的组成、与官方的差距和对拍证据见 [research.md](research.md) §4。本地闭环用来给候选排序、定位失败；成绩以组织方评测为准，每次提交都是正式评测，没有 sandbox。

### 4.2 验证集与命令

验证集 `runs/simdev/rollout/val_all.txt`：split 的 51 条验证 episode（不含 task5-3）× 种子 0、1 = 102 条 rollout，从示教首帧布局开始，每条 210 s。快照之间按 (episode, seed) 配对。布局偏移验证集 `runs/simdev/rollout/val_offset.txt` 是同样 51 条加固定的刚体偏移（行尾 `dx dy dpsi`，[research.md](research.md) §5.5）。task5-3 单独跑 5 条，本地 stage 冻结目的塔（[research.md](research.md) §4.3）。

```bash
FLEET_GPUS="2 3" scripts/fleet.sh runs/r3b/params_44000 jenga-r3b-44000 \
  runs/simdev/rollout/val_all.txt runs/simdev/fleet/r3b-44000_v2 48 0,1
```

参数：params 目录、policy_id、episode 列表、输出目录、每卡并发数（48：两组各 24 的锁步组，实测最快，见 [throughput.md](throughput.md) §8.5）、种子；之后可接 `serve_policy.py` 参数，`--` 后接 `sim/fleet_driver.py` 参数。不设 `FLEET_GPUS` 时用本 pod 全部 GPU，每卡一个 MPS daemon、一个服务（端口 9200+g）、一个驱动。每条 rollout 写 `<variant>_<episode>_s<seed>.{json,npz,log}`，已有 `.json` 的跳过，所以重跑即续跑。完成标记 `FLEET_DONE`（每个驱动先打 `FLEET_DRIVER_DONE`），失败打 `FLEET_FAILED`。用后台任务跑。学习策略的输出目录名带渲染器版本后缀 `_v2`。并发上限和吞吐见 [throughput.md](throughput.md) §8。

打分（用 `runs/simdev/render_venv/bin/python`）：`sim/paired.py <A> <B>` 给两组均值、配对差 B−A ± SE、正负计数、TSR 不一致的对和分变体结果；`sim/funnel.py <dir>...` 给漏斗。

### 4.3 指标

官方评分器没公开，`sim/score.py` 只按积木位姿近似 [H]，用于本地 run 互相比较。

| 指标 | 本地定义 |
|---|---|
| LHPR | `target_mapping` 中成功操作的比例。放置类：终态在容差内（`config_json` 的 `placement_m` × 2 与 `placement_degrees`）；推落类：块离开原位 |
| TSR | 全部操作成功 |
| ODR | 终态位移超过 10 mm 的非目标可动块数 / 操作数，上限 1 |
| reach@20 | 放置类操作中，目标块在任一时刻离目标不超过 20 mm 的比例；比 LHPR 细 |
| 漏斗 | 放置类操作：抬起（高出起点 30 mm）→ 到达（20 mm 内）→ 放成（计分）；"放到又丢"= 曾在 10 mm 内、终态没放成 |

### 4.4 比较规则

学习策略的结果在 [research.md](research.md) §5。学习策略之间比较有两条规则：

- 只在同一渲染器版本内比较。同一个 r2-40000 只把 refine v1 换成 v2，LHPR 就翻倍（research.md §5.1），换渲染器就重跑所有对比快照。
- 只在训练画面来源相同的快照之间比较。见过本地渲染画面的 r3d 比只见过录制图像的 r2 高 0.284 ± 0.044，差的是画面熟悉度（research.md §5.2）。

## 5. 晋级与统计规则

- 选 checkpoint 看执行前缀误差，不看 flow loss（Dream Machines 观察到验证 loss 上升而成功率不变）。
- 离线只用来淘汰：Task4/5 的前缀误差（K=8 看 mae8，K=4 看 mae4）不劣于当前基线，跳变 p95 相对演示步长没有新异常；改配比或续训时 Task1–3 误差退化不超过约 10%，阈值不事后放宽。
- 本地闭环排序：同一渲染器版本、同一 `val_all.txt` 和种子，报 `sim/paired.py` 的配对差和 SE。
- K 和路由的最终取舍靠组织方评测：同一 release 的固定场景配对比较，保存逐 episode 差异。每任务 10 个场景不算大量独立样本，不给过窄的置信区间；reset seed、重试和 worker 数都不算独立场景。
- 闭环结果记 TSR、LHPR、GSR/PSR、ODR、任务族与层数、accepted 与失败 episode 数、总 wall time；策略失败和连接 timeout 分开报。失败分类口径见 [research.md](research.md) §8.2。
- 任何提速若破坏 reset 可复现性、丢相机、改动作单位或导致 deadline 失败，先退回正确基线。

## 6. 未做的实验（学习策略）

每次只改一项，其余（checkpoint SHA256、OpenPI commit 与补丁、prompt、四路相机、预处理、归一化、控制频率、seed 规则）固定。已定的项：合批不变性和训推一致（§2、[throughput.md](throughput.md) §7.2）、推理用 bf16（[throughput.md](throughput.md) §7.1）、权重用 EMA（[recipe.md](recipe.md) §8）。

| 项 | 现状 |
|---|---|
| 执行长度 K=4/8/16 | 默认 8，K=4 备选见 [recipe.md](recipe.md) §9 |
| Euler NFE 10/5/2/1；Heun 等 solver | 未做；去噪步数 4–8 的容差测量见 [throughput.md](throughput.md) §8.1 |
| 按变体路由多个 checkpoint | 未做，需组织方确认 bundle hash |
| 量化、蒸馏等近似加速 | 未做 |
| 夹爪开合时刻偏差；光照扰动下的误差增幅；换 prompt 后继续或停下 | 离线未覆盖 |
