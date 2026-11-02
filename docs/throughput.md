# 训推吞吐

> **Working log (Chinese).** This note was written during the challenge and is kept as it was; only broken links
> and internal paths were edited since. Statements about the "current approach" reflect the date they were written:
> the released policy is the distilled π0.5 described in [method.md](method.md), with results in
> [results.md](results.md). Paths under `runs/` refer to experiment artifacts that are not part of this release.
>
> **工作日志。** 本文写于比赛期间，保留原样，发布时只修改了失效链接和内部路径。文中的"当前方案"以写作当时为准；
> 发布的策略是 [method.md](method.md) 所述的蒸馏 π0.5，结果见 [results.md](results.md)。`runs/` 下的路径指实验产物，
> 不随本仓库发布。

本文记训练、推理、服务和本地闭环仿真的速度，每项改动附等价性证据，以及否决项。§1–§7 和 §9–§10 是学习策略（π0.5）一侧；§8 是本地闭环仿真。方案和精度见 [research.md](research.md)，服务协议见 [evaluation.md](evaluation.md)。

学习策略一侧的对象：OpenPI JAX pi05（3.35B 参数），4 路相机 224²，H=32，10 步去噪，4×H800，FSDP=4，EMA，AdamW。唯一约束是不改模型能力：结构、输入、步数、bf16 计算加 fp32 主权重、EMA 都保持官方配置，只接受数值等价或只差 bf16 舍入的工程改动，每项附等价性证据。

快路径在 `src/jenga/perf.py::enable_fast_path` 和 `patches/openpi-tower.patch`，token 预算在 `src/jenga/dataset.py::MAX_TOKEN_LEN`。原始结果在 `runs/bench/`，表里只写文件名的在 `runs/bench/train/`。

## 1. 测量方法

| 对象 | 脚本 | 口径 |
|---|---|---|
| 训练 | `scripts/bench_train.py` | 直接调 OpenPI 的 `init_train_state` 和 `train_step`；常驻显存的 fake batch，不含数据管线；预热 3 步后取 20 步中位数；首步记为编译时间 |
| 真实数据训练 | `scripts/train.py` | 训练日志每行的 `wall_s` 覆盖 10 步，去掉 step 0 取中位数 |
| 推理 | `scripts/bench_infer.py` | 单卡 bf16 权重，整段 `sample_actions` 10 步，`block_until_ready` 计时 50 次；不含 tokenizer、transform 和网络 |
| 服务 | `scripts/bench_serve.py` | 每卡一个副本，会话拿到 chunk 立刻发下一个请求（仿真时间按 0 算，最坏情况）；全部副本编译完才计时；含 host 到 device 传输 |

等价性：

- 训练：真实 pi05_base 权重，同一 batch、同一 RNG，比 7 步的 loss 和 grad_norm。单项改动在 `--xla_gpu_deterministic_ops=true` 下比。全栈配置在确定性模式下编译期崩溃（`std::out_of_range`，`runs/bench/logs/eq_full.log`），改用噪声底：先测上游自身确定性与非确定性的差，再看优化后的差是否同一量级。
- 推理：固定观察和 noise，比 32×32 动作的最大绝对差。动作幅值约 0.87，bf16 舍入级的差在 1e-2 以下。合批时还比同一请求在批内和单独推理的输出。

Profile 用 `nsys profile --cuda-graph-trace=node`；不加 node 追踪看不到 XLA command buffer 里的 kernel，计算和通信占比会算错。

## 2. 总表

| 场景 | 基线 | 发布配置 | 变化 | 出处 |
|---|---|---|---|---|
| 训练 b32，fake batch | 638.3 ms/step，50.1 samples/s，20.3 GiB/卡 | 348.3 ms/step，91.9 samples/s，55.0 GiB/卡 | 吞吐 +83% | `b32_fsdp4_ema_base.json`，`tok128_b32.json` |
| 训练 b64，真实数据（R1） | 上游 b64 未测 | 726.4 ms/step，88.1 samples/s | 每样本比上游 b32 快 76% | `runs/logs/r1.log` |
| 推理 b1 p50 | 44.5 ms（上游逐路 SigLIP） | 33.0 ms | −26% | `runs/model-smoke.json`，`runs/bench/infer/i_tok128.json` |
| 推理 b8 p50 | 144.5 ms（只合批相机） | 109.8 ms | −24% | `runs/bench/infer/i_up.json`，`i_tok128.json` |
| 8+1 服务，全部请求 p50 / p99 | 75.8 / 117.0 ms（串行 FIFO） | 48.6 / 74.4 ms（动态合批） | −36% / −36% | `runs/bench/serve/dep_{serial,batch}_g*.json` |
| 本地闭环，210 s episode | 1.3 条/分钟/卡（`sim/rollout.py`） | 2.73 条/分钟/卡（`sim/fleet_driver.py`） | 2.1 倍 | §8 |

R1 一行由全部 2506 个 10 步窗口算出，p1–p99 为 725–734 ms/step；另有 10 个 17–24 s 的窗口落在每 2500 步导出参数时，不影响中位数。推理 b1 基线是上游代码 3 次 warm 调用的均值。服务两行都开了快路径，只差调度，token 预算是 200；预算 128 的复测见 §7.2。

## 3. 快路径

训练 b32、fake batch 下逐项叠加，增益是相对上一行。

| 改动 | 开关 | step ms | 增益 | 等价性证据 |
|---|---|---|---|---|
| 基线 | | 638.3 | | |
| 4 路相机合成一次 SigLIP 调用 | `pi0.BATCH_IMAGE_VIEWS` | 558.8 | −12.4% | 第 0 步 loss 逐位相同，之后 loss 差 ≤3e-4，grad_norm 差 ≤0.1%（`eq_batchviews` 对 `eq_perview`） |
| 通信重叠 XLA flags，`CUDA_DEVICE_MAX_CONNECTIONS=1` | `perf.XLA_FLAGS` | 550.6 | −1.5%（单独加在基线上 −3.1%，`s1_ovl`） | 只改调度 |
| SigLIP attention 用 cuDNN | `siglip.ATTENTION_IMPL="cudnn"` | 542.8 | −1.4% | loss 差 ≤3.5e-4（`eq_siglip_cudnn`） |
| Gemma attention 自然布局（单 KV 头 squeeze） | `gemma.ATTENTION_NATURAL_LAYOUT` | 465.2 | −14.3% | loss 差 ≤4.1e-4，grad_norm 差 ≤0.06%，logits 仍为 fp32（`eq_natural`） |
| Gemma 激活加 batch 轴分片约束 | 补丁内常开 | 437.6 | −5.9% | 只改 SPMD 放置 |
| Gemma remat `dots_with_no_batch_dims_saveable` | `gemma.REMAT_POLICY` | 416.0 | −4.9% | dropout 为 0，只改存算取舍 |
| 两塔 layer scan 全展开（Gemma 18，SigLIP 27） | `*.SCAN_UNROLL` | 377.5 | −9.3% | 同一计算；首步编译 58 s → 159 s |
| SigLIP remat `dots_with_no_batch_dims_saveable` | `siglip.REMAT_POLICY` | 366.0 | −3.0% | 只改存算取舍 |
| prompt token 预算 200 → 128 | `dataset.MAX_TOKEN_LEN` | 348.3 | −4.8% | §5 |

结果文件依次为 `s2_batchviews`、`s2_batchviews_ovl`、`s4_siglip_cudnn_ovl`、`s5_natural`、`s7_constr_default`、`s7_constr_dnb`、`s8_both_unroll`、`s9_siglip_dnb`、`tok128_b32`。

全栈等价性（预算 200，`eq_full_nd` 对 `eq_perview`）：loss 最大相对差 1.4e-4，grad_norm 6.4e-4；噪声底（上游自身确定性对非确定性，`eq_perview_nd`）是 1.1e-4 和 6.3e-4。

- 精度：训练主权重、梯度、Adam moment 和 EMA 都是 fp32；计算时权重转 bf16，attention logits 和 softmax 在 fp32。导出的 `params_<step>` 是 bf16，上游 serving 也按 bf16 恢复。
- 通信：HLO（`runs/bench/hlo_base/`）显示 FSDP 的 all-gather 本来就是 bf16，XLA 已把 convert 提到 gather 前面。基线的通信量来自全量 remat 下同一权重每步 gather 三次。
- Gemma remat 放宽的前提是激活分片约束。不加约束时，保存 residual 会让 SPMD partitioner 把 MLP 中间量 `[8,1224,16384]` 按 hidden 维切开再 all-to-all 换回，这个版本 XLA 的 all-to-all thunk 运行时报错（HLO 在 `runs/bench/hlo_dnb/`）。加约束后先 all-gather 权重，all-to-all 消失。

## 4. 训练 b64

`enable_fast_path(per_device_batch=16)` 时 Gemma 保留上游 remat（`nothing_saveable`），因为 Gemma dnb remat 在每卡 16 样本时编译期崩溃（`shape_util.cc:1128 Check failed: return_shape->IsTuple()`，`runs/bench/logs/s11_b64_gdnb.log`）；SigLIP dnb 和两塔全展开照开。每卡 8 样本时 Gemma remat 切到 `dots_with_no_batch_dims_saveable`。

| 配置 | step | samples/s | 峰值 GiB/卡 | 出处 |
|---|---|---|---|---|
| fake batch，预算 200 | 776.8 ms | 82.4 | 62.2 | `tok200_b64_remote.json` |
| fake batch，预算 128 | 722.1 ms | 88.6 | 60.5 | `tok128_b64.json` |
| 真实数据 R1，4 卡，预算 128 | 726.4 ms | 88.1 | | `runs/logs/r1.log` |
| 真实数据 r3a / r3b 续训，4 卡 | 725 ms | | | `runs/r3a/train.jsonl`、`runs/r3b/train.jsonl` |
| 真实数据，8 卡（每卡 8），预算 128 | 348 ms | 184 | | `runs/logs/smoke8.log` |

- 真实数据只比 fake batch 慢 0.6%；b64 每样本吞吐比 b32 低约 4%。8 卡约为 4 卡的 2.09 倍吞吐，fsdp=4 存的完整状态能直接恢复到 8 卡 mesh；8 卡恢复到 4 卡要显式给 orbax restore args。R1 首个 10 步窗口 202 s，含编译和首批数据。
- 显存：b64 的 step 要一块约 50 GiB 的连续临时 buffer，JAX 默认 75% 的池子放不下它和 12.5 GiB 的 state，所以 `train.py` 默认 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.94`。加载权重和初始化会在池子里留空洞，不压紧时第一步 9 次里 4 次 OOM。`train.py` 初始化后把 state `device_get` 再 `device_put`，紧排到池首，耗时 34.8 s，之后每卡 12.51 GiB（`runs/logs/r1.log`）。启动时若有别的进程占着 GPU，最多等 120 s 后报错。
- 梯度累积 2×32（补丁加的 `GRAD_ACCUM_STEPS`）为 80.7 samples/s，比原生 b64 慢，首步编译 364 s（`s12_b64_acc2.json`），留作 b≥128 的通路：原生 b128 OOM（`runs/bench/logs/s8_b128.log`），4×32 累积卡在 XLA 常量折叠，手动中止（`runs/bench/logs/s12_b128_acc4.log`）。

## 5. prompt token 预算 128

prompt 加 state 最多 101 token（见 [recipe.md](recipe.md)），OpenPI 默认补齐到 200。padding 被 mask，位置由 mask 累加，预算改成 128 后有效 token 的位置和输出都不变。前提是零截断，`jenga.dataset` 在 prompt 填满预算时报错。

预算 200 → 128：训练 b32 step 363.4 → 348.3 ms（−4.1%），b64 776.8 → 722.1 ms（−7.0%），b64 峰值显存 62.2 → 60.5 GiB（`tok{200,128}_b{32,64}*.json`）；推理 b1 / b8 p50 从 33.4 / 114.2 ms 到 33.0 / 109.8 ms（`runs/bench/infer/i_tok{200,128}.json`）。两种预算的动作最大差 0.0049（幅值 0.86），是 bf16 级（`runs/bench/infer/i_tok{200,128}.npy`）。

## 6. 数据管线

每个样本 4 次 JPEG 解码（Pillow 11.2.1，libjpeg-turbo），从 HDF5 随机读 state 和 32 步动作，`resize_with_pad` letterbox。16 个 worker；uint8 转 float 放在预取线程，队列深度 2；host 到 device 的拷贝在主线程紧跟 step 派发，免得随时落地把 50 GiB 临时 buffer 要的连续空间切碎。数据不在关键路径上：R1 每个记录步等数据中位数 4 ms，最大 33 ms（`runs/logs/r1.log` 的 `loader_wait_s`）。

## 7. 推理与服务

推理用 `enable_fast_path(training=False)`：相机合批、cuDNN SigLIP attention、Gemma 自然布局、重叠 flags；不开 remat 和 scan 展开。预算 128 的单卡延迟（`runs/bench/infer/i_tok128.json`）：

| batch | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| p50 ms | 33.0 | 42.3 | 64.1 | 109.8 |
| chunks/s | 30.3 | 47.2 | 62.4 | 72.9 |

快路径对只合批相机的版本，动作最大差 0.012（`runs/bench/infer/actions_i_{up,fast}.npy`）。

单请求模型路径（r2-40000，空闲 H800，bucket 1，不含网络和后处理）：主机到设备 1.4 ms，10 步采样 33.4 ms（1 步 17.5 ms，即前缀约 15.8 ms、每个去噪步约 1.8 ms），拷入、计算、拷出合计 35.0 ms，主机侧调度约 1.6 ms。采样已是单个 XLA 可执行体，CUDA graph 或额外编译最多省这 1.6 ms，没做。

模型之外的服务端优化：

| 改动 | 前 | 后 | 证据 |
|---|---|---|---|
| 四路图像预处理（q95 JPEG 往返加缩放）改线程并行，PIL 释放 GIL | 单请求 24 ms | 12 ms，输出逐位相同 | 单请求微基准 |

全量回放（[evaluation.md](evaluation.md) §2.2）里 `model_ms` p50 83 ms、`adapter_ms` p50 34 ms，都是 10 路会话并发下的排队值，不是单请求耗时。

### 7.1 推理精度

问题：bf16 推理在 0.8 mrad 误差水平上是否损失开环精度。对象 R1 25000，56 条留出 episode；fp32 权重取完整训练状态里的 EMA（复现脚本 `runs/eval/precision-tools/`，结果 `runs/eval/precision-compare.json`，延迟 `runs/eval/precision-latency/`）。

| 权重 / 计算 | mae8（mrad） | 对 bf16 的配对差 | b1 p50 | b16 p50 | 峰值显存 |
|---|---|---|---|---|---|
| bf16 / bf16（服务配置） | 0.8193 | | 33.0 ms | 204.8 ms | 8.4 GiB |
| fp32 / bf16 | 0.8228 | +0.0035，56 条中 53 条略差 | 36.3 ms | 207.3 ms | 14.6 GiB |
| fp32 / fp32 | 0.8010 | −0.018（标准差 0.0035），56 条全部更好 | 59.0 ms | 424.6 ms | 16.7 GiB |

参照尺度：同一配置跨进程的执行段逐元素关节差均值 0.028 mrad（XLA 选 kernel 不确定）；相邻快照 22.5k→25k 的 mae8 差 −0.026 mrad。0.018 mrad 乘 0.5 m 力臂约 9 µm，远小于 Task5 约 0.36 mm 的插入容差。

结论：推理用 bf16。全 fp32 确实更准，但只好约 2%，b1 延迟却 +79%、吞吐减半；只换 fp32 权重没有好处。

### 7.2 8+1 服务调度

4 卡各一个副本，GPU0 承载 3 个会话，GPU1–3 各 2 个，每会话 100 次请求；不含 tokenizer、transform 和网络。前两行预算 200（`dep_{serial,batch}_g*.json`），末行预算 128（`dep128_batch_g*.json`）。

| 调度 | 3 会话卡 p50 / p99 | 2 会话卡 p50 / p99 | 全部 p50 / p99 / max | 4 卡 chunks/s |
|---|---|---|---|---|
| 串行 FIFO | 113.9 / 120.5 | 75.5–75.8 / 76.2–81.0 | 75.8 / 117.0 / 122.2 | 105.4 |
| 动态合批（3 ms 窗口，桶 1/2/4） | 72.8 / 74.6 | 48.4–48.6 / 49.0–49.9 | 48.6 / 74.4 / 75.7 | 164.6 |
| 动态合批，预算 128（现配置） | 70.1 / 74.1 | 47.4–47.6 / 48.0–49.2 | | 168.7 |

- 每个会话的 noise 由自己的 sample seed 生成后显式传入，合批不改变随机性；批内与单独推理的最大差 0.0068。
- 等待窗口不能省：非阻塞 drain 下两个闭环会话会锁步错开，永远凑不成批（`runs/bench/serve/sv_s2_batch.json` 的 mean batch 为 1.0）。
- 全部副本预热完再接流量：同机有副本在编译时，单会话 p90 从 38 ms 涨到 100 ms（`sv_s1_serial.json` 对 `sv_s1_serial_solo.json`）。

## 8. 本地闭环仿真

问题：一条本地闭环 episode（210 s，30 Hz 共 6300 个 tick，每 8 个 tick 请求一次策略并渲染四路相机）的时间花在哪，每卡能跑多快。评测协议见 [evaluation.md](evaluation.md) §4。本节的量都是学习策略服务（r2-40000）下测的。

| 驱动 | 吞吐 | 每条墙钟中位 | 渲染 / 策略 / 物理 | GPU 利用率 | CPU |
|---|---|---|---|---|---|
| 旧：`sim/rollout.py`，每卡 4 条各自串行，不合批 | 1.3 条/分钟/卡 | 146 s | 43 / 84 / 19 s | | |
| `sim/fleet_driver.py`，N=48，MPS，两个模型副本 | 2.73 条/分钟/卡（96 条 0 失败，用时 2110 s，另有服务启动 34 s） | 1007 s | 201 / 794 / 17 s | 95% | 6.3 核/卡（驱动 0.86 + worker 3.45 + 服务 2.02） |

`sim/fleet_driver.py` 每卡一个进程，N 个 episode 线程共用一个合批渲染器；下一批 Isaac worker 在当前 episode 运行时提前启动（`--prefetch`），启动不在关键路径上；作业按 variant 轮转，会话按 variant 复用。`scripts/fleet.sh` 每卡起 MPS daemon、策略服务（两个副本，bucket 1–32）和驱动。

这一配置是 GPU 瓶颈：每个请求的 GPU 时间约为模型 12.8 ms 加渲染 9.6 ms（§8.1 的改动后为 12.1 + 7.0 ms）。单卡单副本、空闲 GPU 上模型每样本 b16 13.5 ms、b32 12.8 ms；N=48 时单副本服务的模型线程 97% 的时间在等 GPU。

带来提速的改动：

| 改动 | 效果 | 等价性 |
|---|---|---|
| PhysX 在调用线程步进（`numThreads=0`，`sim/physics_only.kit`），`limit_cpu_threads=2`（`sim/physics_worker.py`） | 每个 worker 从 31 核降到 1.3 核，每 tick 2.1 → 1.5 ms | 3480 tick 状态逐位相同 |
| Isaac worker 隐藏 CUDA 和 Vulkan | 不再在可见卡上占 0.6 GB CUDA 上下文、在 GPU 0 上开 7 MiB 图形上下文 | |
| 服务端 `OPENBLAS_NUM_THREADS=1` | OpenBLAS 空闲线程在 `sched_yield` 里空转，每个服务约 5 核降到 1 核左右 | |
| JAX 持久编译缓存（`JAX_COMPILATION_CACHE_DIR`） | bucket 1–32 冷启动 85 s，热启动 27–34 s | |
| MPS | 服务和渲染器的 kernel 共享 GPU，不再分时 | |
| refine U-Net 按 32 张一块；上采样用 bf16 | E=32 时 `upsample_bilinear2d` 超出 int32 索引，必须分块；bf16 上采样每环境 5.88 → 5.15 ms | 输出逐位相同 |
| 关掉 USD 写回 | 剩下的开销是读 tensor | 输出逐位相同 |

N 的上限：服务端会话上限 64，N 加变体数不能超过它，每卡 N ≤ 48。20 s episode 下扫 N（条/分钟/卡）：N=16 为 20.6，加 MPS 21.6；N=32 为 24.2；N=48 为 24.9，bf16 上采样后 25.4，再加第二个副本 26.7。

渲染开销（每组四路相机，每环境，E=1 / E=8）：

| 光栅 | refine v1 | refine v2 |
|---|---|---|
| 6.5 / 4.3 ms | 12.2 / 9.7 ms | 12.6 / 10.0 ms |

其他：每个 Isaac worker 常驻约 0.7 GB 内存；服务端诊断记录每 96 条 210 s episode 约 16 GB。

episode 不能缩短到 210 s 以下：在 326 条 rollout 上截到 150 s 有 10 条结果改变，截到 120 s 有 23 条；示教本身最长 178 s。吞吐只能靠算力效率提。

### 8.1 锁步、副本进程与渲染

前向本身（服务配置：bf16、10 步去噪、四路相机、prompt 预算 128，`runs/bench/infer/forward_{gpu3,pod7}.json`）。两张卡都顶在软件功耗上限（throttle 0x4，677–697 W，SM 1725 → 1470 MHz），所以每样本时间几乎只取决于批大小：

| 批 | 1 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| 本机 GPU 3，ms/样本 | 33.1 | 13.56 | 12.53 | 11.89 | 11.75 | 11.57 |
| pod GPU 7，ms/样本 | 33.3 | 13.63 | 12.61 | 11.95 | 11.78 | 11.62 |

单卡前向上限约 86 样本/s；prefix（SigLIP + Gemma 看图）占大头，每步去噪约 0.14 ms/样本，prefix 的 KV 在 10 步间已复用（openpi `sample_actions`）。按容差判断的前向候选（64 个固定 val 状态上的动作差，对比同图换噪声种子的噪声底：中位 0.384 mrad、p90 11.2、最大 89.4，每状态 RMS 中位 2.63，`runs/bench/infer/tolerance_{gpu3,pod7}.json`，两卡逐位相同）：

| 去噪步数 | 动作差中位 / p90 / 最大 (mrad) | 每状态 RMS 中位 | ms/样本 (b=24) |
|---|---|---|---|
| 10（现服务） | — | — | 12.07 |
| 8 | 0.138 / 0.54 / 3.96 | 0.27 | 11.83 |
| 6 | 0.212 / 1.07 / 6.69 | 0.50 | 11.62 |
| 5 | 0.256 / 1.53 / 10.5 | 0.76 | 11.43 |
| 4 | 0.339 / 2.08 / 15.6 | 0.93 | 11.24 |

都在噪声底之内，但最多省 7%，而且会和服务一直用的 10 步分叉，没采用。其他候选：fp8 GEMM 见 §9（无收益）；prompt 预算 128 对 85 个真实 token，裁掉上限 2.8%；每请求的 `device_put` 和 host 同步已并入下面的副本进程。

前向内部（b=24，服务配置，本机 GPU 3，nsys `--cuda-graph-trace=node`，`runs/bench/prof/ncu_forward_b24/`）：6351 个 kernel，span 287.6 ms，kernel 时间 284.4 ms，GPU 忙 98.9%。prefix（SigLIP + Gemma 看图）1171 个 kernel 245.0 ms（86.1%）；去噪 10 步各 518 个 kernel，共 39.4 ms（13.9%），每步 3.94 ms，其中 370 个 kernel 不到 10 µs。

| kernel | 次数 | ms/前向 | 占比 | 位置 |
|---|---|---|---|---|
| nvjet 192x192 NNN（coopB，2x1）MLP/attention GEMM | 171 | 109.8 | 38.6% | prefix |
| nvjet 192x192 NNN（1x2，带 bias add） | 36 | 50.0 | 17.6% | prefix |
| loop_multiply_fusion（Gemma prefix attention 的 logits×mask） | 18 | 16.0 | 5.6% | prefix |
| fusion_249（Gemma prefix attention softmax） | 18 | 11.8 | 4.1% | prefix |
| nvjet 160x256 / 192x192 TNN / 192x192 NNN 其他形状 | 63 | 22.9 | 8.1% | prefix |
| gemm_fusion_dot_66_0（去噪步最大的 GEMM） | 180 | 5.7 | 2.0% | loop |
| cuDNN flash SDPA（只有 SigLIP 走 flash） | 27 | 3.4 | 1.2% | prefix |
| cuDNN TF32 隐式 GEMM 卷积（SigLIP patch embed，`sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32`） | — | 0.88 | 0.3% | prefix |

三个大头：两种 192x192 nvjet GEMM 合计 56%，是 Gemma 18 层 prefix 的 MLP 和投影；Gemma prefix attention 没走 flash（multiply + softmax 融合约 28 ms，10%），这是 §9 "cuDNN flash attention" 否决项在推理侧的对应；去噪步是小 kernel 的 launch 串（每步 518 个，GEMM 只占 1/3），靠 XLA command buffer 才有 3.9 ms/步。tensor-pipe 利用率、DRAM 带宽和由此算的 MFU 没有量到：ncu 对这个进程（XLA 60 GB arena、command buffer）在 kernel 回放的保存/恢复上挂起，关 command buffer 加 platform allocator 之后在本机 GPU 3 和 pod GPU 7 上都没在截止前跑完，`analysis/raw_inventory_b24.csv` 只有表头。bf16 patch-embed 卷积的容差也没测。

闭环这一侧的改动（服务端输出对 64 个固定状态逐位相同，`tests/test_server.py` 通过）：

| 改动 | 量法 | 之前 | 之后 |
|---|---|---|---|
| 驱动分两组锁步（`--groups 2`）：一组渲染时另一组的请求整批到模型 | GPU 3，48 个客户端的锁步负载（`load_gen.py`） | 47.05 req/s（动态合批，批碎） | 64.80 req/s，批 24/24 |
| 模型副本搬进 spawn 子进程：前向的 collate、上传、launch、读回都要 GIL，在服务进程里被事件循环、adapter、recorder 线程拉长 2–4 倍（24 样本批 1.4 s 对单独 315 ms） | 同上 | — | 同上一行 |
| 图像以 uint8 过管道和 PCIe，在卡上转 [-1, 1]（同一表达式，jit 一次融合） | 副本日志，24 样本批 | collate 35 ms + 上传 11.5 ms | 1.7 + 1.1 ms；前向 297 ms |
| 否决：副本内流水（第二批在前向时准备） | 同上 | — | 批被切碎（均值 10.5，最小 1），62.35 req/s；加 20 ms 静默仍 14，57.8 |
| 否决：关批规则加 30 ms 静默窗（等锁步组的到达散布） | pod GPU 6，48 环境 60 s episode，t=180/240 s 的稳态 req/s | 37.2 / 39.6（批均值 18–23，最小 1，补齐到 24） | 34.4 / 32.8（批均值 21）；已删除，关批只看 staging |

旧服务（动态合批、两个副本、同一锁步驱动）在同一 60 s 测试里稳态 32–33 req/s。单卡上同时跑渲染和前向时前向是 453 ms/23 样本（独占时 290）：功耗上限下两者的能耗相加，MPS 并发不再有额外收益。

渲染（`sim/render.py`，光栅、阴影加 refine 整条到 uint8；E=16 每环境 ms，本机空闲 GPU；物理不变，判据是 64 个固定 val 状态上的动作差对比同图换噪声种子的噪声底：中位 0.375 / p90 11.3 / 最大 89.3 mrad，每状态 RMS 中位 2.66、最大 32.6，`runs/bench/infer/render_tol_*.json`）：

| 改动 | ms/环境 | 动作差中位 / p90 / 最大 (mrad) | 每状态 RMS 中位 / 最大 | 结论 |
|---|---|---|---|---|
| 基线（df9f641） | 8.41 | 0 / 0 / 0.51（同配置重跑，`dr.antialias` 本身不确定） | 0 / 0.10 | |
| 顶点变换换 Triton kernel（乘积配对 (0+2)+(1+3)，`enable_fp_fusion=False`） | 8.07 | 逐位相同（64×507k×4 个值，两次 pass） | — | 采用，3.58 → 0.72 ms/pass |
| 阴影图 1024 → 512 | 7.99 | 0.149 / 2.11 / 33.3 | 0.65 / 10.4 | 采用 |
| PCF 5x5 → 3x3，采样间距 2× softness | 7.78 | 0.112 / 1.12 / 26.8 | 0.34 / 9.1 | 采用 |
| 三项合并（现状） | 7.00 | 0.190 / 3.00 / 39.0 | 0.94 / 16.4 | 采用，均在噪声底内 |
| 否决：半分辨率渲染再放大 | 3.57 | 1.03 / 21.4 / 161 | 8.77 / 51.3 | 超噪声底 |
| 否决：阴影图 256 | 8.27 | 0.321 / 7.02 / 69.4 | 2.72 / 27.7 | 到噪声底，只省 0.14 ms |
| 否决：CAD 网格再减面 | 8.19 | 0.302 / 6.29 / 153 | 1.94 / 50.0 | 最大值超噪声底 |

每请求 GPU 时间从模型 12.8 + 渲染 9.6 ms（§8）变为 12.1 + 7.0 ms（渲染含 refine，refine 自身 5.40 → 3.91 ms 在 df9f641）。渲染改动没有 102 条配对闭环确认（`sim/paired.py`；在 pod 上因 CPU 争用没跑完），证据只有固定状态上的动作差。

### 8.2 整环 nsys 时间线（结构切片）

问题：fleet 配置（一张卡：MPS、一个服务进程加一个副本进程、`sim/fleet_driver.py` 48 环境两组锁步）里每个请求的时间花在哪一段，GPU 空转由哪些 host 等待造成。工具：`scripts/fleet_nsys.sh`（服务和驱动各一个 `nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node`，副本进程随 `--trace-fork-before-exec` 进服务报告）、`src/jenga/nvtx.py` 和 `sim/{fleet_driver,rollout,render}.py` 里的 NVTX 段、`sim/fleet_nsys_stats.py` 出表。报告 `runs/simdev/prof/fleet48s_{server,driver}.nsys-rep`：pod GPU 4，r2-40000，`val_all.txt` 前 48 条、90 s episode，驱动起后 120 s 起各采 40 s。

只作结构读数，不作定量结论：采集时 pod 的 CPU 被 40 个专家数据 Isaac worker 占满（load 约 100/115 核），而且两份报告的窗口错开 43 s（服务端的 `--delay` 没有从服务启动起算，见脚本头注），所以没有两进程并集的 GPU 占用率。渲染器配置是 §8.1 采用的 shadow 512、3×3 PCF、Triton 顶点变换，与现行 `sim/render.py` 相同。

每个请求（环境线程视角，p50，驱动窗口 1488 个请求 = 37.4 req/s）：

| 段 | ms | 说明 |
|---|---|---|
| poses（FK + 位姿栈） | 0.5 | |
| sync（等本组其他环境到齐） | 52.9 | 锁步等待 |
| render_wait（本组 24 张一批） | 203.7 | 渲染本身 p10/50/90 179/191/287 ms，24/24 不碎，host 侧 9.3 ms/env；readback 2 ms |
| pack（msgpack 3.7 MB） | 25.3 | 48 个线程共用一个 GIL |
| ws_send | 20.0 | 同上 |
| ws_recv（服务端整条链） | 869.1 | 下表 |
| physics（8 tick） | 12.7 | |

服务端（服务窗口 1461 个请求 = 37.8 req/s，63 次前向）：

| 段 | ms | 说明 |
|---|---|---|
| unpack | 2 | p90 5 |
| stage（sha256 + 翻转） | 10 | p90 12 |
| adapter（JPEG 往返 + resize + tokenizer） | 39 | p90 99：24 个请求同时挤 4 个图像线程和 GIL |
| batch_close（首个请求入队到关批） | 167 | p10/50/90 143/167/251；批 p10/50/90 16/24/31，均值 22.9 |
| forward 派发 | 25 | 异步派发 |
| readback（等 GPU 算完） | 352 | 副本每批 377 ms = 16.5 ms/样本，独占时 12.07 |
| pipe_recv（副本等下一批） | 243 | 副本空等占窗口 40% |

GPU：服务窗口里前向 kernel 并集占 47.3%（kernel 之和 18.3 s/38.7 s），驱动窗口里渲染 kernel 并集占 28.5%（10.5 s/39.8 s）；两者不重叠时合计不超过 76%，剩下至少约四分之一是空转，而且前向和渲染同时跑时前向从 12.07 拖到 16.5 ms/样本。链条上可量化的 host 段：batch_close 167 ms 由同组 24 个环境在一个 GIL 下依次 pack+send（每个 45 ms）加 adapter 排队造成；对侧组前向期间它被遮住，但两组同时落在 host 段时 GPU 空转。没做：两份报告对齐后的并集占用、去掉 CPU 争用后的重采，以及任何修改（流重叠、两段流水、JPEG 移出关键路径、组大小）。

### 8.3 本地仿真的 CPU 开销（未并入的分支 a1ac9f5 的测量）

问题：一个 Isaac physics worker 单独运行时 CPU 花在哪，哪些是能免掉的空转；专家批（`sim/expert_batch.py`，每条 rollout 一个 `sim/expert_rollout.py` 加一个 worker）在 115 核的 pod 上开多少并发吞吐最高。对象是一个 8×H800、cgroup 115 核的节点，测量时 pod 上同时跑着别的作业（数据集写入 12 进程约 11 核、其他 agent 的 fleet），单条测量的 load 在 23–37，批测量的 load 在 30–85，表里都附。工具：py-spy 0.4.2（`--native`，100 Hz，装在 Isaac venv），`/proc/<pid>/task/*/stat` 按线程名汇总，`ps`。原始数据在非持久的临时目录里，未随本仓库发布。

**单条专家 rollout（task1-3 ep1022，3708 tick 即 121 s 仿真）的 worker 去向**（`ep/prof/pyspy_worker.txt`，2739 个有效样本，`classify.py` 汇总）：

| 阶段 | 样本占比 | 内部 |
|---|---|---|
| 启动：`SimulationApp` | 38% | libc 读文件/lstat 40%、Python import 29%、USD 20%、carb 11%；没有单一热点 |
| 启动：开 stage、建物理场景、play | 13% | USD 64%、PhysX 23% |
| episode：`tick` | 42% | PhysX `simulate`+`fetch_results` 75%、Python 胶水 15%（其中每步经 simulation manager 读 USD 的 `get_physics_dt` 3%、tensor API 包装 4%）、USD 4%、numpy 3%、carb settings 2% |
| episode：`state` 读回 | 3% | numpy、Python |
| 关闭 | 2.5% | carb |

按线程（`ep/base/worker_threads.txt`，worker 存活 21 s，CPU 13.1 s）：主线程 83%，`tbb.worker` 11%（USD 加载期间），`carb.tasking` 3%，Kit 的 `Profiler` 线程 0.2%，`OC`/`heartbeat`/`FileSystemWatch`/`structured` 为 0。Kit 的 update loop 在 episode 里根本不转：`SimulationContext.step(render=False)` 只是 `simulate()`+`fetch_results()`，headless 下没有可关的扩展循环。专家进程本身每条 8.2 CPU-s，78% 在 numpy。

结论：worker 的 CPU 一半是启动，另一半里四分之三是 PhysX 本身。episode 循环 3708 tick 只要 8.8 s 墙钟（`ep/base/ep.json`），worker 却活了 21 s，worker 的墙钟由启动主导。

**启动时间线**（`ep/base/ep.physics.log`，load 35）：扩展加载 0–3.9 s；`Simulation App Starting` 到 `app ready` 3.9–9.4 s；play 到 11.5 s；关闭 1.7 s。其中三次 `createGpuFoundation` 失败探测（启动、play、关闭各一次）各 1.3–4.7 s：Vulkan ICD 被置空后 gpu.foundation 的 SystemValidation 转而 `/usr/bin/nvidia-smi -q`（`libgpu.foundation.plugin.so` 里的字符串），8 卡节点上一次 1–2 s，40 个 worker 同时启动时 pod 上同时有 20–38 个 `nvidia-smi -q`（`census_sweep.log`，合计 260% CPU）。看到的"nvidia-smi -q 轮询"就是它，不是人。`/physics/cudaDevice=-1`、`/persistent/physics/overrideGPUSettings=1`、`/renderer/gpuEnumeration/glInterop/enabled=false`、关 crashreporter、`/validate/*/enabled=false` 都不能去掉这三次探测（`boot/*.log`、`boot_crash.sh`、`boot_validate.sh`）。

**改动与等价性**（专家 rollout 的 qpos/blocks 用数据集 `replay_comparison_json` 的容差：关节 0.01 rad、指宽 0.5 mm、积木 2 mm / 1°；`compare.py`）：

| 改动 | 效果 | 等价性 |
|---|---|---|
| `Station.tick` 直接调 `simulate(dt, t)`+`fetch_results()`，不再每步经 `SimulationContext.step` 查 timeline 和读 USD 的步长（`sim/scene.py`） | 静止场景每 tick 1.33 → 1.10 ms（`boot/control*.log` 对 `boot/v_*.log`，负载不同）；完整 episode 的 worker CPU 在单条噪声内（`ep/head_*` 对 `ep/new_*`，1161–1272 对 1231–1260 tick） | 3708 tick 逐位相同（`ep/base` 对 `ep/direct2`） |
| 只在目标变化时写 drive target | 静止场景每 tick 0.36 ms | **否决**：写 target 也是让关节链保持清醒的动作，不写则积木姿态最大差 1.57°（`ep/direct`），超容差 |
| `sim/render.py` 延迟 `torch.compile`，`sim/rollout.py` 只在 main 里 import 渲染器（407d4a6 的两处，本分支原来没有） | 每个 `expert_rollout.py` 进程不再起 1 个 inductor 池主进程加 32 个 worker（`census_sweep.log` 07:10:25：16 条 rollout 对应 541 个 `compile_worker`；改后 `census_after.log` 里只剩写入作业的 12–18 个）；专家进程每条 CPU 8.2 → 3.5 s（`ep/base`、`ep/fixed` 的 `expert_threads.txt`）；渲染进程从 32 个 worker 降到 2 个 | 四路相机两组画面逐字节相同（`render_eq/`，407d4a6^ 对本树） |
| worker 的 PATH 前置一个立即失败的 `nvidia-smi` 桩 | **否决**：PATH 传到了 worker（`shutil.which` 给桩），但 gpu.foundation 按绝对路径执行 `/usr/bin/nvidia-smi`（`check_smi.sh`：`exe=/usr/bin/nvidia-smi`），桩不生效；系统二进制不能动 | |
| `sim/expert_batch.py --nice 10`（默认） | 专家批让位给 fleet 的渲染线程 | 只改调度 |

fleet 路径（`sim/rollout.py` 一条 210 s episode，r2-40000 单副本在 GPU 2，task1-3 ep1022，seed 0；`fleet/head.log` 对 `fleet/fixed.log`）：改前改后得分相同（lhpr 0、tsr 0、六个操作都失败），物理耗时 10.8 → 9.6 s；轨迹本身不可比，策略跨进程不确定（§7.1），闭环会放大，所以 fleet 路径的逐位证据是专家路径的那条（同一个 `sim/physics_worker.py`）。

**批并发**（`batch_sweep.sh`，`expert_batch.py --start 20`，每变体 jobs/5 条，`--nice 0`；吞吐按最早 worker 启动到最后一个结果落盘，`batch_stats.py`）：

| 代码 | jobs | 条数 | 墙钟 | 条/分钟 | Kit 启动到 ready 中位 / 最大 | episode 循环中位 | load 前 → 后 |
|---|---|---|---|---|---|---|---|
| 改前（inductor 池 + nvidia-smi） | 16 | 30 | 217 s | 8.3 | 27.3 / 88.1 s | 4.2 s | 29 → 60 |
| 改前 | 24 | 40 | 148 s | 16.2 | 35.2 / 39.1 s | 4.3 s | 60 → 43 |
| 改前 | 40 | 80 | 268 s | 17.9 | 54.9 / 60.6 s | 4.5 s | 43 → 59 |

| 改后（`after_j*`） | 16 | 30 | 107 s | 16.8 | 26.1 / 29.9 s | 4.3 s | 27 → 33 |
| 改后 | 24 | 40 | 130 s | 18.4 | 35.8 / 40.0 s | 4.2 s | 33 → 34 |
| 改后 | 40 | 80 | 202 s | 23.7 | 51.1 / 62.2 s | 4.2 s | 34 → 55 |

改后三档里 pod 整个 cgroup 的平均占用只有 18–21 核（`cpu.stat` 差分，`batch.log` 上方一行），load 却到 55–60：40 个 worker 同时启动时 pod 上同时有 32–38 个 `/usr/bin/nvidia-smi -q`（`census_after.log` 07:30–07:32，合计 325–350% CPU），它们在驱动里串行等待，worker 启动到 ready 从单条的 9.4 s 拖到 51 s。episode 循环只占每条墙钟的十分之一。推荐并发 40（每小时约 1400 条，`--nice 10` 让位给 fleet）；再往上启动会继续变慢，pod 上有 fleet 时用 24。下一步（没做）：在 `expert_batch.py` 里限制同时启动的 worker 数（比如 8 个），让 40 条在飞但启动不堆在驱动上。

**没做的**：线程数环境变量——专家进程已经 `OPENBLAS/OMP/MKL_NUM_THREADS=1`（2 个线程），worker 已经 `limit_cpu_threads=2`、`numThreads=0`，渲染进程的 364 个线程不空转（主线程占 95%，`threads_top.sh`）；Kit 的 Profiler 线程只占 worker 的 0.2%，不值得改 kit 文件。一个 Isaac 进程里放 N 个 station（第 3 问）：PhysX 本身只占每条 21 CPU-s 的约 4.5 s（21%），启动占 6.7 s（32%），专家规划 8 s（38%）；N 个 station 共享一个场景省的只是启动的分摊，PhysX 步进随刚体数线性增长不会变少，而启动分摊用一个常驻 worker 跨 episode 复用（`sim/physics_worker.py --init` 已经做到同一 episode 内复用）就能拿到，改动小得多，所以不做多 station。fleet 路径的物理每条只有 17 s（§8），不是瓶颈。

### 8.4 专家批的常驻物理 worker（`sim/expert_pool.py`，分支 perf-phys）

问题：专家 rollout（本地 Isaac PhysX，纯 CPU）每条墙钟花在哪，怎样在物理结果不变的前提下提高每 pod 的条数。工具：`scripts/phys_prof.sh`（py-spy 0.4.2 `--native --idle` 采驱动和 Isaac worker 的主线程，主线程样本数即墙钟；`/proc` 线程 CPU；Kit 日志时间戳），`scripts/phys_prof_stats.py` 按调用栈归类。对象：同一个 115 核 cgroup 的节点，测量时 pod 上还有其他 agent 的作业（load 15–118），原始数据在 `runs/simdev/perf_phys/`。

**旧路径一条 rollout（task1-3 ep1022，wide 偏移 + DART 噪声）** [M]，`prof/idle0`、`prof/load40`：

| 段 | 单条（s） | 40 并发里的一条（s） |
|---|---|---|
| 驱动 import（`sim/rollout.py` 顶层 import torch），在 worker 启动之前串行 | 6.5 | 4.0 |
| 等 worker 启动（connect） | 17.5 | 74.2 |
| 　其中 worker：扩展加载 / 新 stage 内 CUDA 初始化（libcuda ioctl）/ gpu.foundation 探测 / resourcemonitor | 8.1 / — / 9.0（三次）/ 3.7 | 11.1 / 28.7 / 25.1 / 1.1 |
| episode：等物理（worker tick 6.2–6.7） | 7.2 | 7.3 |
| episode：专家规划 | 1.8 | 1.7 |
| 收尾：逐帧 active 标签（`Expert.held` 的 FK） | 2.1 | 1.6 |
| 等 worker 关闭（其中一次 gpu.foundation 探测） | 2.1 | 4.0 |
| 合计墙钟 | 37.2 | 92.8 |

episode 本身约 9 s，其余全是每条都要付一次的进程启动和关闭。gpu.foundation 探测（Kit 隐藏 CUDA 和 Vulkan 后每次 stage attach 和 timeline stop 都会做）在等 `/usr/bin/nvidia-smi -q`（经 carb.launcher fork+exec）。并发 `nvidia-smi -q` 在 pod 上饱和 [M]：N=1/4/16/48 时 1.05/2.23/3.03/2.45 次/s（`boot/smi_conc.sh`），整个 pod 一共只有约 3 次/s。`/physics/cudaDevice=-1`（启动参数或运行时设置）让 PhysX 走另一条 CUDA 初始化路径，进程以 `_PyGILState_NoteThreadState` 崩溃；假 libcuda 会让 physx.tensors 加载失败；没有 mount namespace 权限替换系统二进制；Vulkan loader 1.2.162 不支持按设备过滤（`boot/`）。

**改动**（按 profile 的大项排序）：

| 改动 | 去掉的 profile 项 | 效果 [M] | 等价性 |
|---|---|---|---|
| 常驻 worker：`sim/physics_worker.py` 以 demo `-` 启动后接收 `EPISODE:` 消息，每个 episode 重新打开 stage（stop、清 SimulationContext、关 stage、把 episode 改过的 layer `Reload`）；JENGA_* 探针变量（layer jitter、fidelity、friction 等）随消息逐 episode 传入；`sim/expert_pool.py` 用 N 个驱动进程各带一个 worker，复用 `expert_batch.py` 的作业列表和输出布局 | 启动和关闭每个 worker 只付一次 | 无 shim、40 worker、200 条 wide：每条 setup p50 17.6 s，episode 4.7 s；3595 条/小时（旧路径 1602） | |
| `Physics` 移到 `sim/physics_client.py`（只依赖 numpy），`expert_rollout.py` 不再经 `rollout.py` import torch | 驱动 import 6.5 s | 新池里每个驱动 import 1.8 s，只付一次 | |
| `Expert.held` 先判夹爪宽度，宽度不在 35–56 mm 时不算 FK（返回值不变） | 收尾 active 标签 2.1 s | 6 条合计 2.1 s，即每条约 0.35 s | `active` 逐元素相同 |
| `sim/nosmi.c`：worker 用 LD_PRELOAD 加载，exec 的路径或参数里出现 nvidia-smi 时改为执行 `/bin/true`（探测照样失败，只是立刻失败）；`JENGA_NOSMI=0` 关闭 | 每 episode 两次 gpu.foundation 探测（40 并发时 69 s/197 s） | 4 worker 时 setup p50 3.24 → 0.78 s；40 worker 时 0.86 s | |

正确性门禁（`scripts/phys_gate.sh`、`scripts/phys_gate.py`）[M]：同一组 20 条 rollout（10 个变体各 2 条，wide 偏移 + DART 噪声），旧路径对 4 个常驻 worker（每个 worker 连跑约 5 个不同变体的 episode）：ops/lhpr/tsr/ticks/done_s 相同，qpos、blocks、actions、executed 的 max|Δ| 都是 0，active 也相同；旧路径自己跑两遍同样是 0 差。JENGA_LAYER_JITTER 下 6 条也全部相同。加 shim 后再跑 20 条，同样 20/20。rebase 到 distill f575e68 后用 d2 再跑 20 条（`--perturb d2`：task1–4 带 layer jitter，task5-2 带 JENGA_PHYS_FIDELITY=1 并做 settle），旧路径对池同样 20/20，max|Δ| 为 0（`gate_d2/`）。

**吞吐**（200 条 d2 作业：10 个变体 × 10 个 episode × 2 种布局，`scripts/phys_bench.sh`，节点上同时有其他作业）[M]：

| 路径 | 并发 | 墙钟（含 worker 启动） | 条/小时（全程） | 条/小时（25%→100% 完成区间） | 进程树 CPU-s/条 | pod cgroup 平均核数 |
|---|---|---|---|---|---|---|
| 旧 `expert_batch.py` | jobs 40 | 499 s | 1442 | 1870 | 27.2 | 50.7 |
| `expert_pool.py` + shim | workers 40 | 102 s | 7057 | 12792 | 11.0 | 37.3 |
| `expert_pool.py` + shim | workers 80 | 163 s | 4412 | 13000 | 18.1 | 41.0 |

池里每条 rollout（40 worker）[M]：setup（换 episode 重开 stage）p50 0.86 s，episode 循环 p50 4.95 s（其中专家规划 0.53 s），存盘 0.06 s；setup 的均值 9.6 s 来自每个 worker 第一条要等 Isaac 启动。80 worker 的稳态吞吐和 40 一样，但 80 个 worker 同时启动，每个第一条要等约 47 s（setup 均值 47.3 s），200 条的批里反而更慢。推荐在 115 核 pod 上用 `--workers 40`；57 核 pod 用 `--workers 20`（[H]：按 40 worker 时 pod 平均约 37 核推算，没有在 57 核 pod 上实测）。d2 s1 用这条路径 7 分钟跑完 1900 条（主线报告的数字）。

没做：PhysX 场景跨 episode 复用（每 episode 仍有 stage 打开和 timeline stop，也就仍有两次探测，只是已经不再排队等 nvidia-smi）；`scene.py` 直接调用 `simulate`/`fetch_results`（a1ac9f5，§8.3，每 tick 约省 0.2 ms），没有并入。

### 8.5 评测 fleet 的开局、收尾与常驻 worker（分支 perf-eval）

问题：学习策略快照的本地闭环评测（`scripts/fleet.sh`，51 条 val_offset，每卡 48 并发）每卡每分钟能跑几条，整次 run 的墙钟花在哪。对象：d1-2000，同一节点。观测：服务端副本日志（每 50 批一行）、远端 `/proc` 采样、py-spy（`--nonblocking`，60 s，25 Hz）。原始数据在 `runs/simdev/perf_eval/` 和一个非持久临时目录。

**旧路径一次 run 的分解**（作业 637，GPU 7 锁频 1590 MHz，pod 其余 7 卡在跑数据集写入）[M]：

| 段 | 时间 | 说明 |
|---|---|---|
| 服务编译到 ready | 38 s | 这时驱动还没启动，GPU 只在编译 |
| 驱动启动到满速 | 154 s | 渲染器加载 37 s；48 个 Isaac worker 同时启动，各条 episode 在 100 s 的范围内陆续开局。这段只完成 1515 个请求，满速应为 7115，相当于丢了约 121 s 满速时间 |
| 稳态 | 46.2 req/s | 批 24/24，副本前向 420 ms/批（17.5 ms/样本，与渲染并发），副本忙 81%；渲染约 177 ms/批 24 |
| 收尾 | 开局散布在收尾时重现，51 > 48，剩下 3 条以 b≈1–2 单独跑 | 前 48 条在 15.7 min 跑完，最后一条在 17.5 min |
| 合计 | 1130 s（作业墙钟），2.71 条/分钟/卡 | 稳态工作量 51×788 req / 46.2 = 870 s，开销 260 s（23%） |

稳态下驱动 48 个 env 线程里平均 34.8 个在等策略，8.8 个等渲染，1.3 个在锁步 sync，0.6 个在物理；驱动 GIL 忙约 27%。pod CPU 用了 28/115 核，51 个 worker 合计 2.9 核，物理和 CPU 都不在关键路径上。fleet 的渲染已经走 writer 那套 shade/refine 快路径（5192571、3175573 的 raster context 和同步修复），`render_rgbx_async` 只多一个 pinned 异步 H2D，没改。

**改动**：

| 改动 | 去掉的段 |
|---|---|
| `sim/fleet_driver.py`：每个 env 线程带一个常驻 Isaac worker（`Physics(None)`，加 nosmi shim），每条 episode 用 `EPISODE:` 重开 stage（约 1 s）；第一批 episode 全部设好后，等渲染器和服务就绪再同轮开局（`threading.Barrier`），所以同轮结束。`--fresh-workers` 恢复旧方案 | 开局和收尾的散布；多 wave 时 Isaac 重启 |
| `scripts/fleet.sh`：驱动和服务同时启动，驱动等服务 ready（连接轮询，900 s 上限） | 服务 ready 前驱动的串行等待 |
| `scripts/fleet.sh`：锁步组大小上限 24（`FLEET_GROUP_MAX`）；`FLEET_BALANCE=1` 可选均分 wave，默认关 | 见下面的否决 |
| `sim/physics_client.py`：nosmi 库编译加线程锁 | 48 个线程同时构建时的竞争（门禁里实际撞到过） |

**等价性** [M]：
- 物理：`scripts/fleet_phys_gate.py`，9 个变体各一条（带 val_offset 偏移），每条用 demo 的 leader qpos 驱动最多 2400 tick。每 episode 一个 worker（旧路径，无 shim）对 3 个常驻 worker（各连跑 3 个不同变体）：状态 max|Δ| 全部为 0（`runs/simdev/perf_eval/phys_gate9/gate.json`）。
- 闭环：`scripts/fleet_gate_compare.py`，按每个 episode 的 LHPR/TSR/ODR，加服务端记录里的动作。val_offset 51 条上，旧对旧（job 637 对 `r3_old51`）43/51 一致，新对新（`new51` 对 `r3_new51`）43/51，旧对新 43–44/51，不一致的集中在 task4 推落和 task5-2。gate16 上旧对新 13/16，新对新 14/16。首个请求的动作差 ≤0.54 mrad，在同图换种子的噪声底内（中位 0.384，p90 11.2）。轨迹在几十到几百个请求后超过噪声底最大值、开始分叉，旧路径自己重跑也一样。所以闭环本来就不能逐位复现，新路径的不一致率和旧路径重跑相同。

**吞吐**（同一张卡：该节点 GPU 3，1980 MHz，d1-2000，与 09:28–09:37 的污染窗口不重叠）[M]：

| run | 墙钟（作业步） | 条/分钟/卡 | 说明 |
|---|---|---|---|
| gate16 旧 | 477 s | 2.01 | 开局散布约 60 s |
| gate16 新 | 341 s | 2.82 | −29% |
| val_offset 51 旧（`r3_old51`，envs 48） | 1101 s | 2.78 | 稳态 47.2–47.6 req/s |
| 51 新，envs 48（`r4_new51_e48`） | 1005 s | 3.05 | −8.7%；48 条在 807 s 同时跑完，尾部 3 条在 b≈1.5 下又用 120 s |
| 51 新，一个 wave 51、两组 25/26（`r3_new51`） | 1040 s | 2.94 | 服务端批碎成均值 22、最小 1，稳态 42.5–42.9 req/s |
| 51 新，一个 wave 51、三组各 17（`r5_new51`，前 300 s 后取消） | — | — | 稳态 42.5–45.0 req/s，批 17/17 但每样本更贵；按此推算约 990 s，与 envs 48 持平 [H] |

推荐：`scripts/fleet.sh <params> <id> <list> <out> 48 <seeds>`（默认设置）。单卡 51 条的增益是 8.7%，作业分到多卡、每卡十几条时（main 为缩短等待常用的分法）增益是 29%。剩下的开销是尾部（51 = 48 + 3）。一个 wave 装下 51 条反而更慢：两组 25/26 时，同组请求在驱动的 GIL 下串行 pack/send，到达散得更开，服务端"无 staging 即关批"的规则把批切碎。三组 17 的批是整的，但小批每样本的前向更贵。

没做：稳态里副本约 19% 的空闲（host 段间隙：pack/send 在一个 GIL 下串行、adapter 的 JPEG 往返），这也是组大于 24 时碎批的根源；渲染占 GPU 时间约 39% [H]（按 177 ms 渲染 + 约 276 ms 独占前向估算）。服务端诊断记录（`--records`）每秒约 184 张 q95 JPEG，约 1 核，不在 GPU 路径上。

## 9. 否决项

| 选项 | 结果 | 出处 |
|---|---|---|
| FP8 GEMM（e4m3，per-tensor 缩放） | 算上量化开销，7 个真实形状里只有 Gemma MLP down 快 17%，其余多数更慢；相对误差 0.036，约为 bf16 的 22 倍 | `runs/bench/fp8_gemm.json` |
| Gemma attention 用 cuDNN flash attention | 自然布局后 XLA 单层 fwd+bwd 2.27 ms，cuDNN 2.39 ms；全 mask 的 padding 行还要额外处理 | `runs/bench/logs/attn_micro2.json` |
| collective combine 阈值 512 MiB | 629.7 ms，劣于不设的 618.6 ms | `s1_ovl_cmb.json` |
| `NCCL_PROTO=Simple` | 644.4 ms | `s1_nccl_simple.json` |
| NVLS（`NCCL_NVLS_ENABLE=1`） | 366.6 ms，对 366.0 无变化 | `s10_nvls.json` |
| fsdp=2（2×DP） | 662.6 ms，峰值 37.3 GiB | `s1_fsdp2.json` |
| 每步只 gather 一次权重 | 全栈下 411.0 ms，比 366.0 慢 | `s10_gather_bf16.json`，`s3_gather1*.json` |
| 完全不 remat | OOM，申请 110 GB | `runs/bench/logs/s1_remat_none.log` |
| SigLIP 不 remat | 781.4 ms | `s6_remat_s_none.json` |
| Gemma remat `dots_saveable` | 451.2 ms，峰值 55.2 GiB，存下了 fp32 logits | `s7_constr_dots.json` |
| `--xla_gpu_enable_triton_gemm=false` | 387.1 ms | `s9_notriton.json` |
| 推理 `while_loop` 换静态 `fori_loop` | 输出逐位相同，b1 37.9 对 38.0 ms | `runs/bench/infer/infer_{fori,while}.json` |
| 推理 scan 全展开 | b1 32.8 ms（不展开 33.0 ms），编译从约 11 s 变成 55–70 s | `runs/bench/infer/i_fast_unroll.json` |
| 关 EMA | 只省约 3 ms（635.3 ms），且 EMA 参数就是部署权重 | `s1_noema_attrib.json` |

推理 fp32 和闭环 episode 截短也否决了，见 §7.1 和 §8。

## 10. 剩余时间

| 阶段 | 计算忙 | NCCL 未重叠 | 空闲 | 出处 |
|---|---|---|---|---|
| 基线 638 ms | 73.3% | 21.5% | 5.2% | `runs/bench/prof/b32_fsdp4_node.nsys-rep` |
| 全栈 366 ms（预算 200） | 90.9% | 5.1% | 4.1% | `runs/bench/prof/s9_best.nsys-rep` |

占比和分项耗时用 `nsys stats` 从这两个文件算出，汇总表没落盘，引用前要重跑。基线每步大头是 GEMM 262 ms、NCCL 136 ms、transpose 94 ms；transpose 来自 SigLIP attention（head_dim 72）、4 路相机逐路调用和 Gemma prefix attention 的 fp32 logits 布局，§3 的合批和自然布局针对的就是这几处。全栈后 GEMM 约 232 ms/step，其余计算约 100 ms。Gemma MLP up 的 bf16 GEMM（9792×2048×16384，657 GFLOP）实测 0.88 ms，约 750 TFLOPS（`runs/bench/fp8_gemm.json`）。在不改模型、不改数值的约束下，训练吞吐已接近这套栈的上限。

没试过的：Auto-PGLE 和 `--xla_gpu_use_memcpy_local_p2p=true`（只改调度，上限是剩下 5% 未重叠的 NCCL）；服务端 host 侧开销（整树一次 `device_put`，msgpack 解码移出事件循环，可压缩的上限约 5 ms）。
