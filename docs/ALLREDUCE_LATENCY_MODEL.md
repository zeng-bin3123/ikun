# all_reduce 延迟模型：从 trace 到尺子（issue #1 理论依据）

**结论（3 条，均可由 `tests/python/test_latency_micro.py` 复算）**

0. 通信结构 $N=2L+1$、$s=2(p-1)$ 已由**三份独立实测**确认：$L{=}40,p{=}4$ 的 972 fence（标定）、$L{=}4,p{=}2$ 的 36 fence（零参数外推，§3.1）、infiniccl 路径每次 allreduce 多 24.3 µs（§3.2）。
1. decode 的 allreduce 处在**延迟区**。每次消息 $S_{\text{dec}} = 4\,\text{KiB}$，比交叉点 $S^\ast \approx 5.65\,\text{MiB}$ 小约 1400 倍，带宽项只占 0.07%。
2. 两份**独立**数据给出同一个 $\alpha$：带宽实测拟合 $\hat\alpha = 268.5\,\mu s$，trace 微结构反推 $\alpha_{\text{trace}} = 267.9\,\mu s$。
3. 通信归零时的 Amdahl 上限是 $17.1$ tok/s（当前 12.5，最多 +37%）。其中 one-shot 算法理论可到 $16.1$ tok/s。

---

## 1 模型

采用 Hockney 的 $\alpha$-$\beta$ 模型：

```math
T(S) = \alpha + \beta S
```

对 $p$ 个 rank 的 ring allreduce，先做 $p-1$ 步 reduce-scatter，再做 $p-1$ 步 all-gather。设每步固定开销为 $\alpha_s$、链路带宽为 $B$：

```math
T_{\text{ring}}(S) = 2(p-1)\,\alpha_s + \frac{2(p-1)}{p}\cdot\frac{S}{B}
\quad\Longrightarrow\quad
\alpha = 2(p-1)\,\alpha_s,\qquad \beta = \frac{2(p-1)}{p\,B}
```

$p=4$ 时，$\alpha = 6\alpha_s$，$\beta = 1.5/B$。

带宽项等于延迟项的**交叉点**为：

```math
S^\ast = \frac{\alpha}{\beta} = p\,\alpha_s B
```

$S \ll S^\ast$ 时，$T \approx \alpha$，这时链路带宽无关紧要。

一般地，每种 allreduce 算法 $A$ 都可以写成步数 $k_A$ 与数据系数 $c_A$ 的组合：

```math
T_A(S) = k_A\,\alpha_s + c_A\,\frac{S}{B}
```

| 算法 | $k_A$ | $c_A$ |
|---|---|---|
| ring | $2(p-1)$ | $2(p-1)/p$ |
| recursive halving-doubling | $2\log_2 p$ | $2(p-1)/p$ |
| two-shot（P2P reduce-scatter + all-gather） | $2$ | $2(p-1)/p$ |
| one-shot（各 rank 直接读全部 peer 后本地归约） | $1$ | $p-1$ |

## 2 标定：用已有带宽实测拟合 $\alpha,\beta$

带宽基准打印的数值是 $G = 2SR/(t\cdot 2^{30})$，其中 $R=20$ 为轮数。据此反推单次耗时：

```math
T_i = \frac{t_i}{R} = \frac{2S_i}{G_i\cdot 2^{30}}
```

| $S$ | $G$ (打印) | $T$ (µs) |
|---|---|---|
| 1 MiB | 6.1 | 320.2 |
| 16 MiB | 31.7 | 985.8 |
| 64 MiB | 37.5 | 3333.3 |
| 256 MiB | 39.1 | 12787.7 |

**噪声模型。** 计时噪声是乘性的：$T_i = (\alpha+\beta S_i)(1+\varepsilon_i)$，$\operatorname{Var}\varepsilon_i = \sigma^2$。这意味着 $\operatorname{Var}T_i \propto T_i^2$，因此最优估计是权重 $w_i = 1/T_i^2$ 的加权最小二乘。正规方程为：

```math
\begin{pmatrix}\sum w_i & \sum w_i S_i\\ \sum w_i S_i & \sum w_i S_i^2\end{pmatrix}
\begin{pmatrix}\hat\alpha\\ \hat\beta\end{pmatrix}
=
\begin{pmatrix}\sum w_i T_i\\ \sum w_i S_i T_i\end{pmatrix}
```

**结果。**

```math
\hat\alpha = 268.5\,\mu s,\quad
\hat\beta = 47.54\,\mu s/\text{MiB},\quad
B_{\text{eff}} = 1/\hat\beta = 22.06\,\text{GB/s},\quad
B = 1.5\,B_{\text{eff}} = 33.1\,\text{GB/s}
```

- 推出的链路带宽 $B$ 与 PCIe 4.0 x16 单向理论值 31.5 GB/s 相符（偏差在拟合误差内）。
- 相对残差为 $+1.3\%,\,-4.4\%,\,+0.7\%,\,+2.7\%$，由此估得 $\hat\sigma_{\text{rel}} = 3.8\%$。
- 标准误为 $\operatorname{SE}(\hat\alpha) = 12.1\,\mu s$（4.5%），$\operatorname{SE}(\hat\beta) = 2.6\%$。
- 由此得 $\hat S^\ast = 5.65$ MiB，进而 $\alpha_s = \hat\alpha/6 = 44.75\,\mu s$。

作为对照，普通最小二乘给出 $\alpha = 225\,\mu s$，但在 1 MiB 处残差达 14%。原因是它被 256 MiB 那一个点的绝对误差主导，而这不符合乘性噪声的假设。

## 3 独立交叉验证：trace 微结构

数据来自 `docs/PROFILE_DECODE_STEP_20260923.md`，是单个 decode step 内的计数。设每步 allreduce 次数为 $N = 2L+1 = 81$，对应每层 attention 输出一次、MoE 输出一次，再加 embedding 一次。各计数除以 $N$：

```math
\frac{972}{81} = 12 = 2\cdot 6\ (\text{fenceWait}),\quad
\frac{486}{81} = 6 = 1\cdot 6\ (\text{fenceOps}),\quad
\frac{243}{81} = 3 = p-1\ (\text{kernelCopy}),\quad
\frac{243}{81} = 3 = p-1\ (\text{elementWise})
```

这正好是 $2(p-1) = 6$ 步 ring 的结构：
- 每步有 2 次 fenceWait 和 1 次 fenceOps。
- reduce-scatter 的 3 步各有 1 个 elementWise（归约），all-gather 的 3 步各有 1 个 kernelCopy。

在 $p=4$、每步 1 个 kernel 的前提下，只有 $N=81$ 能让四个比值都是整数且符合这个结构。

把这四类耗时加总再除以 $N$：

```math
\alpha_{\text{trace}} = \frac{(10.9 + 5.1 + 2.9 + 2.8)\,\text{ms}}{81} = 267.9\,\mu s,
\qquad
\frac{|\alpha_{\text{trace}} - \hat\alpha|}{\hat\alpha} = 0.2\%
```

单步开销约为 $\alpha_s \approx 3\times 11\,\mu s\ (\text{fence}) + 12\,\mu s\ (\text{kernel}) = 45\,\mu s$，与 §2 的 44.75 µs 一致。

**注意：** trace 里的耗时已四舍五入到 0.1 ms，所以 0.2% 的吻合带有巧合成分。真正的硬证据是整数比结构：它确定了 $N$ 和 $k_{\text{ring}}$。

### 3.1 零参数外推验证（另一套配置）

上面的结构只在一套配置（$L=40$, $p=4$）上标定过。PR 10 session 5 另有一次**完全不同配置**的实测：4 层 mini 模型、TP=2，报告 fence 数为 36。把公式原样外推，不引入任何新参数：

```math
\text{fenceWait} = 2s N = 2\cdot 2(p-1)\cdot(2L+1)
```

| 配置 | $s=2(p-1)$ | $N=2L+1$ | 预测 | 实测 | |
|---|---|---|---|---|---|
| $L=40,\ p=4$（标定用） | 6 | 81 | 972 | 972 | ✓ |
| $L=4,\ p=2$（外推） | 2 | 9 | **36** | 36 | ✓ |

**可辨识性。** 在假设族 $N=aL+b,\ s=c(p-1)+d$（$a,c\in[0,3]$，$b,d\in[-2,2]$，共 400 组）中穷举：只用 972 这一个点，唯一解是 $(a,b,c,d)=(2,1,2,0)$；再加上 36 这个点，解不变。两个差异极大的配置同时落在同一组参数上，说明这不是过拟合。被排除的典型替代假设包括：每层一次 allreduce（$N=L$，预测 480 ≠ 972），以及 fence 数与 $p$ 无关（预测 108 ≠ 36）。

**session 5 当时的解读需要修正。** 原话说 972→36 的下降"有误导性，因为 mini 模型只有 4 层"。方向是对的，但按上式，层数只解释其中 9 倍（$81/9$），另外 3 倍来自 $p$ 从 4 降到 2 使步数减半（$6/2$）。这 3 倍正是 §5 里 TP=2 方案的收益来源，它已经在真实硬件上被观测到了。

### 3.2 第三份数据：infiniccl 的额外开销

session 5 报告 infiniccl 路径 12.2 tok/s，基线 12.5 tok/s，并归因于 Python dispatch 多了一层。换算：

```math
\Delta T_{\text{step}} = \frac{1000}{12.2}-\frac{1000}{12.5} = 1.97\ \text{ms},
\qquad \frac{\Delta T_{\text{step}}}{N} = \frac{1970\,\mu s}{81} = 24.3\,\mu s
```

每次 allreduce 多出 24.3 µs，与 PRD 里记录的"每次 PyTorch 函数调用约 25 µs"一致。这从第三个角度确认了 $N=81$：若每步只有 40 次 allreduce，则每次要多花 49 µs，就与已知的 dispatch 开销对不上了。

## 4 decode 通信代价

```math
T_{\text{comm}}(b) = N\big(\alpha + \beta\, b\, h\, s_d\big),\qquad N = 2L+1 = 81,\ h = 2048,\ s_d = 2
```

**$b = 1$ 时：**
- 带宽项 $\beta h s_d = 0.186\,\mu s$，只占 $0.07\%$。
- $T_{\text{comm}} = 81 \times 268.7\,\mu s = 21.76\,\text{ms}$，占 80.2 ms 的 27.1%。trace 里 NCCL 占 25.3%，加上 copy/elementWise 约 27%，两者一致。

**批大小的交叉点：**

```math
b^\ast = \frac{\alpha}{\beta h s_d} = 1446
```

对任何现实的 batch，都有 $T_{\text{comm}}(b)/b \approx N\alpha/b$，即通信成本主要靠 batch 摊薄。

**Amdahl 上限：** 每步其余部分耗时 $T_{\text{other}} = 80.2 - 21.76 = 58.4$ ms，所以

```math
\text{tok/s} \le \frac{1}{T_{\text{other}}} = 17.1\ \text{tok/s}\quad(+37\%)
```

**推论：** A800 与 BI-V100 的 8–10 倍差距，主要不来自通信（NVLink），而来自其余这 58 ms（GEMM、memcpy、cast、launch）。

## 5 换算法的理论收益

在 $\alpha_s = 44.75\,\mu s$ 且对各算法不变的假设下：

| 算法 | $k$ | $T(4\text{KiB})$ | $81\times T$ | step | tok/s |
|---|---|---|---|---|---|
| ring $p=4$（现状） | 6 | 268.7 µs | 21.76 ms | 80.2 ms | 12.5 |
| RHD $p=4$ | 4 | 179.2 µs | 14.51 ms | 73.0 ms | 13.7 |
| two-shot | 2 | 89.7 µs | 7.26 ms | 65.7 ms | 15.2 |
| one-shot | 1 | 45.1 µs | 3.65 ms | 62.1 ms | 16.1 |
| ring TP=2（需 W8） | 2 | 89.6 µs | 7.26 ms | 65.7 ms | 15.2 |

两点说明：
- **one-shot 需要与 $p-1$ 个 peer 同步**，有效步数 $k_{\text{eff}} \in [1, p-1]$，所以收益区间是 3.65–10.9 ms（对应 16.1–14.4 tok/s）。
- **TP=2 + W8 的可行性。** 每卡读取的权重字节数与现状相同：$\tfrac{35\text{B}\times 1\,\text{B}}{2} = \tfrac{35\text{B}\times 2\,\text{B}}{4}$，所以 GEMM 在访存上不变慢，前提是天数上有可用的 W8 GEMM。

## 6 尺子的统计设计

### 6.1 中位数的估计误差

把单次延迟建模为对数正态：$X \sim \operatorname{LogNormal}(\ln m, s^2)$。$n$ 个样本的中位数渐近正态，方差为 $1/(4nf(m)^2)$，其中 $f(m) = 1/(m s\sqrt{2\pi})$。由此：

```math
\operatorname{CV}_{\text{med}} = \sqrt{\frac{\pi}{2}}\,\frac{s}{\sqrt n},
\qquad n = 200:\ \operatorname{CV}_{\text{med}} = 0.0886\,s
```

也可以用非参数方法：由二项分布，$[X_{(86)}, X_{(115)}]$ 覆盖真实中位数的概率为 $0.960$。

$s$ 用四分位距做稳健估计：$\hat s = \ln(p_{75}/p_{25})/(2z_{0.75})$，其中 $z_{0.75} = 0.6745$。

### 6.2 门禁的通过概率

第 $r$ 次重跑的 p50 写成 $M_r = m(1 + \delta_r + \eta_r)$：
- $\delta_r \sim \mathcal N(0, \tau^2)$ 是轮间漂移（频率变化、其他租户等）。
- $\eta_r \sim \mathcal N(0, \operatorname{CV}_{\text{med}}^2)$ 是采样误差。

合起来 $\sigma_{\text{rel}}^2 = \tau^2 + \operatorname{CV}_{\text{med}}^2$。门禁统计量近似为：

```math
V = \frac{\max_r M_r - \min_r M_r}{\operatorname{median}_r M_r} \approx \sigma_{\text{rel}}\, W_3
```

其中 $W_3$ 是 3 个 iid $\mathcal N(0,1)$ 的极差，其分布为：

```math
F_{W_3}(w) = 3\int_{-\infty}^{\infty}\varphi(x)\big[\Phi(x+w) - \Phi(x)\big]^2\,dx,
\qquad \mathbb E[W_3] = 1.693,\quad q_{0.95} = 3.314
```

因此：

```math
P(V \le 5\%) \ge 0.95 \iff \sigma_{\text{rel}} \le \frac{5\%}{3.314} = 1.51\%
```

### 6.3 设计含义

- 若轮间无漂移（$\tau = 0$），200 次迭代能容忍的单次抖动为 $s \le 1.51\%/0.0886 = 17.0\%$。
- 所需最小迭代数为 $n_{\min}(s) = \lceil \tfrac{\pi}{2}(s/\sigma^\ast)^2 \rceil$。脚本会打印每档的 $\hat s$，当 $n_{\min} > n$ 时给出 `⚠iters≥n_min` 警告。
- 增大 $n$ 压不住 $\tau$。漂移只能靠锁频、独占节点来消除。
- 实例：在 1 核 CPU 上用 gloo 试跑，测得 $\hat s = 87.6\%$，脚本给出 $n_{\min} = 5291$，门禁判为不通过。这与理论预期一致。

### 6.4 可辨识性

对 5 档微基准（4 KiB–1 MiB）做 WLS，取每点 $\sigma_{\text{rel}} = 1.5\%$，由 Fisher 信息得：

```math
\operatorname{SE}(\hat\alpha) = 2.2\,\mu s\ (0.83\%),\qquad \operatorname{SE}(\hat\beta) \approx 11\%
```

原因是所有档位都满足 $S < S^\ast$，1 MiB 处带宽项也只占 $T$ 的 15%。所以分工是：**$\alpha$ 由本基准定，$\beta$ 由大包带宽基准定。**

### 6.5 计时偏差

- 主机计时会多算一次同步开销 $\varepsilon_{\text{sync}}$。脚本在空队列上单独测量它，拟合前扣除。
- 每个分位数取所有 rank 中的最大值：集合通信由最慢的 rank 决定，所以这是一个保守上界。

## 7 可证伪预测（在 4×BI-V100 上跑 `python3 bench/latency_micro.py` 检验，安全协议见 §9）

```math
\hat T(S) = 268.5 + 47.54\cdot S/\text{MiB}\ \ [\mu s]
```

| 档位 | 4 KB | 16 KB | 64 KB | 256 KB | 1 MB |
|---|---|---|---|---|---|
| 预测 p50 (µs) | 268.7 | 269.2 | 271.5 | 280.4 | 316.0 |

**判据。** 带宽拟合只有 2 个自由度，$t_{0.975,2} = 4.30$，所以 $\hat\alpha$ 的 95% 区间是 $268.5 \pm 52\,\mu s = [216, 321]\,\mu s$。

| 实测 $\alpha_{\text{micro}}$ | 解读 |
|---|---|
| 落在区间内，且接近 $\alpha_{\text{trace}}$ | 模型成立。优化杠杆是**步数 $k$**，不是链路 $B$，按 §5 排优先级 |
| $\alpha_{\text{micro}} < 216\,\mu s$ | 带宽基准的 1 MiB 点里混有其他开销，trace 中的 fence 另有来源，需要重查 |
| $\alpha_{\text{micro}} > 321\,\mu s$ | `torch.distributed` 调用路径上有额外的逐次开销，推理框架实际走的路径可能更快，需要对照 |

## 8 假设与局限

- 假设带宽基准与 trace 来自同一拓扑（4×BI-V100，TP=4，同一 NCCL/ixccl 版本）。数据类型不影响 $\alpha$。
- trace 的平均耗时已四舍五入。§3 中的整数比是硬证据，0.2% 的吻合不是。
- 假设 $\alpha_s$ 与算法无关，这对 one-shot 偏乐观（见 §5 的区间）。
- 单次延迟服从对数正态是一个假设。IQR 估计 $\hat s$ 对重尾是稳健的。
- decode 实际运行时 allreduce 之后不做主机同步，但 fence 本身就是同步，所以主机计时与设备端耗时应当接近。本基准会直接检验这一点：看 $\alpha_{\text{micro}}$ 与 $\alpha_{\text{trace}}$ 是否一致。

## 9 安全协议：为什么这个基准不会把机器卡死

### 9.1 致坏机制与三条中断链

PR 10 的记录显示，在 corex 3.2.x 上，**中断在途的 NCCL/P2P 通信会让驱动状态永久损坏**，而且 GPU 资源不会被释放，已经因此损失 3 台机器。所以关键不在于"卡住以后怎么恢复"（没有软件手段可以恢复），而在于：**任何进程在其他 rank 还卡在与它的通信中时，都不能被强行终止。**

常见做法里有三条链路会触发这种中断：

| 链路 | 触发条件 | 本基准的处理 |
|---|---|---|
| torchrun | 任一 worker 失败，就向其余 worker 发 SIGTERM，超时后发 SIGKILL | 不用 torchrun。自带启动器从不 kill 子进程 |
| torch ProcessGroupNCCL 看门狗 | 集合通信超时或心跳丢失时 abort 通信器或杀进程 | 设 `TORCH_NCCL_ASYNC_ERROR_HANDLING=0`、`TORCH_NCCL_ENABLE_MONITORING=0`，数据组超时设为 24h |
| 外部看门狗（PR 10 的 `tools/gpu_watchdog.py`） | 60 秒无输出就 SIGTERM，再 SIGKILL | 不要套用它。本基准 stdout 只在结束时输出 6 行，套用必然误杀 |

另外，PR 10 的 `nccl_cleanup` 用 Python 信号处理函数去调 `ncclCommAbort`。这有两个问题：
- 主线程阻塞在卡住的集合通信里时，Python 信号处理函数根本得不到执行机会。
- 即使执行了，它在对端仍在通信时 abort，本身就是一次中断。它自己的文档也承认 abort/reset 无效。

### 9.2 协议

记数据阶段为 $P_1,\dots,P_K$，依次是 canary（NCCL 通信器初始化加一次 4 KiB 自检）和 $R\times 5$ 个扫描阶段。每个阶段之前有一次 CPU 共识 $C_k$：在 gloo 控制面上对向量 $(\text{fail}_r, \text{stop}_r)$ 做 MAX-allreduce。

```math
\text{rank } r \text{ 进入 } P_k \iff C_k \text{ 在 } r \text{ 上返回 } (0,0)
```

阶段内部只有被测的 all_reduce 和 synchronize。阶段结束时各 rank 同步自己的 stream，所以**每个共识点上，参与投票的 rank 的 GPU 上都没有在途操作**。

### 9.3 不变式

**命题。** 若 rank $r$ 进入 $P_k$，则每个 rank $r'$ 在向 $C_k$ 贡献时都满足：健康（$\text{fail}_{r'}=0$），且未收到停止信号（$\text{stop}_{r'}=0$）。

**证明。** MAX-allreduce 的结果为 $(0,0)$，当且仅当所有贡献都是 $(0,0)$。$\square$

**推论（卡死的必要条件）。** $P_k$ 中发生卡死，必须有某个 rank 在窗口 $W_k = [t_{C_k},\, t_{\text{end}}(P_k)]$ 内失效。在这个窗口里：
- SIGINT/SIGTERM 只会置标志，不会导致退出。
- 本基准不发 SIGKILL，也不存在 torchrun 或 torch 看门狗的 abort。

因此，剩下的失效原因只有外部 SIGKILL（如 OOM killer 或人工 `kill -9`）以及硬件/驱动故障。窗口之外发生的任何失败，都会在下一个共识点上被 CPU 侧发现：对端缺席时 gloo 报错，所有 rank 带退出码 2/3/4 自行退出。

**暴露窗口。** 按 §7 的预测延迟估算（$w=20$，$n=200$，$R=3$）：

```math
|W| \approx \sum_k |W_k| \approx R\,(w+n)\sum_{S} \hat T(S) = 3 \times 220 \times 1405.8\,\mu s \approx 0.93\,\text{s}
```

在此之外，只需再加 canary 里通信器初始化的时间。整次运行中，其余时间（import、CUDA 初始化、预分配、统计汇总）出任何错都只发生在 CPU 侧。

### 9.4 残余风险与卡住后的处理

- **阶段内异常**（例如设备报错）：本 rank 不退出，原地停车。理由是对端可能正卡在与它的通信中，拆掉自己这一端是更大的风险。
- **疑似卡住**（单个阶段超过 `--hang-after` 秒）：只在 stderr 报告，并写入 `/tmp/latency_micro.rank{r}.hang.json`。启动器到 `--deadline` 时同样只报告各 rank 的进程状态（`/proc/<pid>/status`），不 kill。
- **卡住后请保留现场**：记录现场文件、`/proc/<pid>/status`、`ixsmi` 的输出。不要 `kill -9`。"卡住但未被中断"的进程能否靠重启恢复，在 corex 上尚无数据；但"被中断"已知会致坏。

### 9.5 分级上机顺序

1. 在目标机器上先跑 `python3 bench/latency_micro.py --cpu`：验证协议和环境，不碰 GPU。
2. 在已受损、但仍有可用卡的机器上，用 `CUDA_VISIBLE_DEVICES` 选两张卡跑 `--nproc 2`：先在低价值硬件上验证真实的 NCCL 路径。
3. 最后在干净的 4 卡机器上跑 `python3 bench/latency_micro.py`。

**验证。** `tests/python/test_latency_micro_safety.py` 用故障注入（崩溃、异常、健康检查失败、SIGTERM）在 CPU/gloo 上检验 §9.3 的不变式：所有 rank 进入过的数据阶段序列必须完全相同，且不包含故障发生处的阶段；所有进程必须自行退出。
