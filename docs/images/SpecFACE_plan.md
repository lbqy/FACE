# SpecFACE: FACE + Speculative Decoding 单 Instance 微架构探索方案

## 0. 目标与设计边界

本文档提出一个精炼版 SpecFACE 设计，用于在现有 FACE simulator-level 架构上探索 speculative decoding (SD) 在单个 instance 内的资源映射、流水 overlap、动态 gamma、fallback 与 KV journal 机制。这里不引入高维复杂 LUT；核心思想是用一个轻量 Operator Mapping Engine (OME) 做小规模搜索，并用 analytical evaluator 直接估计每个候选映射的成本。

本阶段目标不是复现完整系统论文图表，而是形成可写成小论文的微架构探索框架：

- 固定单个 instance 的主要架构参数，保留 JSON 配置接口用于扫参。
- 默认接受率 `alpha = 0.8`，保留静态与动态扫参接口。
- 研究 target prefill、target verify、draft prefill、draft decode、KV commit/free 在 core、SRAM、HBM、NoC 上的映射与 overlap。
- 单个 request 内保持 SD 依赖顺序；通过多 cohort 并发挖掘 target 内部、draft 内部、target-draft 之间的 overlap。
- draft speculative KV 优先写入 SRAM journal，verify 后只将 accepted prefix 推入 HBM，rejected suffix 直接释放。

当前 FACE baseline 的 HBM、SRAM、compute、NoC 参数仍来自 `examples/llm_serving/face/face_wsc_config.json`，但本实验的单 instance 拓扑按 core-group 粒度建模：`2x2` dies，每个 die 有 `2x4` core groups，总计 `32` core groups，每个 group 内 `16` cores。SpecFACE 第一阶段固定为单 instance 内探索，OME 搜索变量是 workflow 需要多少 core groups，而不是直接枚举裸 core 数；跨 instance 调度留作后续。

## 1. 执行流与状态机

对每个请求 `r`，SpecFACE 的单请求依赖顺序为：

```text
TargetPrefill(r) and DraftPrefill(r)
  -> repeat until output complete:
       DraftDecode(r, gamma)
       TargetVerify(r, gamma)
       KVCommitFree(r)
       DraftSync(r, 1)     // target correction/bonus token 的 draft KV 补齐
```

说明：

- `TargetPrefill` 生成 target model prompt KV。
- `DraftPrefill` 生成 draft model prompt KV。
- `DraftDecode(gamma)` 自回归地产生 `gamma` 个 speculative tokens，并把 draft speculative KV 写入 SRAM journal。
- `TargetVerify(gamma)` 用 target model 对 draft tokens 做 batched verification，同时产生 target speculative KV。
- `KVCommitFree` 将 accepted prefix 的 target KV 与 draft KV 标记为 committed；draft rejected suffix 从 SRAM journal 释放；target rejected suffix 若已写入 HBM speculative buffer，则释放或回收。
- `DraftSync(1)` 为 target 生成的 correction/bonus token 补一份 draft KV，保证下一轮 draft context 与 target committed context 对齐。

单个请求内部不能乱序：同一请求的 `TargetVerify` 必须等待其 `DraftDecode` 完成，`KVCommitFree` 必须等待 `TargetVerify` 完成。但不同请求或 cohort 间可以 overlap：例如 cohort A 的 `TargetVerify` 与 cohort B 的 `DraftDecode` 同时运行，cohort C 的 `TargetPrefill` 与 cohort D 的 `TargetVerify` 在 target model 内部 overlap。

## 2. 接受率模型与 SD 收益上界

设每个 draft token 被 target 接受的概率为 `alpha`，默认 `alpha = 0.8`。一次 SD round 提议 `gamma` 个 tokens。令 `K` 为连续接受的 draft token 数，截断于 `gamma`：

```text
P(K = k) = alpha^k (1 - alpha), 0 <= k < gamma
P(K = gamma) = alpha^gamma
```

若采用标准 speculative decoding 的 bonus/correction token 规则，每轮最终提交 token 数为：

```text
R = K + 1
```

其中当 `K < gamma` 时，额外 1 个 token 是 target correction token；当 `K = gamma` 时，额外 1 个 token 是 target bonus token。于是：

```text
E[K] = sum_{i=1}^{gamma} P(K >= i)
     = sum_{i=1}^{gamma} alpha^i
     = alpha (1 - alpha^gamma) / (1 - alpha)

E[R] = 1 + E[K]
     = (1 - alpha^(gamma + 1)) / (1 - alpha)
```

当 `alpha = 0.8` 时：

| gamma | E[K] | E[R] | E[R]/gamma |
|---:|---:|---:|---:|
| 1 | 0.800 | 1.800 | 1.800 |
| 2 | 1.440 | 2.440 | 1.220 |
| 4 | 2.362 | 3.362 | 0.840 |
| 8 | 3.329 | 4.329 | 0.541 |

`E[R]` 随 `gamma` 增长趋近 `1/(1-alpha)`，但 draft decode 成本、verify 成本和 SRAM journal 压力会随 `gamma` 增长。因此动态 gamma 目标不是最大化 `E[R]`，而是最小化每个 committed token 的期望时间：

```text
Cost_SD(gamma) = T_round(gamma) / E[R(gamma, alpha)]
```

若 `Cost_SD(gamma)` 不低于纯 FACE decode 的 per-token 成本，应 fallback。

## 3. Operator 代价模型

### 3.1 统一 roofline 形式

对 operator `o`，给定分配 core 数 `c_o`，计算量 `Ops_o`，HBM 流量 `BytesHBM_o`，NoC 流量 `BytesNoC_o`，使用 roofline 估计：

```text
T_o(c_o) = max(
  Ops_o / (c_o * F_core * eta_compute_o),
  BytesHBM_o / (B_HBM * beta_hbm_o),
  BytesNoC_o / (B_NoC * beta_noc_o)
) + T_barrier_o
```

其中：

- `F_core` 是每 core 计算吞吐，默认 740 GFLOP/s。
- `B_HBM` 是单 instance 总 HBM 带宽。
- `B_NoC` 由 NoC width、core 数和频率等参数估计。
- `eta_compute_o` 是 operator 的 compute efficiency，可扫参。
- `beta_hbm_o`、`beta_noc_o` 表示多流并发时该 operator 获得的有效带宽份额。
- `T_barrier_o` 覆盖 layer barrier、all-reduce、local synchronization 等固定成本。

本设计不使用复杂 LUT。OME 对每个候选 core partition 直接调用该 evaluator，得到 operator duration，再用小型事件模拟计算 overlap 与 makespan。

### 3.2 Target/Draft prefill

对模型 `m in {target, draft}`，prompt chunk 长度为 `P`，hidden size 为 `H_m`，层数为 `L_m`。沿用当前 FACE evaluator 的简化 transformer 成本：

```text
Ops_prefill_m(P)
  = L_m * [12 * P * H_m^2 + 2 * P^2 * H_m]
```

其中 `12 * P * H^2` 近似 QKV/O projection + MLP，`2 * P^2 * H` 近似 attention score/value。对应时间：

```text
T_target_prefill = T_o(Ops_prefill_target(P), c_TP)
T_draft_prefill  = T_o(Ops_prefill_draft(P),  c_DP)
```

### 3.3 Draft decode

Draft decode 自回归地产生 `gamma` 个 token，不能完全当成一个无依赖 batch。上下文长度为 `N` 时，第 `i` 个 draft token 的 attention context 约为 `N+i`：

```text
Ops_draft_decode(gamma, N)
  = sum_{i=0}^{gamma-1} L_d * [12 * H_d^2 + 2 * (N+i) * H_d]
  = L_d * [12 * gamma * H_d^2
           + 2 * H_d * (gamma * N + gamma(gamma-1)/2)]
```

对应 HBM 读取包括 draft committed KV 读取；draft speculative KV 写入 SRAM journal，不立即写 HBM：

```text
BytesHBM_DD ~= gamma * KV_read_draft(N) + activation_bytes
BytesSRAMJournal_DD = gamma * KV_draft_per_token
```

### 3.4 Target verify

Target verify 对 `gamma` 个 draft tokens 做 batched verification，近似为一个 query length 为 `gamma` 的 mini-prefill，且需要 attend 到 committed context `N` 与 speculative prefix：

```text
Ops_target_verify(gamma, N)
  = L_t * [12 * gamma * H_t^2
           + 2 * H_t * (gamma * N + gamma(gamma+1)/2)]
```

如果 target verify 生成的 speculative target KV 先进入 HBM speculative buffer：

```text
BytesHBM_TV_write = gamma * KV_target_per_token
```

如果后续实现也允许 target KV 走 SRAM staging，可把这部分写入拆成 `BytesSRAM_TV_write` 与 commit 时的 HBM write。本阶段先固定 draft KV 走 SRAM journal，target KV 可先按 HBM speculative buffer 建模。

### 3.5 KV commit/free

KV commit/free 通常不消耗核心计算，但消耗 SRAM/HBM/metadata pipeline。期望 accepted draft KV 数为 `E[K]`，rejected draft KV 数为：

```text
E[RejectedDraftKV] = gamma - E[K]
```

SRAM journal 节省的 draft HBM 写流量期望值为：

```text
SavedHBM_DraftKV(gamma, alpha)
  = (gamma - E[K]) * KV_draft_per_token
```

commit/free 时间建模为：

```text
T_commit_free = max(
  E[K] * KV_draft_per_token / B_HBM_commit,
  MetadataOps(gamma) / B_meta,
  FreeBytes(gamma, alpha) / B_free
)
```

如果 `T_commit_free` 较小，可放到 copy/metadata timeline，而不是占用 core partition。

### 3.6 DraftSync

每轮 target 会产生一个 correction/bonus token。Draft model 需要为这个 target-generated token 补齐 draft KV，否则下一轮 draft context 不一致。将其建模为一个 known-token forward：

```text
Ops_draft_sync(N) ~= L_d * [12 * H_d^2 + 2 * (N + E[R]) * H_d]
```

它可以与其他 cohort 的 target verify 或 target prefill overlap。

## 4. Core-group 级资源分配搜索

### 4.1 搜索变量

单 instance 总 core group 数为：

```text
G = die_rows * die_cols * group_rows_per_die * group_cols_per_die
```

将 core groups 分配给四个主要 compute pools：

```text
g_TP: target prefill core groups
g_TV: target verify core groups
g_DP: draft prefill core groups
g_DD: draft decode/sync core groups
```

约束为：

```text
g_TP + g_TV + g_DP + g_DD <= G - g_reserved
g_i >= g_i_min
g_i mod granularity == 0
```

其中 `granularity` 是 core-group 搜索粒度，例如 1/2/4 groups。`KVCommitFree` 默认不分配 core group，只消耗 memory/copy timeline；如果后续发现 metadata/copy 成为瓶颈，可加入 `g_KV` 或 DMA lane 数。

### 4.2 轻量搜索方法

不采用高维 LUT，而采用“解析 seed + 局部枚举”的 OME 搜索。

1. 估计各阶段 work：

```text
A_TP = Ops_target_prefill(P)
A_TV = Ops_target_verify(gamma, N)
A_DP = Ops_draft_prefill(P)
A_DD = Ops_draft_decode(gamma, N) + Ops_draft_sync(N)
```

2. 生成三个 seed partition：

延迟优化 seed，适合单 cohort 总时延近似 `sum A_i/g_i`：

```text
g_i ∝ sqrt(A_i)
```

流水吞吐 seed，适合多 cohort pipeline 近似 `max A_i/g_i`：

```text
g_i ∝ A_i
```

FACE 继承 seed，保留 target-heavy 设计：

```text
g_TP + g_TV ~= 70%~85% G
g_DP + g_DD ~= 15%~30% G
```

3. 将 seed snap 到 `granularity` 的整数倍，满足 `g_i_min`。

4. 在每个 seed 附近做 bounded local search：

```text
for each seed:
  for delta_TP, delta_TV, delta_DP, delta_DD in {-2g, -g, 0, g, 2g}:
    if constraints satisfied:
      evaluate(candidate)
select min objective
```

5. 目标函数：

```text
Objective = T_effective_per_token
          + lambda_tail * P95_queue_delay
          + lambda_mem  * JournalPressure
          + lambda_hbm  * HBMPressure
```

其中：

```text
T_effective_per_token = Makespan(window) / CommittedTokens(window)
JournalPressure = max(0, JournalBytesNeeded / JournalCapacity - 1)
HBMPressure = max(0, HBMBytesNeeded / HBMCapacity - 1)
```

这个搜索空间很小：若 `G=32`、`granularity=1`，全量枚举四元组也很小；局部枚举更轻，可以 runtime 触发。

### 4.3 启发式 core layout

使用矩形区域映射，避免细碎 tiling：

```text
+---------------------------+
| Target Prefill            |
| large contiguous region   |
+-------------+-------------+
| Target      | Draft       |
| Verify      | Decode      |
|             | + Journal   |
+-------------+-------------+
| Draft Prefill / Sync      |
+---------------------------+
```

布局原则：

- `TargetPrefill` 区域最大，保持与 FACE 中 target-heavy 数据流兼容。
- `TargetVerify` 放在靠近 target KV/HBM path 的区域，减少 verify 读 committed KV 的 NoC 距离。
- `DraftDecode` 做成紧凑 island，旁边绑定一组 SRAM journal cores/banks，降低 speculative KV 写入和释放成本。
- `DraftPrefill` 与 `DraftDecode` 邻近，复用 draft weights/KV path。
- KV commit/free 不占 compute region，使用 memory/copy timeline；必要时给 journal island 配置 metadata lane。
- 所有 region 尽量保持长宽比不过分极端，便于 ring/mesh collective 与广播。

## 5. Pipeline 与 overlap 机制

### 5.1 三类 overlap

SpecFACE 不再用一个 scalar overlap 描述全部收益，而记录三类 overlap：

```text
TargetInternalOverlap:
  TargetPrefill(cohort j) overlaps TargetVerify(cohort i)

DraftInternalOverlap:
  DraftPrefill(cohort j) overlaps DraftDecode/DraftSync(cohort i)

CrossModelOverlap:
  TargetPrefill/Verify overlaps DraftPrefill/Decode/Sync
```

总体 overlap 用资源 timeline 的 union 计算，避免重复计数：

```text
OverlapRatio_resource
  = 1 - BusyUnionTime_resource / SumBusyTime_resource
```

至少记录 compute、HBM、NoC、SRAM journal 四条 timeline。

### 5.2 多 cohort 调度

Cohort 是同一阶段、相近上下文长度和相近 gamma 的请求集合。单个请求内部依赖严格，但调度器可在不同 cohort 间并发：

- target pool：`TargetPrefill` 与 `TargetVerify` 共享 target cores，可并行或分时。
- draft pool：`DraftPrefill`、`DraftDecode`、`DraftSync` 共享 draft cores，可并行或分时。
- memory pool：`KVCommitFree` 与 HBM/KV read/write 共享带宽。

优先级建议：

```text
TargetVerify > KVCommitFree > DraftDecode > TargetPrefill > DraftPrefill > DraftSync
```

原因：`TargetVerify` 决定已投机 token 能否提交；`KVCommitFree` 释放 SRAM journal，避免后续 draft decode 被 journal 阻塞；`DraftDecode` 影响下一轮 token 供给；prefill 影响新请求 TTFT，但在 decode-heavy serving 中不应饿死 verify。

### 5.3 事件调度伪代码

```pseudo
procedure SCHEDULE_WINDOW(active_requests, instance_state):
    alpha_hat <- acceptance_monitor.estimate()
    gamma <- CHOOSE_GAMMA(alpha_hat, instance_state)
    partition <- OME_SEARCH_PARTITION(gamma, active_requests, instance_state)
    layout <- MAP_RECTANGULAR_REGIONS(partition)

    while event_queue not empty and time < window_end:
        ready_ops <- COLLECT_READY_OPERATORS(active_requests)
        cohorts <- FORM_COHORTS(ready_ops, by=(phase, context_bucket, gamma))

        for cohort in SORT_BY_PRIORITY(cohorts):
            if not HAS_DEPENDENCIES_SATISFIED(cohort):
                continue
            if not HAS_MEMORY_BUDGET(cohort, gamma):
                STALL(cohort, reason="memory")
                continue
            if not HAS_RESOURCE_SLOT(cohort, partition):
                continue

            estimate <- EVALUATE_OPERATOR_OR_PIPELET(cohort, partition, layout)
            RESERVE_TIMELINES(estimate.compute, estimate.hbm, estimate.noc,
                              estimate.sram_journal)
            ENQUEUE_COMPLETION_EVENT(cohort, time + estimate.duration)

        PROCESS_NEXT_EVENT()
```

## 6. 动态 gamma

### 6.1 接受率统计器

维护 EWMA 接受率：

```text
alpha_hat_t = rho * alpha_hat_{t-1} + (1-rho) * (accepted_tokens_t / proposed_tokens_t)
```

默认 `rho = 0.9`。可按 request class 或 context length bucket 维护多个统计器：

```text
alpha_hat[bucket(prompt_len, context_len, temperature)]
```

### 6.2 gamma 选择目标

候选集合：

```text
GammaCandidates = {2, 3, 4, 6, 8, 12, 16}
```

对每个 `gamma` 估算。`gamma=1` 只作为 one-token lookahead 的退化 sanity point，不进入有效 SpecFACE OME 搜索；如果所有 `gamma >= 2` 都无收益，应返回 `gamma=0` fallback 到 FACE。

```text
E_R = (1 - alpha_hat^(gamma + 1)) / (1 - alpha_hat)
T_round = T_DraftDecode(gamma)
        + T_TargetVerify(gamma)
        + T_KVCommitFree(gamma, alpha_hat)
        + T_DraftSync
        - T_overlap_credit(gamma)

Score(gamma) = T_round / E_R
```

同时加入硬约束：

```text
gamma * ActiveCohorts * KV_draft_per_token <= SRAMJournalCapacity * safety_factor
ExpectedTargetSpecKV(gamma) <= HBMFree * safety_factor
gamma <= gamma_max_latency_budget
gamma <= gamma_max_config
```

选择：

```text
gamma* = argmin Score(gamma)
```

### 6.3 gamma 选择伪代码

```pseudo
procedure CHOOSE_GAMMA(alpha_hat, state):
    best_gamma <- 0
    best_score <- INF

    for gamma in GammaCandidates:
        if not FITS_SRAM_JOURNAL(gamma, state):
            continue
        if not FITS_HBM_SPEC_BUFFER(gamma, state):
            continue

        expected_tokens <- EXPECTED_COMMITTED(gamma, alpha_hat)
        round_time <- ESTIMATE_SD_ROUND_TIME(gamma, alpha_hat, state)
        score <- round_time / expected_tokens

        if score < best_score:
            best_score <- score
            best_gamma <- gamma

    if SHOULD_FALLBACK(best_score, state.face_decode_cost):
        return 0   // gamma=0 means pure FACE decode

    return best_gamma
```

## 7. Fallback 到纯 FACE

Fallback 判断基于收益、稳定性和资源压力三类条件。

### 7.1 收益阈值

设纯 FACE decode 每 token 成本为：

```text
Cost_FACE = T_face_decode_step(batch, context) / batch
```

令 `min_speedup = 1 + epsilon_speedup`。若：

```text
Cost_SD_best * min_speedup >= Cost_FACE
```

则 fallback。默认 `min_speedup = 1.05`，即预测收益小于 5% 时不启用 SD。

### 7.2 接受率阈值

若：

```text
alpha_hat < alpha_min
```

则 fallback。默认 `alpha_min = 0.55~0.65` 扫参。低接受率下，verify 和 draft decode 叠加成本通常抵消收益。

### 7.3 资源压力阈值

若出现以下任一情况，也 fallback：

```text
SRAMJournalPressure > theta_sram
HBMPressure > theta_hbm
TargetVerifyQueueDelay > theta_verify_delay
FallbackHysteresis not expired
```

使用 hysteresis 避免模式震荡：连续 `M_on` 个 window 预测 SD 有收益才开启；连续 `M_off` 个 window 收益不足才关闭。

```pseudo
procedure SHOULD_FALLBACK(sd_cost, face_cost):
    if alpha_hat < alpha_min:
        return true
    if journal_pressure > theta_sram or hbm_pressure > theta_hbm:
        return true
    if sd_cost * min_speedup >= face_cost:
        fallback_bad_windows += 1
    else:
        fallback_good_windows += 1

    if currently_sd:
        return fallback_bad_windows >= M_off
    else:
        return fallback_good_windows < M_on
```

## 8. SRAM Journal 设计

### 8.1 Journal entry

Draft speculative KV 不立即写 HBM，而写入 SRAM journal：

```text
JournalEntry {
  request_id
  round_id
  token_position
  token_id
  kv_bytes
  state: speculative | accepted | rejected | synced
  sram_bank
}
```

每个 active cohort 的 journal 需求为：

```text
JournalBytes = gamma * KV_draft_per_token
```

总容量约束：

```text
sum_active JournalBytes <= C_DD * S_core * journal_fraction
```

其中 `G_DD` 是 draft decode island core group 数，`S_group` 是每 core group 可用 SRAM，`journal_fraction` 默认 0.25~0.5 扫参。

### 8.2 Commit/free 行为

在 target verify 后：

```pseudo
procedure KV_COMMIT_FREE(request, gamma, accepted_k):
    for i in [0, accepted_k):
        MARK_TARGET_KV_COMMITTED(request, i)
        PUSH_DRAFT_KV_FROM_SRAM_TO_HBM(request, i)
        MARK_DRAFT_KV_COMMITTED(request, i)

    for i in [accepted_k, gamma):
        FREE_DRAFT_JOURNAL_ENTRY(request, i)
        FREE_OR_DISCARD_TARGET_SPEC_KV(request, i)

    correction_or_bonus <- GET_TARGET_GENERATED_TOKEN()
    ENQUEUE_DRAFT_SYNC(request, correction_or_bonus)
```

该机制的 HBM 节省来自未接受 draft KV 不进入 HBM：

```text
ExpectedSavedBytes = (gamma - E[K]) * KV_draft_per_token
```

但它会增加 SRAM 压力，因此 dynamic gamma 必须受 journal 容量约束。

## 9. 精炼 OME 接口建议

不做复杂 LUT，新增一个小型 OME，接口可以是：

```cpp
struct SpecFaceStageCost {
    uint64_t compute_ns;
    uint64_t hbm_ns;
    uint64_t noc_ns;
    uint64_t sram_ns;
    uint64_t duration_ns;
    uint64_t hbm_bytes;
    uint64_t sram_journal_bytes;
};

struct SpecFacePartition {
    int target_prefill_cores;
    int target_verify_cores;
    int draft_prefill_cores;
    int draft_decode_cores;
};

struct SpecFaceMappingResult {
    SpecFacePartition partition;
    int gamma;
    double alpha_hat;
    uint64_t expected_round_ns;
    double expected_committed_tokens;
    double expected_ns_per_token;
    double target_internal_overlap_ratio;
    double draft_internal_overlap_ratio;
    double cross_model_overlap_ratio;
    uint64_t hbm_bytes;
    uint64_t sram_journal_peak_bytes;
    bool fallback_to_face;
};
```

OME 主流程：

```pseudo
procedure OME_MAP(active_cohorts, state):
    alpha_hat <- state.acceptance_monitor.alpha_hat
    gamma <- CHOOSE_GAMMA(alpha_hat, state)

    if gamma == 0:
        return FACE_MAPPING()

    candidates <- GENERATE_PARTITIONS(active_cohorts, gamma, state)
    best <- NONE

    for p in candidates:
        layout <- HEURISTIC_LAYOUT(p)
        sim <- MICRO_SCHEDULE(active_cohorts, p, layout, gamma)
        score <- sim.expected_ns_per_token
                 + lambda_mem * sim.journal_pressure
                 + lambda_tail * sim.tail_delay
        if best is NONE or score < best.score:
            best <- sim

    if SHOULD_FALLBACK(best.expected_ns_per_token, state.face_cost):
        best.fallback_to_face <- true

    return best
```

## 10. 配置接口建议

在现有 `llm_server_config.json` 中新增：

```json
"speculative": {
  "enabled": true,
  "acceptance_rate": 0.8,
  "acceptance_ewma_rho": 0.9,
  "gamma_candidates": [2, 3, 4, 6, 8, 12, 16],
  "gamma_max": 16,
  "dynamic_gamma": true,
  "fallback_enabled": true,
  "fallback_min_speedup": 1.05,
  "fallback_alpha_min": 0.6,
  "fallback_hysteresis_on_windows": 2,
  "fallback_hysteresis_off_windows": 2,
  "core_granularity": 64,
  "journal_sram_fraction": 0.35,
  "target_spec_kv_in_hbm": true,
  "draft_sync_enabled": true,
  "partition_search": "seeded-local",
  "layout_policy": "rectangular-islands"
},
"draft_model": {
  "name": "draft-llm",
  "num_layers": 8,
  "hidden_size": 2048,
  "num_heads": 16,
  "num_kv_heads": 16,
  "head_dim": 128,
  "weight_bytes": 1500000000,
  "kv_bytes_per_token": 131072
}
```

其中 draft model 参数应能扫 `L_d/L_t`、`H_d/H_t`、`KV_d/KV_t` 与 `weight_bytes`。

## 11. 实验框架

### 11.1 Baselines

至少包含：

1. `Serial`: 无 FACE overlap，无 SD。
2. `FACE`: 当前 target-only FACE prefill/decode overlap。
3. `SpecFACE-static-gamma`: 固定 gamma，启用 SD 与 journal。
4. `SpecFACE-dynamic-gamma`: 动态 gamma + journal。
5. `SpecFACE-no-journal`: draft speculative KV 直接写 HBM。
6. `SpecFACE-no-fallback`: 禁用 fallback，观察低接受率反例。
7. `SpecFACE-fixed-partition`: 固定 core 分区，不做 OME search。
8. `SpecFACE-full`: 动态 gamma + OME search + journal + fallback。

### 11.2 参数 sweep

接受率：

```text
alpha in {0.50, 0.60, 0.70, 0.80, 0.90, 0.95}
```

Gamma：

```text
gamma_static in {2, 3, 4, 8, 12, 16}
```

Draft model 尺寸：

```text
FLOPs_draft / FLOPs_target in {0.05, 0.10, 0.20, 0.30}
KV_draft / KV_target in {0.25, 0.50, 1.00}
```

Core 分区：

```text
target_core_fraction in {0.60, 0.70, 0.80, 0.90}
verify_within_target_fraction in {0.20, 0.35, 0.50}
draft_decode_within_draft_fraction in {0.50, 0.70, 0.90}
```

SRAM journal：

```text
journal_sram_fraction in {0.10, 0.25, 0.35, 0.50}
core_sram_mb in {0.5, 0.75, 1.0, 1.5}
```

Workload：

```text
prompt_len distribution: short / mixed / long
output_len distribution: short / mixed / long
arrival rate: low / medium / high saturation
cohort size: 1 / 2 / 4 / 8 / dynamic
```

### 11.3 Metrics

性能：

```text
throughput req/s
tokens/s
average/p50/p95 E2E latency
TTFT
TPOT
expected ns per committed token
speedup over FACE
fallback rate
```

Overlap：

```text
target_internal_overlap_ratio
draft_internal_overlap_ratio
cross_model_overlap_ratio
overall_compute_overlap_ratio
HBM overlap/pressure
NoC overlap/pressure
```

SD 行为：

```text
gamma distribution
alpha_hat trace
accepted tokens per verify
E[R] vs observed committed tokens
verify waste = rejected target KV bytes or compute share
draft wasted compute = rejected draft tokens / proposed draft tokens
```

Memory/KV：

```text
SRAM journal peak bytes
journal stall count/time
HBM draft KV bytes saved
KV commit/free bytes
speculative target KV free bytes
HBM capacity pressure
```

OME：

```text
selected core partition distribution
layout policy
partition search time
bottleneck stage distribution
```

### 11.4 关键图表

小论文建议图表：

1. 架构图：single instance 内 target/draft core islands、SRAM journal、HBM KV path。
2. 时序图：多 cohort 下 target verify、draft decode、prefill、commit/free 的 pipeline overlap。
3. 数学图：`E[R]` 与 `Cost_SD` 随 gamma/alpha 变化。
4. 性能图：SpecFACE-full vs FACE tokens/s 与 TPOT。
5. 消融图：dynamic gamma、journal、fallback、OME search 分别带来的贡献。
6. 资源图：core partition heatmap 与 bottleneck resource。
7. 内存图：journal peak、HBM KV write savings、fallback 触发原因。

## 12. 论文叙事主线

可以把贡献写成三点：

1. **SD-aware operator mapping for wafer-scale LLM serving**：把 target prefill/verify 与 draft prefill/decode 映射到单 instance 内的 core islands，避免把 SD 当成 target decode 的外层算法。
2. **Cohort-level multi-flow pipeline**：单请求保持 SD 依赖顺序，但在多 cohort 中显式挖掘 target 内部、draft 内部、target-draft 之间的 overlap。
3. **Acceptance-aware adaptive execution**：用实时接受率估计器动态选择 gamma，并在低收益时 fallback；用 SRAM journal 把 draft speculative KV 写入从 HBM critical path 移开。

核心论点：

```text
Speculative decoding 的系统收益不是简单的 target forward 次数减少，
而是 draft/target 多数据流在同一 WSC instance 内能否被正确映射、overlap、并受 KV 生命周期约束地执行。
```

## 13. 第一阶段实现里程碑

M1: 配置与 evaluator

- 增加 `speculative` 与 `draft_model` 配置。
- 增加 target/draft operator cost evaluator。
- 增加 `E[R]`、`E[K]`、journal bytes、commit/free bytes 公式。

M2: 单 instance OME micro-search

- 实现 seeded-local core partition search。
- 实现 rectangular-island layout 元数据。
- 输出 selected partition、bottleneck、expected ns/token。

M3: 多 cohort scheduler

- 增加 request SD 状态机。
- 实现 target/draft/memory timeline。
- 统计三类 overlap。

M4: Dynamic gamma + fallback

- 实现 EWMA acceptance monitor。
- 实现 gamma candidate scoring。
- 实现 fallback hysteresis。

M5: SRAM journal

- 实现 draft speculative KV journal ledger。
- 实现 accepted push to HBM、rejected free、journal stall。
- 输出 HBM bytes saved 与 journal pressure。

M6: 实验脚本与论文图表数据

- 增加 baseline 配置。
- 增加 alpha/gamma/SRAM/core partition sweep。
- 生成 CSV：per-request、per-cohort、per-partition、per-resource timeline。

## 14. 完成标准

第一版 SpecFACE 微探索完成时应能回答：

- 给定 `alpha=0.8`，动态 gamma 是否优于固定 gamma？最常选 gamma 是多少？
- SRAM journal 能节省多少 draft KV HBM 写入？是否会引入 SRAM stall？
- OME search 选择的 target/draft core 分区是否随 workload 和 alpha 改变？
- target internal overlap、draft internal overlap、cross-model overlap 分别贡献多少？
- 哪些条件下 SD 收益低于 FACE，fallback 是否能避免 tail latency 恶化？

这些问题对应小论文的 evaluation backbone。
