#!/usr/bin/env python3
"""Single-instance SpecFACE micro-experiment harness.

This script intentionally stays lightweight: it uses a roofline-style analytical
model plus a small Operator Mapping Engine (OME) search to explore speculative
decoding design points inside one FACE instance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FP16_BYTES = 2


@dataclass(frozen=True)
class Hardware:
    die_rows: int
    die_cols: int
    hbm_chiplets_per_die: int
    dram_capacity_gb_per_hbm: float
    dram_bandwidth_gbps_per_hbm: float
    d2d_bandwidth_tbps: float
    core_sram_mb: float
    core_compute_gflops: float
    noc_width_bits: int
    core_rows: int
    core_cols: int
    instance_shape: Tuple[int, int]
    sram_bw_bytes_per_core_ns: float

    @property
    def die_count(self) -> int:
        return self.instance_shape[0] * self.instance_shape[1]

    @property
    def core_count(self) -> int:
        return self.core_rows * self.core_cols * self.die_count

    @property
    def hbm_bandwidth_bytes_per_ns(self) -> float:
        return (self.dram_bandwidth_gbps_per_hbm *
                self.hbm_chiplets_per_die * self.die_count)

    @property
    def noc_bandwidth_bytes_per_ns(self) -> float:
        return max(1.0, (self.noc_width_bits / 8.0) * self.core_count)

    @property
    def sram_bytes_per_core(self) -> float:
        return self.core_sram_mb * 1_000_000.0


@dataclass(frozen=True)
class Model:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    weight_bytes: float
    kv_bytes_per_token: float


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_time_ns: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SpecParams:
    alphas: Sequence[float]
    gamma_candidates: Sequence[int]
    static_gammas: Sequence[int]
    default_alpha: float
    dynamic_gamma: bool
    fallback_enabled: bool
    fallback_min_speedup: float
    fallback_alpha_min: float
    core_granularity: int
    journal_sram_fraction: float
    active_cohorts: int
    safety_factor: float
    target_prefill_eff: float
    target_verify_eff: float
    target_decode_eff: float
    draft_prefill_eff: float
    draft_decode_eff: float
    hbm_efficiency: float
    noc_traffic_fraction: float
    fixed_partition: Dict[str, float]


@dataclass(frozen=True)
class Partition:
    target_prefill: int
    target_verify: int
    draft_prefill: int
    draft_decode: int

    @property
    def total(self) -> int:
        return (self.target_prefill + self.target_verify +
                self.draft_prefill + self.draft_decode)


@dataclass
class MappingResult:
    mode: str
    alpha: float
    gamma: int
    fallback: bool
    journal: bool
    partition: Partition
    expected_accepted: float
    expected_committed: float
    face_ns_per_token: float
    expected_ns_per_token: float
    speedup_over_face: float
    prefill_ns: float
    pipeline_round_ns: float
    serial_round_ns: float
    target_compute_ns: float
    draft_compute_ns: float
    memory_ns: float
    noc_ns: float
    journal_ns: float
    journal_stall_ns: float
    journal_peak_bytes: float
    journal_capacity_bytes: float
    hbm_bytes_per_round: float
    hbm_saved_bytes_per_round: float
    target_internal_overlap: float
    draft_internal_overlap: float
    cross_model_overlap: float
    overall_overlap: float
    bottleneck: str
    score: float


def load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def parse_hardware(base_cfg: Dict, exp_cfg: Dict) -> Hardware:
    hw = base_cfg["hardware"]
    shape = tuple(exp_cfg.get("instance_shape", hw.get("die_array", [1, 1])))
    return Hardware(
        die_rows=hw.get("die_array", [1, 1])[0],
        die_cols=hw.get("die_array", [1, 1])[1],
        hbm_chiplets_per_die=hw.get("hbm_chiplets_per_die", 1),
        dram_capacity_gb_per_hbm=hw.get("dram_capacity_gb_per_hbm", 16.0),
        dram_bandwidth_gbps_per_hbm=hw.get("dram_bandwidth_gbps_per_hbm", 410.0),
        d2d_bandwidth_tbps=hw.get("d2d_bandwidth_tbps", 1.0),
        core_sram_mb=hw.get("core_sram_mb", 0.75),
        core_compute_gflops=hw.get("core_compute_gflops", 740.0),
        noc_width_bits=hw.get("noc_width_bits", 512),
        core_rows=hw.get("core_array", [16, 16])[0],
        core_cols=hw.get("core_array", [16, 16])[1],
        instance_shape=(int(shape[0]), int(shape[1])),
        sram_bw_bytes_per_core_ns=exp_cfg.get("sram_bw_bytes_per_core_ns", 4096.0),
    )


def parse_model(cfg: Dict, name: str = "model") -> Model:
    return Model(
        name=cfg.get("name", name),
        num_layers=int(cfg.get("num_layers", 1)),
        hidden_size=int(cfg.get("hidden_size", 1)),
        num_heads=int(cfg.get("num_heads", 1)),
        num_kv_heads=int(cfg.get("num_kv_heads", cfg.get("num_heads", 1))),
        head_dim=int(cfg.get("head_dim", 128)),
        weight_bytes=float(cfg.get("weight_bytes", 0.0)),
        kv_bytes_per_token=float(cfg.get("kv_bytes_per_token", 0.0)),
    )


def parse_params(exp_cfg: Dict) -> SpecParams:
    spec = exp_cfg.get("speculative", {})
    return SpecParams(
        alphas=spec.get("alphas", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
        gamma_candidates=spec.get("gamma_candidates", [1, 2, 3, 4, 6, 8, 12, 16]),
        static_gammas=spec.get("static_gammas", [1, 2, 4, 8, 12, 16]),
        default_alpha=float(spec.get("acceptance_rate", 0.8)),
        dynamic_gamma=bool(spec.get("dynamic_gamma", True)),
        fallback_enabled=bool(spec.get("fallback_enabled", True)),
        fallback_min_speedup=float(spec.get("fallback_min_speedup", 1.05)),
        fallback_alpha_min=float(spec.get("fallback_alpha_min", 0.6)),
        core_granularity=int(spec.get("core_granularity", 64)),
        journal_sram_fraction=float(spec.get("journal_sram_fraction", 0.35)),
        active_cohorts=int(spec.get("active_cohorts", 4)),
        safety_factor=float(spec.get("safety_factor", 0.85)),
        target_prefill_eff=float(spec.get("target_prefill_eff", 0.72)),
        target_verify_eff=float(spec.get("target_verify_eff", 0.62)),
        target_decode_eff=float(spec.get("target_decode_eff", 0.60)),
        draft_prefill_eff=float(spec.get("draft_prefill_eff", 0.74)),
        draft_decode_eff=float(spec.get("draft_decode_eff", 0.68)),
        hbm_efficiency=float(spec.get("hbm_efficiency", 0.82)),
        noc_traffic_fraction=float(spec.get("noc_traffic_fraction", 0.12)),
        fixed_partition=spec.get(
            "fixed_partition",
            {
                "target_prefill": 0.40,
                "target_verify": 0.28,
                "draft_prefill": 0.12,
                "draft_decode": 0.20,
            },
        ),
    )


def load_requests(path: Path) -> List[Request]:
    requests: List[Request] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rid = int(row.get("request_id", row.get("id", i)))
            requests.append(Request(
                request_id=rid,
                arrival_time_ns=int(row.get("arrival_time_ns", row.get("arrival", 0))),
                input_tokens=int(row.get("input_tokens", row.get("prompt_tokens", 0))),
                output_tokens=int(row.get("output_tokens", row.get("decode_tokens", 0))),
            ))
    return requests


def expected_accepted(alpha: float, gamma: int) -> float:
    if gamma <= 0:
        return 0.0
    if alpha >= 1.0:
        return float(gamma)
    if alpha <= 0.0:
        return 0.0
    return alpha * (1.0 - alpha ** gamma) / (1.0 - alpha)


def expected_committed(alpha: float, gamma: int) -> float:
    return 1.0 + expected_accepted(alpha, gamma)


def prefill_ops(model: Model, tokens: int) -> float:
    t = max(1.0, float(tokens))
    h = float(model.hidden_size)
    return model.num_layers * (12.0 * t * h * h + 2.0 * t * t * h)


def decode_ops(model: Model, gamma: int, context_tokens: int) -> float:
    g = max(1.0, float(gamma))
    n = max(1.0, float(context_tokens))
    h = float(model.hidden_size)
    return model.num_layers * (12.0 * g * h * h +
                               2.0 * h * (g * n + g * (g - 1.0) / 2.0))


def verify_ops(model: Model, gamma: int, context_tokens: int) -> float:
    g = max(1.0, float(gamma))
    n = max(1.0, float(context_tokens))
    h = float(model.hidden_size)
    return model.num_layers * (12.0 * g * h * h +
                               2.0 * h * (g * n + g * (g + 1.0) / 2.0))


def sync_ops(model: Model, context_tokens: int, committed_tokens: float) -> float:
    n = max(1.0, float(context_tokens) + committed_tokens)
    h = float(model.hidden_size)
    return model.num_layers * (12.0 * h * h + 2.0 * n * h)


def compute_ns(ops: float, cores: int, hw: Hardware, efficiency: float) -> float:
    usable_cores = max(1, cores)
    return ops / max(1e-9, usable_cores * hw.core_compute_gflops * efficiency)


def hbm_ns(bytes_count: float, hw: Hardware, params: SpecParams) -> float:
    return bytes_count / max(1e-9, hw.hbm_bandwidth_bytes_per_ns * params.hbm_efficiency)


def noc_ns(bytes_count: float, hw: Hardware) -> float:
    return bytes_count / max(1e-9, hw.noc_bandwidth_bytes_per_ns)


def activation_bytes(model: Model, tokens: float) -> float:
    return 4.0 * tokens * model.hidden_size * FP16_BYTES * model.num_layers


def prefill_bytes(model: Model, tokens: int) -> float:
    return model.weight_bytes + activation_bytes(model, tokens) + tokens * model.kv_bytes_per_token


def decode_bytes(model: Model, gamma: int, context_tokens: int) -> float:
    return gamma * context_tokens * model.kv_bytes_per_token + activation_bytes(model, gamma)


def verify_bytes(model: Model, gamma: int, context_tokens: int) -> float:
    # Target verification checks gamma speculative queries as a compact block.
    # The committed context K/V can be streamed once and reused across those
    # queries, unlike autoregressive draft decode where each new token performs
    # a separate context read.
    return (context_tokens * model.kv_bytes_per_token +
            gamma * model.kv_bytes_per_token + activation_bytes(model, gamma))


def round_up_to_granularity(value: float, granularity: int, minimum: int) -> int:
    rounded = int(round(value / granularity) * granularity)
    return max(minimum, rounded)


def make_partition(values: Sequence[float], hw: Hardware, granularity: int) -> Partition:
    minimum = granularity
    total = hw.core_count
    raw_sum = sum(max(0.0, v) for v in values)
    if raw_sum <= 0.0:
        values = [0.4, 0.25, 0.1, 0.25]
        raw_sum = sum(values)
    scaled = [v / raw_sum * total for v in values]
    parts = [round_up_to_granularity(v, granularity, minimum) for v in scaled]
    while sum(parts) > total:
        idx = max(range(4), key=lambda i: parts[i])
        if parts[idx] <= minimum:
            break
        parts[idx] -= granularity
    while sum(parts) + granularity <= total:
        idx = max(range(4), key=lambda i: scaled[i] - parts[i])
        parts[idx] += granularity
    return Partition(parts[0], parts[1], parts[2], parts[3])


def candidate_partitions(hw: Hardware,
                         target: Model,
                         draft: Model,
                         params: SpecParams,
                         gamma: int,
                         prompt_tokens: int,
                         context_tokens: int) -> List[Partition]:
    gran = params.core_granularity
    e_commit = expected_committed(params.default_alpha, gamma)
    works = [
        prefill_ops(target, prompt_tokens),
        verify_ops(target, gamma, context_tokens),
        prefill_ops(draft, prompt_tokens),
        decode_ops(draft, gamma, context_tokens) + sync_ops(draft, context_tokens, e_commit),
    ]
    seeds = [
        [math.sqrt(w) for w in works],
        works,
        [0.42, 0.30, 0.10, 0.18],
        [0.35, 0.25, 0.12, 0.28],
        [0.50, 0.20, 0.10, 0.20],
    ]
    fixed = params.fixed_partition
    seeds.append([
        fixed.get("target_prefill", 0.40),
        fixed.get("target_verify", 0.28),
        fixed.get("draft_prefill", 0.12),
        fixed.get("draft_decode", 0.20),
    ])

    candidates = set()
    for seed in seeds:
        base = make_partition(seed, hw, gran)
        base_tuple = (base.target_prefill, base.target_verify,
                      base.draft_prefill, base.draft_decode)
        for d0 in (-2, -1, 0, 1, 2):
            for d1 in (-2, -1, 0, 1, 2):
                for d2 in (-1, 0, 1):
                    for d3 in (-2, -1, 0, 1, 2):
                        vals = [base_tuple[0] + d0 * gran,
                                base_tuple[1] + d1 * gran,
                                base_tuple[2] + d2 * gran,
                                base_tuple[3] + d3 * gran]
                        if min(vals) < gran or sum(vals) > hw.core_count:
                            continue
                        leftover = hw.core_count - sum(vals)
                        if leftover >= gran:
                            # Give spare cores to the currently dominant target stage.
                            vals[0 if works[0] >= works[1] else 1] += leftover
                        candidates.add(tuple(vals))
    return [Partition(*c) for c in sorted(candidates)]


def fixed_partition(hw: Hardware, params: SpecParams) -> Partition:
    fp = params.fixed_partition
    return make_partition([
        fp.get("target_prefill", 0.40),
        fp.get("target_verify", 0.28),
        fp.get("draft_prefill", 0.12),
        fp.get("draft_decode", 0.20),
    ], hw, params.core_granularity)


def face_decode_cost(hw: Hardware, target: Model, params: SpecParams, context_tokens: int) -> float:
    ops = decode_ops(target, 1, context_tokens)
    comp = compute_ns(ops, hw.core_count, hw, params.target_decode_eff)
    mem = hbm_ns(decode_bytes(target, 1, context_tokens), hw, params)
    noc = noc_ns(decode_bytes(target, 1, context_tokens) * params.noc_traffic_fraction, hw)
    return max(comp, mem, noc)


def face_prefill_cost(hw: Hardware, target: Model, params: SpecParams, prompt_tokens: int) -> float:
    comp = compute_ns(prefill_ops(target, prompt_tokens), hw.core_count, hw, params.target_prefill_eff)
    mem = hbm_ns(prefill_bytes(target, prompt_tokens), hw, params)
    noc = noc_ns(prefill_bytes(target, prompt_tokens) * params.noc_traffic_fraction, hw)
    return max(comp, mem, noc)


def evaluate_partition(mode: str,
                       hw: Hardware,
                       target: Model,
                       draft: Model,
                       params: SpecParams,
                       alpha: float,
                       gamma: int,
                       part: Partition,
                       prompt_tokens: int,
                       context_tokens: int,
                       journal: bool,
                       fallback_allowed: bool) -> MappingResult:
    e_acc = expected_accepted(alpha, gamma)
    e_commit = expected_committed(alpha, gamma)
    face_cost = face_decode_cost(hw, target, params, context_tokens)

    tp_compute = compute_ns(prefill_ops(target, prompt_tokens), part.target_prefill, hw,
                            params.target_prefill_eff)
    dp_compute = compute_ns(prefill_ops(draft, prompt_tokens), part.draft_prefill, hw,
                            params.draft_prefill_eff)
    prefill_mem = hbm_ns(prefill_bytes(target, prompt_tokens) +
                         prefill_bytes(draft, prompt_tokens), hw, params)
    prefill_noc = noc_ns((prefill_bytes(target, prompt_tokens) +
                          prefill_bytes(draft, prompt_tokens)) *
                         params.noc_traffic_fraction, hw)
    prefill_time = max(tp_compute, dp_compute, prefill_mem, prefill_noc)

    tv_compute = compute_ns(verify_ops(target, gamma, context_tokens),
                            part.target_verify, hw, params.target_verify_eff)
    dd_compute = compute_ns(decode_ops(draft, gamma, context_tokens),
                            part.draft_decode, hw, params.draft_decode_eff)
    ds_compute = compute_ns(sync_ops(draft, context_tokens, e_commit),
                            part.draft_decode, hw, params.draft_decode_eff)
    draft_compute_time = dd_compute + ds_compute
    target_compute_time = tv_compute

    target_verify_bytes = verify_bytes(target, gamma, context_tokens)
    draft_decode_hbm_bytes = decode_bytes(draft, gamma, context_tokens)
    draft_sync_hbm_bytes = decode_bytes(draft, 1, int(context_tokens + e_commit))
    if journal:
        commit_hbm_bytes = e_acc * draft.kv_bytes_per_token
        draft_kv_write_bytes = 0.0
        hbm_saved = max(0.0, (gamma - e_acc) * draft.kv_bytes_per_token)
    else:
        commit_hbm_bytes = 0.0
        draft_kv_write_bytes = gamma * draft.kv_bytes_per_token
        hbm_saved = 0.0

    hbm_bytes = (target_verify_bytes + draft_decode_hbm_bytes +
                 draft_sync_hbm_bytes + commit_hbm_bytes + draft_kv_write_bytes)
    mem_time = hbm_ns(hbm_bytes, hw, params)
    noc_time = noc_ns(hbm_bytes * params.noc_traffic_fraction, hw)

    journal_bytes_per_round = gamma * draft.kv_bytes_per_token if journal else 0.0
    journal_peak = journal_bytes_per_round * params.active_cohorts
    journal_capacity = (part.draft_decode * hw.sram_bytes_per_core *
                        params.journal_sram_fraction)
    sram_bw = max(1.0, part.draft_decode * hw.sram_bw_bytes_per_core_ns)
    journal_time = journal_bytes_per_round / sram_bw if journal else 0.0
    excess_journal = max(0.0, journal_peak - journal_capacity * params.safety_factor)
    journal_stall = excess_journal / sram_bw if journal else 0.0

    pipeline_round = max(target_compute_time, draft_compute_time, mem_time,
                         noc_time, journal_time) + journal_stall
    serial_round = (target_compute_time + draft_compute_time + mem_time +
                    max(noc_time, journal_time) + journal_stall)
    ns_per_token = pipeline_round / max(1e-9, e_commit)

    fallback = False
    if fallback_allowed:
        if alpha < params.fallback_alpha_min:
            fallback = True
        if ns_per_token * params.fallback_min_speedup >= face_cost:
            fallback = True
        if excess_journal > 0.0:
            fallback = True
    if fallback:
        ns_per_token = face_cost
        pipeline_round = face_cost
        serial_round = face_cost
        e_acc = 0.0
        e_commit = 1.0
        hbm_saved = 0.0

    target_internal = 1.0 - max(tp_compute, tv_compute) / max(1e-9, tp_compute + tv_compute)
    draft_internal = 1.0 - max(dp_compute, draft_compute_time) / max(1e-9, dp_compute + draft_compute_time)
    cross_model = 1.0 - max(target_compute_time, draft_compute_time) / max(
        1e-9, target_compute_time + draft_compute_time)
    total_components = target_compute_time + draft_compute_time + mem_time + max(noc_time, journal_time)
    overall = 1.0 - max(target_compute_time, draft_compute_time, mem_time,
                        noc_time, journal_time) / max(1e-9, total_components)

    bottleneck_values = {
        "target_compute": target_compute_time,
        "draft_compute": draft_compute_time,
        "hbm": mem_time,
        "noc": noc_time,
        "journal": journal_time + journal_stall,
    }
    bottleneck = max(bottleneck_values, key=bottleneck_values.get)
    speedup = face_cost / max(1e-9, ns_per_token)
    score = ns_per_token + 0.02 * max(0.0, journal_peak / max(1.0, journal_capacity) - 1.0) * ns_per_token

    return MappingResult(
        mode=mode,
        alpha=alpha,
        gamma=gamma,
        fallback=fallback,
        journal=journal,
        partition=part,
        expected_accepted=e_acc,
        expected_committed=e_commit,
        face_ns_per_token=face_cost,
        expected_ns_per_token=ns_per_token,
        speedup_over_face=speedup,
        prefill_ns=prefill_time,
        pipeline_round_ns=pipeline_round,
        serial_round_ns=serial_round,
        target_compute_ns=target_compute_time,
        draft_compute_ns=draft_compute_time,
        memory_ns=mem_time,
        noc_ns=noc_time,
        journal_ns=journal_time,
        journal_stall_ns=journal_stall,
        journal_peak_bytes=journal_peak,
        journal_capacity_bytes=journal_capacity,
        hbm_bytes_per_round=hbm_bytes,
        hbm_saved_bytes_per_round=hbm_saved,
        target_internal_overlap=target_internal,
        draft_internal_overlap=draft_internal,
        cross_model_overlap=cross_model,
        overall_overlap=overall,
        bottleneck=bottleneck,
        score=score,
    )


def search_best(mode: str,
                hw: Hardware,
                target: Model,
                draft: Model,
                params: SpecParams,
                alpha: float,
                gamma: int,
                prompt_tokens: int,
                context_tokens: int,
                journal: bool,
                fallback_allowed: bool,
                use_fixed_partition: bool = False) -> MappingResult:
    if use_fixed_partition:
        parts = [fixed_partition(hw, params)]
    else:
        parts = candidate_partitions(hw, target, draft, params, gamma,
                                     prompt_tokens, context_tokens)
    best: Optional[MappingResult] = None
    for part in parts:
        result = evaluate_partition(mode, hw, target, draft, params, alpha, gamma,
                                    part, prompt_tokens, context_tokens,
                                    journal, fallback_allowed)
        if best is None or result.score < best.score:
            best = result
    assert best is not None
    return best


def dynamic_gamma(mode: str,
                  hw: Hardware,
                  target: Model,
                  draft: Model,
                  params: SpecParams,
                  alpha: float,
                  prompt_tokens: int,
                  context_tokens: int,
                  journal: bool,
                  fallback_allowed: bool,
                  use_fixed_partition: bool = False) -> Tuple[MappingResult, List[MappingResult]]:
    candidates = []
    for gamma in params.gamma_candidates:
        result = search_best(mode, hw, target, draft, params, alpha, gamma,
                             prompt_tokens, context_tokens, journal,
                             fallback_allowed=False,
                             use_fixed_partition=use_fixed_partition)
        # Dynamic gamma should not select an infeasible journal point.
        if journal and result.journal_peak_bytes > result.journal_capacity_bytes * params.safety_factor:
            candidates.append(result)
            continue
        candidates.append(result)
    feasible = [r for r in candidates
                if (not journal or r.journal_peak_bytes <= r.journal_capacity_bytes * params.safety_factor)]
    if not feasible:
        best = min(candidates, key=lambda r: r.score)
    else:
        best = min(feasible, key=lambda r: r.score)
    if fallback_allowed:
        best = evaluate_partition(mode, hw, target, draft, params, alpha,
                                  best.gamma, best.partition, prompt_tokens,
                                  context_tokens, journal, fallback_allowed=True)
    return best, candidates


def face_row(hw: Hardware,
             target: Model,
             params: SpecParams,
             alpha: float,
             prompt_tokens: int,
             context_tokens: int) -> MappingResult:
    decode_hbm_bytes = decode_bytes(target, 1, context_tokens)
    comp = compute_ns(decode_ops(target, 1, context_tokens), hw.core_count, hw,
                      params.target_decode_eff)
    mem = hbm_ns(decode_hbm_bytes, hw, params)
    noc = noc_ns(decode_hbm_bytes * params.noc_traffic_fraction, hw)
    face = max(comp, mem, noc)
    prefill = face_prefill_cost(hw, target, params, prompt_tokens)
    bottleneck_values = {"compute": comp, "hbm": mem, "noc": noc}
    bottleneck = max(bottleneck_values, key=bottleneck_values.get)
    part = Partition(hw.core_count, 0, 0, 0)
    return MappingResult(
        mode="FACE",
        alpha=alpha,
        gamma=0,
        fallback=False,
        journal=False,
        partition=part,
        expected_accepted=0.0,
        expected_committed=1.0,
        face_ns_per_token=face,
        expected_ns_per_token=face,
        speedup_over_face=1.0,
        prefill_ns=prefill,
        pipeline_round_ns=face,
        serial_round_ns=face,
        target_compute_ns=comp,
        draft_compute_ns=0.0,
        memory_ns=mem,
        noc_ns=noc,
        journal_ns=0.0,
        journal_stall_ns=0.0,
        journal_peak_bytes=0.0,
        journal_capacity_bytes=0.0,
        hbm_bytes_per_round=decode_hbm_bytes,
        hbm_saved_bytes_per_round=0.0,
        target_internal_overlap=0.0,
        draft_internal_overlap=0.0,
        cross_model_overlap=0.0,
        overall_overlap=0.0,
        bottleneck=bottleneck,
        score=face,
    )


def result_to_row(result: MappingResult) -> Dict[str, object]:
    return {
        "mode": result.mode,
        "alpha": f"{result.alpha:.4f}",
        "gamma": result.gamma,
        "fallback": int(result.fallback),
        "journal": int(result.journal),
        "target_prefill_cores": result.partition.target_prefill,
        "target_verify_cores": result.partition.target_verify,
        "draft_prefill_cores": result.partition.draft_prefill,
        "draft_decode_cores": result.partition.draft_decode,
        "expected_accepted": f"{result.expected_accepted:.6f}",
        "expected_committed": f"{result.expected_committed:.6f}",
        "face_ns_per_token": f"{result.face_ns_per_token:.3f}",
        "expected_ns_per_token": f"{result.expected_ns_per_token:.3f}",
        "speedup_over_face": f"{result.speedup_over_face:.6f}",
        "prefill_ns": f"{result.prefill_ns:.3f}",
        "pipeline_round_ns": f"{result.pipeline_round_ns:.3f}",
        "serial_round_ns": f"{result.serial_round_ns:.3f}",
        "target_compute_ns": f"{result.target_compute_ns:.3f}",
        "draft_compute_ns": f"{result.draft_compute_ns:.3f}",
        "memory_ns": f"{result.memory_ns:.3f}",
        "noc_ns": f"{result.noc_ns:.3f}",
        "journal_ns": f"{result.journal_ns:.3f}",
        "journal_stall_ns": f"{result.journal_stall_ns:.3f}",
        "journal_peak_bytes": f"{result.journal_peak_bytes:.0f}",
        "journal_capacity_bytes": f"{result.journal_capacity_bytes:.0f}",
        "hbm_bytes_per_round": f"{result.hbm_bytes_per_round:.0f}",
        "hbm_saved_bytes_per_round": f"{result.hbm_saved_bytes_per_round:.0f}",
        "target_internal_overlap": f"{result.target_internal_overlap:.6f}",
        "draft_internal_overlap": f"{result.draft_internal_overlap:.6f}",
        "cross_model_overlap": f"{result.cross_model_overlap:.6f}",
        "overall_overlap": f"{result.overall_overlap:.6f}",
        "bottleneck": result.bottleneck,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_requests(requests: Sequence[Request],
                       mapping: MappingResult,
                       face: MappingResult) -> List[Dict[str, object]]:
    rows = []
    for req in requests:
        if mapping.mode == "FACE" or mapping.fallback:
            ttft = face.prefill_ns + face.expected_ns_per_token
            tpot = face.expected_ns_per_token
            e2e = face.prefill_ns + req.output_tokens * face.expected_ns_per_token
            rounds = req.output_tokens
            committed = 1.0
        else:
            rounds = math.ceil(req.output_tokens / max(1e-9, mapping.expected_committed))
            ttft = mapping.prefill_ns + mapping.pipeline_round_ns
            tpot = mapping.expected_ns_per_token
            e2e = mapping.prefill_ns + rounds * mapping.pipeline_round_ns
            committed = mapping.expected_committed
        rows.append({
            "request_id": req.request_id,
            "input_tokens": req.input_tokens,
            "output_tokens": req.output_tokens,
            "mode": mapping.mode,
            "alpha": f"{mapping.alpha:.4f}",
            "gamma": mapping.gamma,
            "rounds": rounds,
            "expected_committed_per_round": f"{committed:.6f}",
            "ttft_ns": f"{ttft:.3f}",
            "tpot_ns": f"{tpot:.3f}",
            "e2e_latency_ns": f"{e2e:.3f}",
        })
    return rows



def select_default_workflow_rows(summary: Sequence[MappingResult],
                                 default_alpha: float) -> List[MappingResult]:
    rows = [r for r in summary if abs(r.alpha - default_alpha) < 1e-9]
    selected: List[MappingResult] = []
    for mode in ["FACE", "SpecFACE-full", "SpecFACE-fixed-partition",
                 "SpecFACE-no-journal", "SpecFACE-no-fallback"]:
        candidates = [r for r in rows if r.mode == mode]
        if candidates:
            selected.append(min(candidates, key=lambda r: r.expected_ns_per_token))
    static = [r for r in rows if r.mode == "SpecFACE-static-gamma" and not r.fallback]
    if static:
        selected.append(min(static, key=lambda r: r.expected_ns_per_token))
    return selected


def workload_variant_specs(exp_cfg: Dict,
                           base_prompt: int,
                           base_output: int,
                           base_context: int) -> List[Dict[str, object]]:
    variants = exp_cfg.get("workload_variants")
    if variants:
        return variants
    return [
        {
            "name": "trace_mixed",
            "prompt_tokens": base_prompt,
            "output_tokens": base_output,
            "context_tokens": base_context,
        },
        {
            "name": "short_prompt",
            "prompt_tokens": 256,
            "output_tokens": 64,
            "context_tokens": 256,
        },
        {
            "name": "long_prompt",
            "prompt_tokens": 2048,
            "output_tokens": 64,
            "context_tokens": 2048,
        },
        {
            "name": "long_decode",
            "prompt_tokens": 512,
            "output_tokens": 256,
            "context_tokens": 512,
        },
        {
            "name": "balanced_long",
            "prompt_tokens": 1024,
            "output_tokens": 128,
            "context_tokens": 1024,
        },
    ]


def estimate_e2e_ns(mapping: MappingResult,
                    face: MappingResult,
                    output_tokens: int) -> float:
    if mapping.mode == "FACE" or mapping.fallback:
        return face.prefill_ns + output_tokens * face.expected_ns_per_token
    rounds = math.ceil(output_tokens / max(1e-9, mapping.expected_committed))
    return mapping.prefill_ns + rounds * mapping.pipeline_round_ns


def run_workload_variants(hw: Hardware,
                          target: Model,
                          draft: Model,
                          params: SpecParams,
                          variants: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in variants:
        name = str(variant["name"])
        prompt_tokens = int(variant["prompt_tokens"])
        output_tokens = int(variant["output_tokens"])
        context_tokens = int(variant.get("context_tokens", prompt_tokens))
        alpha = float(variant.get("alpha", params.default_alpha))
        face = face_row(hw, target, params, alpha, prompt_tokens, context_tokens)
        spec, _ = dynamic_gamma("SpecFACE-full", hw, target, draft, params,
                                alpha, prompt_tokens, context_tokens,
                                journal=True,
                                fallback_allowed=params.fallback_enabled)
        face_e2e = estimate_e2e_ns(face, face, output_tokens)
        spec_e2e = estimate_e2e_ns(spec, face, output_tokens)
        rows.append({
            "workload": name,
            "alpha": f"{alpha:.4f}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "context_tokens": context_tokens,
            "gamma": spec.gamma,
            "fallback": int(spec.fallback),
            "face_ns_per_token": f"{face.expected_ns_per_token:.3f}",
            "specface_ns_per_token": f"{spec.expected_ns_per_token:.3f}",
            "tpot_speedup": f"{face.expected_ns_per_token / max(1e-9, spec.expected_ns_per_token):.6f}",
            "face_e2e_ns": f"{face_e2e:.3f}",
            "specface_e2e_ns": f"{spec_e2e:.3f}",
            "e2e_speedup": f"{face_e2e / max(1e-9, spec_e2e):.6f}",
            "target_prefill_cores": spec.partition.target_prefill,
            "target_verify_cores": spec.partition.target_verify,
            "draft_prefill_cores": spec.partition.draft_prefill,
            "draft_decode_cores": spec.partition.draft_decode,
            "bottleneck": spec.bottleneck,
        })
    return rows


def make_plots(summary: Sequence[MappingResult],
               workload_rows: Sequence[Dict[str, object]],
               default_alpha: float,
               plot_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    selected = select_default_workflow_rows(summary, default_alpha)
    spec_selected = [r for r in selected if r.mode != "FACE"]
    colors = {
        "target_prefill": "#2b6cb0",
        "target_verify": "#63b3ed",
        "draft_prefill": "#2f855a",
        "draft_decode": "#9ae6b4",
        "target_compute": "#2b6cb0",
        "draft_compute": "#2f855a",
        "hbm": "#c05621",
        "sram": "#805ad5",
        "noc": "#718096",
    }

    if selected:
        paths.append(draw_mapping_layouts(selected, plot_dir, colors))
        paths.append(draw_workflow_timelines(selected, plot_dir, colors))

    if spec_selected:
        labels = [r.mode.replace("SpecFACE-", "") for r in spec_selected]
        data = [
            [r.partition.target_prefill for r in spec_selected],
            [r.partition.target_verify for r in spec_selected],
            [r.partition.draft_prefill for r in spec_selected],
            [r.partition.draft_decode for r in spec_selected],
        ]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        bottom = [0] * len(labels)
        names = ["target_prefill", "target_verify", "draft_prefill", "draft_decode"]
        for name, values in zip(names, data):
            ax.bar(labels, values, bottom=bottom, label=name, color=colors[name])
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_ylabel("cores")
        ax.set_title(f"SpecFACE workflow core partitions at alpha={default_alpha}")
        ax.legend(ncol=2, fontsize=8)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        path = plot_dir / "workflow_partitions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    full_rows = [r for r in summary if r.mode == "SpecFACE-full"]
    if full_rows:
        full_rows = sorted(full_rows, key=lambda r: r.alpha)
        x = [r.alpha for r in full_rows]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(x, [r.target_internal_overlap for r in full_rows], marker="o", label="target internal")
        ax.plot(x, [r.draft_internal_overlap for r in full_rows], marker="o", label="draft internal")
        ax.plot(x, [r.cross_model_overlap for r in full_rows], marker="o", label="cross model")
        ax.plot(x, [r.overall_overlap for r in full_rows], marker="o", label="overall")
        ax.set_xlabel("acceptance rate alpha")
        ax.set_ylabel("overlap ratio")
        ax.set_ylim(0, 1)
        ax.set_title("Online SpecFACE overlap by acceptance rate")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / "online_overlap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if selected:
        labels = [r.mode.replace("SpecFACE-", "") for r in selected]
        target_values = [r.target_compute_ns for r in selected]
        draft_values = [r.draft_compute_ns for r in selected]
        hbm_values = [r.memory_ns for r in selected]
        sram_values = [r.journal_ns + r.journal_stall_ns for r in selected]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        bottom = [0.0] * len(labels)
        for name, values in [("target_compute", target_values),
                             ("draft_compute", draft_values),
                             ("hbm", hbm_values),
                             ("sram", sram_values)]:
            ax.bar(labels, values, bottom=bottom, label=name, color=colors[name])
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_ylabel("ns per round/token component")
        ax.set_title(f"Compute/HBM/SRAM resource time at alpha={default_alpha}")
        ax.legend(ncol=2, fontsize=8)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        path = plot_dir / "resource_breakdown.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if workload_rows:
        labels = [str(r["workload"]) for r in workload_rows]
        tpot = [float(r["tpot_speedup"]) for r in workload_rows]
        e2e = [float(r["e2e_speedup"]) for r in workload_rows]
        xs = list(range(len(labels)))
        width = 0.36
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar([x - width / 2 for x in xs], tpot, width, label="TPOT speedup", color="#2b6cb0")
        ax.bar([x + width / 2 for x in xs], e2e, width, label="E2E speedup", color="#c05621")
        ax.axhline(1.0, color="#4a5568", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("speedup over FACE")
        ax.set_title(f"SpecFACE speedup across workload classes at alpha={default_alpha}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / "workload_speedup.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def workflow_label(result: MappingResult) -> str:
    if result.mode == "FACE":
        return "FACE"
    label = result.mode.replace("SpecFACE-", "")
    if result.mode == "SpecFACE-static-gamma":
        label = f"static-g{result.gamma}"
    return label


def layout_rectangles(result: MappingResult) -> List[Dict[str, object]]:
    total = max(1, result.partition.total)
    width = 64.0
    height = 64.0
    if result.mode == "FACE":
        return [{
            "name": "target decode",
            "x": 0.0,
            "y": 0.0,
            "w": width,
            "h": height,
            "color": "target_prefill",
            "cores": result.partition.target_prefill,
        }]

    tp = result.partition.target_prefill
    tv = result.partition.target_verify
    dp = result.partition.draft_prefill
    dd = result.partition.draft_decode
    rects: List[Dict[str, object]] = []
    y = height

    def add_rect(name: str, x: float, y0: float, w: float, h: float,
                 color: str, cores: int) -> None:
        if cores <= 0 or h <= 0.0 or w <= 0.0:
            return
        rects.append({
            "name": name,
            "x": x,
            "y": y0,
            "w": w,
            "h": h,
            "color": color,
            "cores": cores,
        })

    tp_h = height * tp / total
    y -= tp_h
    add_rect("target prefill", 0.0, y, width, tp_h, "target_prefill", tp)

    middle = tv + dd
    middle_h = height * middle / total
    y -= middle_h
    if middle > 0:
        tv_w = width * tv / middle
        add_rect("target verify", 0.0, y, tv_w, middle_h, "target_verify", tv)
        add_rect("draft decode", tv_w, y, width - tv_w, middle_h,
                 "draft_decode", dd)

    dp_h = height * dp / total
    y -= dp_h
    add_rect("draft prefill/sync", 0.0, y, width, dp_h, "draft_prefill", dp)
    return rects


def timeline_events(result: MappingResult) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []

    def add(lane: str, name: str, start: float, duration: float, color: str) -> None:
        if duration <= 0.0:
            return
        events.append({
            "workflow": workflow_label(result),
            "lane": lane,
            "name": name,
            "start_ns": start,
            "end_ns": start + duration,
            "duration_ns": duration,
            "color": color,
        })

    if result.mode == "FACE" or result.fallback:
        add("Target", "target decode", 0.0, result.target_compute_ns,
            "target_verify")
        add("HBM", "KV read/write", 0.0, result.memory_ns, "hbm")
        add("NoC", "collective/data move", 0.0, result.noc_ns, "noc")
        return events

    prefill_target = result.prefill_ns * 0.78
    prefill_draft = result.prefill_ns * 0.35
    add("Target", "target prefill", 0.0, prefill_target,
        "target_prefill")
    add("Draft", "draft prefill", 0.0, prefill_draft,
        "draft_prefill")
    add("HBM", "prefill weights/KV", 0.0, result.prefill_ns * 0.55, "hbm")

    gap = result.prefill_ns * 0.08
    round_start = result.prefill_ns + gap
    draft_decode = max(0.0, result.draft_compute_ns * 0.86)
    draft_sync = max(0.0, result.draft_compute_ns - draft_decode)
    target_verify = result.target_compute_ns
    hbm_start = round_start
    sram = result.journal_ns + result.journal_stall_ns

    add("Draft", f"draft decode g={result.gamma}", round_start,
        draft_decode, "draft_decode")
    add("SRAM", "journal write", round_start, sram, "sram")
    add("Target", "target verify", round_start + draft_decode * 0.12,
        target_verify, "target_verify")
    add("Draft", "draft sync", round_start + draft_decode,
        draft_sync, "draft_prefill")
    add("HBM", "KV read / commit", hbm_start, result.memory_ns, "hbm")
    add("NoC", "broadcast/reduce", hbm_start, result.noc_ns, "noc")
    return events


def write_timeline_csv(path: Path, selected: Sequence[MappingResult]) -> None:
    rows: List[Dict[str, object]] = []
    for result in selected:
        for event in timeline_events(result):
            row = dict(event)
            for key in ["start_ns", "end_ns", "duration_ns"]:
                row[key] = f"{float(row[key]):.3f}"
            rows.append(row)
    write_csv(path, rows)



def write_layout_csv(path: Path, selected: Sequence[MappingResult]) -> None:
    rows: List[Dict[str, object]] = []
    for result in selected:
        for rect in layout_rectangles(result):
            rows.append({
                "workflow": workflow_label(result),
                "gamma": result.gamma,
                "fallback": int(result.fallback),
                "region": rect["name"],
                "cores": rect["cores"],
                "x": f"{float(rect['x']):.3f}",
                "y": f"{float(rect['y']):.3f}",
                "w": f"{float(rect['w']):.3f}",
                "h": f"{float(rect['h']):.3f}",
            })
    write_csv(path, rows)

def draw_mapping_layouts(selected: Sequence[MappingResult], plot_dir: Path,
                         colors: Dict[str, str]) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    cols = min(3, max(1, len(selected)))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.4 * rows))
    if not isinstance(axes, (list, tuple)):
        axes_list = list(getattr(axes, "flat", [axes]))
    else:
        axes_list = list(axes)
    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)

    for ax, result in zip(axes_list, selected):
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{workflow_label(result)}\ngamma={result.gamma}, speedup={result.speedup_over_face:.2f}x",
                     fontsize=10)
        for rect in layout_rectangles(result):
            patch = patches.Rectangle((rect["x"], rect["y"]), rect["w"], rect["h"],
                                      linewidth=1.2, edgecolor="#1a202c",
                                      facecolor=colors[str(rect["color"])], alpha=0.88)
            ax.add_patch(patch)
            area = float(rect["w"]) * float(rect["h"])
            if area > 110:
                ax.text(float(rect["x"]) + float(rect["w"]) / 2,
                        float(rect["y"]) + float(rect["h"]) / 2,
                        f"{rect['name']}\n{rect['cores']} cores",
                        ha="center", va="center", fontsize=8, color="#111827")
        ax.text(0, -4.8, "topology: TP top, TV/DD middle, DP bottom",
                fontsize=7, color="#4a5568")

    for ax in axes_list[len(selected):]:
        ax.axis("off")
    fig.suptitle("Intuitive core-island mapping layouts", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = plot_dir / "workflow_mapping_layouts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_workflow_timelines(selected: Sequence[MappingResult], plot_dir: Path,
                            colors: Dict[str, str]) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    lanes = ["Target", "Draft", "HBM", "SRAM", "NoC"]
    lane_y = {lane: idx for idx, lane in enumerate(reversed(lanes))}
    fig, axes = plt.subplots(len(selected), 1,
                             figsize=(12, max(3.0, 1.75 * len(selected))),
                             sharex=False)
    if len(selected) == 1:
        axes = [axes]
    for ax, result in zip(axes, selected):
        events = timeline_events(result)
        max_end = max((float(e["end_ns"]) for e in events), default=1.0)
        scale = 1000.0
        for event in events:
            y = lane_y[str(event["lane"])]
            start = float(event["start_ns"]) / scale
            dur = float(event["duration_ns"]) / scale
            patch = patches.Rectangle((start, y - 0.34), dur, 0.68,
                                      facecolor=colors[str(event["color"])],
                                      edgecolor="#1a202c", linewidth=0.8, alpha=0.9)
            ax.add_patch(patch)
            if dur > max_end / scale * 0.08:
                ax.text(start + dur / 2, y, str(event["name"]),
                        ha="center", va="center", fontsize=7)
        ax.set_yticks([lane_y[l] for l in lanes])
        ax.set_yticklabels(lanes)
        ax.set_xlim(0, max_end / scale * 1.05)
        ax.set_ylim(-0.7, len(lanes) - 0.3)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(f"{workflow_label(result)} timeline: gamma={result.gamma}, "
                     f"fallback={int(result.fallback)}, bottleneck={result.bottleneck}",
                     fontsize=10)
        ax.set_xlabel("time (us)")
    fig.suptitle("Workflow execution over time", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = plot_dir / "workflow_timelines.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def write_markdown_summary(path: Path,
                           default_alpha: float,
                           summary: List[MappingResult],
                           workload_rows: Sequence[Dict[str, object]],
                           output_dir: Path,
                           plot_dir: Path,
                           plot_paths: Sequence[Path]) -> None:
    rows = [r for r in summary if abs(r.alpha - default_alpha) < 1e-9]
    rows.sort(key=lambda r: (r.mode, r.gamma))
    best = min((r for r in rows if r.mode.startswith("SpecFACE-full")),
               key=lambda r: r.expected_ns_per_token,
               default=None)
    face = next((r for r in rows if r.mode == "FACE"), None)
    lines = [
        "# SpecFACE Micro Experiment 001",
        "",
        "This single-instance analytical experiment uses the lightweight OME "
        "search from `docs/images/SpecFACE_plan.md` and the FACE WSC baseline "
        "configuration. It now emits plots for workflow partitioning, online "
        "overlap, resource usage, and workload-class speedups.",
        "",
        f"Output directory: `{output_dir}`",
        f"Plot directory: `{plot_dir}`",
        "",
        "## Default-alpha Snapshot",
        "",
        f"Default acceptance rate: `{default_alpha}`",
        "",
        "| mode | gamma | fallback | ns/token | speedup vs FACE | partition TP/TV/DP/DD | bottleneck |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        part = f"{r.partition.target_prefill}/{r.partition.target_verify}/{r.partition.draft_prefill}/{r.partition.draft_decode}"
        lines.append(
            f"| {r.mode} | {r.gamma} | {int(r.fallback)} | "
            f"{r.expected_ns_per_token:.1f} | {r.speedup_over_face:.3f} | "
            f"{part} | {r.bottleneck} |")

    lines.extend(["", "## Figures", ""])
    for plot in plot_paths:
        rel = plot.relative_to(path.parent)
        title = plot.stem.replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{title}]({rel})", ""])

    if workload_rows:
        lines.extend([
            "## Workload-Class Speedups",
            "",
            "| workload | prompt | output | context | gamma | fallback | TPOT speedup | E2E speedup | bottleneck |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for r in workload_rows:
            lines.append(
                f"| {r['workload']} | {r['prompt_tokens']} | {r['output_tokens']} | "
                f"{r['context_tokens']} | {r['gamma']} | {r['fallback']} | "
                f"{float(r['tpot_speedup']):.3f} | {float(r['e2e_speedup']):.3f} | "
                f"{r['bottleneck']} |")

    lines.extend(["", "## Notes", ""])
    if face and best:
        lines.append(
            f"At alpha={default_alpha}, the best full SpecFACE point selects "
            f"gamma={best.gamma} and estimates {best.speedup_over_face:.3f}x "
            f"speedup over FACE decode ns/token. Journal peak is "
            f"{best.journal_peak_bytes / 1e6:.2f} MB against "
            f"{best.journal_capacity_bytes / 1e6:.2f} MB available in the "
            "draft decode island.")
    lines.append(
        "These numbers are not cycle-accurate; they are meant to drive the first "
        "microarchitecture sweep and identify useful regions for the simulator implementation.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(config_path: Path) -> None:
    exp_cfg = load_json(config_path)
    cfg_dir = config_path.parent.resolve()
    base_cfg_path = resolve_path(cfg_dir, exp_cfg.get("base_config", "../face/face_wsc_config.json"))
    base_cfg = load_json(base_cfg_path)
    output_dir = resolve_path(cfg_dir, exp_cfg.get("output_dir", "outputs"))
    trace_path = resolve_path(cfg_dir, exp_cfg.get(
        "request_trace",
        base_cfg.get("workload", {}).get("request_trace", "requests.csv"),
    ))
    if not trace_path.exists():
        trace_path = resolve_path(base_cfg_path.parent, base_cfg["workload"]["request_trace"])

    hw = parse_hardware(base_cfg, exp_cfg)
    target = parse_model(base_cfg["model"], "target")
    draft = parse_model(exp_cfg.get("draft_model", {}), "draft")
    params = parse_params(exp_cfg)
    requests = load_requests(trace_path)
    if not requests:
        raise RuntimeError(f"no requests loaded from {trace_path}")

    prompt_tokens = int(round(sum(r.input_tokens for r in requests) / len(requests)))
    output_tokens = int(round(sum(r.output_tokens for r in requests) / len(requests)))
    context_tokens = int(exp_cfg.get("context_tokens", prompt_tokens))

    summary_results: List[MappingResult] = []
    gamma_sweep_rows: List[Dict[str, object]] = []

    for alpha in params.alphas:
        face = face_row(hw, target, params, alpha, prompt_tokens, context_tokens)
        summary_results.append(face)

        for gamma in params.static_gammas:
            static = search_best("SpecFACE-static-gamma", hw, target, draft,
                                 params, alpha, gamma, prompt_tokens,
                                 context_tokens, journal=True,
                                 fallback_allowed=params.fallback_enabled)
            summary_results.append(static)

        dynamic, candidates = dynamic_gamma("SpecFACE-full", hw, target, draft,
                                            params, alpha, prompt_tokens,
                                            context_tokens, journal=True,
                                            fallback_allowed=params.fallback_enabled)
        summary_results.append(dynamic)
        for cand in candidates:
            row = result_to_row(cand)
            row["sweep_parent_mode"] = "SpecFACE-full"
            gamma_sweep_rows.append(row)

        no_journal, no_journal_candidates = dynamic_gamma(
            "SpecFACE-no-journal", hw, target, draft, params, alpha,
            prompt_tokens, context_tokens, journal=False,
            fallback_allowed=params.fallback_enabled)
        summary_results.append(no_journal)
        for cand in no_journal_candidates:
            row = result_to_row(cand)
            row["sweep_parent_mode"] = "SpecFACE-no-journal"
            gamma_sweep_rows.append(row)

        no_fallback, _ = dynamic_gamma("SpecFACE-no-fallback", hw, target, draft,
                                       params, alpha, prompt_tokens,
                                       context_tokens, journal=True,
                                       fallback_allowed=False)
        summary_results.append(no_fallback)

        fixed, _ = dynamic_gamma("SpecFACE-fixed-partition", hw, target, draft,
                                 params, alpha, prompt_tokens, context_tokens,
                                 journal=True,
                                 fallback_allowed=params.fallback_enabled,
                                 use_fixed_partition=True)
        summary_results.append(fixed)

    summary_rows = [result_to_row(r) for r in summary_results]
    write_csv(output_dir / "specface_summary.csv", summary_rows)
    write_csv(output_dir / "specface_gamma_sweep.csv", gamma_sweep_rows)

    variants = workload_variant_specs(exp_cfg, prompt_tokens, output_tokens, context_tokens)
    workload_rows = run_workload_variants(hw, target, draft, params, variants)
    write_csv(output_dir / "specface_workload_summary.csv", workload_rows)

    default_dynamic = next(
        (r for r in summary_results
         if r.mode == "SpecFACE-full" and abs(r.alpha - params.default_alpha) < 1e-9),
        None,
    )
    default_face = next(
        (r for r in summary_results
         if r.mode == "FACE" and abs(r.alpha - params.default_alpha) < 1e-9),
        None,
    )
    if default_dynamic is not None and default_face is not None:
        request_rows = summarize_requests(requests, default_dynamic, default_face)
        write_csv(output_dir / "specface_request_estimates.csv", request_rows)

    summary_doc = resolve_path(cfg_dir, exp_cfg.get(
        "summary_markdown",
        "../../../docs/images/SpecFACE_experiment_001.md",
    ))
    plot_dir = resolve_path(cfg_dir, exp_cfg.get(
        "plot_dir",
        "../../../docs/images/specface_experiment_001",
    ))
    selected_default = select_default_workflow_rows(summary_results, params.default_alpha)
    write_layout_csv(output_dir / "specface_mapping_rects.csv", selected_default)
    write_timeline_csv(output_dir / "specface_timeline_events.csv", selected_default)

    plot_paths = make_plots(summary_results, workload_rows, params.default_alpha, plot_dir)
    write_markdown_summary(summary_doc, params.default_alpha, summary_results,
                           workload_rows, output_dir, plot_dir, plot_paths)

    print(f"SpecFACE micro experiment complete")
    print(f"  config: {config_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  summary: {output_dir / 'specface_summary.csv'}")
    print(f"  gamma_sweep: {output_dir / 'specface_gamma_sweep.csv'}")
    print(f"  workload_summary: {output_dir / 'specface_workload_summary.csv'}")
    print(f"  mapping_rects: {output_dir / 'specface_mapping_rects.csv'}")
    print(f"  timeline_events: {output_dir / 'specface_timeline_events.csv'}")
    print(f"  plots: {plot_dir}")
    print(f"  report: {summary_doc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpecFACE micro experiments")
    parser.add_argument("--config", required=True, type=Path,
                        help="SpecFACE micro-experiment JSON config")
    args = parser.parse_args()
    run(args.config.resolve())
