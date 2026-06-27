/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/llm/FaceEvaluator.hh"

#include <algorithm>
#include <cmath>

namespace AstraSim {
namespace {

uint64_t ceil_to_u64(double value) {
    if (value <= 1.0) {
        return 1;
    }
    return static_cast<uint64_t>(std::ceil(value));
}

double safe_div(double numerator, double denominator) {
    if (denominator <= 0.0) {
        return numerator;
    }
    return numerator / denominator;
}

}  // namespace

FaceEvaluator::FaceEvaluator(const FaceConfig& config) : config_(config) {}

FaceEvalResult FaceEvaluator::evaluate_attention(
    const FaceInstanceShape& shape,
    int prefill_tokens,
    int decode_batch,
    int decode_tokens,
    int prefill_tile,
    int decode_tile) const {
    const auto prefill_latency =
        evaluate_prefill_chunk(shape, prefill_tokens, prefill_tile);
    const auto decode_latency =
        evaluate_decode_step(shape, decode_batch, decode_tokens, decode_tile);

    const auto latency = std::max(prefill_latency, decode_latency);
    const auto useful = static_cast<double>(prefill_latency + decode_latency);
    FaceEvalResult result;
    result.latency_ns = latency;
    result.utilization =
        std::min(1.0, safe_div(useful, static_cast<double>(2 * latency)));
    result.dram_bytes =
        static_cast<uint64_t>(prefill_tokens + decode_batch * decode_tokens) *
        config_.model.hidden_size * sizeof(uint16_t);
    result.noc_bytes = result.dram_bytes / std::max(1, shape.die_count());
    result.d2d_bytes =
        static_cast<uint64_t>(config_.model.hidden_size) * decode_batch *
        sizeof(uint16_t);
    return result;
}

uint64_t FaceEvaluator::evaluate_prefill_chunk(
    const FaceInstanceShape& shape,
    int prefill_tokens,
    int tile) const {
    const auto& model = config_.model;
    const auto tokens = std::max(1, prefill_tokens);
    const auto tile_tokens = std::max(1, tile);
    const auto hidden = static_cast<double>(model.hidden_size);
    const auto heads = static_cast<double>(std::max(1, model.num_heads));
    const auto head_dim = static_cast<double>(std::max(1, model.head_dim));

    const double qkv_projection_ops = 4.0 * tokens * hidden * hidden;
    const double attention_ops = 2.0 * heads * tokens * tokens * head_dim;
    const double mlp_ops = 8.0 * tokens * hidden * hidden;
    const double ops_per_layer = qkv_projection_ops + attention_ops + mlp_ops;

    const auto compute_ns = safe_div(
        ops_per_layer * model.num_layers, total_compute_gflops(shape));
    const double dram_bytes =
        static_cast<double>(model.weight_bytes) / std::max(1, shape.die_count()) +
        4.0 * tokens * hidden * sizeof(uint16_t) * model.num_layers;
    const auto dram_ns =
        safe_div(dram_bytes, dram_bandwidth_bytes_per_ns(shape));

    const double softmax_bytes =
        static_cast<double>(tokens) * tokens * sizeof(uint16_t) *
        std::max(1, model.num_heads);
    const auto vector_ns =
        safe_div(softmax_bytes, dram_bandwidth_bytes_per_ns(shape));

    const double noc_bytes =
        3.0 * tile_tokens * hidden * sizeof(uint16_t) *
        std::max(1, shape.die_count());
    const auto noc_ns = safe_div(noc_bytes, noc_bandwidth_bytes_per_ns(shape));

    const auto ring_ns = bidirectional_ring_latency_ns(
        shape, static_cast<uint64_t>(tokens * hidden * sizeof(uint16_t)));

    const auto pe_pipeline_ns = compute_ns * 0.72;
    const auto vector_pipeline_ns = vector_ns + noc_ns;
    return ceil_to_u64(std::max(pe_pipeline_ns, vector_pipeline_ns) +
                       0.25 * dram_ns + ring_ns);
}

uint64_t FaceEvaluator::evaluate_decode_step(
    const FaceInstanceShape& shape,
    int decode_batch,
    int context_tokens,
    int tile) const {
    const auto& model = config_.model;
    const auto batch = std::max(1, decode_batch);
    const auto context = std::max(1, context_tokens);
    const auto tile_tokens = std::max(1, tile);
    const auto hidden = static_cast<double>(model.hidden_size);
    const auto heads = static_cast<double>(std::max(1, model.num_heads));
    const auto head_dim = static_cast<double>(std::max(1, model.head_dim));

    const double projection_ops = 4.0 * batch * hidden * hidden;
    const double attention_ops =
        2.0 * batch * heads * context * head_dim;
    const double mlp_ops = 8.0 * batch * hidden * hidden;
    const double ops_per_layer = projection_ops + attention_ops + mlp_ops;

    const auto compute_ns =
        safe_div(ops_per_layer * model.num_layers, total_compute_gflops(shape));
    const double kv_bytes =
        static_cast<double>(kv_bytes_for_tokens(context)) * batch;
    const double activation_bytes =
        4.0 * batch * hidden * sizeof(uint16_t) * model.num_layers;
    const auto dram_ns =
        safe_div(kv_bytes + activation_bytes,
                 dram_bandwidth_bytes_per_ns(shape));
    const double noc_bytes =
        2.0 * tile_tokens * hidden * sizeof(uint16_t) *
        std::max(1, shape.die_count());
    const auto noc_ns = safe_div(noc_bytes, noc_bandwidth_bytes_per_ns(shape));
    const auto ring_ns = bidirectional_ring_latency_ns(
        shape, static_cast<uint64_t>(batch * hidden * sizeof(uint16_t)));

    const auto pe_pipeline_ns = compute_ns * 0.60;
    const auto vector_pipeline_ns = dram_ns + noc_ns;
    return ceil_to_u64(std::max(pe_pipeline_ns, vector_pipeline_ns) + ring_ns);
}

uint64_t FaceEvaluator::kv_bytes_for_tokens(int tokens) const {
    return static_cast<uint64_t>(std::max(1, tokens)) *
           config_.model.kv_bytes_per_token;
}

double FaceEvaluator::dram_bandwidth_bytes_per_ns(
    const FaceInstanceShape& shape) const {
    return config_.hardware.dram_bandwidth_gbps_per_hbm *
           config_.hardware.hbm_chiplets_per_die * shape.die_count();
}

double FaceEvaluator::d2d_bandwidth_bytes_per_ns() const {
    return config_.hardware.d2d_bandwidth_tbps * 1000.0;
}

double FaceEvaluator::total_compute_gflops(
    const FaceInstanceShape& shape) const {
    return config_.hardware.core_compute_gflops *
           config_.hardware.core_rows * config_.hardware.core_cols *
           shape.die_count();
}

double FaceEvaluator::noc_bandwidth_bytes_per_ns(
    const FaceInstanceShape& shape) const {
    const auto bytes_per_cycle =
        std::max(1, config_.hardware.noc_width_bits) / 8.0;
    const auto core_count =
        config_.hardware.core_rows * config_.hardware.core_cols *
        shape.die_count();
    return bytes_per_cycle * std::max(1, core_count);
}

uint64_t FaceEvaluator::bidirectional_ring_latency_ns(
    const FaceInstanceShape& shape,
    uint64_t bytes) const {
    if (shape.die_count() <= 1 || bytes == 0) {
        return 0;
    }
    const auto steps = std::max(shape.rows, shape.cols);
    const auto effective_bandwidth =
        std::max(1.0, 2.0 * d2d_bandwidth_bytes_per_ns());
    return ceil_to_u64(static_cast<double>(bytes) * steps /
                       effective_bandwidth);
}

}  // namespace AstraSim
