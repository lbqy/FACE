/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/llm/FaceDAS.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace AstraSim {
namespace {

uint64_t multiply_latency(uint64_t latency_ns, int count) {
    return latency_ns * static_cast<uint64_t>(std::max(1, count));
}

uint64_t interval_overlap(uint64_t start_a,
                          uint64_t end_a,
                          uint64_t start_b,
                          uint64_t end_b) {
    const auto start = std::max(start_a, start_b);
    const auto end = std::min(end_a, end_b);
    return end > start ? end - start : 0;
}

int max_batch_size(const FaceConfig& config) {
    if (config.cse.decode_batch_sizes.empty()) {
        return 1;
    }
    return *std::max_element(config.cse.decode_batch_sizes.begin(),
                             config.cse.decode_batch_sizes.end());
}

}  // namespace

FaceDAS::FaceDAS(const FaceConfig& config,
                 const FaceEvaluator& evaluator,
                 const FaceCSE& cse,
                 FaceOMM& omm)
    : config_(config), evaluator_(evaluator), cse_(cse), omm_(omm) {}

void FaceDAS::schedule_request(FaceCseResult& cse_result,
                               FaceRequest& request) {
    if (cse_result.instances.empty()) {
        throw std::runtime_error("FACE DAS cannot schedule without instances");
    }

    const auto chunks = std::max(
        1, static_cast<int>(std::ceil(
               static_cast<double>(request.input_tokens) /
               std::max(1, config_.workload.prefill_chunk_tokens))));

    if (config_.scheduler == FaceSchedulerMode::Serial) {
        const auto instance_id = choose_prefill_instance(cse_result, request);
        auto& instance = cse_result.instances[instance_id];
        const auto entry = lookup_or_evaluate(cse_result, instance.shape, 1,
                                              request.input_tokens);
        const auto prefill_latency = multiply_latency(
            evaluator_.evaluate_prefill_chunk(instance.shape,
                                              config_.workload
                                                  .prefill_chunk_tokens,
                                              entry.prefill_tile),
            chunks);
        const auto decode_step = evaluator_.evaluate_decode_step(
            instance.shape, 1, request.input_tokens, entry.decode_tile);
        request.prefill_instance = instance_id;
        request.decode_instance = instance_id;
        request.prefill_start_ns =
            std::max(request.arrival_time_ns, instance.serial_available_ns);
        request.prefill_end_ns = request.prefill_start_ns + prefill_latency;
        request.decode_start_ns = request.prefill_end_ns;
        request.first_token_end_ns = request.decode_start_ns + decode_step;
        request.decode_end_ns =
            request.decode_start_ns + multiply_latency(decode_step,
                                                       request.output_tokens);
        request.selected_prefill_tile = entry.prefill_tile;
        request.selected_decode_tile = entry.decode_tile;
        instance.serial_available_ns = request.decode_end_ns;
        instance.prefill_busy_ns += prefill_latency;
        instance.decode_busy_ns += multiply_latency(decode_step,
                                                    request.output_tokens);
        instance.dram_traffic_bytes +=
            evaluator_.kv_bytes_for_tokens(request.input_tokens);
        omm_.place_kv(cse_result, request.id, instance_id,
                      request.input_tokens + request.output_tokens);
        return;
    }

    const auto prefill_instance_id = choose_prefill_instance(cse_result, request);
    auto& prefill_instance = cse_result.instances[prefill_instance_id];
    auto entry = lookup_or_evaluate(cse_result, prefill_instance.shape, 1,
                                    request.input_tokens);
    const auto prefill_latency = multiply_latency(
        evaluator_.evaluate_prefill_chunk(prefill_instance.shape,
                                          config_.workload.prefill_chunk_tokens,
                                          entry.prefill_tile),
        chunks);

    request.prefill_instance = prefill_instance_id;
    request.prefill_start_ns = std::max(request.arrival_time_ns,
                                        prefill_instance.prefill_available_ns);
    request.prefill_end_ns = request.prefill_start_ns + prefill_latency;
    const auto prefill_decode_overlap = interval_overlap(
        request.prefill_start_ns, request.prefill_end_ns,
        request.prefill_start_ns, prefill_instance.decode_available_ns);
    prefill_instance.overlap_ns += prefill_decode_overlap;
    prefill_instance.prefill_available_ns = request.prefill_end_ns;
    prefill_instance.prefill_busy_ns += prefill_latency;
    prefill_instance.dram_traffic_bytes +=
        evaluator_.kv_bytes_for_tokens(request.input_tokens);

    omm_.place_kv(cse_result, request.id, prefill_instance_id,
                  request.input_tokens + request.output_tokens);

    int active_decodes = 0;
    for (const auto& instance : cse_result.instances) {
        if (instance.decode_available_ns > request.prefill_end_ns) {
            active_decodes++;
        }
    }
    const auto decode_batch = std::min(max_batch_size(config_),
                                       std::max(1, active_decodes + 1));

    std::optional<FaceLutEntry> selected_entry;
    int decode_instance_id = prefill_instance_id;
    if (config_.scheduler == FaceSchedulerMode::Disaggregated) {
        const auto split = std::max(1, static_cast<int>(cse_result.instances.size()) / 2);
        uint64_t best_available = std::numeric_limits<uint64_t>::max();
        for (int i = split; i < static_cast<int>(cse_result.instances.size()); i++) {
            if (cse_result.instances[i].decode_available_ns < best_available) {
                best_available = cse_result.instances[i].decode_available_ns;
                decode_instance_id = i;
            }
        }
        selected_entry = lookup_or_evaluate(cse_result,
                                            cse_result.instances[decode_instance_id].shape,
                                            decode_batch, request.input_tokens);
    } else {
        decode_instance_id = choose_decode_instance(cse_result, request,
                                                    decode_batch, entry,
                                                    &selected_entry);
    }

    if (!selected_entry.has_value()) {
        selected_entry = lookup_or_evaluate(cse_result,
                                            cse_result.instances[decode_instance_id].shape,
                                            decode_batch, request.input_tokens);
    }

    bool remote_kv = false;
    bool offloaded = false;
    omm_.reserve_for_decode(cse_result, request.id, decode_instance_id,
                            &remote_kv, &offloaded);
    auto& decode_instance = cse_result.instances[decode_instance_id];
    const auto decode_step = evaluator_.evaluate_decode_step(
        decode_instance.shape, decode_batch, request.input_tokens,
        selected_entry->decode_tile);
    const auto remote_penalty = remote_kv
        ? static_cast<uint64_t>(std::ceil(
              static_cast<double>(omm_.remote_access_bytes(request.id)) /
              std::max(1.0, evaluator_.d2d_bandwidth_bytes_per_ns())))
        : 0;

    request.decode_instance = decode_instance_id;
    request.remote_kv = remote_kv;
    request.kv_offloaded = offloaded;
    request.selected_prefill_tile = selected_entry->prefill_tile;
    request.selected_decode_tile = selected_entry->decode_tile;
    request.decode_start_ns =
        std::max(request.prefill_end_ns, decode_instance.decode_available_ns) +
        remote_penalty;
    request.first_token_end_ns = request.decode_start_ns + decode_step;
    request.decode_end_ns =
        request.decode_start_ns + multiply_latency(decode_step,
                                                   request.output_tokens);

    const auto decode_prefill_overlap = interval_overlap(
        request.decode_start_ns, request.decode_end_ns,
        request.decode_start_ns, decode_instance.prefill_available_ns);
    decode_instance.overlap_ns += decode_prefill_overlap;
    decode_instance.decode_available_ns = request.decode_end_ns;
    decode_instance.decode_busy_ns += multiply_latency(decode_step,
                                                       request.output_tokens);
    decode_instance.dram_traffic_bytes +=
        evaluator_.kv_bytes_for_tokens(request.input_tokens) *
        static_cast<uint64_t>(std::max(1, request.output_tokens));
}

int FaceDAS::choose_prefill_instance(const FaceCseResult& cse_result,
                                     const FaceRequest& request) const {
    int begin = 0;
    int end = static_cast<int>(cse_result.instances.size());
    if (config_.scheduler == FaceSchedulerMode::Disaggregated && end > 1) {
        end = std::max(1, end / 2);
    }

    int best = begin;
    uint64_t best_time = std::numeric_limits<uint64_t>::max();
    for (int i = begin; i < end; i++) {
        const auto& instance = cse_result.instances[i];
        const auto available = config_.scheduler == FaceSchedulerMode::Serial
                                   ? instance.serial_available_ns
                                   : instance.prefill_available_ns;
        const auto start = std::max(request.arrival_time_ns, available);
        if (start < best_time ||
            (start == best_time && instance.id < cse_result.instances[best].id)) {
            best = i;
            best_time = start;
        }
    }
    return best;
}

int FaceDAS::choose_decode_instance(
    const FaceCseResult& cse_result,
    const FaceRequest& request,
    int decode_batch_size,
    const FaceLutEntry& current_entry,
    std::optional<FaceLutEntry>* selected_entry) const {
    (void)current_entry;
    const auto candidates = omm_.schedulable_instances(
        cse_result, request.id, request.prefill_instance);
    int best = request.prefill_instance;
    uint64_t best_delta = std::numeric_limits<uint64_t>::max();
    std::optional<FaceLutEntry> best_entry;
    for (const auto candidate : candidates) {
        if (candidate < 0 ||
            candidate >= static_cast<int>(cse_result.instances.size())) {
            continue;
        }
        const auto& instance = cse_result.instances[candidate];
        const auto entry = lookup_or_evaluate(cse_result, instance.shape,
                                              decode_batch_size,
                                              request.input_tokens);
        const auto step = evaluator_.evaluate_decode_step(
            instance.shape, decode_batch_size, request.input_tokens,
            entry.decode_tile);
        const auto start = std::max(request.prefill_end_ns,
                                    instance.decode_available_ns);
        const auto finish = start + multiply_latency(step, request.output_tokens);
        const auto delta = finish > instance.decode_available_ns
                               ? finish - instance.decode_available_ns
                               : 0;
        if (delta < best_delta ||
            (delta == best_delta && candidate < best)) {
            best = candidate;
            best_delta = delta;
            best_entry = entry;
        }
    }
    if (selected_entry != nullptr) {
        *selected_entry = best_entry;
    }
    return best;
}

FaceLutEntry FaceDAS::lookup_or_evaluate(const FaceCseResult& cse_result,
                                         const FaceInstanceShape& shape,
                                         int decode_batch_size,
                                         int decode_token_count) const {
    auto entry = cse_.lookup(cse_result, shape,
                            config_.workload.prefill_chunk_tokens,
                            decode_batch_size, decode_token_count);
    if (entry.has_value()) {
        return *entry;
    }

    const auto prefill_tile = config_.cse.prefill_tile_candidates.empty()
                                  ? config_.workload.prefill_chunk_tokens
                                  : config_.cse.prefill_tile_candidates.front();
    const auto decode_tile = config_.cse.decode_tile_candidates.empty()
                                 ? 1
                                 : config_.cse.decode_tile_candidates.front();
    const auto eval = evaluator_.evaluate_attention(
        shape, config_.workload.prefill_chunk_tokens, decode_batch_size,
        decode_token_count, prefill_tile, decode_tile);

    FaceLutEntry fallback;
    fallback.key = FaceLutKey{shape.rows, shape.cols,
                              config_.workload.prefill_chunk_tokens,
                              decode_batch_size, decode_token_count};
    fallback.prefill_tile = prefill_tile;
    fallback.decode_tile = decode_tile;
    fallback.attention_latency_ns = eval.latency_ns;
    fallback.utilization = eval.utilization;
    return fallback;
}

void FaceDAS::record_overlap(FaceInstance& instance,
                             uint64_t prefill_start,
                             uint64_t prefill_end,
                             uint64_t decode_start,
                             uint64_t decode_end) const {
    instance.overlap_ns += interval_overlap(prefill_start, prefill_end,
                                            decode_start, decode_end);
}

}  // namespace AstraSim
