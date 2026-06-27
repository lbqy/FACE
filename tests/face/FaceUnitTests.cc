#include "astra-sim/llm/FaceCSE.hh"
#include "astra-sim/llm/FaceDAS.hh"
#include "astra-sim/llm/FaceEvaluator.hh"
#include "astra-sim/llm/FaceOMM.hh"

#include <cassert>
#include <iostream>

using namespace AstraSim;

namespace {

FaceConfig make_config(FaceSchedulerMode mode) {
    FaceConfig config;
    config.hardware.die_rows = 4;
    config.hardware.die_cols = 4;
    config.hardware.hbm_chiplets_per_die = 2;
    config.hardware.dram_capacity_gb_per_hbm = 8;
    config.hardware.dram_bandwidth_gbps_per_hbm = 400;
    config.hardware.d2d_bandwidth_tbps = 4;
    config.hardware.core_sram_mb = 0.75;
    config.hardware.core_compute_gflops = 740;
    config.hardware.noc_width_bits = 512;
    config.hardware.core_rows = 8;
    config.hardware.core_cols = 8;
    config.model.name = "unit-test-llm";
    config.model.num_layers = 4;
    config.model.hidden_size = 1024;
    config.model.num_heads = 8;
    config.model.num_kv_heads = 8;
    config.model.head_dim = 128;
    config.model.weight_bytes = 2ull * 1000 * 1000 * 1000;
    config.model.kv_bytes_per_token = 65536;
    config.workload.prefill_chunk_tokens = 128;
    config.cse.decode_batch_sizes = {1, 2, 4};
    config.cse.decode_token_counts = {128, 256, 512};
    config.cse.prefill_tile_candidates = {64, 128};
    config.cse.decode_tile_candidates = {16, 32};
    config.scheduler = mode;
    return config;
}

void test_cse_and_lut() {
    auto config = make_config(FaceSchedulerMode::Face);
    FaceEvaluator evaluator(config);
    FaceCSE cse(config, evaluator);
    auto result = cse.run();
    assert(!result.feasible_shapes.empty());
    assert(!result.instances.empty());
    assert(!result.lut.empty());
    auto entry = cse.lookup(result, result.selected_shape,
                            config.workload.prefill_chunk_tokens, 2, 256);
    assert(entry.has_value());
    assert(entry->attention_latency_ns > 0);
}

void test_omm_range_and_capacity() {
    auto config = make_config(FaceSchedulerMode::Face);
    FaceEvaluator evaluator(config);
    FaceCSE cse(config, evaluator);
    auto result = cse.run();
    FaceOMM omm(config, evaluator);
    omm.place_kv(result, 7, 0, 256);
    auto candidates = omm.schedulable_instances(result, 7, 0);
    assert(!candidates.empty());
    bool remote = false;
    bool offloaded = false;
    assert(omm.reserve_for_decode(result, 7, candidates.back(), &remote,
                                  &offloaded));
}

void test_das_schedules_request() {
    auto config = make_config(FaceSchedulerMode::Face);
    FaceEvaluator evaluator(config);
    FaceCSE cse(config, evaluator);
    auto result = cse.run();
    FaceOMM omm(config, evaluator);
    FaceDAS das(config, evaluator, cse, omm);
    FaceRequest request;
    request.id = 1;
    request.arrival_time_ns = 0;
    request.input_tokens = 256;
    request.output_tokens = 8;
    das.schedule_request(result, request);
    assert(request.prefill_instance >= 0);
    assert(request.decode_instance >= 0);
    assert(request.decode_end_ns > request.arrival_time_ns);
}

}  // namespace

int main() {
    test_cse_and_lut();
    test_omm_range_and_capacity();
    test_das_schedules_request();
    std::cout << "FACE unit tests passed\n";
    return 0;
}
