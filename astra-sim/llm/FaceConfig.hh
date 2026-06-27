/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_SIM_LLM_FACE_CONFIG_HH__
#define __ASTRA_SIM_LLM_FACE_CONFIG_HH__

#include <cstdint>
#include <string>
#include <vector>

namespace AstraSim {

enum class FaceSchedulerMode { Face, Serial, Disaggregated };

struct FaceHardwareConfig {
    int die_rows = 1;
    int die_cols = 1;
    int hbm_chiplets_per_die = 1;
    double dram_capacity_gb_per_hbm = 16.0;
    double dram_bandwidth_gbps_per_hbm = 410.0;
    double d2d_bandwidth_tbps = 1.0;
    double core_sram_mb = 0.75;
    double core_compute_gflops = 740.0;
    int noc_width_bits = 512;
    int core_rows = 16;
    int core_cols = 16;
};

struct FaceModelConfig {
    std::string name = "llm";
    int num_layers = 32;
    int hidden_size = 4096;
    int num_heads = 32;
    int num_kv_heads = 32;
    int head_dim = 128;
    uint64_t weight_bytes = 0;
    uint64_t kv_bytes_per_token = 0;
};

struct FaceWorkloadConfig {
    std::string request_trace;
    std::string output_dir = "face_outputs";
    int prefill_chunk_tokens = 1024;
};

struct FaceCseConfig {
    std::vector<int> decode_batch_sizes;
    std::vector<int> decode_token_counts;
    std::vector<int> prefill_tile_candidates;
    std::vector<int> decode_tile_candidates;
};

struct FaceRequest {
    int id = -1;
    uint64_t arrival_time_ns = 0;
    int input_tokens = 0;
    int output_tokens = 0;

    int prefill_instance = -1;
    int decode_instance = -1;
    uint64_t prefill_start_ns = 0;
    uint64_t prefill_end_ns = 0;
    uint64_t decode_start_ns = 0;
    uint64_t first_token_end_ns = 0;
    uint64_t decode_end_ns = 0;
    bool remote_kv = false;
    bool kv_offloaded = false;
    int selected_prefill_tile = 0;
    int selected_decode_tile = 0;
};

struct FaceConfig {
    FaceHardwareConfig hardware;
    FaceModelConfig model;
    FaceWorkloadConfig workload;
    FaceCseConfig cse;
    FaceSchedulerMode scheduler = FaceSchedulerMode::Face;
    std::string config_dir = ".";

    static FaceConfig load(const std::string& filename);
    static std::vector<FaceRequest> load_requests(const FaceConfig& config);
};

std::string to_string(FaceSchedulerMode mode);

}  // namespace AstraSim

#endif /* __ASTRA_SIM_LLM_FACE_CONFIG_HH__ */
