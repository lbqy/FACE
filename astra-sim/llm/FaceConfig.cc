/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/llm/FaceConfig.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <json/json.hpp>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace AstraSim {
namespace {

std::vector<int> get_int_vector(const json& j,
                                const std::string& key,
                                std::vector<int> fallback) {
    if (!j.contains(key)) {
        return fallback;
    }
    return j.at(key).get<std::vector<int>>();
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> columns;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        columns.push_back(trim(item));
    }
    return columns;
}

bool looks_like_header(const std::vector<std::string>& columns) {
    if (columns.empty()) {
        return true;
    }
    return !std::all_of(columns[0].begin(), columns[0].end(), [](char c) {
        return c == '-' || (c >= '0' && c <= '9');
    });
}

std::string resolve_path(const std::string& base_dir,
                         const std::string& maybe_relative) {
    std::filesystem::path path(maybe_relative);
    if (path.is_absolute()) {
        return path.string();
    }
    return (std::filesystem::path(base_dir) / path).lexically_normal().string();
}

FaceSchedulerMode parse_scheduler(const json& j) {
    std::string mode = "face";
    if (j.contains("scheduler")) {
        if (j.at("scheduler").is_string()) {
            mode = j.at("scheduler").get<std::string>();
        } else if (j.at("scheduler").is_object() &&
                   j.at("scheduler").contains("mode")) {
            mode = j.at("scheduler").at("mode").get<std::string>();
        }
    }
    if (mode == "face") {
        return FaceSchedulerMode::Face;
    }
    if (mode == "serial") {
        return FaceSchedulerMode::Serial;
    }
    if (mode == "disaggregated") {
        return FaceSchedulerMode::Disaggregated;
    }
    throw std::runtime_error("Unknown FACE scheduler mode: " + mode);
}

}  // namespace

std::string to_string(FaceSchedulerMode mode) {
    switch (mode) {
    case FaceSchedulerMode::Face:
        return "face";
    case FaceSchedulerMode::Serial:
        return "serial";
    case FaceSchedulerMode::Disaggregated:
        return "disaggregated";
    }
    return "unknown";
}

FaceConfig FaceConfig::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Unable to open LLM server configuration: " +
                                 filename);
    }

    json root;
    in >> root;

    FaceConfig config;
    const auto config_path = std::filesystem::path(filename);
    config.config_dir =
        config_path.has_parent_path() ? config_path.parent_path().string() : ".";

    const auto& hardware = root.at("hardware");
    const auto die_array =
        get_int_vector(hardware, "die_array", {config.hardware.die_rows,
                                                config.hardware.die_cols});
    if (die_array.size() != 2) {
        throw std::runtime_error("hardware.die_array must contain two values");
    }
    config.hardware.die_rows = die_array[0];
    config.hardware.die_cols = die_array[1];
    config.hardware.hbm_chiplets_per_die =
        hardware.value("hbm_chiplets_per_die",
                       config.hardware.hbm_chiplets_per_die);
    config.hardware.dram_capacity_gb_per_hbm =
        hardware.value("dram_capacity_gb_per_hbm",
                       config.hardware.dram_capacity_gb_per_hbm);
    config.hardware.dram_bandwidth_gbps_per_hbm =
        hardware.value("dram_bandwidth_gbps_per_hbm",
                       config.hardware.dram_bandwidth_gbps_per_hbm);
    config.hardware.d2d_bandwidth_tbps =
        hardware.value("d2d_bandwidth_tbps",
                       config.hardware.d2d_bandwidth_tbps);
    config.hardware.core_sram_mb =
        hardware.value("core_sram_mb", config.hardware.core_sram_mb);
    config.hardware.core_compute_gflops =
        hardware.value("core_compute_gflops",
                       config.hardware.core_compute_gflops);
    config.hardware.noc_width_bits =
        hardware.value("noc_width_bits", config.hardware.noc_width_bits);
    const auto core_array =
        get_int_vector(hardware, "core_array", {config.hardware.core_rows,
                                                 config.hardware.core_cols});
    if (core_array.size() != 2) {
        throw std::runtime_error("hardware.core_array must contain two values");
    }
    config.hardware.core_rows = core_array[0];
    config.hardware.core_cols = core_array[1];

    const auto& model = root.at("model");
    config.model.name = model.value("name", config.model.name);
    config.model.num_layers = model.value("num_layers", config.model.num_layers);
    config.model.hidden_size =
        model.value("hidden_size", config.model.hidden_size);
    config.model.num_heads = model.value("num_heads", config.model.num_heads);
    config.model.num_kv_heads =
        model.value("num_kv_heads", config.model.num_kv_heads);
    config.model.head_dim = model.value("head_dim", config.model.head_dim);
    config.model.weight_bytes =
        model.value("weight_bytes", config.model.weight_bytes);
    if (config.model.weight_bytes == 0) {
        const auto hidden = static_cast<uint64_t>(config.model.hidden_size);
        config.model.weight_bytes =
            static_cast<uint64_t>(12) * config.model.num_layers * hidden *
            hidden * sizeof(uint16_t);
    }
    config.model.kv_bytes_per_token =
        model.value("kv_bytes_per_token", config.model.kv_bytes_per_token);
    if (config.model.kv_bytes_per_token == 0) {
        config.model.kv_bytes_per_token =
            static_cast<uint64_t>(2) * config.model.num_layers *
            config.model.num_kv_heads * config.model.head_dim * sizeof(uint16_t);
    }

    const auto& workload = root.at("workload");
    config.workload.request_trace =
        resolve_path(config.config_dir,
                     workload.at("request_trace").get<std::string>());
    config.workload.output_dir = resolve_path(
        config.config_dir,
        workload.value("output_dir", config.workload.output_dir));
    config.workload.prefill_chunk_tokens =
        workload.value("prefill_chunk_tokens",
                       config.workload.prefill_chunk_tokens);

    const auto& cse = root.at("cse");
    config.cse.decode_batch_sizes =
        get_int_vector(cse, "decode_batch_sizes", {1, 2, 4, 8});
    config.cse.decode_token_counts =
        get_int_vector(cse, "decode_token_counts", {128, 512, 1024, 2048});
    config.cse.prefill_tile_candidates =
        get_int_vector(cse, "prefill_tile_candidates", {128, 256, 512});
    config.cse.decode_tile_candidates =
        get_int_vector(cse, "decode_tile_candidates", {16, 32, 64, 128});
    config.scheduler = parse_scheduler(root);

    return config;
}

std::vector<FaceRequest> FaceConfig::load_requests(const FaceConfig& config) {
    std::ifstream in(config.workload.request_trace);
    if (!in) {
        throw std::runtime_error("Unable to open FACE request trace: " +
                                 config.workload.request_trace);
    }

    std::vector<FaceRequest> requests;
    std::string line;
    int auto_id = 0;
    bool first_line = true;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto columns = split_csv_line(line);
        if (first_line && looks_like_header(columns)) {
            first_line = false;
            continue;
        }
        first_line = false;
        if (columns.size() < 4) {
            throw std::runtime_error(
                "FACE request trace rows must contain request_id,"
                "arrival_time_ns,input_tokens,output_tokens");
        }
        FaceRequest request;
        request.id = std::stoi(columns[0]);
        if (request.id < 0) {
            request.id = auto_id;
        }
        request.arrival_time_ns = std::stoull(columns[1]);
        request.input_tokens = std::stoi(columns[2]);
        request.output_tokens = std::stoi(columns[3]);
        requests.push_back(request);
        auto_id++;
    }

    std::sort(requests.begin(), requests.end(),
              [](const FaceRequest& a, const FaceRequest& b) {
                  if (a.arrival_time_ns == b.arrival_time_ns) {
                      return a.id < b.id;
                  }
                  return a.arrival_time_ns < b.arrival_time_ns;
              });
    return requests;
}

}  // namespace AstraSim
