/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/llm/FaceServerCoordinator.hh"

#include "astra-sim/common/Logging.hh"
#include "astra-sim/llm/FaceDAS.hh"
#include "astra-sim/llm/FaceEvaluator.hh"
#include "astra-sim/llm/FaceOMM.hh"
#include "astra-sim/system/Sys.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace AstraSim {
namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto index = static_cast<size_t>(
        std::min<double>(values.size() - 1, p * (values.size() - 1)));
    return values[index];
}

uint64_t max_decode_end(const std::vector<FaceRequest>& requests) {
    uint64_t end = 0;
    for (const auto& request : requests) {
        end = std::max(end, request.decode_end_ns);
    }
    return end;
}

uint64_t min_arrival(const std::vector<FaceRequest>& requests) {
    if (requests.empty()) {
        return 0;
    }
    uint64_t arrival = requests.front().arrival_time_ns;
    for (const auto& request : requests) {
        arrival = std::min(arrival, request.arrival_time_ns);
    }
    return arrival;
}

}  // namespace

FaceServerCoordinator::FaceServerCoordinator(
    std::vector<Sys*> systems,
    std::string llm_server_configuration)
    : systems_(std::move(systems)),
      llm_server_configuration_(std::move(llm_server_configuration)) {}

void FaceServerCoordinator::start() {
    if (systems_.empty() || systems_[0] == nullptr) {
        throw std::runtime_error("FACE coordinator requires at least one Sys");
    }
    // The analytical event queue advances only to strictly future timestamps.
    systems_[0]->register_event(this, EventType::General, nullptr, 1);
}

void FaceServerCoordinator::call(EventType event, CallData* data) {
    (void)event;
    (void)data;
    if (has_run_) {
        return;
    }
    has_run_ = true;
    run();
}

void FaceServerCoordinator::run() {
    auto logger = LoggerFactory::get_logger("llm");
    logger->info("FACE LLM serving mode starting with config {}",
                 llm_server_configuration_);

    const auto config = FaceConfig::load(llm_server_configuration_);
    auto requests = FaceConfig::load_requests(config);
    FaceEvaluator evaluator(config);
    FaceCSE cse(config, evaluator);
    auto cse_result = cse.run();
    FaceOMM omm(config, evaluator);
    FaceDAS das(config, evaluator, cse, omm);

    for (auto& request : requests) {
        das.schedule_request(cse_result, request);
    }

    write_outputs(cse_result, requests);
    report_summary(cse_result, requests);
    notify_finished();
}

void FaceServerCoordinator::write_outputs(
    const FaceCseResult& cse_result,
    const std::vector<FaceRequest>& requests) const {
    const auto config = FaceConfig::load(llm_server_configuration_);
    std::filesystem::create_directories(config.workload.output_dir);

    const auto request_path =
        std::filesystem::path(config.workload.output_dir) / "face_requests.csv";
    std::ofstream request_out(request_path);
    request_out << "request_id,arrival_time_ns,input_tokens,output_tokens,"
                << "prefill_instance,decode_instance,prefill_start_ns,"
                << "prefill_end_ns,decode_start_ns,first_token_end_ns,"
                << "decode_end_ns,e2e_latency_ns,ttft_ns,tpot_ns,remote_kv,"
                << "kv_offloaded,prefill_tile,decode_tile\n";
    for (const auto& request : requests) {
        const auto e2e = request.decode_end_ns - request.arrival_time_ns;
        const auto ttft = request.first_token_end_ns - request.arrival_time_ns;
        const auto tpot = request.output_tokens > 1
                              ? (request.decode_end_ns -
                                 request.first_token_end_ns) /
                                    static_cast<uint64_t>(request.output_tokens - 1)
                              : 0;
        request_out << request.id << ',' << request.arrival_time_ns << ','
                    << request.input_tokens << ',' << request.output_tokens
                    << ',' << request.prefill_instance << ','
                    << request.decode_instance << ',' << request.prefill_start_ns
                    << ',' << request.prefill_end_ns << ','
                    << request.decode_start_ns << ',' << request.first_token_end_ns
                    << ',' << request.decode_end_ns << ',' << e2e << ','
                    << ttft << ',' << tpot << ',' << request.remote_kv << ','
                    << request.kv_offloaded << ','
                    << request.selected_prefill_tile << ','
                    << request.selected_decode_tile << '\n';
    }

    const auto instance_path =
        std::filesystem::path(config.workload.output_dir) /
        "face_instances.csv";
    std::ofstream instance_out(instance_path);
    instance_out << "instance_id,row,col,shape_rows,shape_cols,kv_capacity_bytes,"
                 << "kv_used_bytes,prefill_busy_ns,decode_busy_ns,overlap_ns,"
                 << "overlap_ratio,dram_traffic_bytes,d2d_traffic_bytes,"
                 << "rejected_or_delayed_attempts\n";
    for (const auto& instance : cse_result.instances) {
        const auto busy = instance.prefill_busy_ns + instance.decode_busy_ns;
        const auto overlap_ratio = busy == 0
                                       ? 0.0
                                       : static_cast<double>(instance.overlap_ns) /
                                             static_cast<double>(busy);
        instance_out << instance.id << ',' << instance.row << ','
                     << instance.col << ',' << instance.shape.rows << ','
                     << instance.shape.cols << ','
                     << instance.kv_capacity_bytes << ','
                     << instance.kv_used_bytes << ','
                     << instance.prefill_busy_ns << ','
                     << instance.decode_busy_ns << ',' << instance.overlap_ns
                     << ',' << overlap_ratio << ','
                     << instance.dram_traffic_bytes << ','
                     << instance.d2d_traffic_bytes << ','
                     << instance.rejected_or_delayed_attempts << '\n';
    }
}

void FaceServerCoordinator::report_summary(
    const FaceCseResult& cse_result,
    const std::vector<FaceRequest>& requests) const {
    const auto config = FaceConfig::load(llm_server_configuration_);
    auto logger = LoggerFactory::get_logger("llm");
    std::vector<double> e2e_latencies;
    std::vector<double> ttft_latencies;
    std::vector<double> tpot_latencies;
    for (const auto& request : requests) {
        e2e_latencies.push_back(
            static_cast<double>(request.decode_end_ns - request.arrival_time_ns));
        ttft_latencies.push_back(static_cast<double>(
            request.first_token_end_ns - request.arrival_time_ns));
        if (request.output_tokens > 1) {
            tpot_latencies.push_back(static_cast<double>(
                (request.decode_end_ns - request.first_token_end_ns) /
                static_cast<uint64_t>(request.output_tokens - 1)));
        }
    }

    const auto sum_e2e =
        std::accumulate(e2e_latencies.begin(), e2e_latencies.end(), 0.0);
    const auto sum_ttft =
        std::accumulate(ttft_latencies.begin(), ttft_latencies.end(), 0.0);
    const auto sum_tpot =
        std::accumulate(tpot_latencies.begin(), tpot_latencies.end(), 0.0);
    const auto start = min_arrival(requests);
    const auto end = max_decode_end(requests);
    const auto duration_s = end > start
                                ? static_cast<double>(end - start) / 1e9
                                : 0.0;
    const auto throughput = duration_s > 0.0
                                ? static_cast<double>(requests.size()) / duration_s
                                : 0.0;

    uint64_t total_overlap = 0;
    uint64_t total_prefill = 0;
    uint64_t total_decode = 0;
    uint64_t total_d2d = 0;
    for (const auto& instance : cse_result.instances) {
        total_overlap += instance.overlap_ns;
        total_prefill += instance.prefill_busy_ns;
        total_decode += instance.decode_busy_ns;
        total_d2d += instance.d2d_traffic_bytes;
    }

    logger->info("FACE scheduler mode: {}", to_string(config.scheduler));
    logger->info("FACE selected instance shape: {}x{}, instances={}, LUT entries={}",
                 cse_result.selected_shape.rows,
                 cse_result.selected_shape.cols, cse_result.instances.size(),
                 cse_result.lut.size());
    logger->info("FACE completed requests: {}, throughput: {:.3f} req/s",
                 requests.size(), throughput);
    logger->info("FACE E2E latency ns avg={:.3f} p50={:.3f} p95={:.3f}",
                 e2e_latencies.empty() ? 0.0 : sum_e2e / e2e_latencies.size(),
                 percentile(e2e_latencies, 0.50),
                 percentile(e2e_latencies, 0.95));
    logger->info("FACE TTFT ns avg={:.3f}, TPOT ns avg={:.3f}",
                 ttft_latencies.empty() ? 0.0
                                        : sum_ttft / ttft_latencies.size(),
                 tpot_latencies.empty() ? 0.0
                                        : sum_tpot / tpot_latencies.size());
    logger->info("FACE busy ns prefill={}, decode={}, overlap={}, D2D bytes={}",
                 total_prefill, total_decode, total_overlap, total_d2d);
    logger->info("FACE CSV outputs written under {}", config.workload.output_dir);
}

void FaceServerCoordinator::notify_finished() const {
    for (auto* system : systems_) {
        if (system != nullptr && system->comm_NI != nullptr) {
            system->comm_NI->sim_notify_finished();
        }
    }
}

}  // namespace AstraSim
