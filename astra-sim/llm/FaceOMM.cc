/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/llm/FaceOMM.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace AstraSim {

FaceOMM::FaceOMM(const FaceConfig& config, const FaceEvaluator& evaluator)
    : config_(config), evaluator_(evaluator) {}

void FaceOMM::place_kv(FaceCseResult& cse,
                       int request_id,
                       int home_instance,
                       int tokens) {
    if (home_instance < 0 ||
        home_instance >= static_cast<int>(cse.instances.size())) {
        throw std::runtime_error("FACE OMM received an invalid home instance");
    }
    FaceKvRecord record;
    record.request_id = request_id;
    record.home_instance = home_instance;
    record.current_instance = home_instance;
    record.tokens = tokens;
    record.bytes = evaluator_.kv_bytes_for_tokens(tokens);
    record.offloaded = false;
    kv_records_[request_id] = record;

    auto& instance = cse.instances[home_instance];
    if (instance.kv_used_bytes + record.bytes <= instance.kv_capacity_bytes) {
        instance.kv_used_bytes += record.bytes;
    } else {
        record.offloaded = true;
        kv_records_[request_id] = record;
        instance.rejected_or_delayed_attempts++;
    }
}

std::vector<int> FaceOMM::schedulable_instances(
    const FaceCseResult& cse,
    int request_id,
    int home_instance) const {
    std::vector<int> candidates;
    if (home_instance < 0 ||
        home_instance >= static_cast<int>(cse.instances.size())) {
        return candidates;
    }

    const auto& home = cse.instances[home_instance];
    const auto d2d = evaluator_.d2d_bandwidth_bytes_per_ns();
    const auto dram = evaluator_.dram_bandwidth_bytes_per_ns(home.shape);
    const auto max_distance = std::max(0, static_cast<int>(std::floor(d2d / dram)));

    for (const auto& instance : cse.instances) {
        if (manhattan_distance(home, instance) <= max_distance) {
            candidates.push_back(instance.id);
        }
    }
    if (candidates.empty()) {
        candidates.push_back(home_instance);
    }
    return candidates;
}

bool FaceOMM::reserve_for_decode(FaceCseResult& cse,
                                 int request_id,
                                 int target_instance,
                                 bool* remote_kv,
                                 bool* offloaded) {
    auto it = kv_records_.find(request_id);
    if (it == kv_records_.end()) {
        throw std::runtime_error("FACE OMM cannot reserve an unknown KV record");
    }
    if (target_instance < 0 ||
        target_instance >= static_cast<int>(cse.instances.size())) {
        return false;
    }

    auto& record = it->second;
    auto& target = cse.instances[target_instance];
    const bool is_remote = target_instance != record.home_instance;
    if (remote_kv != nullptr) {
        *remote_kv = is_remote;
    }

    bool is_offloaded = record.offloaded;
    if (is_remote) {
        target.d2d_traffic_bytes += record.bytes;
        target.dram_traffic_bytes += record.bytes;
    }

    if (is_remote && !has_capacity(target, record)) {
        is_offloaded = true;
        target.rejected_or_delayed_attempts++;
        target.d2d_traffic_bytes += record.bytes;
    } else if (is_remote) {
        target.kv_used_bytes += record.bytes;
        record.current_instance = target_instance;
    }

    record.offloaded = is_offloaded;
    if (offloaded != nullptr) {
        *offloaded = is_offloaded;
    }
    return true;
}

uint64_t FaceOMM::remote_access_bytes(int request_id) const {
    auto it = kv_records_.find(request_id);
    if (it == kv_records_.end()) {
        return 0;
    }
    return it->second.bytes;
}

const std::map<int, FaceKvRecord>& FaceOMM::kv_records() const {
    return kv_records_;
}

int FaceOMM::manhattan_distance(const FaceInstance& a,
                                const FaceInstance& b) const {
    const auto ar = a.row + a.shape.rows / 2;
    const auto ac = a.col + a.shape.cols / 2;
    const auto br = b.row + b.shape.rows / 2;
    const auto bc = b.col + b.shape.cols / 2;
    return std::abs(ar - br) + std::abs(ac - bc);
}

bool FaceOMM::has_capacity(const FaceInstance& instance,
                           const FaceKvRecord& record) const {
    return instance.kv_used_bytes + record.bytes <= instance.kv_capacity_bytes;
}

}  // namespace AstraSim
