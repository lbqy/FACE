/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/llm/FaceCSE.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace AstraSim {
namespace {

uint64_t bytes_from_gb(double gb) {
    return static_cast<uint64_t>(gb * 1000.0 * 1000.0 * 1000.0);
}

bool ring_closed(const FaceInstanceShape& shape) {
    return shape.die_count() == 1 || shape.rows > 1 || shape.cols > 1;
}

int max_or_default(const std::vector<int>& values, int fallback) {
    if (values.empty()) {
        return fallback;
    }
    return *std::max_element(values.begin(), values.end());
}

}  // namespace

FaceCSE::FaceCSE(const FaceConfig& config, const FaceEvaluator& evaluator)
    : config_(config), evaluator_(evaluator) {}

FaceCseResult FaceCSE::run() const {
    FaceCseResult result;
    result.feasible_shapes = enumerate_feasible_shapes();
    if (result.feasible_shapes.empty()) {
        throw std::runtime_error(
            "FACE CSE found no feasible instance shape; model weights do not "
            "fit into the configured wafer DRAM capacity");
    }
    result.lut = build_lut(result.feasible_shapes);
    result.selected_shape = select_shape(result.feasible_shapes, result.lut);
    result.instances = build_layout(result.selected_shape);
    if (result.instances.empty()) {
        throw std::runtime_error("FACE CSE selected an empty instance layout");
    }
    return result;
}

std::optional<FaceLutEntry> FaceCSE::lookup(
    const FaceCseResult& result,
    const FaceInstanceShape& shape,
    int prefill_chunk_tokens,
    int decode_batch_size,
    int decode_token_count) const {
    std::optional<FaceLutEntry> best;
    double best_distance = std::numeric_limits<double>::max();
    for (const auto& [key, entry] : result.lut) {
        if (key.shape_rows != shape.rows || key.shape_cols != shape.cols ||
            key.prefill_chunk_tokens != prefill_chunk_tokens) {
            continue;
        }
        const auto batch_delta = key.decode_batch_size - decode_batch_size;
        const auto token_delta = key.decode_token_count - decode_token_count;
        const auto distance = std::sqrt(static_cast<double>(
            batch_delta * batch_delta + token_delta * token_delta));
        if (!best.has_value() || distance < best_distance ||
            (distance == best_distance &&
             entry.attention_latency_ns < best->attention_latency_ns)) {
            best = entry;
            best_distance = distance;
        }
    }
    return best;
}

std::vector<FaceInstanceShape> FaceCSE::enumerate_feasible_shapes() const {
    std::vector<FaceInstanceShape> shapes;
    const auto& hw = config_.hardware;
    const auto capacity_per_die =
        bytes_from_gb(hw.dram_capacity_gb_per_hbm) *
        static_cast<uint64_t>(std::max(1, hw.hbm_chiplets_per_die));
    for (int rows = 1; rows <= hw.die_rows; rows++) {
        for (int cols = 1; cols <= hw.die_cols; cols++) {
            FaceInstanceShape shape{rows, cols};
            const auto capacity = capacity_per_die * shape.die_count();
            if (capacity < config_.model.weight_bytes) {
                continue;
            }
            if (!ring_closed(shape)) {
                continue;
            }
            shapes.push_back(shape);
        }
    }
    std::sort(shapes.begin(), shapes.end(), [](const auto& a, const auto& b) {
        if (a.die_count() == b.die_count()) {
            return std::tie(a.rows, a.cols) < std::tie(b.rows, b.cols);
        }
        return a.die_count() < b.die_count();
    });
    return shapes;
}

std::vector<FaceInstance> FaceCSE::build_layout(
    const FaceInstanceShape& shape) const {
    std::vector<FaceInstance> instances;
    const auto& hw = config_.hardware;
    if (shape.rows <= 0 || shape.cols <= 0 || hw.die_rows % shape.rows != 0 ||
        hw.die_cols % shape.cols != 0) {
        return instances;
    }

    const auto capacity_per_die =
        bytes_from_gb(hw.dram_capacity_gb_per_hbm) *
        static_cast<uint64_t>(std::max(1, hw.hbm_chiplets_per_die));
    const auto raw_capacity = capacity_per_die * shape.die_count();
    const auto kv_capacity = raw_capacity > config_.model.weight_bytes
                                 ? raw_capacity - config_.model.weight_bytes
                                 : 0;

    int id = 0;
    for (int row = 0; row < hw.die_rows; row += shape.rows) {
        for (int col = 0; col < hw.die_cols; col += shape.cols) {
            FaceInstance instance;
            instance.id = id++;
            instance.shape = shape;
            instance.row = row;
            instance.col = col;
            instance.kv_capacity_bytes = kv_capacity;
            instances.push_back(instance);
        }
    }
    return instances;
}

std::map<FaceLutKey, FaceLutEntry> FaceCSE::build_lut(
    const std::vector<FaceInstanceShape>& shapes) const {
    std::map<FaceLutKey, FaceLutEntry> lut;
    for (const auto& shape : shapes) {
        for (const auto decode_batch : config_.cse.decode_batch_sizes) {
            for (const auto decode_tokens : config_.cse.decode_token_counts) {
                for (const auto prefill_tile :
                     config_.cse.prefill_tile_candidates) {
                    for (const auto decode_tile :
                         config_.cse.decode_tile_candidates) {
                        const auto eval = evaluator_.evaluate_attention(
                            shape, config_.workload.prefill_chunk_tokens,
                            decode_batch, decode_tokens, prefill_tile,
                            decode_tile);
                        const FaceLutKey key{shape.rows,
                                             shape.cols,
                                             config_.workload
                                                 .prefill_chunk_tokens,
                                             decode_batch,
                                             decode_tokens};
                        FaceLutEntry entry;
                        entry.key = key;
                        entry.prefill_tile = prefill_tile;
                        entry.decode_tile = decode_tile;
                        entry.attention_latency_ns = eval.latency_ns;
                        entry.utilization = eval.utilization;
                        auto it = lut.find(key);
                        if (it == lut.end() ||
                            entry.attention_latency_ns <
                                it->second.attention_latency_ns) {
                            lut[key] = entry;
                        }
                    }
                }
            }
        }
    }
    return lut;
}

FaceInstanceShape FaceCSE::select_shape(
    const std::vector<FaceInstanceShape>& shapes,
    const std::map<FaceLutKey, FaceLutEntry>& lut) const {
    const auto batch = max_or_default(config_.cse.decode_batch_sizes, 1);
    const auto tokens = max_or_default(config_.cse.decode_token_counts,
                                       config_.workload.prefill_chunk_tokens);
    double best_score = -1.0;
    FaceInstanceShape best_shape = shapes.front();
    for (const auto& shape : shapes) {
        const auto layout = build_layout(shape);
        if (layout.empty()) {
            continue;
        }
        const FaceLutKey key{shape.rows, shape.cols,
                             config_.workload.prefill_chunk_tokens, batch,
                             tokens};
        auto it = lut.find(key);
        if (it == lut.end()) {
            continue;
        }
        const auto per_die_latency =
            static_cast<double>(it->second.attention_latency_ns) /
            std::max(1, shape.die_count());
        const auto score = static_cast<double>(layout.size()) /
                           std::max(1.0, per_die_latency);
        if (score > best_score) {
            best_score = score;
            best_shape = shape;
        }
    }
    return best_shape;
}

}  // namespace AstraSim
