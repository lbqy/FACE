/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_SIM_LLM_FACE_CSE_HH__
#define __ASTRA_SIM_LLM_FACE_CSE_HH__

#include "astra-sim/llm/FaceConfig.hh"
#include "astra-sim/llm/FaceEvaluator.hh"

#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace AstraSim {

struct FaceLutKey {
    int shape_rows = 1;
    int shape_cols = 1;
    int prefill_chunk_tokens = 0;
    int decode_batch_size = 0;
    int decode_token_count = 0;

    bool operator<(const FaceLutKey& other) const {
        return std::tie(shape_rows, shape_cols, prefill_chunk_tokens,
                        decode_batch_size, decode_token_count) <
               std::tie(other.shape_rows, other.shape_cols,
                        other.prefill_chunk_tokens, other.decode_batch_size,
                        other.decode_token_count);
    }
};

struct FaceLutEntry {
    FaceLutKey key;
    int prefill_tile = 0;
    int decode_tile = 0;
    uint64_t attention_latency_ns = 1;
    double utilization = 0.0;
};

struct FaceInstance {
    int id = -1;
    FaceInstanceShape shape;
    int row = 0;
    int col = 0;
    uint64_t kv_capacity_bytes = 0;
    uint64_t kv_used_bytes = 0;

    uint64_t prefill_available_ns = 0;
    uint64_t decode_available_ns = 0;
    uint64_t serial_available_ns = 0;

    uint64_t prefill_busy_ns = 0;
    uint64_t decode_busy_ns = 0;
    uint64_t overlap_ns = 0;
    uint64_t dram_traffic_bytes = 0;
    uint64_t d2d_traffic_bytes = 0;
    int rejected_or_delayed_attempts = 0;
};

struct FaceCseResult {
    std::vector<FaceInstanceShape> feasible_shapes;
    std::vector<FaceInstance> instances;
    FaceInstanceShape selected_shape;
    std::map<FaceLutKey, FaceLutEntry> lut;
};

class FaceCSE {
  public:
    FaceCSE(const FaceConfig& config, const FaceEvaluator& evaluator);

    FaceCseResult run() const;

    std::optional<FaceLutEntry> lookup(
        const FaceCseResult& result,
        const FaceInstanceShape& shape,
        int prefill_chunk_tokens,
        int decode_batch_size,
        int decode_token_count) const;

  private:
    std::vector<FaceInstanceShape> enumerate_feasible_shapes() const;
    std::vector<FaceInstance> build_layout(const FaceInstanceShape& shape) const;
    std::map<FaceLutKey, FaceLutEntry> build_lut(
        const std::vector<FaceInstanceShape>& shapes) const;
    FaceInstanceShape select_shape(
        const std::vector<FaceInstanceShape>& shapes,
        const std::map<FaceLutKey, FaceLutEntry>& lut) const;

    const FaceConfig& config_;
    const FaceEvaluator& evaluator_;
};

}  // namespace AstraSim

#endif /* __ASTRA_SIM_LLM_FACE_CSE_HH__ */
