/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_SIM_LLM_FACE_EVALUATOR_HH__
#define __ASTRA_SIM_LLM_FACE_EVALUATOR_HH__

#include "astra-sim/llm/FaceConfig.hh"

#include <cstdint>

namespace AstraSim {

struct FaceInstanceShape {
    int rows = 1;
    int cols = 1;

    int die_count() const {
        return rows * cols;
    }

    bool operator==(const FaceInstanceShape& other) const {
        return rows == other.rows && cols == other.cols;
    }
};

struct FaceEvalResult {
    uint64_t latency_ns = 1;
    double utilization = 0.0;
    uint64_t dram_bytes = 0;
    uint64_t noc_bytes = 0;
    uint64_t d2d_bytes = 0;
};

class FaceEvaluator {
  public:
    explicit FaceEvaluator(const FaceConfig& config);

    FaceEvalResult evaluate_attention(const FaceInstanceShape& shape,
                                      int prefill_tokens,
                                      int decode_batch,
                                      int decode_tokens,
                                      int prefill_tile,
                                      int decode_tile) const;

    uint64_t evaluate_prefill_chunk(const FaceInstanceShape& shape,
                                    int prefill_tokens,
                                    int tile) const;

    uint64_t evaluate_decode_step(const FaceInstanceShape& shape,
                                  int decode_batch,
                                  int context_tokens,
                                  int tile) const;

    uint64_t kv_bytes_for_tokens(int tokens) const;
    double dram_bandwidth_bytes_per_ns(const FaceInstanceShape& shape) const;
    double d2d_bandwidth_bytes_per_ns() const;

  private:
    double total_compute_gflops(const FaceInstanceShape& shape) const;
    double noc_bandwidth_bytes_per_ns(const FaceInstanceShape& shape) const;
    uint64_t bidirectional_ring_latency_ns(const FaceInstanceShape& shape,
                                           uint64_t bytes) const;

    const FaceConfig& config_;
};

}  // namespace AstraSim

#endif /* __ASTRA_SIM_LLM_FACE_EVALUATOR_HH__ */
