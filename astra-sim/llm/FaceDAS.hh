/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_SIM_LLM_FACE_DAS_HH__
#define __ASTRA_SIM_LLM_FACE_DAS_HH__

#include "astra-sim/llm/FaceCSE.hh"
#include "astra-sim/llm/FaceConfig.hh"
#include "astra-sim/llm/FaceEvaluator.hh"
#include "astra-sim/llm/FaceOMM.hh"

#include <cstdint>
#include <optional>

namespace AstraSim {

class FaceDAS {
  public:
    FaceDAS(const FaceConfig& config,
            const FaceEvaluator& evaluator,
            const FaceCSE& cse,
            FaceOMM& omm);

    void schedule_request(FaceCseResult& cse_result, FaceRequest& request);

  private:
    int choose_prefill_instance(const FaceCseResult& cse_result,
                                const FaceRequest& request) const;
    int choose_decode_instance(const FaceCseResult& cse_result,
                               const FaceRequest& request,
                               int decode_batch_size,
                               const FaceLutEntry& current_entry,
                               std::optional<FaceLutEntry>* selected_entry) const;

    FaceLutEntry lookup_or_evaluate(const FaceCseResult& cse_result,
                                    const FaceInstanceShape& shape,
                                    int decode_batch_size,
                                    int decode_token_count) const;

    void record_overlap(FaceInstance& instance,
                        uint64_t prefill_start,
                        uint64_t prefill_end,
                        uint64_t decode_start,
                        uint64_t decode_end) const;

    const FaceConfig& config_;
    const FaceEvaluator& evaluator_;
    const FaceCSE& cse_;
    FaceOMM& omm_;
};

}  // namespace AstraSim

#endif /* __ASTRA_SIM_LLM_FACE_DAS_HH__ */
