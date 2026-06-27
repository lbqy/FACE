/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_SIM_LLM_FACE_OMM_HH__
#define __ASTRA_SIM_LLM_FACE_OMM_HH__

#include "astra-sim/llm/FaceCSE.hh"
#include "astra-sim/llm/FaceEvaluator.hh"

#include <cstdint>
#include <map>
#include <vector>

namespace AstraSim {

struct FaceKvRecord {
    int request_id = -1;
    int home_instance = -1;
    int current_instance = -1;
    int tokens = 0;
    uint64_t bytes = 0;
    bool offloaded = false;
};

class FaceOMM {
  public:
    FaceOMM(const FaceConfig& config, const FaceEvaluator& evaluator);

    void place_kv(FaceCseResult& cse,
                  int request_id,
                  int home_instance,
                  int tokens);

    std::vector<int> schedulable_instances(const FaceCseResult& cse,
                                           int request_id,
                                           int home_instance) const;

    bool reserve_for_decode(FaceCseResult& cse,
                            int request_id,
                            int target_instance,
                            bool* remote_kv,
                            bool* offloaded);

    uint64_t remote_access_bytes(int request_id) const;
    const std::map<int, FaceKvRecord>& kv_records() const;

  private:
    int manhattan_distance(const FaceInstance& a, const FaceInstance& b) const;
    bool has_capacity(const FaceInstance& instance,
                      const FaceKvRecord& record) const;

    const FaceConfig& config_;
    const FaceEvaluator& evaluator_;
    std::map<int, FaceKvRecord> kv_records_;
};

}  // namespace AstraSim

#endif /* __ASTRA_SIM_LLM_FACE_OMM_HH__ */
