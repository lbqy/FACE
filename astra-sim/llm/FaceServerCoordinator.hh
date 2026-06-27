/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_SIM_LLM_FACE_SERVER_COORDINATOR_HH__
#define __ASTRA_SIM_LLM_FACE_SERVER_COORDINATOR_HH__

#include "astra-sim/llm/FaceCSE.hh"
#include "astra-sim/llm/FaceConfig.hh"
#include "astra-sim/system/Callable.hh"

#include <memory>
#include <string>
#include <vector>

namespace AstraSim {

class Sys;

class FaceServerCoordinator : public Callable {
  public:
    FaceServerCoordinator(std::vector<Sys*> systems,
                          std::string llm_server_configuration);

    void start();
    void call(EventType event, CallData* data) override;

  private:
    void run();
    void write_outputs(const FaceCseResult& cse_result,
                       const std::vector<FaceRequest>& requests) const;
    void report_summary(const FaceCseResult& cse_result,
                        const std::vector<FaceRequest>& requests) const;
    void notify_finished() const;

    std::vector<Sys*> systems_;
    std::string llm_server_configuration_;
    bool has_run_ = false;
};

}  // namespace AstraSim

#endif /* __ASTRA_SIM_LLM_FACE_SERVER_COORDINATOR_HH__ */
