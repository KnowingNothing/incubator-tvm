#include "interpreter.h"


namespace tvm {


namespace tg {

void interpret(te::Schedule &sch, Array<te::Tensor> tensors, TIRGraph subgraph, Target target, MultiScheduleEntity entity) {
  std::cerr << "Enter C++ interpret\n";
  const auto* f = runtime::Registry::Get("tg.autoschedule.interpret");
  std::cerr << f << '\n';
  if(f == nullptr) {
    std::cerr << "Can't get tg.autoschedule.interpret.";
    abort();
  }
  (*f)(sch, tensors, subgraph, target, entity);
  std::cerr << "End C++ interpret\n";
}

}  // namespace tg


}  // namespace tvm