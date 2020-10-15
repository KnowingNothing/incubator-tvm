#include "tvm/te/operation.h"
#include "tvm/te/schedule.h"
#include "tvm/te/tensor.h"
#include "topi/nn.h"
#include "tvm/auto_scheduler/compute_dag.h"
#include "tvm/auto_scheduler/loop_state.h"


using namespace std;


std::pair<tvm::te::Tensor, tvm::Array<tvm::te::Tensor>> get_model() {
  int m = 300, n = 512;
  auto Input = tvm::te::placeholder({n, n, 3});
  auto Filter1 = tvm::te::placeholder({3, 3, 3, 3});
  auto di1 = tvm::te::reduce_axis({0, 3});
  auto dj1 = tvm::te::reduce_axis({0, 3});
  auto dk1 = tvm::te::reduce_axis({0, 3});

  auto Conv1 = tvm::te::compute({n - 2, n - 2, 3}, [=](auto i, auto j, auto k) {
    return tvm::sum(Input(i + di1, j + dj1, dk1) * Filter1(k, di1, dj1, dk1), {di1, dj1, dk1});
  });

  auto Relu1 = topi::leaky_relu(Conv1);

  auto Filter2 = tvm::te::placeholder({3, 3, 3, 3});
  auto di2 = tvm::te::reduce_axis({0, 3});
  auto dj2 = tvm::te::reduce_axis({0, 3});
  auto dk2 = tvm::te::reduce_axis({0, 3});

  auto Conv2 = tvm::te::compute({n - 4, n - 4, 3}, [=](auto i, auto j, auto k) {
    return tvm::sum(Relu1(i + di2, j + dj2, dk2) * Filter2(k, di2, dj2, dk2), {di2, dj2, dk2});
  });

  auto Relu2 = topi::leaky_relu(Conv2);

  auto Affine1 = tvm::te::placeholder({3, 3});
  auto dl1 = tvm::te::reduce_axis({0, 3});

  auto FC1 = tvm::te::compute({m - 4, m - 4, 3}, [=](auto x, auto y, auto z) {
    return tvm::sum(Relu2(x, y, dl1) * Affine1(z, dl1), {dl1});
  });

  auto Relu3 = topi::leaky_relu(FC1);

  auto Affine2 = tvm::te::placeholder({3, 3});
  auto dl2 = tvm::te::reduce_axis({0, 3});

  auto Output = tvm::te::compute({m - 4, m - 4, 3}, [=](auto x, auto y, auto z) {
    return tvm::sum(Relu3(x, y, dl2) * Affine2(z, dl2), {dl2});
  });

  auto buffers = tvm::Array<tvm::te::Tensor>({Input, Filter1, Filter2, Affine1, Affine2, Output});

  return std::move(std::make_pair(Output, buffers));
}

int main() {
  auto tensors = get_model().second;
  auto access_analyzer = tvm::auto_scheduler::AccessAnalyzer(std::move(tensors));
  auto ops = access_analyzer->ops_topo_order;
  auto init_state = tvm::auto_scheduler::State(ops);
  std::cout << init_state << std::endl;
}