#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/parallel_sort.hpp>
#include <torch/torch.h>
#include <iostream>

// N is batch size; D_in is input dimension
// H is hidden dimension; D_out is output dimension
const int64_t N = 64;
const int64_t D_in = 1000;
const int64_t H = 100;
const int64_t D_out = 10;

struct TwoLayerNet : torch::nn::Module {
  TwoLayerNet() : linear1(D_in, H), linear2(H, D_out) {
    register_module("linear1", linear1);
    register_module("linear2", linear2);
  }
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(linear1->forward(x));
    x = linear2->forward(x);
    return x;
  }
  torch::nn::Linear linear1;
  torch::nn::Linear linear2;
};

int main() {
  torch::manual_seed(1);

  torch::Tensor x = torch::rand({N, D_in});
  torch::Tensor y = torch::rand({N, D_out});
  // change this to torch::kCUDA if GPU is available
  torch::Device device(torch::kCPU);

  TwoLayerNet model;
  model.to(device);

  float_t learning_rate = 1e-4;
  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(learning_rate));

  hpx::parallel::v2::for_loop_n(hpx::parallel::execution::par,1,100,[&](size_t epoch)
  {
    optimizer.zero_grad();
    auto y_pred = model.forward(x);
    auto loss = torch::mse_loss(y_pred, y.detach());
    if (epoch%1 == 0)
      hpx::cout << "epoch = " << epoch << " " << "loss = " << loss << "\n";
    loss.backward();
    optimizer.step();
  });
}
