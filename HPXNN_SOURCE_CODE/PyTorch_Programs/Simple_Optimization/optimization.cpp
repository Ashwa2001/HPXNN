//hpx header files
#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/parallel_sort.hpp>
//--------------------------------
#include <torch/torch.h>
#include <cstdlib>
#include <iostream>

constexpr double kLearningRate = 0.001;
constexpr int kMaxIterations = 1000;

void native_run(double minimal) {
  // Initial x value
  auto x = torch::randn({1, 1}, torch::requires_grad(true));

	hpx::parallel::v2::for_loop_n(hpx::parallel::execution::par, 0, kMaxIterations,[&](size_t t ){
    // Expression/value to be minimized
    auto y = (x - minimal) * (x - minimal);
    //if (y.item<double>() < 1e-3) {
    // break;
    //}
    // Calculate gradient
    y.backward();

    // Step x value without considering gradient
    torch::NoGradGuard no_grad_guard;
    x -= kLearningRate * x.grad();

    // Reset the gradient of variable x
    x.grad().reset();
  });

  hpx::cout << "[native] Actual minimal x value: " << minimal
            << ", calculated optimal x value: " << x.item<double>() << hpx::endl;
}

void optimizer_run(double minimal) {
  // Initial x value
  std::vector<torch::Tensor> x;
  x.push_back(torch::randn({1, 1}, torch::requires_grad(true)));
  auto opt = torch::optim::SGD(x, torch::optim::SGDOptions(kLearningRate));

	hpx::parallel::v2::for_loop_n(hpx::parallel::execution::par, 0, kMaxIterations,[&](size_t t ){
    // Expression/value to be minimized
    auto y = (x[0] - minimal) * (x[0] - minimal);
    //if (y.item<double>() < 1e-3) {
    //  break;
    //}
    // Calculate gradient
    y.backward();

    // Step x value without considering gradient
    opt.step();
    // Reset the gradient of variable x
    opt.zero_grad();
  });

  hpx::cout << "[optimizer] Actual minimal x value: " << minimal
            << ", calculated optimal x value: " << x[0].item<double>() << hpx::endl;
}

// optimize y = (x - 10)^2
int main(int argc, char* argv[]) {
  if (argc < 2) {
    hpx::cout << "Usage: " << argv[0] << " minimal_value\n";
    return 1;
  }
	hpx::future<void> f2091 = hpx::async( native_run, atof(argv[1]) );
	hpx::future<void> f2123 = hpx::async( optimizer_run, atof(argv[1]) );
  return 0;
}
