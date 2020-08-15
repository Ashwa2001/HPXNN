#include <torch/torch.h>
#include <iostream>
//hpx files
#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/parallel_sort.hpp>


template <typename T>
void pretty_print(const std::string& info, T&& data) {
  hpx::cout << info << hpx::endl;
  hpx::cout << data << hpx::endl << hpx::endl;
}

int main() {
  // Create an eye tensor
  torch::Tensor tensor = torch::eye(3);
  pretty_print("Eye tensor: ", tensor);

  // Tensor view is like reshape in numpy, which changes the dimension representation of the tensor
  // without touching its underlying memory structure.
  tensor = torch::range(1, 9, 1);
  hpx::future<void> f1 = hpx::async(pretty_print,"Tensor range 1x9: ", tensor);
  hpx::future<void> f2 = hpx::async(pretty_print,"Tensor view 3x3: ", tensor.view({3, 3}));
  hpx::future<void> f3 = hpx::async(pretty_print,"Tensor view 3x3 with D0 and D1 transposed: ", tensor.view({3, 3}).transpose(0, 1));
  tensor = torch::range(1, 27, 1);
  hpx::future<void> f4 = hpx::async(pretty_print,"Tensor range 1x27: ", tensor);
  hpx::future<void> f5 = hpx::async(pretty_print,"Tensor view 3x3x3: ", tensor.view({3, 3, 3}));
  hpx::future<void> f6 = hpx::async(pretty_print,"Tensor view 3x3x3 with D0 and D1 transposed: ",
               tensor.view({3, 3, 3}).transpose(0, 1));
  hpx::future<void> f7 = hpx::async(pretty_print,"Tensor view 3x1x9: ", tensor.view({3, 1, -1}));
}