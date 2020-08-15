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

// Hyper parameters
const int input_size = 784;
const int hidden_size = 500;
const int num_classes = 10;
const int num_epochs = 5;
const int batch_size = 100;
const double learning_rate = 0.001;

struct NeuralNet: torch::nn::Module {
  // Declare all the layers of nerual network
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};

  // Construct all the layers
  NeuralNet() {
    fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_size, num_classes));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return torch::log_softmax(x, 1);
  }
};

int main() {
  hpx::cout << "FeedForward Neural Network" << hpx::endl;

  // Device
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    hpx::cout << "CUDA available. Training on GPU." << hpx::endl;
    device_type = torch::kCUDA;
  } else {
    hpx::cout << "Training on CPU." << hpx::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  const std::string MNIST_data_path = "../data/mnist/";

  // MNIST Dataset (images and labels)
  auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());

  // Data loader (input pipeline)
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), batch_size);
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_dataset), batch_size);

  // Neural Network model
  auto model = std::make_shared<NeuralNet>();
  model->to(device);

  // Loss and optimizer
  auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(learning_rate));

  // Train the model
  hpx::parallel::v2::for_loop_n(hpx::parallel::execution::par,0,num_epochs,
	[&](int epoch)
    {
    int i = 0;
    for (auto& batch : *train_loader) {
      auto data = batch.data.reshape({batch_size, -1}).to(device);
      auto labels = batch.target.to(device);

      // Forward pass
      auto outputs = model->forward(data);
      auto loss = torch::nll_loss(outputs, labels);

      // Backward and optimize
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      if ((i+1) % 100 == 0) {
        hpx::cout << "Epoch [" << (epoch+1) << "/" << num_epochs << "], Batch: "
          << (i+1) << ", Loss: " << loss.item().toFloat() << hpx::endl;
      }
      ++i;
    }
  });

  // Test the model
  model->eval();
  torch::NoGradGuard no_grad;

  int correct = 0;
  int total = 0;
  for (const auto& batch : *test_loader) {
    auto data = batch.data.reshape({batch_size, -1}).to(device);
    auto labels = batch.target.to(device);
    auto outputs = model->forward(data);
    auto predicted = outputs.argmax(1);
    total += labels.size(0);
    correct += predicted.eq(labels).sum().item<int>();
  }

  hpx::cout << "Accuracy of the model on the 10000 test images: " <<
    static_cast<double>(100 * correct / total) << hpx::endl;
}
