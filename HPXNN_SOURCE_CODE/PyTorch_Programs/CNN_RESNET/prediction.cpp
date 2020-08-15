#include <torch/script.h>

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/parallel_sort.hpp>



// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 3

int LoadImage(std::string file_name, cv::Mat &image) {
    image = cv::imread(file_name);  // CV_8UC3
    if (image.empty() || !image.data) {
        return 0;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    hpx::cout << "== image size: " << image.size() << " ==" << hpx::endl;

    // scale image to fit
    cv::Size scale(kIMAGE_SIZE, kIMAGE_SIZE);
    cv::resize(image, image, scale);
    hpx::cout << "== simply resize: " << image.size() << " ==" << hpx::endl;

    // convert [unsigned int] to [float]
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    return 1;
}

int LoadImageNetLabel(std::string file_name,
                       std::vector<std::string> &labels) {
    std::ifstream ifs(file_name);
    if (!ifs) {
        return 0;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        labels.push_back(line);
    }
    return 1;
}

int main(int argc, const char *argv[]) {
 
time_t start, end; 

time(&start); 

    if (argc != 3) {
        std::cerr << "Usage: classifier <path-to-exported-script-module> "
                     "<path-to-lable-file>"
                  << hpx::endl;
        return -1;
    }

std::vector<std::string> labels;
std::string arg2=argv[2];
hpx::future<int> f1 = hpx::async([&]{return LoadImageNetLabel(arg2, labels);});
        int a1 = f1.get();

    torch::jit::script::Module module = torch::jit::load(argv[1]);
    hpx::cout << "== Switch to GPU mode" << hpx::endl;
    // to GPU
   // module.to(at::kCUDA);

    hpx::cout << "== ResNet50 loaded!\n";
    

	



    if (a1) {
        hpx::cout << "== Label loaded! Let's try it\n";
    } else {
        std::cerr << "Please check your label file path." << hpx::endl;
        return -1;
    }

    std::string file_name = "";
    cv::Mat image;


//std::string arr[2] = {"/home/privacy2/hpx_intern/sharkdog/normal/PyTorch-CPP/pic/dog.jpg","/home/privacy2/hpx_intern/sharkdog/normal/PyTorch-CPP/pic/shark.jpg"};
std::string arr[2] = {"/Volumes/ANDROID/Pytorch/PyTorch-CPP/pic/dog.jpg","/Volumes/ANDROID/Pytorch/PyTorch-CPP/pic/shark.jpg"};

    /*while (true) {
        hpx::cout << "== Input image path: [enter Q to exit]" << hpx::endl;
        std::cin >> file_name;
        if (file_name == "Q") {
            break;
                              }
*/

for(int i=0;i<2;i++)
{

hpx::future<int> f2 = hpx::async([&]{return LoadImage(arr[i], image);});	
	int a2 = f2.get();
        if (a2) {
            auto input_tensor = torch::from_blob(
                    image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
            input_tensor = input_tensor.permute({0, 3, 1, 2});
            input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
            input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
            input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

            // to GPU
          //  input_tensor = input_tensor.to(at::kCUDA);

            torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();

            auto results = out_tensor.sort(-1, true);
            auto softmaxs = std::get<0>(results)[0].softmax(0);
            auto indexs = std::get<1>(results)[0];

            for (int i = 0; i < kTOP_K; ++i) {
                auto idx = indexs[i].item<int>();
                hpx::cout << "    ============= Top-" << i + 1
                          << " =============" << hpx::endl;
                hpx::cout << "    Label:  " << labels[idx] << hpx::endl;
                hpx::cout << "    With Probability:  "
                          << softmaxs[i].item<float>() * 100.0f << "%" << hpx::endl;
            }

        } else {
            hpx::cout << "Can't load the image, please check your path." << hpx::endl;
        }
    }

time(&end);
double time_taken = double(end - start); 
    hpx::cout << "Time taken by program is : " << std::fixed 
         << time_taken << std::setprecision(5); 
    hpx::cout << " sec " << hpx::endl; 


    return 0;
}