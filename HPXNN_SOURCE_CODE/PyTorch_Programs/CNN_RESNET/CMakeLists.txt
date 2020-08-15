cmake_minimum_required(VERSION 3.0)
project(predict_demo CXX)

find_package(OpenCV REQUIRED)
find_package(HPX REQUIRED)
find_package(Torch REQUIRED PATHS /Users/amriteshamrit/pytorch)

include_directories( ${OpenCV_INCLUDE_DIRS} )
#add_executable(classifier prediction.cpp)
add_hpx_executable(predict_demo
        ESSENTIAL
        SOURCES prediction.cpp
        DEPENDENCIES ${Torch_LIBS}
        COMPONENT_DEPENDENCIES iostreams)
target_link_libraries(predict_demo PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# set C++14 to compile PyTorch
set_property(TARGET predict_demo PROPERTY CXX_STANDARD 14)
