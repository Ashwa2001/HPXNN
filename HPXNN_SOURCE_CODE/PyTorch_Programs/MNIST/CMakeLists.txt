cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(mnist CXX)
#set(CMAKE_CXX_STANDARD 14)

# Instruct cmake to find the HPX settings
find_package(HPX REQUIRED)
find_package(Torch REQUIRED PATHS /path/to/pytorch)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

# Build your application using HPX
add_hpx_executable(mnist
        ESSENTIAL
        SOURCES main.cpp
        DEPENDENCIES ${Torch_LIBS}
        COMPONENT_DEPENDENCIES iostreams)

target_compile_features(mnist PRIVATE cxx_range_for)
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
