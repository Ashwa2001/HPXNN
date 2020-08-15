cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(EXECUTABLE_NAME logistic-regression)

project(log CXX)

# Instruct cmake to find the HPX settings
find_package(HPX REQUIRED)
find_package(Torch REQUIRED PATHS /path/to/pytorch)

# Build your application using HPX
add_hpx_executable(log
        ESSENTIAL
        SOURCES main.cpp
        DEPENDENCIES ${Torch_LIBS}
        COMPONENT_DEPENDENCIES iostreams)

target_link_libraries(log PRIVATE "${TORCH_LIBRARIES}")
