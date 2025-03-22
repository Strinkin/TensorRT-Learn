cmake_minimum_required(VERSION 3.10)
project(mnist_strinkin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/local/TensorRT-8.6.1.6/include)
link_directories(/usr/local/TensorRT-8.6.1.6/lib)

add_executable(mnist_strinkin src/mnist.cpp)
target_include_directories(mnist_strinkin PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnist_strinkin PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mnist_strinkin ${OpenCV_LIBS})
target_link_libraries(mnist_strinkin cudart nvinfer)
