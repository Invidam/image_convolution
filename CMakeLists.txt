cmake_minimum_required(VERSION 3.26)
project(image_convolution)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(image_convolution main.cpp)

target_link_libraries(image_convolution ${OpenCV_LIBS})
