cmake_minimum_required(VERSION 3.26)
project(image_convolution)
set(CMAKE_CXX_STANDARD 17)

# Set runtime output directory to project root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# List of source files
set(SOURCE_FILES
        main.cpp
        Calculator.cpp
        Timer.cpp
)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable
add_executable(image_convolution ${SOURCE_FILES})

# Link OpenCV and OpenMP libraries
target_link_libraries(image_convolution PRIVATE ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})
