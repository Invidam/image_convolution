cmake_minimum_required(VERSION 3.25)
project(image_convolution)

# Set runtime output directory to project root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# List of source files
set(SOURCE_FILES
        main.cpp
        Calculator.cpp
        Timer.cpp
        )

set(OpenCV_DIR $ENV{OPENCV_4.5.5_HOME})
set(OpenCV_INCLUDE ${OpenCV_DIR}/include)
set(OpenCV_BIN ${OpenCV_DIR}/x64/mingw/bin)

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

# Create executable
add_executable(image_convolution ${SOURCE_FILES})

# Define compile feature (C++ standard version)
target_compile_features(image_convolution PRIVATE cxx_std_17)

# Include OpenCV directory
target_include_directories(image_convolution PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link OpenCV and OpenMP libraries
target_link_libraries(image_convolution PRIVATE ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})
