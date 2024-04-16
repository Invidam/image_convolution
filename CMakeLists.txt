cmake_minimum_required(VERSION 3.25)
project(image_convolution)

# Set runtime output directory to project root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++ -lwinpthread")
set(CMAKE_CXX_STANDARD_LIBRARIES "-static-libgcc -static-libstdc++ -lwsock32 -lws2_32 ${CMAKE_CXX_STANDARD_LIBRARIES}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive")

# List of source files
set(SOURCE_FILES
        src/main.cpp
        src/Transformer.cpp
        src/Timer.cpp
        src/Filter.cpp
)

set(OpenCV_DIR $ENV{OPENCV_4.5.5_HOME})
set(OpenCV_INCLUDE ${OpenCV_DIR}/include)
set(OpenCV_BIN ${OpenCV_DIR}/x64/mingw/bin)

# Find OpenMP and enable it
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    add_compile_definitions(OPENMP_VERSION=${OpenMP_CXX_VERSION})
    message(STATUS "OpenMP v${OpenMP_CXX_VERSION} found")
else()
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
