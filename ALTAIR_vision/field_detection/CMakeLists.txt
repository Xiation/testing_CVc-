cmake_minimum_required(VERSION 3.19)
project(field_detection)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(field_det field.cpp)

# Link libraries
target_link_libraries(field_det ${OpenCV_LIBS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)