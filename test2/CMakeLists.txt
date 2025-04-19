cmake_minimum_required(VERSION 3.19)
project(test2)

# set(OPENCV_EXTRA_MODULES_PATH $ENV{OPENCV_EXTRA_MODULES_PATH} CACHE STRING "Path to OpenCV contrib modules")
# mark_as_advanced(OPENCV_EXTRA_MODULES_PATH)

# Find OpenCV, including the bgsegm module
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(my_test2 test2.cpp)

# Link libraries
target_link_libraries(my_test2 ${OpenCV_LIBS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
