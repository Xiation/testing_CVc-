cmake_minimum_required(VERSION 3.19)
project(ball_detection)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add executable
add_executable(ball_det ball.cpp)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(ball_det PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(ball_det ${OpenCV_LIBS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)