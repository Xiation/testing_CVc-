cmake_minimum_required(VERSION 3.5) # Or newer
project(test1)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(my_cv_program cv.cpp)
target_link_libraries(my_cv_program ${OpenCV_LIBS})
