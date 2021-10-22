cmake_minimum_required(VERSION 3.14)
project(pose-emti-v2)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

add_executable(pose-emti-v2 main.cpp circledec.h circledec.cpp)
target_link_libraries(pose-emti-v2 ${OpenCV_LIBS})