cmake_minimum_required(VERSION 3.14)

set(VCPKG_ROOT "C:/vcpkg")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake CACHE STRING "")
set(ENV{VCPKG_DEFAULT_TRIPLET} "x64-windows")
if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
endif()

#set(CMAKE_GENERATOR "Visual Studio 16 2019" CACHE STRING "")

project(pose-emti-v2)
set(SOURCE_FILES main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#set_target_properties(COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
add_executable(pose-emti-v2 ${SOURCE_FILES})
target_link_libraries(pose-emti-v2 PRIVATE ${OpenCV_LIBS})