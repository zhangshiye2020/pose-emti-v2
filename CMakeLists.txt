cmake_minimum_required(VERSION 3.14)

# vcpkg相关
set(ENV{VCPKG_ROOT} "C:/vcpkg")
if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG NOT FOUND")
endif()
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake CACHE STRING "")

# 架构
set(CMAKE_GENERATOR_PLATFORM x64)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ENV{VCPKG_DEFAULT_TRIPLET} "x64-windows")
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(ENV{VCPKG_DEFAULT_TRIPLET} "x64-linux")
endif()

#set(CMAKE_BUILD_TYPE Release CACHE STRING "")

if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
endif()

project(pose-emti-v2)
set(SOURCE_FILES main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pose-emti-v2 ${SOURCE_FILES})
target_link_libraries(pose-emti-v2 PRIVATE ${OpenCV_LIBS})
