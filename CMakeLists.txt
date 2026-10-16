cmake_minimum_required(VERSION 3.16)
project(ikun LANGUAGES C CXX)

# Options
option(USE_ILU "Enable Iluvatar BI-V100 specific code paths" ON)
option(BUILD_TESTING "Enable unit tests" ON)

# Core inference engine (if present)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/CMakeLists.txt)
    add_subdirectory(core)
endif()

# Collective runner (always built)
add_subdirectory(src)

# InfiniCCL (if present)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/InfiniCCL/CMakeLists.txt)
    add_subdirectory(third_party/InfiniCCL)
endif()
