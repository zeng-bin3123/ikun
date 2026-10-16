# ikun collective runner — standalone C library + binary
cmake_minimum_required(VERSION 3.16)

add_library(ikun_collective STATIC ikun.c)
target_include_directories(ikun_collective PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_compile_features(ikun_collective PUBLIC c_std_11)

add_executable(ikun_runner main.c)
target_link_libraries(ikun_runner PRIVATE ikun_collective)
