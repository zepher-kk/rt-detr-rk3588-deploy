cmake_minimum_required(VERSION 3.10)
project(rtdetr_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -Wall -Wno-unused-parameter -Wno-unused-variable)
add_compile_definitions(RTDETR_INPUT_SIZE=640)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /usr/include/rga
    /usr/include/libdrm
)

add_executable(rtdetr_pipeline
    src/drm_alloc.cc
    src/rga_utils.cc
    src/rknn_detector.cc
    src/pipeline.cc
    src/main.cc
)

# Use the system runtime (2.5.0). The bundled 2.3.2 in include/ has no GPU GridSample.
set(RKNN_RT_LIB "/usr/lib/librknnrt.so" CACHE FILEPATH "RKNN runtime library")

target_link_libraries(rtdetr_pipeline
    ${RKNN_RT_LIB}
    /usr/lib/aarch64-linux-gnu/librga.so
    drm
    pthread
    dl
    m
    ${OpenCV_LIBS}
)
