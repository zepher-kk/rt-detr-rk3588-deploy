cmake_minimum_required(VERSION 3.10)
project(rtdetr_rk3588_rga_dma CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# 编译选项
# ============================================================================
add_compile_options(
    -O3
    -g
    -Wall
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-deprecated-declarations
    -fopenmp
)

# ARM NEON 选项（仅对 32 位 ARM 启用，aarch64 默认支持 NEON，无需额外选项）
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_options(-mfloat-abi=hard -mfpu=neon-fp-armv8)
endif()

# ============================================================================
# RK3588 工具链路径（按需修改）—— 若未设置环境变量则自动跳过交叉编译
# ============================================================================
set(TOOLCHAIN_DIR "$ENV{RK3588_TOOLCHAIN}" CACHE PATH "RK3588 toolchain root")
if(NOT TOOLCHAIN_DIR STREQUAL "")
    set(CMAKE_C_COMPILER   ${TOOLCHAIN_DIR}/bin/aarch64-rockchip-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-rockchip-linux-gnu-g++)
    set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR})
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()

# ============================================================================
# SDK 路径 —— 已根据实际开发板路径固化为以下默认值
# ============================================================================
set(RKNN_RT_LIB_DIR "${CMAKE_SOURCE_DIR}/include" CACHE PATH "RKNN runtime lib dir (contains librknnrt.so and rknn_api.h)")
set(RGA_INCLUDE_DIR "/usr/include/rga" CACHE PATH "RGA include dir")
set(RGA_LIB_DIR "/usr/lib/aarch64-linux-gnu" CACHE PATH "RGA lib dir")

# ============================================================================
# OpenCV
# ============================================================================
find_package(OpenCV REQUIRED)

# ============================================================================
# GStreamer（任务 4：RK MPP 硬解/硬编）
# ============================================================================
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
link_directories(${GST_LIBRARY_DIRS})

# ============================================================================
# 头文件搜索路径
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${RGA_INCLUDE_DIR}
    ${RKNN_RT_LIB_DIR}
    ${GST_INCLUDE_DIRS}
    /usr/include/libdrm
)

# ============================================================================
# 源文件
# ============================================================================
set(SOURCES
    src/drm_alloc.cc
    src/rga_utils.cc
    src/v4l2_capture.cc
    src/rknn_detector.cc
    src/postprocess.cc
    src/npu_pipeline.cc
    src/gst_io.cc
    src/main.cc
)

# ============================================================================
# 主可执行文件
# ============================================================================
add_executable(rtdetr_pipeline ${SOURCES})

# ============================================================================
# 链接库
# ============================================================================
target_link_libraries(rtdetr_pipeline
    ${RKNN_RT_LIB_DIR}/librknnrt.so
    ${RGA_LIB_DIR}/librga.so
    drm
    v4l2
    v4lconvert
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    gomp
    pthread
    dl
    m
)

# ============================================================================
# 单元测试可执行文件
# ============================================================================
add_executable(test_unit
    tests/test_unit.cc
    src/drm_alloc.cc
    src/rga_utils.cc
    src/v4l2_capture.cc
    src/rknn_detector.cc
    src/postprocess.cc
    src/npu_pipeline.cc
    src/gst_io.cc
)

target_link_libraries(test_unit
    ${RKNN_RT_LIB_DIR}/librknnrt.so
    ${RGA_LIB_DIR}/librga.so
    drm
    v4l2
    v4lconvert
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    gomp
    pthread
    dl
    m
)

# ============================================================================
# 安装
# ============================================================================
install(TARGETS rtdetr_pipeline DESTINATION bin)
