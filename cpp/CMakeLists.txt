cmake_minimum_required(VERSION 3.10)
project(rknn_rtdetr_demo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 寻找 OpenCV 库
find_package(OpenCV REQUIRED)

# 打印一下 OpenCV 的信息，方便排错
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 2. 指定头文件目录 (引入你的 include 文件夹和 OpenCV)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# 3. 收集 src 目录下的所有源文件
file(GLOB_RECURSE SRC_FILES "src/*.cc")

# 4. 定义要生成的可执行文件
add_executable(${PROJECT_NAME} ${SRC_FILES})

# 5. 链接所有需要的库 
# 包括 OpenCV、多线程 pthread，以及你放在 include 目录下的 RKNN 动态库！
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/librknnrt.so
    pthread
)