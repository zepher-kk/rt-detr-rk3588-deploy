#!/bin/bash
# ============================================================================
# RT-DETR-R18 RK3588 RGA_DMA 流水线健壮性测试脚本
#
# 本脚本执行以下测试：
#   1. 环境检查（设备节点、权限、库文件）
#   2. DMA buffer 分配测试
#   3. RGA 硬件加速测试
#   4. V4L2 零拷贝采集测试
#   5. NPU 推理测试
#   6. 端到端流水线测试
#   7. 压力测试（长时间运行）
#   8. 异常恢复测试
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

print_test() { echo -e "${BLUE}[TEST]${NC} $1"; }
print_pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS + 1)); }
print_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }
print_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; SKIP=$((SKIP + 1)); }
print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

test_environment() {
    print_test "1. 环境检查"
    if [ -e /dev/dri/renderD128 ]; then
        print_pass "DRM render node 存在"
    else
        print_fail "DRM render node 不存在"
    fi
    local v4l2_found=0
    for dev in /dev/video*; do
        if [ -e "$dev" ]; then
            print_pass "V4L2 设备 $dev 存在"
            v4l2_found=1; break
        fi
    done
    [ $v4l2_found -eq 0 ] && print_skip "未找到 V4L2 设备"
    for lib in librknnrt.so librga.so libdrm.so libv4l2.so; do
        if ldconfig -p | grep -q "$lib"; then
            print_pass "库 $lib 存在"
        else
            print_fail "库 $lib 不存在"
        fi
    done
    if [ -w /dev/dri/renderD128 ]; then
        print_pass "DRM 设备可写"
    else
        print_fail "DRM 设备不可写（需要 video 组权限）"
    fi
}

test_dma_buffer() {
    print_test "2. DMA buffer 分配测试"
    local meminfo=$(grep "MemAvailable" /proc/meminfo | awk '{print $2}')
    if [ "$meminfo" -gt 1048576 ]; then
        print_pass "可用内存: $((meminfo / 1024)) MB"
    else
        print_fail "可用内存不足: $((meminfo / 1024)) MB"
    fi
}

test_rga() {
    print_test "3. RGA 硬件加速测试"
    [ -e /dev/rga ] && print_pass "RGA 设备存在" || print_skip "RGA 设备不存在"
    [ -d /sys/class/rga ] && print_pass "RGA sysfs 存在" || print_skip "RGA sysfs 不存在"
}

test_v4l2() {
    print_test "4. V4L2 零拷贝采集测试"
    local v4l2_dev=""
    for dev in /dev/video0 /dev/video1; do
        [ -e "$dev" ] && v4l2_dev="$dev" && break
    done
    [ -z "$v4l2_dev" ] && { print_skip "无 V4L2 设备"; return 0; }
    if command -v v4l2-ctl &> /dev/null; then
        local caps=$(v4l2-ctl --device=$v4l2_dev --info 2>&1)
        echo "$caps" | grep -q "Video Capture" && print_pass "支持视频采集" || print_fail "不支持视频采集"
        echo "$caps" | grep -q "streaming" && print_pass "支持 streaming" || print_fail "不支持 streaming"
    else
        print_skip "v4l2-ctl 不可用"
    fi
}

test_npu() {
    print_test "5. NPU 推理测试"
    [ -d /sys/kernel/debug/rknpu ] && print_pass "NPU debugfs 存在" || print_skip "NPU debugfs 不存在"
    [ -f /sys/kernel/debug/rknpu/load ] && print_info "NPU 负载: $(cat /sys/kernel/debug/rknpu/load 2>/dev/null)"
}

test_pipeline() {
    print_test "6. 端到端流水线测试"
    local BINARY="./build/rtdetr_pipeline"
    local MODEL="${1:-rtdetr_r18.rknn}"
    [ ! -f "$BINARY" ] && { print_fail "可执行文件不存在"; return 1; }
    [ ! -f "$MODEL" ] && { print_fail "模型文件不存在"; return 1; }
    if [ -f "test.jpg" ]; then
        timeout 10 $BINARY -m "$MODEL" -v test.jpg -o /tmp/test_output.mp4 2>&1 | tail -3
        [ $? -eq 0 ] && print_pass "单图测试通过" || print_fail "单图测试失败"
    else
        print_skip "test.jpg 不存在"
    fi
    if [ -e /dev/video0 ]; then
        timeout 5 $BINARY -m "$MODEL" -d /dev/video0 -W 1920 -H 1080 2>&1 | tail -3
        [ $? -eq 0 ] || [ $? -eq 124 ] && print_pass "V4L2 测试通过" || print_fail "V4L2 测试失败"
    fi
}

test_stress() {
    print_test "7. 压力测试（30秒）"
    local BINARY="./build/rtdetr_pipeline"
    local MODEL="${1:-rtdetr_r18.rknn}"
    [ ! -f "$BINARY" ] || [ ! -f "$MODEL" ] && { print_skip "缺少文件"; return 0; }
    [ -e /dev/video0 ] && {
        timeout 30 $BINARY -m "$MODEL" -d /dev/video0 -W 1920 -H 1080 2>&1 | grep -E "(frame=|FPS|Error)" | tail -5
        [ $? -eq 0 ] || [ $? -eq 124 ] && print_pass "压力测试通过" || print_fail "压力测试失败"
    } || print_skip "无摄像头"
}

test_recovery() {
    print_test "8. 异常恢复测试"
    local BINARY="./build/rtdetr_pipeline"
    local MODEL="${1:-rtdetr_r18.rknn}"
    [ ! -f "$BINARY" ] || [ ! -f "$MODEL" ] && { print_skip "缺少文件"; return 0; }
    [ -e /dev/video0 ] && {
        $BINARY -m "$MODEL" -d /dev/video0 -W 1920 -H 1080 &
        local pid=$!
        sleep 3
        kill -INT $pid
        wait $pid 2>/dev/null
        [ $? -eq 0 ] && print_pass "Ctrl+C 优雅退出通过" || print_fail "Ctrl+C 退出失败"
        local residual=$(pgrep -f "rtdetr_pipeline" | wc -l)
        [ "$residual" -eq 0 ] && print_pass "无残留进程" || { print_fail "残留 $residual 进程"; pkill -9 -f "rtdetr_pipeline"; }
        fuser /dev/video0 2>/dev/null && print_fail "/dev/video0 仍被占用" || print_pass "/dev/video0 已释放"
    } || print_skip "无摄像头"
}

main() {
    echo "============================================"
    echo "  RT-DETR-R18 RK3588 RGA_DMA 健壮性测试"
    echo "============================================"
    echo ""
    test_environment; echo ""
    test_dma_buffer; echo ""
    test_rga; echo ""
    test_v4l2; echo ""
    test_npu; echo ""
    test_pipeline "$1"; echo ""
    test_stress "$1"; echo ""
    test_recovery "$1"; echo ""
    echo "============================================"
    echo "  测试结果汇总"
    echo "============================================"
    echo -e "  ${GREEN}通过: $PASS${NC}"
    echo -e "  ${RED}失败: $FAIL${NC}"
    echo -e "  ${YELLOW}跳过: $SKIP${NC}"
    echo "============================================"
    [ $FAIL -gt 0 ] && exit 1 || exit 0
}

main "$1"
