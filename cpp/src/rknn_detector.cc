#include "rknn_detector.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// uint8 RGB -> FP16, multiplied by `scale` (1/255 keeps the model's expected
// normalization when the runtime conversion is bypassed via rknn_set_io_mem).
static void u8_to_f16_scaled(const uint8_t* src, uint16_t* dst, size_t n, float scale) {
#if defined(__aarch64__)
    const float32x4_t vscale = vdupq_n_f32(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t u8 = vld1_u8(src + i);
        uint16x8_t u16 = vmovl_u8(u8);
        float32x4_t lo = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16))), vscale);
        float32x4_t hi = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16))), vscale);
        float16x8_t h = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
        vst1q_f16((float16_t*)(dst + i), h);
    }
    for (; i < n; ++i) {
        __fp16 h = (__fp16)(src[i] * scale);
        memcpy(dst + i, &h, 2);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        __fp16 h = (__fp16)(src[i] * scale);
        memcpy(dst + i, &h, 2);
    }
#endif
}

static unsigned char* load_file(const char* path, int* size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[rknn] cannot open %s\n", path); return nullptr; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    auto* buf = (unsigned char*)malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return nullptr; }
    fclose(f);
    *size = (int)n;
    return buf;
}

RKNNDetector::~RKNNDetector() {
    if (calls_) {
        fprintf(stderr, "[rknn] avg per call: prep=%.2fms run=%.2fms get=%.2fms (n=%lld) io_mem=%d\n",
                t_set_ / 1000.0 / calls_, t_run_ / 1000.0 / calls_,
                t_get_ / 1000.0 / calls_, (long long)calls_, (int)in_mem_ready_);
    }
    if (in_mem_) {
        rknn_destroy_mem(ctx_, in_mem_);
        in_mem_ = nullptr;
    }
    if (ctx_) rknn_destroy(ctx_);
    free(model_data_);
}

bool RKNNDetector::init(const std::string& model_path, rknn_core_mask mask) {
    int size = 0;
    model_data_ = load_file(model_path.c_str(), &size);
    if (!model_data_) return false;

    if (rknn_init(&ctx_, model_data_, (uint32_t)size, 0, nullptr) != 0) {
        fprintf(stderr, "[rknn] init failed\n");
        return false;
    }
    mask_ = mask;
    if (mask != RKNN_NPU_CORE_AUTO) {
        int r = rknn_set_core_mask(ctx_, mask);
        if (r != 0) fprintf(stderr, "[rknn] set_core_mask(%d) failed: %d\n", (int)mask, r);
    }
    if (!query_io()) return false;
    ready_ = true;
    return true;
}

bool RKNNDetector::query_io() {
    rknn_input_output_num io;
    if (rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != 0) return false;
    n_input_ = (int)io.n_input;
    n_output_ = (int)io.n_output;
    fprintf(stderr, "[rknn] core_mask=%d n_in=%d n_out=%d\n", (int)mask_, n_input_, n_output_);

    if (n_input_ > 0) {
        memset(&input_attr_, 0, sizeof(input_attr_));
        input_attr_.index = 0;
        if (rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)) != 0) {
            fprintf(stderr, "[rknn] query input attr failed\n");
            return false;
        }
        memset(&native_in_attr_, 0, sizeof(native_in_attr_));
        native_in_attr_.index = 0;
        rknn_query(ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &native_in_attr_, sizeof(native_in_attr_));
        fprintf(stderr, "[rknn] input type=%d fmt=%d size=%u | native type=%d fmt=%d size=%u\n",
                (int)input_attr_.type, (int)input_attr_.fmt, input_attr_.size,
                (int)native_in_attr_.type, (int)native_in_attr_.fmt, native_in_attr_.size);

        // Prepare a native-format input buffer for rknn_set_io_mem.
        // The model input is FP16 NHWC, so we convert uint8 RGB -> FP16 /255
        // ourselves instead of letting rknn_inputs_set do it every frame.
        if (getenv("RTDETR_IO_MEM") == nullptr ||
            std::string(getenv("RTDETR_IO_MEM")) != "0") {
            if (native_in_attr_.type == RKNN_TENSOR_FLOAT16 && native_in_attr_.size > 0) {
                in_mem_ = rknn_create_mem(ctx_, native_in_attr_.size);
                if (in_mem_) {
                    rknn_tensor_attr a = native_in_attr_;
                    a.index = 0;
                    a.pass_through = 1;
                    if (rknn_set_io_mem(ctx_, in_mem_, &a) == 0) {
                        in_mem_ready_ = true;
                        fprintf(stderr, "[rknn] io_mem input ready: %u bytes (fp16)\n",
                                native_in_attr_.size);
                    } else {
                        fprintf(stderr, "[rknn] set_io_mem failed, using inputs_set\n");
                        rknn_destroy_mem(ctx_, in_mem_);
                        in_mem_ = nullptr;
                    }
                }
            }
        }
    }

    out_attrs_.resize(n_output_);
    out_buffers_.resize(n_output_);
    boxes_idx_ = logits_idx_ = -1;

    for (int i = 0; i < n_output_; ++i) {
        auto& a = out_attrs_[i];
        memset(&a, 0, sizeof(a));
        a.index = i;
        if (rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) != 0) return false;
        fprintf(stderr, "[rknn] out[%d] n_elems=%u type=%d fmt=%d name=%s\n",
                i, a.n_elems, (int)a.type, (int)a.fmt, a.name);
        out_buffers_[i].assign(a.n_elems, 0.f);

        // RT-DETR: boxes = 300*4, logits = 300*10 (last match wins, like v3)
        if (a.n_elems == 300u * 4u) boxes_idx_ = i;
        if (a.n_elems == 300u * NUM_CLASSES) logits_idx_ = i;
    }

    if (boxes_idx_ < 0 && logits_idx_ < 0) {
        // fall back to pair inference
        for (int i = 0; i < n_output_; ++i) {
            if (out_attrs_[i].n_elems % 4 != 0) continue;
            for (int j = 0; j < n_output_; ++j) {
                if (out_attrs_[j].n_elems % NUM_CLASSES != 0) continue;
                if (out_attrs_[i].n_elems / 4 == out_attrs_[j].n_elems / NUM_CLASSES) {
                    boxes_idx_ = i;
                    logits_idx_ = j;
                }
            }
        }
    }
    if (boxes_idx_ < 0 || logits_idx_ < 0) {
        fprintf(stderr, "[rknn] FATAL: cannot identify boxes/logits outputs\n");
        return false;
    }
    boxes_elems_ = (int)out_attrs_[boxes_idx_].n_elems;
    logits_elems_ = (int)out_attrs_[logits_idx_].n_elems;
    fprintf(stderr, "[rknn] boxes=out[%d] (%d) logits=out[%d] (%d)\n",
            boxes_idx_, boxes_elems_, logits_idx_, logits_elems_);
    return true;
}

bool RKNNDetector::infer(const DmaBufferPtr& input, FrameBundle& out) {
    if (!ready_ || !input) return false;
    int64_t t0 = now_us();

    if (in_mem_ready_) {
        const size_t n = (size_t)INPUT_W * INPUT_H * 3;
        u8_to_f16_scaled((const uint8_t*)input->ptr, (uint16_t*)in_mem_->virt_addr,
                         n, 1.0f / 255.0f);
    } else {
        rknn_input in;
        memset(&in, 0, sizeof(in));
        in.index = 0;
        in.type = RKNN_TENSOR_UINT8;
        in.fmt = RKNN_TENSOR_NHWC;
        in.size = (uint32_t)input->size;
        in.buf = input->ptr;
        in.pass_through = 0;
        if (rknn_inputs_set(ctx_, 1, &in) != 0) return false;
    }
    int64_t t1 = now_us();
    if (rknn_run(ctx_, nullptr) != 0) return false;
    int64_t t2 = now_us();

    std::vector<rknn_output> outs(n_output_);
    for (int i = 0; i < n_output_; ++i) {
        outs[i].want_float = 1;
        outs[i].index = i;
        outs[i].is_prealloc = 1;
        outs[i].buf = out_buffers_[i].data();
        outs[i].size = out_attrs_[i].n_elems * sizeof(float);
    }
    if (rknn_outputs_get(ctx_, n_output_, outs.data(), nullptr) != 0) return false;
    int64_t t3 = now_us();
    t_set_ += t1 - t0;
    t_run_ += t2 - t1;
    t_get_ += t3 - t2;
    calls_ += 1;

    out.pred_boxes.resize(boxes_elems_);
    out.pred_logits.resize(logits_elems_);
    memcpy(out.pred_boxes.data(), out_buffers_[boxes_idx_].data(),
           boxes_elems_ * sizeof(float));
    memcpy(out.pred_logits.data(), out_buffers_[logits_idx_].data(),
           logits_elems_ * sizeof(float));
    out.num_boxes = boxes_elems_ / 4;
    out.num_classes = logits_elems_ / out.num_boxes;

    rknn_outputs_release(ctx_, n_output_, outs.data());
    return true;
}
