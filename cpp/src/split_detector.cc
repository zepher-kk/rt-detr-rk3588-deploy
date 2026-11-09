#include "split_detector.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static unsigned char* load_file(const char* path, int* size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[split] cannot open %s\n", path);
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    auto* buf = (unsigned char*)malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    *size = (int)n;
    return buf;
}

// Dequantise one standard-NCHW INT8 feature map into the decoder's native
// NC1HWC2 FP16 layout. `rknn_outputs_get` returns the backbone outputs as
// contiguous NCHW, so the walk is sequential on both sides.
static void dequant_nchw(const int8_t* src, __fp16* dst, int H, int W, int C,
                         float scale, float bias) {
    const size_t plane = (size_t)H * (size_t)W;
    for (int c = 0; c < C; ++c) {
        const int8_t* s = src + (size_t)c * plane;
        __fp16* d = dst + (size_t)(c / 8) * plane * 8 + (size_t)(c % 8);
        for (size_t p = 0; p < plane; ++p) d[p * 8] = (__fp16)(s[p] * scale + bias);
    }
}

// ---------------------------------------------------------------------------
// SplitDetector
// ---------------------------------------------------------------------------

SplitDetector::~SplitDetector() {
    for (auto* m : b_in_mem_)
        if (m) rknn_destroy_mem(b_ctx_, m);
    b_in_mem_.clear();
    if (a_in_mem_) rknn_destroy_mem(a_ctx_, a_in_mem_);
    if (b_ctx_) rknn_destroy(b_ctx_);
    if (a_ctx_) rknn_destroy(a_ctx_);
    free(a_data_);
    free(b_data_);
}

bool SplitDetector::load_and_query(const std::string& path, rknn_context& ctx,
                                   unsigned char*& data, int& n_in, int& n_out) {
    int size = 0;
    data = load_file(path.c_str(), &size);
    if (!data) return false;
    if (rknn_init(&ctx, data, (uint32_t)size, 0, nullptr) != 0) {
        fprintf(stderr, "[split] rknn_init failed: %s\n", path.c_str());
        return false;
    }
    rknn_input_output_num io{};
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != 0) return false;
    n_in = (int)io.n_input;
    n_out = (int)io.n_output;
    return true;
}

static std::string query_custom_string(rknn_context ctx) {
    rknn_custom_string cs{};
    if (rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &cs, sizeof(cs)) != 0)
        return std::string();
    return std::string(cs.string);
}

bool SplitDetector::init(const std::string& backbone_path,
                         const std::string& decoder_path, rknn_core_mask mask) {
    mask_ = mask;
    if (!load_and_query(backbone_path, a_ctx_, a_data_, a_in_, a_out_)) return false;
    if (!load_and_query(decoder_path, b_ctx_, b_data_, b_in_, b_out_)) return false;

    if (mask_ != RKNN_NPU_CORE_AUTO) {
        rknn_set_core_mask(a_ctx_, mask_);
        rknn_set_core_mask(b_ctx_, mask_);
    }
    if (a_in_ != 1 || a_out_ != 3 || b_in_ != 3) {
        fprintf(stderr, "[split] unexpected io: backbone in/out = %d/%d, decoder in = %d\n",
                a_in_, a_out_, b_in_);
        return false;
    }

    // refuse to run a mismatched pair: the two models must be built together
    const std::string id_a = query_custom_string(a_ctx_);
    const std::string id_b = query_custom_string(b_ctx_);
    if (!id_a.empty() && !id_b.empty() && id_a != id_b) {
        fprintf(stderr, "[split] model mismatch: backbone '%s' vs decoder '%s'\n",
                id_a.c_str(), id_b.c_str());
        return false;
    }
    fprintf(stderr, "[split] build id: %s\n", id_a.empty() ? "(unset)" : id_a.c_str());

    a_in_attr_ = {};
    a_in_attr_.index = 0;
    rknn_query(a_ctx_, RKNN_QUERY_INPUT_ATTR, &a_in_attr_, sizeof(a_in_attr_));
    a_native_in_ = {};
    a_native_in_.index = 0;
    rknn_query(a_ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &a_native_in_, sizeof(a_native_in_));

    a_out_attrs_.resize(a_out_);
    a_native_out_.resize(a_out_);
    a_out_raw_.resize(a_out_);
    feat_h_.resize(a_out_);
    feat_w_.resize(a_out_);
    feat_c_.resize(a_out_);
    int total = 0;
    for (int i = 0; i < a_out_; ++i) {
        auto& a = a_out_attrs_[i];
        a = {};
        a.index = i;
        rknn_query(a_ctx_, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
        auto& n = a_native_out_[i];
        n = {};
        n.index = i;
        rknn_query(a_ctx_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &n, sizeof(n));
        // standard attrs are NCHW: [1, C, H, W]
        feat_c_[i] = (int)a.dims[1];
        feat_h_[i] = (int)a.dims[2];
        feat_w_[i] = (int)a.dims[3];
        a_out_raw_[i].assign(
            std::max(n.size_with_stride, a.size_with_stride), 0);
        total += (int)a_out_raw_[i].size();
        fprintf(stderr,
                "[split] feat%d %dx%dx%d int8 -> scale=%.6g zp=%d  (%zu B raw, %u B fp16)\n",
                i, feat_c_[i], feat_h_[i], feat_w_[i], (double)a.scale, (int)a.zp,
                a_out_raw_[i].size(), n.size_with_stride * 2u);
    }

    b_in_attrs_.resize(b_in_);
    b_native_in_.resize(b_in_);
    for (int i = 0; i < b_in_; ++i) {
        auto& a = b_in_attrs_[i];
        a = {};
        a.index = i;
        rknn_query(b_ctx_, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
        auto& n = b_native_in_[i];
        n = {};
        n.index = i;
        rknn_query(b_ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &n, sizeof(n));
        if ((int)a.dims[1] != feat_h_[i] || (int)a.dims[2] != feat_w_[i] ||
            (int)a.dims[3] != feat_c_[i]) {
            fprintf(stderr, "[split] decoder input %d shape mismatch\n", i);
            return false;
        }
    }

    b_out_attrs_.resize(b_out_);
    b_out_buf_.resize(b_out_);
    for (int i = 0; i < b_out_; ++i) {
        auto& a = b_out_attrs_[i];
        a = {};
        a.index = i;
        rknn_query(b_ctx_, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
        b_out_buf_[i].assign(a.n_elems, 0.f);
    }
    for (int i = 0; i < b_out_; ++i) {
        const uint32_t e = b_out_attrs_[i].n_elems;
        if (e == 0 || e % (4 + NUM_CLASSES) != 0) continue;
        const uint32_t q = e / (4 + NUM_CLASSES);
        if (q < 1 || q > 2000) continue;
        combined_idx_ = i;
        queries_ = (int)q;
        break;
    }
    if (combined_idx_ < 0) {
        fprintf(stderr, "[split] decoder: no y=(N,4+classes) output\n");
        return false;
    }
    boxes_elems_ = queries_ * 4;
    logits_elems_ = queries_ * NUM_CLASSES;

    if (!prepare_buffers()) return false;

    fprintf(stderr, "[split] ready: feat bytes=%d, y=out[%d] queries=%d\n",
            total, combined_idx_, queries_);
    ready_ = true;
    return true;
}

bool SplitDetector::prepare_buffers() {
    // backbone input: native uint8 NHWC, filled straight from the RGA buffer
    a_in_mem_ = rknn_create_mem(a_ctx_, a_native_in_.size_with_stride
                                            ? a_native_in_.size_with_stride
                                            : a_native_in_.size);
    if (!a_in_mem_) return false;
    rknn_tensor_attr ia = a_native_in_;
    ia.index = 0;
    ia.pass_through = 1;
    if (rknn_set_io_mem(a_ctx_, a_in_mem_, &ia) != 0) {
        fprintf(stderr, "[split] backbone set_io_mem failed\n");
        return false;
    }

    // decoder inputs: native fp16 NC1HWC2, written by dequant_nchw
    b_in_mem_.resize(b_in_, nullptr);
    for (int i = 0; i < b_in_; ++i) {
        uint32_t bytes = b_native_in_[i].size_with_stride
                             ? b_native_in_[i].size_with_stride
                             : b_native_in_[i].size;
        b_in_mem_[i] = rknn_create_mem(b_ctx_, bytes);
        if (!b_in_mem_[i]) return false;
        rknn_tensor_attr ia = b_native_in_[i];
        ia.index = i;
        ia.pass_through = 1;
        if (rknn_set_io_mem(b_ctx_, b_in_mem_[i], &ia) != 0) {
            fprintf(stderr, "[split] decoder set_io_mem %d failed\n", i);
            return false;
        }
    }
    io_mem_ready_ = true;
    return true;
}

bool SplitDetector::infer(const DmaBufferPtr& input, FrameBundle& out) {
    if (!ready_ || !input || !io_mem_ready_) return false;

    // backbone input is uint8 RGB NHWC but the tensor is INT8 with zp=-128,
    // so flip the sign bit instead of copying the bytes as-is
    {
        const uint8_t* s = (const uint8_t*)input->ptr;
        uint8_t* d = (uint8_t*)a_in_mem_->virt_addr;
        const size_t n = (size_t)INPUT_W * INPUT_H * 3;
        for (size_t i = 0; i < n; ++i) d[i] = (uint8_t)(s[i] ^ 0x80u);
    }
    if (rknn_run(a_ctx_, nullptr) != 0) return false;

    rknn_output a_outs[3];
    memset(a_outs, 0, sizeof(a_outs));
    for (int i = 0; i < a_out_; ++i) {
        a_outs[i].index = i;
        a_outs[i].want_float = 0;
        a_outs[i].is_prealloc = 1;
        a_outs[i].buf = a_out_raw_[i].data();
        a_outs[i].size = (uint32_t)a_out_raw_[i].size();
    }
    if (rknn_outputs_get(a_ctx_, (uint32_t)a_out_, a_outs, nullptr) != 0) return false;

    for (int i = 0; i < a_out_; ++i) {
        const float scale = a_out_attrs_[i].scale;
        const float bias = -(float)a_out_attrs_[i].zp * scale;
        const int8_t* src = (const int8_t*)a_out_raw_[i].data();
        __fp16* dst = (__fp16*)b_in_mem_[i]->virt_addr;
        dequant_nchw(src, dst, feat_h_[i], feat_w_[i], feat_c_[i], scale, bias);
        if (getenv("RTDETR_SPLIT_DEBUG") && calls_ < 2) {
            float mn = 1e9f, mx = -1e9f;
            size_t n = (size_t)feat_h_[i] * feat_w_[i] * feat_c_[i];
            for (size_t k = 0; k < n; ++k) {
                float v = (float)dst[k];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            fprintf(stderr, "[dbg] B in%d fp16 min=%.4f max=%.4f\n",
                    i, (double)mn, (double)mx);
        }
    }

    if (rknn_run(b_ctx_, nullptr) != 0) return false;

    std::vector<rknn_output> b_outs(b_out_);
    for (int i = 0; i < b_out_; ++i) {
        b_outs[i].want_float = 1;
        b_outs[i].index = i;
        b_outs[i].is_prealloc = 1;
        b_outs[i].buf = b_out_buf_[i].data();
        b_outs[i].size = b_out_attrs_[i].n_elems * sizeof(float);
    }
    if (rknn_outputs_get(b_ctx_, (uint32_t)b_out_, b_outs.data(), nullptr) != 0)
        return false;

    out.pred_boxes.resize(boxes_elems_);
    out.pred_logits.resize(logits_elems_);
    const float* y = b_out_buf_[combined_idx_].data();
    const int stride = 4 + NUM_CLASSES;
    for (int i = 0; i < queries_; ++i) {
        memcpy(out.pred_boxes.data() + (size_t)i * 4, y + (size_t)i * stride,
               4 * sizeof(float));
        memcpy(out.pred_logits.data() + (size_t)i * NUM_CLASSES,
               y + (size_t)i * stride + 4, NUM_CLASSES * sizeof(float));
    }
    out.num_boxes = queries_;
    out.num_classes = NUM_CLASSES;
    if (getenv("RTDETR_SPLIT_DEBUG") && calls_ < 2) {
        float mn = 1e9f, mx = -1e9f;
        for (int i = 0; i < queries_; ++i)
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float v = out.pred_logits[(size_t)i * NUM_CLASSES + c];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        fprintf(stderr, "[dbg] y scores min=%.4f max=%.4f\n", (double)mn, (double)mx);
    }
    calls_++;
    return true;
}
