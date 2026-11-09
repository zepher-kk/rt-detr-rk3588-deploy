// Dump input/output tensor attributes (standard + native) of RKNN models.
// Used to work out how to chain a backbone model into a decoder model.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rknn_api.h"

static const char* fmt_name(rknn_tensor_format f) {
    switch (f) {
        case RKNN_TENSOR_NCHW: return "NCHW";
        case RKNN_TENSOR_NHWC: return "NHWC";
        case RKNN_TENSOR_NC1HWC2: return "NC1HWC2";
        case RKNN_TENSOR_UNDEFINED: return "UNDEFINED";
        default: return "UNKNOWN";
    }
}

static const char* type_name(rknn_tensor_type t) {
    switch (t) {
        case RKNN_TENSOR_FLOAT32: return "FLOAT32";
        case RKNN_TENSOR_FLOAT16: return "FLOAT16";
        case RKNN_TENSOR_INT8: return "INT8";
        case RKNN_TENSOR_UINT8: return "UINT8";
        default: return "OTHER";
    }
}

static unsigned char* load(const char* path, int* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    auto* buf = (unsigned char*)malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return nullptr; }
    fclose(f);
    *size = (int)n;
    return buf;
}

static void show(const char* tag, const rknn_tensor_attr& a) {
    printf("  %-12s idx=%d name=%-52s fmt=%-8s type=%-7s dims=[", tag, a.index, a.name,
           fmt_name(a.fmt), type_name(a.type));
    for (uint32_t i = 0; i < a.n_dims; ++i) printf("%s%u", i ? "," : "", a.dims[i]);
    printf("] n_elems=%u size=%u stride_w=%u size_stride=%u qnt=%d scale=%.8g zp=%d\n",
           a.n_elems, a.size, a.w_stride, a.size_with_stride,
           (int)a.qnt_type, (double)a.scale, (int)a.zp);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        int size = 0;
        unsigned char* data = load(argv[i], &size);
        if (!data) { printf("cannot open %s\n", argv[i]); continue; }
        rknn_context ctx = 0;
        if (rknn_init(&ctx, data, (uint32_t)size, 0, nullptr) != 0) {
            printf("init failed: %s\n", argv[i]);
            free(data);
            continue;
        }
        rknn_input_output_num io{};
        rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
        printf("=== %s  n_in=%d n_out=%d\n", argv[i], io.n_input, io.n_output);

        for (int k = 0; k < io.n_input; ++k) {
            rknn_tensor_attr a{}; a.index = k;
            rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
            show("input", a);
            rknn_tensor_attr n{}; n.index = k;
            if (rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &n, sizeof(n)) == 0)
                show("native_in", n);
        }
        for (int k = 0; k < io.n_output; ++k) {
            rknn_tensor_attr a{}; a.index = k;
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
            show("output", a);
            rknn_tensor_attr n{}; n.index = k;
            if (rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &n, sizeof(n)) == 0)
                show("native_out", n);
        }
        rknn_destroy(ctx);
        free(data);
        printf("\n");
    }
    return 0;
}
