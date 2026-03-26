import sys

from rknn.api import RKNN

DATASET_PATH = '/mnt/d/learn/rknn/rknn_model_zoo/datasets/Visdrone/Visdrone_subset_20.txt'
DEFAULT_RKNN_PATH = './rtdetr2/weights/best_raw_17.onnx'
DEFAULT_QUANT = True


def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]));
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
        # -------------------------------------------------
        # ⬇️ 必选项：关闭优化器 ⬇️
        optimization_level=0,
        disable_rules=['fuse_mul_into_gemm'])
        # -------------------------------------------------
    print('done')

    # Load model
    print('--> Loading model')
    # ⬇️⬇️⬇️ 关键修改在这里 ⬇️⬇️⬇️
    # 必须显式告诉 RKNN 输入图片的大小，防止它在内部解析 MatMul 时猜错维度
    ret = rknn.load_onnx(
        model=model_path,
        inputs=['images'],                  # 指定输入节点名
        input_size_list=[[1, 3, 640, 640]]  # 强制定死形状 (Batch, C, H, W)
        
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # ⬇️ 建议先设为 False (FP16模式) 跑通，确认不报错后再开 True (i8)
    # 因为量化(i8)会触发额外的图优化，容易诱发 MatMul 报错
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
