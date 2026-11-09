import os
import cv2
import time
import argparse
import numpy as np

# 自动兼容：板端使用 RKNNLite，PC WSL 端使用 RKNN
try:
    from rknnlite.api import RKNNLite as RKNN
    IS_BOARD = True
except ImportError:
    from rknn.api import RKNN
    IS_BOARD = False

# ==========================================
# 类别配置 (VisDrone 数据集 10 类别)
# ==========================================
CLASSES = (
    "Pedestrian", "People", "Bicycle", "Car", "Van", 
    "Truck", "Tricycle", "Awning-tricycle", "Bus", "Motor"
)

# ==========================================
# 核心：RT-DETR 后处理 (智能形状寻型)
# ==========================================
def post_process(outputs, orig_shape, conf_thres=0.45):
    pred_boxes = None
    pred_logits = None
    
    # 🌟 终极防御：按维度特征智能寻型
    for out in outputs:
        # 去掉可能的冗余维度
        squeezed = np.squeeze(out)
        if squeezed.ndim == 2:
            if squeezed.shape[-1] == 4 and squeezed.shape[0] == 300:
                pred_boxes = squeezed
            elif squeezed.shape[-1] == len(CLASSES) and squeezed.shape[0] == 300:
                pred_logits = squeezed

    if pred_boxes is None or pred_logits is None:
        # 兼容 (1, 300, 4) / (1, 300, 10)
        for out in outputs:
            if out.shape[-1] == 4 and 300 in out.shape:
                pred_boxes = np.reshape(out, (300, 4))
            elif out.shape[-1] == len(CLASSES) and 300 in out.shape:
                pred_logits = np.reshape(out, (300, len(CLASSES)))

    if pred_boxes is None or pred_logits is None:
        print("❌ 致命错误：在模型输出中找不到目标张量！")
        return []

    boxes = pred_boxes.copy()
    scores = pred_logits.copy()

    # 获取每个框的最大类别和分数
    class_ids = np.argmax(scores, axis=-1)
    max_scores = np.max(scores, axis=-1)
    
    # 阈值过滤
    mask = max_scores > conf_thres
    valid_boxes = boxes[mask]
    valid_scores = max_scores[mask]
    valid_class_ids = class_ids[mask]
    
    if len(valid_boxes) == 0:
        return []

    # 坐标反归一化 (直接基于 NPU 输出的归一化坐标计算)
    h_orig, w_orig = orig_shape
    valid_boxes[:, 0] *= w_orig  # cx
    valid_boxes[:, 1] *= h_orig  # cy
    valid_boxes[:, 2] *= w_orig  # w
    valid_boxes[:, 3] *= h_orig  # h
    
    x_min = valid_boxes[:, 0] - valid_boxes[:, 2] / 2
    y_min = valid_boxes[:, 1] - valid_boxes[:, 3] / 2
    x_max = valid_boxes[:, 0] + valid_boxes[:, 2] / 2
    y_max = valid_boxes[:, 1] + valid_boxes[:, 3] / 2
    
    dets = []
    for i in range(len(valid_boxes)):
        dets.append([
            float(x_min[i]), float(y_min[i]),
            float(x_max[i]), float(y_max[i]),
            float(valid_scores[i]), int(valid_class_ids[i])
        ])
    return dets

def draw_detections(img, detections):
    for det in detections:
        xmin, ymin, xmax, ymax, score, clsid = det
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        
        label = f"{CLASSES[clsid]}: {score:.2f}"
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + w, ymin), (0, 255, 0), -1)
        cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def process_frame(rknn, frame, img_size, conf_thres):
    orig_shape = frame.shape[:2]
    
    # 纯正 Letterbox 保持原图比例
    h, w = orig_shape
    scale = min(img_size / h, img_size / w)
    nh, nw = int(h * scale), int(w * scale)
    
    resized_img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    input_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    top = (img_size - nh) // 2
    left = (img_size - nw) // 2
    input_img[top:top+nh, left:left+nw] = resized_img
    
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(input_img, axis=0)

    # 推理
    outputs = rknn.inference(inputs=[input_data])
    
    # 后处理
    detections = post_process(outputs, (img_size, img_size), conf_thres)
    
    # 将坐标映射回原图尺寸
    mapped_dets = []
    for det in detections:
        xmin, ymin, xmax, ymax, score, clsid = det
        xmin = (xmin - left) / scale
        ymin = (ymin - top) / scale
        xmax = (xmax - left) / scale
        ymax = (ymax - top) / scale
        
        xmin = max(0, min(w, xmin))
        ymin = max(0, min(h, ymin))
        xmax = max(0, min(w, xmax))
        ymax = max(0, min(h, ymax))
        
        mapped_dets.append([xmin, ymin, xmax, ymax, score, clsid])

    res_img = draw_detections(frame.copy(), mapped_dets)
    return res_img, len(mapped_dets)

def main(args):
    rknn = RKNN()

    print(f"-> 1. 加载 RKNN 模型: {args.model_path}")
    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        print("❌ 加载模型失败!")
        return

    if IS_BOARD:
        print("-> 运行在 RK3588 板端环境 (使用全部 3 核心 NPU)...")
        ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_0_1_2)
    else:
        print("-> 运行在 PC 环境 (若连接了板端 ADB 请传入 --target rk3588)...")
        ret = rknn.init_runtime(target=args.target) if args.target else rknn.init_runtime()
        
    if ret != 0:
        print("❌ 初始化 Runtime 失败！")
        return

    img_size = args.imgsz
    is_image = args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

    if is_image:
        print(f"-> 2. 读取输入图片: {args.source}")
        orig_img = cv2.imread(args.source)
        if orig_img is None:
            print("❌ 读取图片失败！")
            return
            
        print("-> 3. 执行推理与后处理...")
        res_img, num_targets = process_frame(rknn, orig_img, img_size, args.conf_thres)
        print(f"-> 共检测到 {num_targets} 个目标。")

        out_name = "result_" + os.path.basename(args.source)
        cv2.imwrite(out_name, res_img)
        print(f"✅ 结果已保存为 {out_name}")

    else:
        print(f"-> 2. 打开视频流: {args.source}")
        source_val = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source_val)
        
        if not cap.isOpened():
            print("❌ 视频流打开失败！")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        out_name = "result_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

        print("-> 3. 视频推理中...")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            res_img, _ = process_frame(rknn, frame, img_size, args.conf_thres)
            writer.write(res_img)
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"   [运行中] 已处理 {frame_count} 帧, 当前平均速度: {current_fps:.2f} FPS")

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"✅ 视频处理完成! 共检测 {frame_count} 帧, 整体平均 FPS: {avg_fps:.2f}")
        print(f"✅ 结果已保存为 {out_name}")

        cap.release()
        writer.release()

    rknn.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RK3588 RT-DETR 推理脚本 (兼容板端与 PC)")
    parser.add_argument('--model_path', type=str, required=True, help='你的 .rknn 模型路径')
    parser.add_argument('--source', type=str, required=True, help='输入源 (图片路径或视频路径/摄像头索引)')
    parser.add_argument('--conf_thres', type=float, default=0.45, help='置信度阈值')
    parser.add_argument('--imgsz', type=int, default=640, help='模型输入尺寸')
    parser.add_argument('--target', type=str, default=None, help='PC 端通过 ADB 连接的目标设备类型 (如 rk3588)')
    
    args = parser.parse_args()
    main(args)
