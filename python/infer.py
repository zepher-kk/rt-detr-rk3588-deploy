import os
import cv2
import time
import argparse
import numpy as np

# 从板子上跑
from rknnlite.api import RKNNLite as RKNN

# ==========================================
# 类别配置 (保持与你的模型对应)
# ==========================================
CLASSES = (
    "Pedestrian", "People", "Bicycle", "Car", "Van", 
    "Truck", "Tricycle", "Awning-tricycle", "Bus", "Motor"
)

# ==========================================
# 核心：RT-DETR 后处理 (双张量接收版)
# ==========================================
# ==========================================
# 核心：RT-DETR 后处理 (调试探针版)
# ==========================================
# ==========================================
# 核心：RT-DETR 后处理 (智能寻型终极版)
# ==========================================
def post_process(outputs, orig_shape, conf_thres=0.45):
    pred_boxes = None
    pred_logits = None
    
    # 🌟 终极防御：不看顺序，直接用形状“滴血认亲”！
    for out in outputs:
        if out.shape == (1, 300, 4):
            pred_boxes = out
        elif out.shape == (1, 300, 10):
            pred_logits = out
            
    if pred_boxes is None or pred_logits is None:
        print("❌ 致命错误：在 NPU 输出中找不到目标张量！")
        return []

    # 挤掉 Batch 维度
    boxes = np.squeeze(pred_boxes, axis=0)      # 变成 [300, 4]
    scores = np.squeeze(pred_logits, axis=0)    # 变成 [300, 10]

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
    
    # 裁剪到图像物理边界内
    x_min = np.clip(x_min, 0, w_orig)
    y_min = np.clip(y_min, 0, h_orig)
    x_max = np.clip(x_max, 0, w_orig)
    y_max = np.clip(y_max, 0, h_orig)
    
    results = []
    for i in range(len(valid_boxes)):
        results.append({
            'class_id': int(valid_class_ids[i]),
            'score': float(valid_scores[i]),
            'box': [int(x_min[i]), int(y_min[i]), int(x_max[i]), int(y_max[i])]
        })
    return results

def draw(image, results):
    for res in results:
        box = res['box']
        class_id = res['class_id']
        score = res['score']
        
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = f"{CLASSES[class_id]} {score:.2f}"
        cv2.putText(image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def process_frame(rknn, orig_img, img_size, conf_thres):
    orig_shape = orig_img.shape[:2]
    
    # 前处理
    img = cv2.resize(orig_img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 🌟 修复 2：绝不能除以 255！因为我们在 convert.py 里已经配置了 std_values
    img = np.expand_dims(img, axis=0)

    # 推理
    outputs = rknn.inference(inputs=[img])
    
    # 后处理
    results = post_process(outputs, orig_shape, conf_thres=conf_thres)
    res_img = draw(orig_img.copy(), results)
    
    return res_img, len(results)

def main(args):
    print("-> 1. 初始化 RKNN 引擎...")
    rknn = RKNN()
    
    print(f"-> 正在加载模型: {args.model_path}")
    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        print("❌ Load RKNN failed!")
        return
        
    print("-> 正在初始化运行时环境...")
    ret = rknn.init_runtime()
    if ret != 0:
        print("❌ Init runtime failed!")
        return

    img_size = (args.img_size, args.img_size)
    source_ext = os.path.splitext(args.source)[-1].lower()
    is_video = source_ext in ['.mp4', '.avi', '.mov', '.mkv'] or args.source.isdigit()

    if not is_video:
        # ==================== 单图处理 ====================
        print(f"-> 2. 读取图片: {args.source}")
        orig_img = cv2.imread(args.source)
        if orig_img is None:
            return
            
        print("-> 3. 执行推理与后处理...")
        res_img, num_targets = process_frame(rknn, orig_img, img_size, args.conf_thres)
        print(f"-> 共检测到 {num_targets} 个目标。")

        out_name = "result_today_" + os.path.basename(args.source)
        cv2.imwrite(out_name, res_img)
        print(f"✅ 测试完成，不进行单图测速，结果已保存为 {out_name}")

    else:
        # ==================== 视频处理 (附带测速) ====================
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

        print("-> 3. 视频推理与测速轰鸣中...")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            res_img, _ = process_frame(rknn, frame, img_size, args.conf_thres)
            writer.write(res_img)
            frame_count += 1
            
            # 每 30 帧打印一次动态平均 FPS
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
    parser = argparse.ArgumentParser(description="RK3588 RT-DETR 推理脚本")
    parser.add_argument('--model_path', type=str, required=True, help='你的 .rknn 模型路径')
    parser.add_argument('--source', type=str, required=True, help='输入源')
    parser.add_argument('--conf_thres', type=float, default=0.45, help='置信度阈值')
    parser.add_argument('--img_size', type=int, default=640, help='模型输入尺寸')
    
    args = parser.parse_args()
    main(args)