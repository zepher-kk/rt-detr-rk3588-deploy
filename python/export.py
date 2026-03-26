import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import RTDETR

# ==========================================
# 🧙‍♂️ 黑魔法 1：手动展开 GridSample (无布尔运算极简版)
# 根除 RKNN 的 dataconvert type -1 报错
# ==========================================
def custom_grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    N, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]

    if align_corners:
        ix = ((ix + 1) / 2) * (W_in - 1)
        iy = ((iy + 1) / 2) * (H_in - 1)
    else:
        ix = ((ix + 1) * W_in - 1) / 2
        iy = ((iy + 1) * H_in - 1) / 2

    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_ne = ix_nw + 1
    iy_sw = iy_nw + 1
    ix_se = ix_ne
    iy_se = iy_sw

    nw_weight = (ix_ne - ix) * (iy_sw - iy)
    ne_weight = (ix - ix_nw) * (iy_sw - iy)
    sw_weight = (ix_ne - ix) * (iy - iy_nw)
    se_weight = (ix - ix_nw) * (iy - iy_nw)

    # 边界裁剪：保证索引不越界，天然实现 padding_mode='border'
    ix_nw = torch.clamp(ix_nw, 0, W_in - 1)
    iy_nw = torch.clamp(iy_nw, 0, H_in - 1)
    ix_ne = torch.clamp(ix_ne, 0, W_in - 1)
    iy_sw = torch.clamp(iy_sw, 0, H_in - 1)
    ix_se = torch.clamp(ix_se, 0, W_in - 1)
    iy_se = torch.clamp(iy_se, 0, H_in - 1)

    idx_nw = (iy_nw * W_in + ix_nw).reshape(N, -1)
    idx_ne = (iy_nw * W_in + ix_ne).reshape(N, -1)
    idx_sw = (iy_sw * W_in + ix_nw).reshape(N, -1)
    idx_se = (iy_se * W_in + ix_se).reshape(N, -1)

    flat_input = input.reshape(N, C, -1)

    idx_nw = idx_nw.unsqueeze(1).expand(-1, C, -1).long()
    idx_ne = idx_ne.unsqueeze(1).expand(-1, C, -1).long()
    idx_sw = idx_sw.unsqueeze(1).expand(-1, C, -1).long()
    idx_se = idx_se.unsqueeze(1).expand(-1, C, -1).long()

    nw_val = torch.gather(flat_input, 2, idx_nw).view(N, C, H_out, W_out)
    ne_val = torch.gather(flat_input, 2, idx_ne).view(N, C, H_out, W_out)
    sw_val = torch.gather(flat_input, 2, idx_sw).view(N, C, H_out, W_out)
    se_val = torch.gather(flat_input, 2, idx_se).view(N, C, H_out, W_out)

    nw_weight = nw_weight.unsqueeze(1)
    ne_weight = ne_weight.unsqueeze(1)
    sw_weight = sw_weight.unsqueeze(1)
    se_weight = se_weight.unsqueeze(1)

    out = (nw_val * nw_weight +
           ne_val * ne_weight +
           sw_val * sw_weight +
           se_val * se_weight)

    return out

F.grid_sample = custom_grid_sample
print("✅ 成功注入极简版 GridSample 补丁！")

# ==========================================
# 🧙‍♂️ 黑魔法 2：改良版 Linear 补丁 (彻底消灭 -1 动态维度)
# ==========================================
original_linear_forward = nn.Linear.forward

def patched_linear_forward(self, x):
    if x.dim() > 2:
        original_shape = x.shape
        flatten_dim = 1
        for i in range(len(original_shape) - 1):
            flatten_dim *= original_shape[i]
        x_flatten = x.reshape(flatten_dim, original_shape[-1])
        out_flatten = original_linear_forward(self, x_flatten)
        out_shape = list(original_shape[:-1]) + [self.out_features]
        return out_flatten.reshape(out_shape)
    else:
        return original_linear_forward(self, x)

nn.Linear.forward = patched_linear_forward
print("✅ 成功注入 nn.Linear 静态扁平化补丁！")

# ==========================================
# 主流程：只生成唯一的 ONNX 模型
# ==========================================
pt_path = r"D:\learn\detr\UAV-DETR-rt\runs\train\rtdetr2\weights\best.pt"
# 名字越简单越好，不要 raw 也不要 sim 了
onnx_final_path = r"D:\learn\detr\UAV-DETR-rt\runs\train\rtdetr2\weights\best_rtdetr_npu.onnx"

print("\n1. 加载 PyTorch 模型...")
model = RTDETR(pt_path)
torch_model = model.model.cpu().eval()

if hasattr(torch_model, 'head'):
    torch_model.head.export = True
    torch_model.head.format = 'onnx'

dummy_input = torch.randn(1, 3, 640, 640)

print(f"2. 导出 ONNX 模型至 {onnx_final_path} ...")
torch.onnx.export(
    torch_model,
    dummy_input,
    onnx_final_path,
    opset_version=12,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['pred_logits', 'pred_boxes']
)

print(f"\n🎉 大功告成！拿好你的 {onnx_final_path}，直接去给 convert.py 转换吧！")