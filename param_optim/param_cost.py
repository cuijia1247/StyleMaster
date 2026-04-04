# 计算 SSC ResNet 和 SSC Transformer 两套管线的计算代价
# 覆盖指标：Parameters / FLOPs / Inference Time / GPU Memory
# 运行方式：python param_optim/param_cost.py （在项目根目录下执行）
# Author: cuijia1247  Date: 2026-04-04

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm
from fvcore.nn import FlopCountAnalysis, parameter_count

from ssc.Sscreg import SscReg as ResNetSscReg          # resnet50 版 SscReg
from ssc.Sscreg_transformer import SscReg as VitSscReg  # swin/vit 版 SscReg
from ssc.classifier import EfficientClassifier


# ─── 配置 ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 1          # 推理时序批大小
IMAGE_SIZE  = 64         # 与训练脚本中 image_size 一致
CLASS_NUM   = 15         # 示例类别数（与 WikiArt3 对齐）
WARMUP_RUNS = 10         # GPU 预热次数
MEASURE_RUNS= 100        # 推理计时次数


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """统计模型全部参数量（包含冻结参数）"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: nn.Module) -> int:
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_flops(model: nn.Module, dummy_input: torch.Tensor) -> int:
    """使用 fvcore 计算单次前向 FLOPs"""
    model.eval()
    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total()

def measure_inference_time(model: nn.Module, dummy_input: torch.Tensor) -> float:
    """
    测量平均推理延迟（ms）。
    GPU 模式使用 CUDA Event 计时，CPU 模式使用 time.perf_counter。
    """
    model.eval()
    use_cuda = dummy_input.device.type == 'cuda'

    with torch.no_grad():
        # 预热
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_input)

        if use_cuda:
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(MEASURE_RUNS):
                _ = model(dummy_input)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt) / MEASURE_RUNS
        else:
            t0 = time.perf_counter()
            for _ in range(MEASURE_RUNS):
                _ = model(dummy_input)
            elapsed_ms = (time.perf_counter() - t0) / MEASURE_RUNS * 1000.0

    return elapsed_ms

def measure_gpu_memory(model: nn.Module, dummy_input: torch.Tensor) -> float:
    """
    测量单次前向推理的 GPU 显存峰值（MB）。
    CPU 设备返回 0.0。
    """
    if dummy_input.device.type != 'cuda':
        return 0.0

    model.eval()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(dummy_input)

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    return peak_bytes / 1024 ** 2  # bytes → MB

def fmt_params(n: int) -> str:
    """将参数量格式化为 M 单位字符串"""
    return f"{n / 1e6:.2f} M"

def fmt_flops(f: int) -> str:
    """将 FLOPs 格式化为 G 单位字符串"""
    return f"{f / 1e9:.3f} GFLOPs"

def print_result(name: str, total_params: int, trainable_params: int,
                 flops: int, infer_ms: float, mem_mb: float):
    """格式化打印单个模型的计算代价"""
    sep = "─" * 56
    print(f"\n{'═'*56}")
    print(f"  Model : {name}")
    print(sep)
    print(f"  Parameters (total)    : {fmt_params(total_params)}")
    print(f"  Parameters (trainable): {fmt_params(trainable_params)}")
    print(f"  FLOPs (single sample) : {fmt_flops(flops)}")
    print(f"  Inference Time        : {infer_ms:.3f} ms  (batch={BATCH_SIZE}, avg over {MEASURE_RUNS} runs)")
    if mem_mb > 0:
        print(f"  GPU Memory (peak)     : {mem_mb:.1f} MB")
    else:
        print(f"  GPU Memory (peak)     : N/A (CPU mode)")
    print('═' * 56)


# ─── 管线封装 ──────────────────────────────────────────────────────────────────
class ResNetPipeline(nn.Module):
    """
    ResNet 管线：resnet50(backend) + MLP projector + EfficientClassifier
    与 ssc_train_resnet.py 中的使用方式对齐：
      - backend 输出 2048 维，不经过 fc 层（Backend.py 的 forward 跳过了 fc）
      - projector: 2048→2048 (depth=3)
      - classifier 接收 projector 输出作为输入
    """
    def __init__(self, class_num: int):
        super().__init__()
        self.ssc = ResNetSscReg(backend='resnet50', input_size=2048,
                                output_size=2048, pretrained_backend=False)
        self.classifier = EfficientClassifier(input_feature=2048,
                                              class_number=class_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.ssc(x)
        return self.classifier(feat)


class VitPipeline(nn.Module):
    """
    ViT/Swin 管线：swin_base_patch4_window7_224(backend) + MLP projector + EfficientClassifier
    与 ssc_train_transformer.py 中的使用方式对齐：
      - backend 输出 1024 维
      - projector: 1024→1024 (depth=3)
      - classifier 接收 projector 输出
    注意：此处跳过从磁盘加载预训练权重，仅评估架构代价。
    """
    def __init__(self, class_num: int):
        super().__init__()
        # 直接用 timm 构建（不加载预训练权重，避免磁盘依赖）
        backend = timm.create_model('swin_base_patch4_window7_224',
                                    pretrained=False, num_classes=0)
        self.backend = backend

        # 与 Sscreg_transformer.py 中一致：冻结 backend
        for p in self.backend.parameters():
            p.requires_grad = False

        # MLP projector: 1024→1024, depth=3（与 ssc_train_transformer.py 配置一致）
        from ssc.Sscreg_transformer import MLP
        self.projector = MLP(input_size=1024, output_size=1024, depth=3)

        self.classifier = EfficientClassifier(input_feature=1024,
                                              class_number=class_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feat = self.backend(x)
        feat = self.projector(feat)
        return self.classifier(feat)


# ─── 主程序 ───────────────────────────────────────────────────────────────────
def evaluate_model(name: str, model: nn.Module, img_size: int):
    """对单个模型完整执行四项代价评估"""
    model = model.to(DEVICE)
    dummy = torch.randn(BATCH_SIZE, 3, img_size, img_size, device=DEVICE)

    total_params    = count_parameters(model)
    trainable       = count_trainable_parameters(model)
    flops           = measure_flops(model, dummy)
    infer_ms        = measure_inference_time(model, dummy)
    mem_mb          = measure_gpu_memory(model, dummy)

    print_result(name, total_params, trainable, flops, infer_ms, mem_mb)


def main():
    print(f"\n运行设备: {DEVICE}")
    print(f"输入尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}  (推理时 Swin 自动插值到 224x224)")
    print(f"类别数  : {CLASS_NUM}")

    # ── ResNet50 管线 ──────────────────────────────────────────────────────────
    resnet_pipeline = ResNetPipeline(class_num=CLASS_NUM)
    evaluate_model("ResNet50 + MLP Projector + EfficientClassifier",
                   resnet_pipeline, img_size=IMAGE_SIZE)

    # ── Swin-Base (ViT) 管线 ──────────────────────────────────────────────────
    vit_pipeline = VitPipeline(class_num=CLASS_NUM)
    evaluate_model("Swin-Base + MLP Projector + EfficientClassifier",
                   vit_pipeline, img_size=IMAGE_SIZE)


if __name__ == '__main__':
    main()
