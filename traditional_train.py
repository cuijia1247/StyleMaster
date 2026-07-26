from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms

from ssc.classifier import Classifier_Simple

# backbone名称 -> (特征维度, 输入分辨率)
BACKBONE_CONFIGS: dict[str, tuple[int, int]] = {
    "vgg16":        (4096, 224),
    "vgg19":        (4096, 224),
    "resnet50":     (2048, 224),
    "resnet101":    (2048, 224),
    "inception_v3": (2048, 299),
    "vit_b_16":     (768,  224),
    "vit_l_16":     (1024, 224),
}

DEFAULT_DATA_BASE = "/mnt/codes/data/style"
BENCHMARK_DATASETS = [
    ("Painting91", 13),
    ("FashionStyle14", 14),
]
DEFAULT_FAILURE_MD = os.path.join("ieee_access_paperdata", "ivt_failure_case_list.md")


@dataclass
class FailureCase:
    """单条预测错误记录。"""
    file_path: str
    filename: str
    true_label: int
    pred_label: int
    true_name: str
    pred_name: str


@dataclass
class DatasetTrainResult:
    """单数据集训练与测试集预测汇总。"""
    name: str
    num_classes: int
    data_root: str
    backbone: str
    accuracy: float
    total: int
    num_errors: int
    model_path: str
    failures: List[FailureCase] = field(default_factory=list)


def build_backbone(name: str, device: torch.device) -> nn.Module:
    """构建冻结的预训练特征提取器，预训练权重缓存至 ./pretrainModels。"""
    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            m.features, m.avgpool, nn.Flatten(),
            *list(m.classifier.children())[:-1],
        )
    elif name == "vgg19":
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            m.features, m.avgpool, nn.Flatten(),
            *list(m.classifier.children())[:-1],
        )
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    elif name == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    elif name == "inception_v3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        m.aux_logits = False
        m.fc = nn.Identity()
        backbone = m
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()
        backbone = m
    elif name == "vit_l_16":
        m = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()
        backbone = m
    else:
        raise ValueError(f"不支持的 backbone: {name}，可选: {list(BACKBONE_CONFIGS)}")

    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


@torch.no_grad()
def extract_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """使用冻结 backbone 提取特征。"""
    return backbone(images)


@torch.no_grad()
def evaluate(
    classifier: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """在测试特征缓存上评估准确率。"""
    classifier.eval()
    correct = 0
    total = 0
    for feats, labels in test_loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = classifier(feats)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_end2end(
    backbone: nn.Module,
    image_loader: DataLoader,
    classifier: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """在原始图像上端到端推理评估，返回 (accuracy, inference_time_s, per_sample_ms)。"""
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer_start = time.perf_counter()
    for images, labels in image_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = extract_features(backbone, images)
        logits = classifier(feats)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_s = time.perf_counter() - t_infer_start
    acc = correct / total if total > 0 else 0.0
    per_sample_ms = inference_time_s / total * 1000.0 if total > 0 else 0.0
    return acc, inference_time_s, per_sample_ms


@torch.no_grad()
def predict_test_with_failures(
    backbone: nn.Module,
    classifier: nn.Module,
    test_set: datasets.ImageFolder,
    test_image_loader: DataLoader,
    device: torch.device,
) -> tuple[float, List[FailureCase]]:
    """在测试集上端到端预测，收集预测错误的样本。"""
    backbone.eval()
    classifier.eval()
    class_names = test_set.classes
    sample_paths = [os.path.abspath(path) for path, _ in test_set.samples]

    failures: List[FailureCase] = []
    correct = 0
    total = 0
    sample_idx = 0

    for images, labels in test_image_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = extract_features(backbone, images)
        logits = classifier(feats)
        preds = logits.argmax(dim=1)

        for i in range(labels.size(0)):
            true_label = int(labels[i].item())
            pred_label = int(preds[i].item())
            file_path = sample_paths[sample_idx]
            filename = os.path.basename(file_path)

            if true_label != pred_label:
                failures.append(
                    FailureCase(
                        file_path=file_path,
                        filename=filename,
                        true_label=true_label,
                        pred_label=pred_label,
                        true_name=class_names[true_label],
                        pred_name=class_names[pred_label],
                    )
                )
            else:
                correct += 1
            total += 1
            sample_idx += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, failures


@torch.no_grad()
def build_feature_cache(
    backbone: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> TensorDataset:
    """一次性提取并缓存全量特征，后续训练直接读取缓存。"""
    feat_list = []
    label_list = []
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        feats = extract_features(backbone, images)
        feat_list.append(feats.cpu())
        label_list.append(labels)
    features = torch.cat(feat_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return TensorDataset(features, labels)


def build_base_name(backbone: str, data_root: str) -> str:
    """构建统一命名基名：traditional-{backbone}-{数据集名}-{时间戳}"""
    dataset_name = os.path.basename(os.path.normpath(data_root))
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return f"traditional-{backbone}-{dataset_name}-{time_str}"


def write_failure_markdown(
    out_path: str,
    results: List[DatasetTrainResult],
    argv_summary: str,
) -> None:
    """将各数据集预测错误样本写入 Markdown。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# ViT 传统训练测试集预测错误案例",
        "",
        f"_生成时间: {timestamp}_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
        "## 汇总",
        "",
        "| Dataset | num_classes | Test Acc | 错误数 | 测试样本数 | backbone | 模型路径 |",
        "|---------|-------------|----------|--------|-----------|----------|---------|",
    ]

    for result in results:
        lines.append(
            f"| {result.name} | {result.num_classes} | {result.accuracy:.4f} "
            f"| {result.num_errors} | {result.total} "
            f"| {result.backbone} | `{result.model_path}` |"
        )
    lines.append("")

    for result in results:
        lines.extend([
            f"## {result.name}",
            "",
            f"- **data_root**: `{result.data_root}`",
            f"- **Test Accuracy**: {result.accuracy:.4f}",
            f"- **错误数 / 总数**: {result.num_errors} / {result.total}",
            "",
            "| # | 文件名 | 文件路径 | 真实类别 (id) | 预测类别 (id) |",
            "|---|--------|---------|--------------|--------------|",
        ])
        for idx, case in enumerate(result.failures, start=1):
            lines.append(
                f"| {idx} | `{case.filename}` | `{case.file_path}` "
                f"| {case.true_name} ({case.true_label}) | {case.pred_name} ({case.pred_label}) |"
            )
        if not result.failures:
            lines.append("| - | _无错误样本_ | - | - | - |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def train_single_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    num_classes: int,
    data_root: str,
    device: torch.device,
    logger: logging.Logger,
) -> DatasetTrainResult:
    """在单个数据集上训练 ViT 分类器，并用最佳 checkpoint 收集测试集错误案例。"""
    feat_dim, input_size = BACKBONE_CONFIGS[args.backbone]
    base_name = build_base_name(args.backbone, data_root)
    log_path = os.path.join("log", f"{base_name}.log")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(test_dir, transform=transform)

    train_image_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_image_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_number = num_classes if num_classes > 0 else len(train_set.classes)
    backbone = build_backbone(args.backbone, device)
    logger.info("[%s] Extracting train/test features with frozen %s (feat_dim=%d)...",
                dataset_name, args.backbone, feat_dim)
    train_cache_set = build_feature_cache(backbone, train_image_loader, device)
    test_cache_set = build_feature_cache(backbone, test_image_loader, device)
    logger.info("[%s] Feature cache ready: train=%d, test=%d",
                dataset_name, len(train_cache_set), len(test_cache_set))

    train_loader = DataLoader(
        train_cache_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("[%s] Start training: backbone=%s, data=%s, class_num=%d",
                dataset_name, args.backbone, data_root, class_number)
    logger.info("[%s] Config: runs=%d, classifier_iteration=%d",
                dataset_name, args.runs, args.classifier_iteration)
    logger.info("[%s] Log file: %s", dataset_name, log_path)

    run_best_accs: List[float] = []
    overall_best_acc = 0.0
    overall_best_model_path = ""

    for run_idx in range(1, args.runs + 1):
        classifier = Classifier_Simple(input_feature=feat_dim, class_number=class_number).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)

        best_acc = 0.0
        save_path = os.path.join("model", f"{base_name}-run{run_idx}.pth")
        logger.info("[%s] Run %d/%d started", dataset_name, run_idx, args.runs)

        for it in range(args.classifier_iteration):
            classifier.train()
            for feats, labels in train_loader:
                feats = feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = classifier(feats)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (it + 1) % 10 == 0:
                test_acc, inference_time_s, per_sample_ms = evaluate_end2end(
                    backbone, test_image_loader, classifier, device
                )
                logger.info(
                    "[%s] [Run %d] [Iter %03d/%03d] test_acc=%.4f, "
                    "inference_time=%.3fs (%.3fms/image)",
                    dataset_name, run_idx, it + 1, args.classifier_iteration,
                    test_acc, inference_time_s, per_sample_ms,
                )
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(
                        {
                            "run": run_idx,
                            "iteration": it + 1,
                            "best_acc": best_acc,
                            "classifier_state_dict": classifier.state_dict(),
                            "backbone_state_dict": backbone.state_dict(),
                            "class_to_idx": train_set.class_to_idx,
                        },
                        save_path,
                    )
                    logger.info("[%s] [Run %d] Best model saved: %s",
                                dataset_name, run_idx, save_path)

        run_best_accs.append(best_acc)
        logger.info("[%s] [Run %d] best_test_acc=%.4f", dataset_name, run_idx, best_acc)

        if best_acc > 0 and os.path.exists(save_path):
            acc_int = int(round(best_acc * 10000))
            new_save_path = os.path.join(
                "model", f"{base_name}-run{run_idx}-acc{acc_int}.pth"
            )
            os.rename(save_path, new_save_path)
            logger.info("[%s] [Run %d] Model renamed: %s",
                        dataset_name, run_idx, new_save_path)
            if best_acc > overall_best_acc:
                overall_best_acc = best_acc
                overall_best_model_path = new_save_path

    mean_acc = float(np.mean(run_best_accs))
    std_acc = float(np.std(run_best_accs))
    logger.info("[%s] best acc list: %s", dataset_name, [round(x, 4) for x in run_best_accs])
    logger.info("[%s] Final result (mean+-std): %.4f+-%.4f", dataset_name, mean_acc, std_acc)
    print(f"[{dataset_name}] Final result (mean+-std): {mean_acc:.4f}+-{std_acc:.4f}")

    if not overall_best_model_path:
        raise RuntimeError(f"[{dataset_name}] 未找到可用的最佳模型 checkpoint")

    # 从最佳 checkpoint 加载分类器，避免误用最后一轮迭代权重
    ckpt = torch.load(overall_best_model_path, map_location=device)
    overall_best_classifier = Classifier_Simple(
        input_feature=feat_dim, class_number=class_number
    ).to(device)
    overall_best_classifier.load_state_dict(ckpt["classifier_state_dict"])
    overall_best_classifier.eval()

    test_acc, failures = predict_test_with_failures(
        backbone, overall_best_classifier, test_set, test_image_loader, device
    )
    logger.info("[%s] Failure collection done: acc=%.4f, errors=%d/%d",
                dataset_name, test_acc, len(failures), len(test_set))
    logger.removeHandler(file_handler)

    return DatasetTrainResult(
        name=dataset_name,
        num_classes=class_number,
        data_root=data_root,
        backbone=args.backbone,
        accuracy=test_acc,
        total=len(test_set),
        num_errors=len(failures),
        model_path=os.path.abspath(overall_best_model_path),
        failures=failures,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traditional training with frozen backbone + Classifier")
    parser.add_argument("--backbone", type=str, default="vit_l_16", choices=list(BACKBONE_CONFIGS),
                        help="预训练特征提取器类型")
    parser.add_argument("--data_root", type=str, default="/mnt/codes/data/style/Painting91",
                        help="单数据集模式：数据集根目录（需含 train/test 子目录）")
    parser.add_argument("--num_classes", type=int, default=13,
                        help="单数据集模式类别数；<=0 时自动从数据集推断")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--classifier_lr", type=float, default=3e-4)
    parser.add_argument("--classifier_iteration", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_base", type=str, default=DEFAULT_DATA_BASE,
                        help="多数据集模式的数据根目录")
    parser.add_argument(
        "--benchmark_datasets",
        nargs="+",
        default=[],
        choices=[name for name, _ in BENCHMARK_DATASETS],
        help="多数据集模式：依次训练并收集错误案例，如 Painting91 FashionStyle14",
    )
    parser.add_argument("--failure_md", type=str, default=DEFAULT_FAILURE_MD,
                        help="预测错误案例 Markdown 输出路径")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("traditional_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(__file__), "pretrainModels"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("model", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("pretrainModels", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.failure_md)), exist_ok=True)

    logger = setup_logger()
    argv_summary = " ".join(sys.argv)

    if args.benchmark_datasets:
        dataset_map = dict(BENCHMARK_DATASETS)
        selected = [(name, dataset_map[name]) for name in args.benchmark_datasets]
    else:
        selected = [(os.path.basename(os.path.normpath(args.data_root)), args.num_classes)]

    results: List[DatasetTrainResult] = []
    for dataset_name, num_classes in selected:
        if args.benchmark_datasets:
            data_root = os.path.join(args.data_base, dataset_name)
        else:
            data_root = args.data_root

        logger.info("=" * 80)
        logger.info("Dataset: %s (%s)", dataset_name, data_root)
        logger.info("=" * 80)
        results.append(
            train_single_dataset(args, dataset_name, num_classes, data_root, device, logger)
        )

    if len(results) > 1 or args.benchmark_datasets:
        write_failure_markdown(args.failure_md, results, argv_summary)
        logger.info("Failure cases written to: %s", os.path.abspath(args.failure_md))
        print(f"Failure cases written to: {os.path.abspath(args.failure_md)}")
    elif results:
        write_failure_markdown(args.failure_md, results, argv_summary)
        logger.info("Failure cases written to: %s", os.path.abspath(args.failure_md))
        print(f"Failure cases written to: {os.path.abspath(args.failure_md)}")


if __name__ == "__main__":
    main()
