from __future__ import annotations

import argparse
import logging
import os
import time

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


def build_backbone(name: str, device: torch.device) -> nn.Module:
    """构建冻结的预训练特征提取器，预训练权重缓存至 ./pretrainModels。"""
    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            m.features, m.avgpool, nn.Flatten(),
            *list(m.classifier.children())[:-1],  # 去掉最后分类层，保留 4096 维
        )
    elif name == "vgg19":
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            m.features, m.avgpool, nn.Flatten(),
            *list(m.classifier.children())[:-1],
        )
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(*list(m.children())[:-1], nn.Flatten())  # 去掉 FC 层
    elif name == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    elif name == "inception_v3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        m.aux_logits = False  # 关闭辅助分类器
        m.fc = nn.Identity()  # 替换分类头，输出 2048 维
        backbone = m
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()  # 替换分类头，输出 768 维
        backbone = m
    elif name == "vit_l_16":
        m = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()  # 替换分类头，输出 1024 维
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Traditional training with frozen backbone + Classifier")
    parser.add_argument("--backbone", type=str, default="vgg16", choices=list(BACKBONE_CONFIGS),
                        help="预训练特征提取器类型")
    parser.add_argument("--data_root", type=str, default="/mnt/codes/data/style/Painting91",
                        help="数据集根目录（需含 train/test 子目录）")
    parser.add_argument("--num_classes", type=int, default=13,
                        help="数据集类别数（Painting91=13）；<=0 时自动从数据集推断")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--classifier_lr", type=float, default=3e-4)
    parser.add_argument("--classifier_iteration", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # 预训练权重缓存目录
    os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(__file__), "pretrainModels"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("model", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("pretrainModels", exist_ok=True)

    feat_dim, input_size = BACKBONE_CONFIGS[args.backbone]

    base_name = build_base_name(args.backbone, args.data_root)
    log_path = os.path.join("log", f"{base_name}.log")

    logger = logging.getLogger("traditional_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(test_dir, transform=transform)

    # 用于一次性提取特征的图像加载器（只提取一次）
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

    # 优先使用显式传入的类别数，否则从数据集自动推断
    class_number = args.num_classes if args.num_classes > 0 else len(train_set.classes)
    backbone = build_backbone(args.backbone, device)
    logger.info("Extracting train/test features once with frozen %s backbone (feat_dim=%d)...",
                args.backbone, feat_dim)
    train_cache_set = build_feature_cache(backbone, train_image_loader, device)
    test_cache_set = build_feature_cache(backbone, test_image_loader, device)
    logger.info("Feature cache ready: train=%d, test=%d", len(train_cache_set), len(test_cache_set))

    # 训练和评估均基于缓存特征，不再重复过 backbone
    train_loader = DataLoader(
        train_cache_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_cache_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("Start training: backbone=%s, data=%s, class_num=%d",
                args.backbone, args.data_root, class_number)
    logger.info("Config: runs=%d, classifier_iteration=%d", args.runs, args.classifier_iteration)
    logger.info("Log file: %s", log_path)

    run_best_accs = []

    for run_idx in range(1, args.runs + 1):
        classifier = Classifier_Simple(input_feature=feat_dim, class_number=class_number).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)

        best_acc = 0.0
        save_path = os.path.join("model", f"{base_name}-run{run_idx}.pth")
        logger.info("Run %d/%d started", run_idx, args.runs)

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
                test_acc = evaluate(classifier, test_loader, device)
                logger.info(
                    "[Run %d] [Iter %03d/%03d] test_acc=%.4f",
                    run_idx, it + 1, args.classifier_iteration, test_acc
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
                    logger.info("[Run %d] Best model saved: %s", run_idx, save_path)

        run_best_accs.append(best_acc)
        logger.info("[Run %d] best_test_acc=%.4f", run_idx, best_acc)

        # 将保存的模型文件重命名，加入准确率（如 0.6702 → acc6702）
        if best_acc > 0 and os.path.exists(save_path):
            acc_int = int(round(best_acc * 10000))
            new_save_path = os.path.join(
                "model", f"{base_name}-run{run_idx}-acc{acc_int}.pth"
            )
            os.rename(save_path, new_save_path)
            logger.info("[Run %d] Model renamed: %s", run_idx, new_save_path)

    mean_acc = float(np.mean(run_best_accs))
    std_acc  = float(np.std(run_best_accs))
    logger.info("5-run best acc list: %s", [round(x, 4) for x in run_best_accs])
    logger.info("Final result (mean+-std): %.4f+-%.4f", mean_acc, std_acc)
    print(f"Final result (mean+-std): {mean_acc:.4f}+-{std_acc:.4f}")


if __name__ == "__main__":
    main()
