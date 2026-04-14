import argparse
import logging
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms

from ssc.classifier import Classifier


def build_backbone(device: torch.device) -> nn.Module:
    """构建 VGG16 预训练特征提取器（输出 4096 维特征）。"""
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(
        vgg.features,
        vgg.avgpool,
        nn.Flatten(),
        *list(vgg.classifier.children())[:-1],  # 去掉最后分类层，保留 4096 维 embedding
    )
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


def build_base_name(data_root: str) -> str:
    """构建统一命名基名：traditional-vgg16-数据集名-年-月-日-时-分-秒"""
    dataset_name = os.path.basename(os.path.normpath(data_root))
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return f"traditional-vgg16-{dataset_name}-{time_str}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Traditional training with VGG16 + Classifier")
    parser.add_argument("--data_root", type=str, default="/mnt/codes/data/style/Painting91")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--classifier_lr", type=float, default=3e-4)
    parser.add_argument("--classifier_iteration", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("model", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    base_name = build_base_name(args.data_root)
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
            transforms.Resize((224, 224)),
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

    class_number = len(train_set.classes)
    backbone = build_backbone(device)
    logger.info("Extracting train/test features once with frozen VGG16 backbone...")
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

    logger.info("Start training: data=%s, class_num=%d", args.data_root, class_number)
    logger.info("Config: runs=%d, classifier_iteration=%d", args.runs, args.classifier_iteration)
    logger.info("Log file: %s", log_path)

    run_best_accs = []

    for run_idx in range(1, args.runs + 1):
        classifier = Classifier(input_feature=4096, class_number=class_number).to(device)
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

    min_acc = float(np.min(run_best_accs))
    std_acc = float(np.std(run_best_accs))
    logger.info("5-run best acc list: %s", [round(x, 4) for x in run_best_accs])
    logger.info("Final result (min+-std): %.4f+-%.4f", min_acc, std_acc)
    print(f"Final result (min+-std): {min_acc:.4f}+-{std_acc:.4f}")


if __name__ == "__main__":
    main()
