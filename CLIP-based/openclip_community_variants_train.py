# OpenCLIP Community Variants 风格分类（冻结视觉编码器）
# 默认本地权重（pretrainModels/，不联网）：
#   - linear_probe：vit_large_patch16_224.pth（timm ImageNet ViT-L/16，1024 维）
#   - zero_shot：ViT-L-14-openai.pt（OpenCLIP ViT-L-14，768 维）
#
# 用法：
#   python CLIP-based/openclip_community_variants_train.py \
#     --data_root /mnt/codes/data/style/Painting91 --num_classes 13 \
#     --mode linear_probe --runs 3 --dataset_label Painting91

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import cv2
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from ssc.classifier import Classifier

MODEL_FAMILY = "OpenCLIP Community (ViT-H-14)"
DEFAULT_DATA_BASE = "/mnt/codes/data/style/"
DEFAULT_RESULT_MD = os.path.join(
    PROJECT_ROOT, "ieee_access_paperdata", "clip-based_multiple.md"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN_DIR = os.path.join(PROJECT_ROOT, "pretrainModels")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
BATCH_SIZE = 32
CLASSIFIER_LR = 1e-4
CLASSIFIER_EPOCHS = 50
# 默认本地权重（不联网下载）
LOCAL_VIT_PATH = os.path.join(PRETRAIN_DIR, "vit_large_patch16_224.pth")
LOCAL_CLIP_PATH = os.path.join(PRETRAIN_DIR, "ViT-L-14-openai.pt")
TIMM_VIT_NAME = "vit_large_patch16_224"
TIMM_FEAT_DIM = 1024
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_FEAT_DIM = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 224

DATASET_ORDER = ["Painting91", "Pandora", "ArtBench", "FashionStyle14", "Arch"]
DATASET_REL = {
    "Painting91": "Painting91",
    "Pandora": "Pandora",
    "ArtBench": "Artbench",
    "FashionStyle14": "FashionStyle14",
    "Arch": "Arch",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "balanced_accuracy": "Balanced Accuracy",
}

MODE_LABELS = {
    "zero_shot": "Zero-shot",
    "linear_probe": "Linear Probe",
}

# (显示名, model_name, pretrained_tag, feat_dim, 本地权重关键词)
EXPERIMENTS = [
    (
        "ViT-H-14 (laion2b_s32b_b79k)",
        "ViT-H-14",
        "laion2b_s32b_b79k",
        1024,
        ["vit-h-14", "laion2b"],
    ),
    (
        "ViT-g-14 (laion2b_s34b_b88k)",
        "ViT-g-14",
        "laion2b_s34b_b88k",
        1024,
        ["vit-g-14", "laion2b"],
    ),
    (
        "ViT-bigG-14 (laion2b_s39b_b160k)",
        "ViT-bigG-14",
        "laion2b_s39b_b160k",
        1280,
        ["vit-bigg-14", "laion2b"],
    ),
]

# 零样本 prompt 模板（{} 为 1-based 类别编号，与数据目录 class_id 一致）
ZERO_SHOT_TEMPLATES = [
    "a photo of an artwork in style category {}.",
    "a painting in artistic style {}.",
    "an image of visual style {}.",
]

os.makedirs(PRETRAIN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class TimmVisionEncoder(nn.Module):
    """timm ViT 视觉骨干；提供 encode_image 接口供特征提取使用。"""

    def __init__(self, model_name: str, checkpoint_path: str):
        super().__init__()
        import timm

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"未找到本地 ViT 权重: {checkpoint_path}")
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.feat_dim = self.model.num_features

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RunMetrics(TypedDict):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float


@dataclass
class DatasetResult:
    name: str
    num_classes: int
    data_root: str
    all_runs: list[RunMetrics]


class _FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(log_filename: str) -> logging.Logger:
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    lg = logging.getLogger(f"oc_community_train.{log_filename}")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    lg.handlers.clear()
    sh = _FlushStreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    fh = logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg


logger = logging.getLogger("oc_community_train")


def find_local_weight(keywords: list[str]) -> str | None:
    for fname in os.listdir(PRETRAIN_DIR):
        lower = fname.lower()
        if all(kw.lower() in lower for kw in keywords):
            path = os.path.join(PRETRAIN_DIR, fname)
            logger.info("在 pretrainModels/ 中找到已有权重: %s", path)
            return path
    return None


def resolve_experiment(model_name: str | None) -> tuple[str, str, str, int, list[str]]:
    if model_name:
        for exp in EXPERIMENTS:
            if exp[1] == model_name:
                return exp
        raise ValueError(f"未知 model_name={model_name}，可选: {[e[1] for e in EXPERIMENTS]}")
    return EXPERIMENTS[0]


class StyleDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        image_size: int = IMAGE_SIZE,
        norm_type: str = "clip",
    ):
        if norm_type == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = CLIP_MEAN, CLIP_STD
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.samples: list[tuple[str, int]] = []
        skipped = 0
        split_dir = os.path.join(data_root, split)
        class_ids = sorted(int(d) for d in os.listdir(split_dir) if d.isdigit())
        for cid in class_ids:
            cls_dir = os.path.join(split_dir, str(cid))
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                if cv2.imread(path, cv2.IMREAD_COLOR) is None:
                    logger.warning("跳过损坏图片: %s", path)
                    skipped += 1
                    continue
                self.samples.append((path, cid - 1))
        if skipped:
            logger.warning("共跳过 %d 张损坏图片", skipped)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), label


def metrics_from_arrays(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> RunMetrics:
    if not y_true:
        return RunMetrics(
            accuracy=0.0, macro_f1=0.0, weighted_f1=0.0, balanced_accuracy=0.0
        )
    labels_arr = np.asarray(y_true, dtype=np.int64)
    preds_arr = np.asarray(y_pred, dtype=np.int64)
    labels_all = list(range(num_classes))
    return RunMetrics(
        accuracy=float(np.mean(labels_arr == preds_arr)),
        macro_f1=float(
            f1_score(
                labels_arr, preds_arr, average="macro", labels=labels_all, zero_division=0
            )
        ),
        weighted_f1=float(
            f1_score(
                labels_arr,
                preds_arr,
                average="weighted",
                labels=labels_all,
                zero_division=0,
            )
        ),
        balanced_accuracy=float(balanced_accuracy_score(labels_arr, preds_arr)),
    )


def load_frozen_openclip(
    model_name: str, pretrained_tag: str, keywords: list[str]
) -> tuple[nn.Module, object, int, str]:
    local_path = find_local_weight(keywords)
    pretrained = local_path if local_path else pretrained_tag
    encoder, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(model_name)
    feat_dim = {e[1]: e[3] for e in EXPERIMENTS}.get(model_name, CLIP_FEAT_DIM)
    label = f"{model_name} ({'local' if local_path else pretrained_tag})"
    logger.info(
        "OpenCLIP 编码器: %s, 参数量 %.2f B, 冻结=True",
        label,
        sum(p.numel() for p in encoder.parameters()) / 1e9,
    )
    return encoder, tokenizer, feat_dim, label


def load_encoder_for_mode(
    mode: str,
    pretrained_path: str,
    clip_pretrained_path: str,
    model_name: str | None,
) -> tuple[nn.Module, object | None, int, str, str]:
    """
    返回 (encoder, tokenizer, feat_dim, backbone_label, norm_type)。
    默认：linear_probe → timm vit_large_patch16_224.pth；
         zero_shot   → ViT-L-14-openai.pt（OpenCLIP）。
    """
    if model_name:
        _, mn, tag, feat_dim, keywords = resolve_experiment(model_name)
        encoder, tokenizer, feat_dim, label = load_frozen_openclip(mn, tag, keywords)
        return encoder, tokenizer, feat_dim, label, "clip"

    if mode == "linear_probe":
        encoder = TimmVisionEncoder(TIMM_VIT_NAME, pretrained_path).to(DEVICE)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        label = f"vit_large_patch16_224 (local: {pretrained_path})"
        logger.info(
            "timm ViT 编码器: %s, feat_dim=%d, 冻结=True",
            label,
            encoder.feat_dim,
        )
        return encoder, None, encoder.feat_dim, label, "imagenet"

    if not os.path.isfile(clip_pretrained_path):
        raise FileNotFoundError(
            f"zero_shot 需要 OpenCLIP 权重，未找到: {clip_pretrained_path}"
        )
    encoder, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=clip_pretrained_path, device=DEVICE
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    label = f"ViT-L-14 (local: {clip_pretrained_path})"
    logger.info(
        "OpenCLIP 编码器: %s, 参数量 %.2f B, 冻结=True",
        label,
        sum(p.numel() for p in encoder.parameters()) / 1e9,
    )
    return encoder, tokenizer, CLIP_FEAT_DIM, label, "clip"


def load_frozen_encoder(model_name: str, pretrained_tag: str, keywords: list[str]):
    """兼容旧接口；优先使用 load_encoder_for_mode。"""
    encoder, tokenizer, _, _ = load_frozen_openclip(model_name, pretrained_tag, keywords)
    return encoder, tokenizer


@torch.no_grad()
def build_zero_shot_classifier(
    encoder: nn.Module, tokenizer, num_classes: int
) -> torch.Tensor:
    """每类对多模板文本特征取均值，得到 (num_classes, feat_dim) 权重。"""
    class_feats = []
    for cid in range(num_classes):
        texts = [tmpl.format(cid + 1) for tmpl in ZERO_SHOT_TEMPLATES]
        tokens = tokenizer(texts).to(DEVICE)
        feats = encoder.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        class_feats.append(feats.mean(dim=0))
    weights = torch.stack(class_feats, dim=0)
    return weights / weights.norm(dim=-1, keepdim=True)


@torch.no_grad()
def zero_shot_eval(
    encoder: nn.Module,
    tokenizer,
    loader: DataLoader,
    num_classes: int,
) -> RunMetrics:
    text_weights = build_zero_shot_classifier(encoder, tokenizer, num_classes)
    y_true: list[int] = []
    y_pred: list[int] = []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        img_feats = encoder.encode_image(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feats @ text_weights.T
        pred = logits.argmax(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.cpu().tolist())
    return metrics_from_arrays(y_true, y_pred, num_classes)


@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        feats = encoder.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


@torch.no_grad()
def evaluate_classifier_metrics(
    classifier: Classifier,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
) -> RunMetrics:
    classifier.eval()
    loader = DataLoader(
        torch.utils.data.TensorDataset(test_feats, test_labels), batch_size=BATCH_SIZE
    )
    y_true, y_pred = [], []
    for feats, labels in loader:
        feats = feats.to(DEVICE)
        pred = classifier(feats).argmax(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.cpu().tolist())
    return metrics_from_arrays(y_true, y_pred, num_classes)


def train_classifier_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    feat_dim: int,
    num_classes: int,
) -> tuple[Classifier, RunMetrics]:
    classifier = Classifier(feat_dim, num_classes).to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr=CLASSIFIER_LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CLASSIFIER_EPOCHS)
    loader = DataLoader(
        torch.utils.data.TensorDataset(train_feats, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_metrics = RunMetrics(
        accuracy=0.0, macro_f1=0.0, weighted_f1=0.0, balanced_accuracy=0.0
    )
    best_state = None
    for epoch in range(CLASSIFIER_EPOCHS):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += len(labels)
        scheduler.step()
        metrics = evaluate_classifier_metrics(
            classifier, test_feats, test_labels, num_classes
        )
        logger.info(
            "Epoch [%d/%d] loss=%.4f acc=%.4f macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f",
            epoch + 1,
            CLASSIFIER_EPOCHS,
            total_loss / max(total, 1),
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
            metrics["balanced_accuracy"],
        )
        if metrics["accuracy"] > best_metrics["accuracy"]:
            best_metrics = metrics
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    if best_state is not None:
        classifier.load_state_dict(best_state)
    return classifier, best_metrics


def _format_mean_std(values: list[float]) -> str:
    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        return "FAILED" if values else "-"
    arr = np.asarray(valid, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{mean_v:.4f}±{std_v:.4f}"


def _run_cell_value(all_runs: list[RunMetrics], run_idx: int, metric_key: str) -> str:
    if run_idx >= len(all_runs):
        return "-"
    v = all_runs[run_idx][metric_key]
    if np.isnan(v):
        return "FAILED"
    return f"{v:.4f}"


def _format_metric_table_block(
    metric_title: str, results: list[DatasetResult], metric_key: str, runs: int
) -> list[str]:
    run_headers = [f"run{i}" for i in range(1, runs + 1)]
    lines = [
        f"### {metric_title}",
        "",
        "| Dataset | num_classes | "
        + " | ".join(run_headers)
        + " | mean±std | data_root |",
        "|" + "|".join(["---------"] * (4 + runs)) + "|",
    ]
    for result in results:
        values = [m[metric_key] for m in result.all_runs]
        run_cells = [_run_cell_value(result.all_runs, i, metric_key) for i in range(runs)]
        lines.append(
            f"| {result.name} | {result.num_classes} | "
            + " | ".join(run_cells)
            + f" | {_format_mean_std(values)} | `{result.data_root}` |"
        )
    lines.append("")
    return lines


def _format_summary_table(results: list[DatasetResult]) -> list[str]:
    metric_titles = list(METRIC_LABELS.values())
    lines = [
        "## 汇总总表",
        "",
        "| Dataset | num_classes | " + " | ".join(metric_titles) + " |",
        "|---------|-------------|" + "|".join(["---------"] * len(metric_titles)) + "|",
    ]
    for result in results:
        cells = [_format_mean_std([m[k] for m in result.all_runs]) for k in METRIC_LABELS]
        lines.append(
            f"| {result.name} | {result.num_classes} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    return lines


def write_result_markdown(
    result_path: str,
    results: list[DatasetResult],
    data_base: str,
    mode: str,
    backbone: str,
    classifier_epochs: int,
    runs: int,
    completed_runs: int | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    argv_summary = " ".join([sys.argv[0]] + sys.argv[1:])
    dataset_names = ", ".join(r.name for r in results)
    done = completed_runs if completed_runs is not None else max(
        (len(r.all_runs) for r in results), default=0
    )
    progress = f", completed={done}/{runs}" if done < runs else ""
    mode_label = MODE_LABELS.get(mode, mode)
    epoch_part = (
        f"classifier_epochs={classifier_epochs}, "
        if mode == "linear_probe"
        else ""
    )
    lines = [
        f"# CLIP-based 多数据集多次实验",
        "",
        f"## {mode_label} benchmark ({dataset_names}) "
        f"(backbone={backbone}, {epoch_part}runs={runs}{progress}) — {timestamp}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
    ]
    for metric_key, metric_title in METRIC_LABELS.items():
        lines.extend(_format_metric_table_block(metric_title, results, metric_key, runs))
    lines.extend(_format_summary_table(results))
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_run_markdown(
    result_md: str,
    dataset_name: str,
    num_classes: int,
    data_root: str,
    all_runs: list[RunMetrics],
    mode: str,
    backbone: str,
    total_runs: int,
) -> None:
    data_base = os.path.dirname(data_root.rstrip("/")) + os.sep
    results = [
        DatasetResult(
            name=dataset_name,
            num_classes=num_classes,
            data_root=data_root.rstrip("/"),
            all_runs=all_runs,
        )
    ]
    write_result_markdown(
        result_md,
        results,
        data_base,
        mode,
        backbone,
        CLASSIFIER_EPOCHS,
        total_runs,
        completed_runs=len(all_runs),
    )


def merge_batch_partials(
    partial_dir: str,
    merge_result_md: str,
    runs: int,
    mode: str,
    data_base: str = DEFAULT_DATA_BASE,
    backbone: str = "ViT-H-14",
) -> None:
    if not data_base.endswith("/"):
        data_base += "/"

    suffix = f"_{mode}.md"
    partials: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(partial_dir, f"*{suffix}"))):
        name = os.path.splitext(os.path.basename(path))[0][: -len(f"_{mode}")]
        with open(path, encoding="utf-8") as f:
            partials[name] = f.read()

    classifier_epochs = CLASSIFIER_EPOCHS if mode == "linear_probe" else 0
    mode_label = MODE_LABELS.get(mode, mode)
    epoch_part = (
        f"classifier_epochs={classifier_epochs}, "
        if mode == "linear_probe"
        else ""
    )

    def extract_table_row(text: str, section: str) -> str:
        pat = rf"### {re.escape(section)}\s*\n\n(\|.+\|\n\|[-| ]+\|\n)(\|.+\|)"
        hit = re.search(pat, text)
        return hit.group(2).strip() if hit else ""

    def extract_summary_row(text: str) -> str:
        pat = r"## 汇总总表\s*\n\n(\|.+\|\n\|[-| ]+\|\n)(\|.+\|)"
        hit = re.search(pat, text)
        return hit.group(2).strip() if hit else ""

    metric_sections = list(METRIC_LABELS.values())
    run_headers = [f"run{i}" for i in range(1, runs + 1)]
    lines = [
        f"# CLIP-based 多数据集多次实验",
        "",
        f"## {mode_label} benchmark ({', '.join(DATASET_ORDER)}) "
        f"(backbone={backbone}, {epoch_part}runs={runs}) — "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        f"_命令: `./CLIP-based/run_openclip_train_bat.sh` → "
        f"`openclip_community_variants_train.py` mode={mode} × {len(DATASET_ORDER)} 数据集_",
        "",
    ]
    for section in metric_sections:
        lines += [
            f"### {section}",
            "",
            "| Dataset | num_classes | "
            + " | ".join(run_headers)
            + " | mean±std | data_root |",
            "|" + "|".join(["---------"] * (4 + runs)) + "|",
        ]
        for ds in DATASET_ORDER:
            row = extract_table_row(partials.get(ds, ""), section)
            if row:
                lines.append(row)
            else:
                failed = " | ".join(["FAILED"] * runs)
                lines.append(
                    f"| {ds} | ? | {failed} | FAILED | `{data_base}{DATASET_REL.get(ds, ds)}` |"
                )
        lines.append("")

    lines += [
        "## 汇总总表",
        "",
        "| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |",
        "|---------|-------------|---------|---------|---------|---------|",
    ]
    for ds in DATASET_ORDER:
        row = extract_summary_row(partials.get(ds, ""))
        lines.append(row if row else f"| {ds} | ? | FAILED | FAILED | FAILED | FAILED |")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(merge_result_md)), exist_ok=True)
    with open(merge_result_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def merge_all_modes_to_file(
    partial_dir: str,
    merge_result_md: str,
    runs: int,
    data_base: str,
    backbone_by_mode: dict[str, str] | None = None,
) -> None:
    """合并 zero_shot 与 linear_probe 两类 partial → 单一 Markdown。"""
    if backbone_by_mode is None:
        backbone_by_mode = {
            "zero_shot": "ViT-L-14 (local)",
            "linear_probe": "vit_large_patch16_224 (local)",
        }
    blocks: list[str] = []
    for i, mode in enumerate(("zero_shot", "linear_probe")):
        tmp_md = f"{merge_result_md}.{mode}.part"
        merge_batch_partials(
            partial_dir,
            tmp_md,
            runs,
            mode,
            data_base,
            backbone_by_mode[mode],
        )
        with open(tmp_md, encoding="utf-8") as f:
            text = f.read().strip()
        os.remove(tmp_md)
        if i == 0:
            blocks.append(text)
        else:
            # 去掉第二个文件的重复总标题行
            blocks.append("\n".join(text.splitlines()[2:]))
    os.makedirs(os.path.dirname(os.path.abspath(merge_result_md)), exist_ok=True)
    with open(merge_result_md, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")


def run_once(
    encoder,
    tokenizer,
    args: argparse.Namespace,
    feat_dim: int,
    exp_display: str,
    norm_type: str,
) -> RunMetrics:
    train_set = StyleDataset(args.data_root, args.train, norm_type=norm_type)
    test_set = StyleDataset(args.data_root, args.test, norm_type=norm_type)
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    num_classes = args.num_classes

    if args.mode == "zero_shot":
        logger.info("Zero-shot 评测（冻结 CLIP，无训练）")
        return zero_shot_eval(encoder, tokenizer, test_loader, num_classes)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    logger.info("Linear Probe：提取特征后训练分类头 epochs=%d", CLASSIFIER_EPOCHS)
    train_feats, train_labels = extract_features(encoder, train_loader)
    test_feats, test_labels = extract_features(encoder, test_loader)
    classifier, metrics = train_classifier_probe(
        train_feats, train_labels, test_feats, test_labels, feat_dim, num_classes
    )
    if args.save_classifier:
        dataset_name = os.path.basename(args.data_root.rstrip("/"))
        tag = exp_display.replace("-", "").replace(" ", "")
        save_name = (
            f"oc-{dataset_name}-{tag}-{args.mode}-acc{metrics['accuracy']:.4f}.pth"
        )
        torch.save(classifier.state_dict(), os.path.join(MODEL_DIR, save_name))
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenCLIP Community Variants 风格分类（zero-shot / linear_probe）"
    )
    p.add_argument("--data_root", type=str, default="/mnt/codes/data/style/Painting91")
    p.add_argument("--train", type=str, default="train")
    p.add_argument("--test", type=str, default="test")
    p.add_argument("--num_classes", type=int, default=13)
    p.add_argument(
        "--mode",
        type=str,
        choices=["zero_shot", "linear_probe"],
        required=True,
        help="zero_shot=零样本；linear_probe=冻结 encoder 训练分类头",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="可选：OpenCLIP 社区模型名（ViT-H-14 等）；默认用本地 vit_large + ViT-L-14",
    )
    p.add_argument(
        "--pretrained_path",
        type=str,
        default=LOCAL_VIT_PATH,
        help="linear_probe 本地 timm ViT 权重（默认 pretrainModels/vit_large_patch16_224.pth）",
    )
    p.add_argument(
        "--clip_pretrained_path",
        type=str,
        default=LOCAL_CLIP_PATH,
        help="zero_shot 本地 OpenCLIP 权重（默认 pretrainModels/ViT-L-14-openai.pt）",
    )
    p.add_argument("--result_md", type=str, default=DEFAULT_RESULT_MD)
    p.add_argument("--merge_result_md", type=str, default=None)
    p.add_argument("--partial_dir", type=str, default=None)
    p.add_argument("--dataset_label", type=str, default=None)
    p.add_argument("--data_base", type=str, default=DEFAULT_DATA_BASE)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument(
        "--save_classifier",
        action="store_true",
        help="linear_probe 模式下保存最佳分类头",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("TORCH_HOME", PRETRAIN_DIR)

    if args.runs < 1:
        raise SystemExit("错误: --runs 须 >= 1")

    data_root = os.path.abspath(args.data_root.rstrip("/"))
    dataset_name = args.dataset_label or os.path.basename(data_root)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"oc-{dataset_name}-{args.mode}-{timestamp}.log"

    global logger
    logger = setup_logger(log_filename)
    logger.info("Device: %s", DEVICE)
    logger.info("Mode: %s", args.mode)
    logger.info(
        "Local weights: vit=%s  clip=%s",
        args.pretrained_path,
        args.clip_pretrained_path,
    )
    logger.info("Dataset: %s  classes=%d", data_root, args.num_classes)

    all_runs: list[RunMetrics] = []
    backbone_label = "unknown"
    for r in range(1, args.runs + 1):
        logger.info("========== run %d/%d ==========", r, args.runs)
        encoder, tokenizer, feat_dim, backbone_label, norm_type = load_encoder_for_mode(
            args.mode,
            args.pretrained_path,
            args.clip_pretrained_path,
            args.model_name,
        )
        exp_display = backbone_label
        try:
            metrics = run_once(
                encoder, tokenizer, args, feat_dim, exp_display, norm_type
            )
            all_runs.append(metrics)
            print(
                f"[{dataset_name} {args.mode} run{r}/{args.runs}] "
                f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}, "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )
        except Exception:
            logger.exception("run %d failed", r)
            all_runs.append(
                RunMetrics(
                    accuracy=float("nan"),
                    macro_f1=float("nan"),
                    weighted_f1=float("nan"),
                    balanced_accuracy=float("nan"),
                )
            )
            print(f"[{dataset_name} {args.mode} run{r}/{args.runs}] FAILED")
        finally:
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _save_run_markdown(
            args.result_md,
            dataset_name,
            args.num_classes,
            data_root,
            all_runs,
            args.mode,
            backbone_label,
            args.runs,
        )
        print(f"结果已更新: {args.result_md} ({len(all_runs)}/{args.runs} runs)")


if __name__ == "__main__":
    main()
