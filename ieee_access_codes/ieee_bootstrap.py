#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Painting91 测试集上对比 ST-SACLF 与 Ours (SSC)，对 Accuracy 做 bootstrap（95% CI / p-value）。

用法（项目根目录）::
  python ieee_access_codes/ieee_bootstrap.py

或在代码中调用::
  from ieee_access_codes.ieee_bootstrap import painting91_bootstrap
  result = painting91_bootstrap(ours_base=..., ours_classifier=...)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ST_ADAIN = os.path.join(_ROOT, "ST-SACLF-ncc_main", "pytorch-AdaIN")
_DEFAULT_VGG = os.path.join(_ROOT, "ST-SACLF-ncc_main", "models", "vgg_normalised.pth")
os.environ.setdefault("VGG_NORMALISED_PATH", _DEFAULT_VGG)
if _ST_ADAIN not in sys.path:
    sys.path.insert(0, _ST_ADAIN)

import net  # noqa: E402
from SscDataSet_new import SscDataset  # noqa: E402
from ssc.utils import MultiViewDataInjector, get_ssc_transforms  # noqa: E402
from ssc_train_resnet_copy import parameter_load  # noqa: E402
from utils.pretrainFeatureExtraction import load_dataFeatures  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_TEST_DIR = "/mnt/codes/data/style/Painting91/test"
DEFAULT_TRAIN_DIR = "/mnt/codes/data/style/Painting91/train"
DEFAULT_ST_DECODER = os.path.join(
    _ROOT,
    "ST-SACLF-ncc_main/experiments/adain_decoders/"
    "decoder_iter_10000_Painting91_run1.pth",
)
DEFAULT_OURS_BASE = os.path.join(
    _ROOT,
    "model/ssc-Painting91-SSC-resnet50-2026-07-03-08-54-58-run0-"
    "iteration-0-accuracy-7101-SSC-base-best.pth",
)
DEFAULT_OURS_CLASSIFIER = os.path.join(
    _ROOT,
    "model/ssc-Painting91-SSC-resnet50-2026-07-03-08-54-58-run0-"
    "iteration-0-accuracy-7101-SSC-classifier-best.pth",
)
DEFAULT_PRE_FEATURE = os.path.join(_ROOT, "pretrainFeatures")
DEFAULT_OUT_MD = os.path.join(_ROOT, "ieee_access_paperdata", "ieee_bootstrap.md")

NUM_CLASSES = 13


@dataclass
class PredictionResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    sample_keys: List[str]


@dataclass
class BootstrapSummary:
    point: float
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class PairedBootstrapSummary:
    diff_mean: float
    diff_ci_low: float
    diff_ci_high: float
    p_value: float


@dataclass
class Painting91BootstrapResult:
    """painting91_bootstrap 完整输出。"""
    st_result: PredictionResult
    ours_result: PredictionResult
    st_boot: BootstrapSummary
    ours_boot: BootstrapSummary
    paired: PairedBootstrapSummary
    n_samples: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


class NumericImageFolder(ImageFolder):
    """数字类名目录：标签 0..N-1（与 ST-SACLF train.py 一致）。"""

    def find_classes(self, directory: str):
        names = sorted(
            (d.name for d in Path(directory).iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda x: int(x),
        )
        if not names:
            raise FileNotFoundError(f"未在 {directory} 找到数字类别子目录")
        class_to_idx = {name: int(name) - 1 for name in names}
        return names, class_to_idx


def align_predictions(a: PredictionResult, b: PredictionResult) -> Tuple[PredictionResult, PredictionResult]:
    """按文件名对齐两模型预测。"""
    map_a = {k: (int(t), int(p)) for k, t, p in zip(a.sample_keys, a.y_true, a.y_pred)}
    map_b = {k: (int(t), int(p)) for k, t, p in zip(b.sample_keys, b.y_true, b.y_pred)}
    common = sorted(set(map_a) & set(map_b))
    if not common:
        raise RuntimeError("两模型无公共样本键，无法对齐。")
    if len(common) < len(map_a) or len(common) < len(map_b):
        print(f"  警告: 对齐样本 {len(common)}（ST={len(map_a)}, Ours={len(map_b)}）")

    y_true = np.asarray([map_a[k][0] for k in common], dtype=np.int64)
    y_pred_a = np.asarray([map_a[k][1] for k in common], dtype=np.int64)
    y_pred_b = np.asarray([map_b[k][1] for k in common], dtype=np.int64)
    if not np.array_equal(y_true, [map_b[k][0] for k in common]):
        raise RuntimeError("对齐后两模型 y_true 仍不一致。")

    return (
        PredictionResult(a.name, y_true, y_pred_a, compute_accuracy(y_true, y_pred_a), common),
        PredictionResult(b.name, y_true, y_pred_b, compute_accuracy(y_true, y_pred_b), common),
    )


@torch.no_grad()
def predict_ours(
    base_path: str,
    classifier_path: str,
    data_root: str,
    pre_feature_path: str,
    batch_size: int,
    image_size: int,
) -> PredictionResult:
    data_source = data_root.rstrip("/") + "/"
    feature_path = os.path.join(pre_feature_path, "Painting91_resnet50_test_features.pkl")
    feature_dict = load_dataFeatures(feature_path)

    model = torch.load(base_path, map_location=device).to(device).eval()
    classifier = torch.load(classifier_path, map_location=device).to(device).eval()

    _, _, transform_eval = get_ssc_transforms(
        image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    testset = SscDataset(
        data_source, "test", transform=MultiViewDataInjector([transform_eval, transform_eval])
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    y_true: List[int] = []
    y_pred: List[int] = []
    sample_keys: List[str] = []
    for view1, view2, label, names, _ in testloader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        bb = torch.stack([feature_dict[n] for n in names], dim=0).to(device)
        logits = classifier(model(view1), model(view2), bb)
        y_true.extend((label - 1).long().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        sample_keys.extend(list(names))

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    return PredictionResult(
        "Ours (SSC-ResNet50)", y_true_arr, y_pred_arr,
        compute_accuracy(y_true_arr, y_pred_arr), sample_keys,
    )


def _infer_vgg_feat_dim(vgg_module: nn.Module, sample_loader: DataLoader) -> int:
    with torch.no_grad():
        imgs, _ = next(iter(sample_loader))
        _, g = vgg_module(imgs.to(device))
        return int(g.view(g.size(0), -1).size(1))


def predict_st_sacl(
    decoder_path: str,
    train_dir: str,
    test_dir: str,
    num_classes: int,
    batch_size: int,
    clf_epochs: int,
    clf_lr: float,
    seed: int,
) -> PredictionResult:
    if not os.path.isfile(decoder_path):
        raise FileNotFoundError(f"ST-SACLF decoder 不存在: {decoder_path}")

    set_seed(seed)
    vgg = net.vgg
    decoder = net.decoder
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    vgg.eval().to(device)
    decoder.eval().to(device)

    eval_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_ds = NumericImageFolder(train_dir, transform=eval_tf)
    test_ds = NumericImageFolder(test_dir, transform=eval_tf)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available(),
    )

    feat_dim = _infer_vgg_feat_dim(vgg, train_loader)
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=clf_lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(clf_epochs):
        classifier.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                _, g = vgg(imgs)
                feats = g.view(g.size(0), -1)
            loss = criterion(classifier(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            _, g = vgg(imgs)
            pred = classifier(g.view(g.size(0), -1)).argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().tolist())

    sample_keys = [os.path.basename(p) for p, _ in test_ds.samples]
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    return PredictionResult(
        "ST-SACLF (AdaIN)", y_true_arr, y_pred_arr,
        compute_accuracy(y_true_arr, y_pred_arr), sample_keys,
    )


def bootstrap_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int, seed: int
) -> BootstrapSummary:
    point = compute_accuracy(y_true, y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = compute_accuracy(y_true[idx], y_pred[idx])
    return BootstrapSummary(
        point=point,
        mean=float(samples.mean()),
        ci_low=float(np.percentile(samples, 2.5)),
        ci_high=float(np.percentile(samples, 97.5)),
    )


def bootstrap_paired_accuracy(
    y_true: np.ndarray,
    y_pred_ours: np.ndarray,
    y_pred_st: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> PairedBootstrapSummary:
    """配对 bootstrap：Ours − ST-SACLF 的 Accuracy 差值及双侧 p-value。"""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        acc_ours = compute_accuracy(y_true[idx], y_pred_ours[idx])
        acc_st = compute_accuracy(y_true[idx], y_pred_st[idx])
        diffs[i] = acc_ours - acc_st
    p_left = float(np.mean(diffs <= 0.0))
    p_right = float(np.mean(diffs >= 0.0))
    return PairedBootstrapSummary(
        diff_mean=float(diffs.mean()),
        diff_ci_low=float(np.percentile(diffs, 2.5)),
        diff_ci_high=float(np.percentile(diffs, 97.5)),
        p_value=min(1.0, 2.0 * min(p_left, p_right)),
    )


def write_markdown(
    out_path: str,
    test_dir: str,
    st_decoder: str,
    ours_base: str,
    ours_classifier: str,
    st_result: PredictionResult,
    ours_result: PredictionResult,
    st_boot: BootstrapSummary,
    ours_boot: BootstrapSummary,
    paired: PairedBootstrapSummary,
    n_bootstrap: int,
    seed: int,
    append: bool = False,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(st_result.y_true)
    has_content = append and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    lines = [
        *(["", f"---", ""] if has_content else ["# Bootstrap 对比实验 (Painting91 test)", ""]),
        f"## run — {ts}",
        "",
        f"_test_dir: `{test_dir}`_",
        f"_ST-SACLF decoder: `{st_decoder}`_",
        f"_Ours base: `{ours_base}`_",
        f"_Ours classifier: `{ours_classifier}`_",
        f"_bootstrap: n={n_bootstrap}, seed={seed}_",
        "",
        "### Accuracy 点估计",
        "",
        "| Model | AC | N |",
        "|-------|-----|---|",
        f"| ST-SACLF (AdaIN) | {st_result.accuracy:.4f} | {n} |",
        f"| Ours (SSC-ResNet50) | {ours_result.accuracy:.4f} | {n} |",
        "",
        "### Bootstrap 95% CI（Accuracy）",
        "",
        "| Model | AC | 95% CI |",
        "|-------|-----|--------|",
        f"| ST-SACLF (AdaIN) | {st_boot.point:.4f} | [{st_boot.ci_low:.4f}, {st_boot.ci_high:.4f}] |",
        f"| Ours (SSC-ResNet50) | {ours_boot.point:.4f} | [{ours_boot.ci_low:.4f}, {ours_boot.ci_high:.4f}] |",
        "",
        "### 配对 Bootstrap（Ours − ST-SACLF, Accuracy）",
        "",
        f"- Δ mean: {paired.diff_mean:+.4f}",
        f"- 95% CI: [{paired.diff_ci_low:+.4f}, {paired.diff_ci_high:+.4f}]",
        f"- p-value (two-sided): {paired.p_value:.4f}",
        "",
    ]
    mode = "a" if has_content else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_bootstrap_summary(result: Painting91BootstrapResult, out_md: str | None) -> None:
    st_boot, ours_boot, paired = result.st_boot, result.ours_boot, result.paired
    print(f"ST-SACLF  AC={st_boot.point:.4f} [{st_boot.ci_low:.4f}, {st_boot.ci_high:.4f}]")
    print(f"Ours      AC={ours_boot.point:.4f} [{ours_boot.ci_low:.4f}, {ours_boot.ci_high:.4f}]")
    print(
        f"Δ(Ours−ST)={paired.diff_mean:+.4f} [{paired.diff_ci_low:+.4f}, {paired.diff_ci_high:+.4f}], "
        f"p={paired.p_value:.4f}"
    )
    if out_md:
        print(f"结果已写入: {out_md}")


def painting91_bootstrap(
    test_dir: str = DEFAULT_TEST_DIR,
    train_dir: str = DEFAULT_TRAIN_DIR,
    st_decoder: str = DEFAULT_ST_DECODER,
    ours_base: str = DEFAULT_OURS_BASE,
    ours_classifier: str = DEFAULT_OURS_CLASSIFIER,
    pre_feature_path: str = DEFAULT_PRE_FEATURE,
    num_classes: int = NUM_CLASSES,
    batch_size: int | None = None,
    st_clf_epochs: int = 20,
    st_clf_lr: float = 1e-3,
    n_bootstrap: int = 10000,
    seed: int = 42,
    out_md: str | None = DEFAULT_OUT_MD,
    append_md: bool = False,
    verbose: bool = True,
) -> Painting91BootstrapResult:
    """
    Painting91 test 上对比 ST-SACLF 与 Ours (SSC)，对 Accuracy 做 bootstrap。

    流程：ST-SACLF 预测 → Ours 预测 → 样本对齐 → 单模型 95% CI → 配对 p-value。
    若指定 out_md，则将结果写入 markdown。
    """
    _, default_bs, _, _, image_size, *_ = parameter_load()
    bs = batch_size or default_bs

    if verbose:
        print("Bootstrap: ST-SACLF vs Ours (Painting91 test, AC only)")
        print(f"test_dir={test_dir}, device={device}")

    st_result = predict_st_sacl(
        st_decoder, train_dir, test_dir,
        num_classes, bs, st_clf_epochs, st_clf_lr, seed,
    )
    if verbose:
        print(f"ST-SACLF  AC={st_result.accuracy:.4f}, N={len(st_result.y_true)}")

    ours_result = predict_ours(
        ours_base, ours_classifier,
        str(Path(test_dir).parent), pre_feature_path,
        bs, image_size,
    )
    if verbose:
        print(f"Ours      AC={ours_result.accuracy:.4f}, N={len(ours_result.y_true)}")

    st_result, ours_result = align_predictions(st_result, ours_result)
    if verbose:
        print(f"对齐后 N={len(st_result.y_true)}")

    st_boot = bootstrap_accuracy(st_result.y_true, st_result.y_pred, n_bootstrap, seed + 1)
    ours_boot = bootstrap_accuracy(ours_result.y_true, ours_result.y_pred, n_bootstrap, seed + 2)
    paired = bootstrap_paired_accuracy(
        ours_result.y_true, ours_result.y_pred, st_result.y_pred,
        n_bootstrap, seed + 3,
    )

    result = Painting91BootstrapResult(
        st_result=st_result,
        ours_result=ours_result,
        st_boot=st_boot,
        ours_boot=ours_boot,
        paired=paired,
        n_samples=len(st_result.y_true),
    )

    if out_md:
        write_markdown(
            out_md, test_dir, st_decoder, ours_base, ours_classifier,
            st_result, ours_result, st_boot, ours_boot, paired,
            n_bootstrap, seed, append=append_md,
        )

    if verbose:
        print()
        _print_bootstrap_summary(result, out_md)

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Painting91 test bootstrap: ST-SACLF vs Ours (AC only)")
    p.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR)
    p.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--st_decoder", type=str, default=DEFAULT_ST_DECODER)
    p.add_argument("--ours_base", type=str, default=DEFAULT_OURS_BASE)
    p.add_argument("--ours_classifier", type=str, default=DEFAULT_OURS_CLASSIFIER)
    p.add_argument("--pre_feature_path", type=str, default=DEFAULT_PRE_FEATURE)
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--st_clf_epochs", type=int, default=20)
    p.add_argument("--st_clf_lr", type=float, default=1e-3)
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_md", type=str, default=DEFAULT_OUT_MD)
    p.add_argument("--append_result", action="store_true", help="追加写入 out_md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    painting91_bootstrap(
        test_dir=args.test_dir,
        train_dir=args.train_dir,
        st_decoder=args.st_decoder,
        ours_base=args.ours_base,
        ours_classifier=args.ours_classifier,
        pre_feature_path=args.pre_feature_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        st_clf_epochs=args.st_clf_epochs,
        st_clf_lr=args.st_clf_lr,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        out_md=args.out_md,
        append_md=args.append_result,
        verbose=True,
    )


if __name__ == "__main__":
    main()
