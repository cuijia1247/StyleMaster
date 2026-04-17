"""
预提取 DenseNet169（6 通道 RGB+HSV）全局池化特征，供 ssc_train_densnet169_add.py 中分类器融合使用。
输出维度 1664，与 SscRegDensenet169 骨干输出一致。

用法（项目根目录）::
  python utils/extract_densenet169_6ch_features.py --dataset Painting91

生成::
  pretrainFeatures/{dataset}_densenet169_6ch_train_features.pkl
  pretrainFeatures/{dataset}_densenet169_6ch_test_features.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import DenseNet169_Weights, densenet169
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ssc.utils_add import ToRgbHsv6Tensor  # noqa: E402


class DenseNet169Features6ch(nn.Module):
    """仅 features + ReLU + GAP → 1664 维。"""

    def __init__(self):
        super().__init__()
        net = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        orig = net.features.conv0
        net.features.conv0 = nn.Conv2d(
            6,
            orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False,
        )
        with torch.no_grad():
            net.features.conv0.weight[:, :3] = orig.weight
            net.features.conv0.weight[:, 3:] = orig.weight.mean(dim=1, keepdim=True)
        self.features = net.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.flatten(1)


def build_transform(image_size: int = 224) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
    std = [0.229, 0.224, 0.225, 0.25, 0.25, 0.25]
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            ToRgbHsv6Tensor(),
            transforms.Normalize(mean, std),
        ]
    )


@torch.no_grad()
def extract_split(
    model: nn.Module,
    data_dir: str,
    transform,
    device: torch.device,
    batch_size: int,
) -> dict:
    from pathlib import Path

    model.eval()
    paths = []
    names = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, f))
                names.append(f)
    feat_dict = {}
    batch_t = []
    batch_n = []
    for p, n in tqdm(list(zip(paths, names)), desc=os.path.basename(data_dir)):
        try:
            img = Image.open(p).convert("RGB")
            batch_t.append(transform(img))
            batch_n.append(n)
            if len(batch_t) >= batch_size:
                x = torch.stack(batch_t).to(device)
                y = model(x).cpu()
                for fn, row in zip(batch_n, y):
                    feat_dict[fn] = row
                batch_t, batch_n = [], []
        except Exception as e:
            print(f"skip {p}: {e}")
    if batch_t:
        x = torch.stack(batch_t).to(device)
        y = model(x).cpu()
        for fn, row in zip(batch_n, y):
            feat_dict[fn] = row
    return feat_dict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Painting91")
    p.add_argument(
        "--data_root",
        type=str,
        default="/mnt/codes/data/style/",
        help="含 <dataset>/train 与 <dataset>/test",
    )
    p.add_argument("--out_dir", type=str, default="pretrainFeatures")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224)
    args = p.parse_args()

    os.chdir(_ROOT)
    hub_dir = os.path.join(_ROOT, "pretrainModels", "hub")
    os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)
    os.environ.setdefault("TORCH_HOME", os.path.join(_ROOT, "pretrainModels"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet169Features6ch().to(device)
    transform = build_transform(args.image_size)

    tag = f"{args.dataset}_densenet169_6ch"
    train_dir = os.path.join(args.data_root, args.dataset, "train")
    test_dir = os.path.join(args.data_root, args.dataset, "test")
    os.makedirs(args.out_dir, exist_ok=True)

    for split, sub in [("train", train_dir), ("test", test_dir)]:
        fd = extract_split(model, sub, transform, device, args.batch_size)
        out_path = os.path.join(args.out_dir, f"{tag}_{split}_features.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "feature_dict": fd,
                    "feature_dim": 1664,
                    "num_samples": len(fd),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved {out_path} ({len(fd)} samples)")


if __name__ == "__main__":
    main()
