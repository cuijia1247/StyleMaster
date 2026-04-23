#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 WebStyle 数据集在 train / test / train+test 三个视角下，10 个类别各自的图片数量。

数据根目录 DATA_ROOT 从 remote_sh/run_ssc_train_resnet_bat.sh 中内嵌 runner 的
`DATA_ROOT = '...'` 行解析（与批量训练一致）；子目录名为 `webstyle`。
若解析失败，可显式传入 --data_root。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

# 与 SscDataSet_new.getFeature 中可读图片类型一致（按扩展名计数，不强制解码）
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _project_root() -> Path:
    """experiment_result/Webstyle_analysis.py -> 仓库根目录。"""
    return Path(__file__).resolve().parent.parent


def parse_data_root_from_bat_sh(bat_sh: Path) -> str:
    """
    从 run_ssc_train_resnet_bat.sh 中读取内嵌 Python 里的 DATA_ROOT 赋值。
    匹配形如: DATA_ROOT        = '/mnt/codes/data/style/'
    """
    text = bat_sh.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"^DATA_ROOT\s*=\s*['\"]([^'\"]+)['\"]\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not m:
        raise ValueError(f"无法在 {bat_sh} 中解析 DATA_ROOT 赋值行")
    root = m.group(1).strip().rstrip("/") + "/"
    return root


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES


def count_per_class(split_root: Path, class_ids: List[str]) -> Dict[str, int]:
    """split_root 下每个类别子目录中的图片文件数。"""
    out: Dict[str, int] = {}
    for cid in class_ids:
        d = split_root / cid
        if not d.is_dir():
            out[cid] = 0
            continue
        out[cid] = sum(1 for f in d.iterdir() if _is_image_file(f))
    return out


def discover_class_ids(train_dir: Path, test_dir: Path, expect: int) -> List[str]:
    """
    类别文件夹名与 SscDataset 一致：'1'..'N'。若存在则按数字排序；
    否则合并 train/test 下子目录名后排序。
    """
    def subdir_names(root: Path) -> List[str]:
        if not root.is_dir():
            return []
        return sorted(
            [p.name for p in root.iterdir() if p.is_dir()],
            key=lambda s: (int(s) if s.isdigit() else s, s),
        )

    merged = sorted(
        set(subdir_names(train_dir)) | set(subdir_names(test_dir)),
        key=lambda s: (int(s) if s.isdigit() else s, s),
    )
    if merged:
        return merged
    # 无数据时仍输出 1..expect 占位，便于对照训练配置
    return [str(i) for i in range(1, expect + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="WebStyle 数据集各类图片数量统计")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="覆盖从 shell 解析的数据根目录（应含 webstyle 子目录）",
    )
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default="webstyle",
        help="DATA_ROOT 下的数据集子目录名（默认与 run_ssc_train_resnet_bat.sh 一致）",
    )
    parser.add_argument(
        "--class_num",
        type=int,
        default=10,
        help="期望类别数（仅用于无子目录时的占位行）",
    )
    args = parser.parse_args()

    bat_sh = _project_root() / "remote_sh" / "run_ssc_train_resnet_bat.sh"
    if args.data_root:
        data_root = args.data_root.rstrip("/") + "/"
    else:
        try:
            data_root = parse_data_root_from_bat_sh(bat_sh)
        except (OSError, ValueError) as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)

    dataset_root = Path(data_root) / args.dataset_subdir
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    class_ids = discover_class_ids(train_dir, test_dir, args.class_num)
    train_counts = count_per_class(train_dir, class_ids)
    test_counts = count_per_class(test_dir, class_ids)

    print(f"解析来源: {bat_sh if not args.data_root else '(命令行 --data_root)'}")
    print(f"DATA_ROOT = {data_root}")
    print(f"数据集路径 = {dataset_root}")
    print(f"train 目录存在: {train_dir.is_dir()}  |  test 目录存在: {test_dir.is_dir()}")
    print()

    # 表头
    col_w = max(8, max((len(c) for c in class_ids), default=0) + 2)
    hdr = f"{'类别':<{col_w}}{'train':>10}{'test':>10}{'train+test':>12}"
    print(hdr)
    print("-" * len(hdr))

    sum_tr = sum_te = 0
    for cid in class_ids:
        tr = train_counts.get(cid, 0)
        te = test_counts.get(cid, 0)
        sum_tr += tr
        sum_te += te
        print(f"{cid:<{col_w}}{tr:>10}{te:>10}{tr + te:>12}")

    print("-" * len(hdr))
    print(f"{'合计':<{col_w}}{sum_tr:>10}{sum_te:>10}{sum_tr + sum_te:>12}")


if __name__ == "__main__":
    main()
