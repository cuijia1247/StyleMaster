# Author: cuijia1247
# Date: 2025-4-27
# version: 1.0
# 基于 barlowtwins_train.py，使用 VICReg 自监督预训练 + SSC 风格分类器
"""
用法:
    python vicreg_train.py --benchmark_all --epochs 120 --runs 3
    python vicreg_train.py --data_root /mnt/codes/data/style/Pandora --num_classes 12

后台运行（SSH 断开仍继续）:
    ./selfsupervised/run_vicreg_train.sh
    ./selfsupervised/manage_vicreg_train.sh status    # 查看进度
    ./selfsupervised/manage_vicreg_train.sh tail      # 实时日志
    ./selfsupervised/manage_vicreg_train.sh stop      # 停止
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

from vicreg.vicreg import VICReg, vicreg_loss_fun
from ssc.utils import get_byol_transforms, MultiViewDataInjector
from SscDataSet import SscDataset
from ssc.classifier import Classifier

MODEL_NAME = "VICReg"
DEFAULT_DATA_BASE = "/mnt/codes/data/style"

BENCHMARK_DATASETS: list[tuple[str, int, str]] = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("ArtBench", 10, "artbench-10-imagefolder-split"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
]

METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "balanced_accuracy": "Balanced Accuracy",
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@dataclass
class TrainParams:
    """训练超参默认值（修改此处即可全局生效）。"""
    epochs: int = 300
    ssc_input: int = 2048
    ssc_output: int = 2048
    batch_size: int = 64
    offset_bs: int = 512
    base_lr: float = 0.008
    image_size: int = 64
    classifier_iteration: int = 120
    classifier_lr: float = 0.001
    classifier_training_gap: int = 50
    classifier_test_gap: int = 15
    model_name: str = ""
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    vicreg_optimizer_lr: float = 3e-4
    vicreg_weight_decay: float = 1.5e-6


def parameter_load() -> TrainParams:
    return TrainParams()


def _compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> RunMetrics:
    if not y_true:
        return RunMetrics(
            accuracy=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            balanced_accuracy=0.0,
        )
    labels_arr = np.asarray(y_true, dtype=np.int64)
    preds_arr = np.asarray(y_pred, dtype=np.int64)
    labels_all = list(range(num_classes))
    return RunMetrics(
        accuracy=float(np.mean(labels_arr == preds_arr)),
        macro_f1=float(
            f1_score(
                labels_arr,
                preds_arr,
                average="macro",
                labels=labels_all,
                zero_division=0,
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


@torch.no_grad()
def evaluate_test_metrics(
    model: nn.Module,
    classifier: nn.Module,
    resnet50: nn.Module,
    testloader,
    num_classes: int,
) -> RunMetrics:
    """在测试集上计算四项指标。"""
    model.eval()
    classifier.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for view1, view2, label, name, original in testloader:
        view1 = view1.to(device).detach()
        view2 = view2.to(device).detach()
        z1, z2 = model.forward(view1, view2)
        original = original.to(device)
        backbone_view = resnet50(original)
        test_feat = (backbone_view - z1) + (backbone_view - z2)
        prediction = classifier(test_feat)
        labels = (label - 1).cpu().tolist()
        preds = prediction.argmax(dim=1).cpu().tolist()
        y_true.extend(labels)
        y_pred.extend(preds)
    return _compute_metrics(y_true, y_pred, num_classes)


def vicreg_train(
    logger,
    model_path,
    current_time,
    opt_model_name,
    dataset,
    class_number,
    epochs: int | None = None,
    save_models: bool = True,
    params: TrainParams | None = None,
) -> RunMetrics:
    logger.debug(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    logger.debug("THIS IS THE FORMAL TRAINING PROCESS")
    logger.debug(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    logger.info("vicreg parameter setting up...")
    p = params or parameter_load()
    epochs_ = epochs if epochs is not None else p.epochs
    model_name_ = opt_model_name or p.model_name

    logger.info("dataset = %s", dataset)
    logger.info("epochs = %d", epochs_)
    logger.info("batch_size = %d", p.batch_size)
    logger.info("vicreg learning rate = %f", p.base_lr)
    logger.info("classifier training gap = %d", p.classifier_training_gap)
    logger.info("classifier test gap = %d", p.classifier_test_gap)
    logger.info("classifier iteration is %d", p.classifier_iteration)
    logger.info("classifier learning rate = %f", p.classifier_lr)
    logger.info(
        "vicreg loss coeffs: sim=%f std=%f cov=%f",
        p.sim_coeff,
        p.std_coeff,
        p.cov_coeff,
    )
    logger.info("model name is %s", model_name_)

    transformT, transformT1, transformEvalT = get_byol_transforms(
        p.image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )

    dataSource = dataset
    trainset = SscDataset(
        dataSource, "train", transform=MultiViewDataInjector([transformT, transformT1])
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=p.batch_size, shuffle=False
    )
    testset = SscDataset(
        dataSource, "test", transform=MultiViewDataInjector([transformT, transformT1])
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p.batch_size, shuffle=False
    )
    if len(trainset) == 0 or len(testset) == 0:
        raise ValueError(
            f"数据集为空 (train={len(trainset)}, test={len(testset)})，请检查路径: {dataSource}"
        )
    logger.info("vicreg %s is ready (train=%d, test=%d)...", dataSource, len(trainset), len(testset))

    model = VICReg(p.ssc_input, p.ssc_output)
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(p.ssc_input, p.ssc_output)
    resnet50 = resnet50.eval()
    model = model.to(device)
    resnet50 = resnet50.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=p.vicreg_optimizer_lr, weight_decay=p.vicreg_weight_decay
    )
    logger.info("vicreg model is ready...")

    time_str = current_time
    best_metrics = RunMetrics(
        accuracy=0.0,
        macro_f1=0.0,
        weighted_f1=0.0,
        balanced_accuracy=0.0,
    )
    last_accuracy = 0.0
    for epoch in range(epochs_):
        model.train()
        train_loss = []
        for view1, view2, label, name, _ in trainloader:
            view1 = view1.to(device)
            view2 = view2.to(device)
            z1, z2 = model.forward(view1, view2)
            loss = vicreg_loss_fun(
                z1, z2, p.sim_coeff, p.std_coeff, p.cov_coeff
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        if epoch % 10 == 0 or epoch == epochs_ - 1:
            logger.info(
                "The epoch is %d, vicreg train loss is %f", epoch, np.mean(train_loss)
            )

        if (
            epoch % p.classifier_training_gap == 0
            and epoch != 0
            or epoch == epochs_ - 1
        ):
            classifier = Classifier(p.ssc_output, class_number).cuda()
            classifier_criterion = nn.CrossEntropyLoss()
            classifier_optimizer = torch.optim.Adam(
                classifier.parameters(), lr=p.classifier_lr
            )
            style_loss = torch.zeros(1).cuda()
            model.eval()
            for i in range(p.classifier_iteration):
                total_correct = 0.0
                tk1 = trainloader
                tk2 = testloader
                for view1, view2, label, name, original in tk1:
                    correct = 0.0
                    view1 = view1.to(device).detach()
                    view2 = view2.to(device).detach()
                    z1, z2 = model.forward(view1, view2)
                    original = original.to(device)
                    backbone_view = resnet50(original)
                    test1 = backbone_view - z1
                    test2 = backbone_view - z2
                    test = test1 + test2
                    prediction = classifier(test)
                    label = label - 1
                    label = Variable(label).cuda()
                    style_loss = classifier_criterion(prediction, label)
                    classifier_optimizer.zero_grad()
                    style_loss.backward()
                    classifier_optimizer.step()
                    pred = prediction.data.max(1, keepdim=True)[1]
                    correct += pred.eq(label.data.view_as(pred)).cpu().sum()
                    total_correct += correct
                if i % 20 == 19:
                    logger.info(
                        "The classifer-train round is %d, the training accuracy is %d/%d",
                        i,
                        total_correct,
                        len(trainset),
                    )
                if i % p.classifier_test_gap == p.classifier_test_gap - 1:
                    run_metrics = evaluate_test_metrics(
                        model, classifier, resnet50, tk2, class_number
                    )
                    test_accuracy = run_metrics["accuracy"]
                    last_accuracy = test_accuracy
                    if test_accuracy > best_metrics["accuracy"]:
                        best_metrics = run_metrics
                        if save_models:
                            lt_classifier_name = (
                                model_name_
                                + "-SSC-resnet50-"
                                + time_str
                                + "-vicreg-classifier-best.pth"
                            )
                            lt_base_name = (
                                model_name_
                                + "-SSC-resnet50-"
                                + time_str
                                + "-vicreg-base-best.pth"
                            )
                            torch.save(model, model_path + lt_base_name)
                            torch.save(classifier, model_path + lt_classifier_name)
                        logger.info(
                            "+++THE BEST MODEL updated+++. acc=%.4f macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f",
                            run_metrics["accuracy"],
                            run_metrics["macro_f1"],
                            run_metrics["weighted_f1"],
                            run_metrics["balanced_accuracy"],
                        )
                    logger.info(
                        "Test result: round=%d acc=%.4f macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f",
                        i,
                        run_metrics["accuracy"],
                        run_metrics["macro_f1"],
                        run_metrics["weighted_f1"],
                        run_metrics["balanced_accuracy"],
                    )
            if epoch == epochs_ - 1 and save_models:
                lt_classifier_name = (
                    model_name_
                    + "-vicreg-resnet50-"
                    + time_str
                    + "-SSC-classifier-last.pth"
                )
                lt_base_name = (
                    model_name_ + "-vicreg-resnet50-" + time_str + "-SSC-base-last.pth"
                )
                torch.save(model, model_path + lt_base_name)
                torch.save(classifier, model_path + lt_classifier_name)
                logger.info(
                    "The last models are saved. The last accuracy is %f", last_accuracy
                )
    logger.info(
        "Best metrics: acc=%.4f macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f",
        best_metrics["accuracy"],
        best_metrics["macro_f1"],
        best_metrics["weighted_f1"],
        best_metrics["balanced_accuracy"],
    )
    return best_metrics


def _make_logger(log_name: str) -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    os.makedirs(os.path.join(get_project_root(), "log"), exist_ok=True)
    filehandler = logging.FileHandler(os.path.join(get_project_root(), "log", log_name))
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_v, std_v


def _format_mean_std(values: list[float]) -> str:
    if not values or all(np.isnan(v) for v in values):
        return "FAILED"
    mean_v, std_v = compute_mean_std(values)
    return f"{mean_v:.4f}±{std_v:.4f}"


def _format_metric_table_block(
    metric_title: str,
    results: list[DatasetResult],
    metric_key: str,
    runs: int,
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
        values = [run_metrics[metric_key] for run_metrics in result.all_runs]
        run_cells = [
            f"{v:.4f}" if not np.isnan(v) else "FAILED" for v in values
        ]
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
        "| Dataset | num_classes | "
        + " | ".join(metric_titles)
        + " |",
        "|---------|-------------|"
        + "|".join(["---------"] * len(metric_titles))
        + "|",
    ]
    for result in results:
        cells = []
        for metric_key in METRIC_LABELS:
            values = [run_metrics[metric_key] for run_metrics in result.all_runs]
            cells.append(_format_mean_std(values))
        lines.append(
            f"| {result.name} | {result.num_classes} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    return lines


def _format_dataset_metric_row(
    result: DatasetResult, metric_key: str, runs: int
) -> str:
    """生成单个数据集在某指标表中的一行。"""
    values = [run_metrics[metric_key] for run_metrics in result.all_runs]
    run_cells = [f"{v:.4f}" if not np.isnan(v) else "FAILED" for v in values]
    while len(run_cells) < runs:
        run_cells.append("FAILED")
    return (
        f"| {result.name} | {result.num_classes} | "
        + " | ".join(run_cells[:runs])
        + f" | {_format_mean_std(values)} | `{result.data_root}` |"
    )


def _format_dataset_summary_row(result: DatasetResult) -> str:
    """生成汇总总表中单个数据集的一行。"""
    cells = [
        _format_mean_std([m[k] for m in result.all_runs])
        for k in METRIC_LABELS
    ]
    return f"| {result.name} | {result.num_classes} | " + " | ".join(cells) + " |"


def merge_dataset_into_markdown(
    result_path: str, result: DatasetResult, runs: int
) -> None:
    """仅替换已有 Markdown 中指定 Dataset 的行，其余内容保持不变。"""
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f"结果文件不存在: {result_path}")
    with open(result_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    metric_title_to_key = {v: k for k, v in METRIC_LABELS.items()}
    row_prefix = f"| {result.name} |"
    current_metric_key: str | None = None
    in_summary = False
    out: list[str] = []
    for line in lines:
        if line.startswith("### "):
            current_metric_key = metric_title_to_key.get(line[4:].strip())
            in_summary = False
            out.append(line)
        elif line.startswith("## 汇总总表"):
            current_metric_key = None
            in_summary = True
            out.append(line)
        elif line.startswith(row_prefix):
            if in_summary:
                out.append(_format_dataset_summary_row(result))
            elif current_metric_key is not None:
                out.append(
                    _format_dataset_metric_row(result, current_metric_key, runs)
                )
            else:
                out.append(line)
        else:
            out.append(line)

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def write_result_markdown(
    result_path: str,
    results: list[DatasetResult],
    data_base: str,
    epochs: int,
    runs: int,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    argv_summary = " ".join([sys.argv[0]] + sys.argv[1:])
    dataset_names = ", ".join(r.name for r in results)
    lines = [
        f"# {MODEL_NAME} 多数据集多次实验",
        "",
        f"## {MODEL_NAME} benchmark ({dataset_names}) (epochs={epochs}, runs={runs}) — {timestamp}",
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


def run_dataset_benchmark(
    dataset_name: str,
    num_classes: int,
    data_root: str,
    runs: int,
    save_models: bool,
    epochs: int | None = None,
    params: TrainParams | None = None,
) -> DatasetResult:
    train_params = params or parameter_load()
    epochs_effective = epochs if epochs is not None else train_params.epochs
    print("=" * 60)
    print(
        f"数据集: {dataset_name} | 路径: {data_root} | 类别数: {num_classes} | epochs: {epochs_effective}"
    )
    all_runs: list[RunMetrics] = []
    model_path = os.path.join(get_project_root(), "model") + os.sep
    os.makedirs(model_path, exist_ok=True)

    for run_idx in range(1, runs + 1):
        print("-" * 60)
        print(f"[{dataset_name}] Run {run_idx}/{runs} 开始")
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        model_name = f"vicreg_{dataset_name.lower()}_run{run_idx}"
        log_name = f"{model_name}-{current_time}.log"
        logger = _make_logger(log_name)
        try:
            metrics = vicreg_train(
                logger,
                model_path,
                current_time,
                model_name,
                data_root if data_root.endswith("/") else data_root + "/",
                num_classes,
                epochs=epochs,
                save_models=save_models,
                params=train_params,
            )
            all_runs.append(metrics)
            print(
                f"[{dataset_name}] Run {run_idx}/{runs} | "
                f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}, "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )
        except Exception as exc:
            logger.exception("Run failed: %s run %d", dataset_name, run_idx)
            all_runs.append(
                RunMetrics(
                    accuracy=float("nan"),
                    macro_f1=float("nan"),
                    weighted_f1=float("nan"),
                    balanced_accuracy=float("nan"),
                )
            )
            print(f"[{dataset_name}] Run {run_idx}/{runs} FAILED: {exc}")

    return DatasetResult(
        name=dataset_name,
        num_classes=num_classes,
        data_root=os.path.abspath(data_root),
        all_runs=all_runs,
    )


def parse_args() -> argparse.Namespace:
    default_params = parameter_load()
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} 自监督预训练 + SSC 风格分类")
    parser.add_argument(
        "--benchmark_all",
        action="store_true",
        help="依次训练 Painting91 / Pandora / ArtBench / FashionStyle14 / Arch",
    )
    parser.add_argument(
        "--data_base",
        type=str,
        default=DEFAULT_DATA_BASE,
        help="多数据集 benchmark 的数据根目录",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="单数据集模式：数据集根目录（需含 train/test 子目录，末尾 / 可选）",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="单数据集模式下的类别数",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"VICReg 预训练 epoch 数（默认 {default_params.epochs}，见 TrainParams）",
    )
    parser.add_argument("--runs", type=int, default=1, help="重复实验次数，报告 mean±std")
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="benchmark 模式下也保存模型（默认仅单数据集保存）",
    )
    parser.add_argument(
        "--result_md",
        type=str,
        default=os.path.join(
            get_project_root(), "ieee_access_paperdata", "vicreg_multiple.md"
        ),
        help="多次实验结果 Markdown 输出路径",
    )
    parser.add_argument(
        "--merge_result",
        action="store_true",
        help="单数据集多次 run 时仅合并更新 result_md 中对应 Dataset 行",
    )
    return parser.parse_args()


def main() -> None:
    print("#########################################################root simclr.py#########################################################")
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("错误: --runs 须 >= 1")

    os.makedirs(os.path.join(get_project_root(), "model"), exist_ok=True)
    os.makedirs(os.path.join(get_project_root(), "log"), exist_ok=True)

    benchmark_all = args.benchmark_all
    if not benchmark_all and args.data_root is None:
        # 兼容无参数：默认单数据集 Pandora
        args.data_root = os.path.join(DEFAULT_DATA_BASE, "Pandora")
        args.num_classes = 12

    train_params = parameter_load()
    train_epochs = args.epochs if args.epochs is not None else train_params.epochs

    t0 = time.time()
    results: list[DatasetResult] = []

    if benchmark_all:
        save_models = args.save_models
        data_base = os.path.normpath(args.data_base)
        for name, num_classes, rel_path in BENCHMARK_DATASETS:
            data_root = os.path.join(data_base, rel_path.replace("/", os.sep))
            try:
                result = run_dataset_benchmark(
                    name,
                    num_classes,
                    data_root,
                    args.runs,
                    save_models,
                    epochs=train_epochs,
                    params=train_params,
                )
                results.append(result)
            except Exception as exc:
                print(f"[{name}] 失败: {exc}")
        data_base_for_md = data_base + os.sep
    else:
        data_root = args.data_root
        dataset_name = os.path.basename(os.path.normpath(data_root))
        num_classes = args.num_classes or 12
        result = run_dataset_benchmark(
            dataset_name,
            num_classes,
            data_root,
            max(args.runs, 1),
            save_models=True,
            epochs=train_epochs,
            params=train_params,
        )
        results.append(result)
        data_base_for_md = os.path.dirname(os.path.abspath(data_root)) + os.sep

    if not results:
        raise SystemExit("错误: 没有成功完成任何数据集实验")

    if benchmark_all or args.runs > 1:
        if args.merge_result and not benchmark_all and len(results) == 1:
            merge_dataset_into_markdown(args.result_md, results[0], args.runs)
            print(f"\n已合并更新 {results[0].name} → {args.result_md}")
        else:
            write_result_markdown(
                args.result_md,
                results,
                data_base_for_md,
                train_epochs,
                args.runs,
            )
            print(f"\n结果已写入: {args.result_md}")

    print(f"总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
