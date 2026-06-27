# Author: cuijia1247
# Date: 2025-4-27
# version: 1.0
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
from torch import nn
import torchvision.models as models
from torch.autograd import Variable
from sklearn.metrics import balanced_accuracy_score, f1_score
# from ssc.Sscreg import SscReg
from simclr.simclr import SimCLR
from ssc.utils import criterion, get_byol_transforms, MultiViewDataInjector
from SscDataSet import SscDataset
from ssc.classifier import Classifier
from simclr.optimizers import get_optimizer, LR_Scheduler
from simclr.arguments import get_args

#setup device for cuda or cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MODEL_NAME = "SimCLR (SSC)"
DEFAULT_DATA_ROOT = "/mnt/codes/data/style/Painting91/"
DEFAULT_NUM_CLASSES = 13
DEFAULT_NUM_RUNS = 3
DEFAULT_RESULT_MD = os.path.join("ieee_access_paperdata", "simclr_multiple.md")

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "balanced_accuracy": "Balanced Accuracy",
}


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


def _compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> RunMetrics:
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
            f1_score(labels_arr, preds_arr, average="macro", labels=labels_all, zero_division=0)
        ),
        weighted_f1=float(
            f1_score(labels_arr, preds_arr, average="weighted", labels=labels_all, zero_division=0)
        ),
        balanced_accuracy=float(balanced_accuracy_score(labels_arr, preds_arr)),
    )


def _format_mean_std(values: list[float]) -> str:
    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        return "FAILED" if values else "-"
    arr = np.asarray(valid, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{mean_v:.4f}±{std_v:.4f}"


def _run_cell_value(all_runs: list[RunMetrics], run_idx: int, metric_key: str) -> str:
    """run_idx 为 0-based；尚未完成的轮次显示 '-'。"""
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
        "| Dataset | num_classes | "
        + " | ".join(metric_titles)
        + " |",
        "|---------|-------------|"
        + "|".join(["---------"] * len(metric_titles))
        + "|",
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
    pretrain_epochs: int,
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
    lines = [
        f"# {MODEL_NAME} 多数据集多次实验",
        "",
        f"## {MODEL_NAME} benchmark ({dataset_names}) "
        f"(epochs={pretrain_epochs}, runs={runs}{progress}) — {timestamp}",
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


def _make_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def parameter_load():
    epochs = 126 #best, perhaps300
    # backbone = 'resnet50'
    # ssc_backend = 'resnet50'
    ssc_input = 2048
    ssc_output = 2048
    batch_size_ = 64
    # batch_size_sample = 'None'
    # offset_bs = 512
    base_lr = 0.008 #best
    image_size = 64 #best
    classfier_iteration = 300 #best 150
    # classfier_iteration = 300  # best
    classifier_lr = 0.001 #best
    # classifier_structure = '2048-1024-512-13 with dropout'
    classifier_training_gap = 25
    model_name = ''
    return (epochs, batch_size_, base_lr, image_size, classfier_iteration, classifier_lr, model_name,
            classifier_training_gap, ssc_input, ssc_output)#, classifier_structure

def simclr_train(logger, model_path, current_time, opt_model_name, dataset, class_number):
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.debug('THIS IS THE FORMAL TRAINING PROCESS')
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('simlar parameter setting up...')
    # load all the parameters
    (epochs_, batch_size_, base_lr_, image_size_, classifier_iteration_, classifier_lr_, model_name_,
     classifier_training_gap_, ssc_input_, ssc_output_)= parameter_load()
    # the training parameters
    epochs = epochs_
    batch_size = batch_size_
    # offset_bs = offset_bs_
    base_lr = base_lr_
    image_size = image_size_
    model_name_ = opt_model_name  ####optimal
    # display all the necessary parameters & record them in logger
    logger.info('dataset = %s', dataset)
    # logger.info('backbone is %s', backbone_) # for now backbone == backend
    logger.info('epochs = %d', epochs)
    logger.info('batch_size = %d', batch_size)
    # logger.info('SSC backend = %s', ssc_backend_)
    # logger.info('SSC input = %d', ssc_input_)
    # logger.info('SSC output = %d', ssc_output_)
    logger.info('simclr learning rate = %f', base_lr)
    # logger.info('sub patch size = (%d, %d)', image_size, image_size)
    # logger.info('sub pathc sample is %s', batch_size_sample_)
    logger.info('classifier training gap = %d', classifier_training_gap_)
    logger.info('classifier iteration is %d', classifier_iteration_)
    logger.info('classifier learning rate = %f', classifier_lr_)
    # logger.info('classifier structure = %s', classifier_structure_)  ####optimal
    logger.info('model name is %s', model_name_)
    # logger.info('SSC output is %d', ssc_output)

    #normalize and randomcrop input images
    transformT, transformT1, transformEvalT = get_byol_transforms(image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # SscDataset 用字符串拼接路径，dataSource 必须以 '/' 结尾
    dataSource = dataset if dataset.endswith(os.sep) else dataset + os.sep
    trainData = 'train'
    trainset = SscDataset(dataSource, trainData, transform=MultiViewDataInjector([transformT, transformT1]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testData = 'test'
    testset = SscDataset(dataSource, testData, transform=MultiViewDataInjector([transformT, transformT1]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    if len(trainset) == 0 or len(testset) == 0:
        raise ValueError(
            f"数据集为空: train={len(trainset)}, test={len(testset)}, dataSource={dataSource}"
        )
    logger.info('simclr %s is ready (train=%d, test=%d)...', dataSource, len(trainset), len(testset))

    # lr = base_lr*batch_size/offset_bs
    #set up the simclr model
    # define optimizer

    model = SimCLR()
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(ssc_input_, ssc_output_)
    resnet50 = resnet50.eval()
    model = model.to(device)
    resnet50 = resnet50.to(device)
    args = get_args()
    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        lr=args.train.base_lr * args.train.batch_size / 256,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr * args.train.batch_size / 256,
        args.train.num_epochs, args.train.base_lr * args.train.batch_size / 256,
                                  args.train.final_lr * args.train.batch_size / 256,
        len(trainloader),
        constant_predictor_lr=True  # see the end of section 4.2 predictor
    )
    # resnet50 = resnet50.to(device)
    # params = model.parameters()
    # optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
    logger.info('simclr model is ready...')


    # time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    time_str = current_time
    best_accuracy = 0.0
    last_accuracy = 0.0
    best_metrics = RunMetrics(
        accuracy=0.0, macro_f1=0.0, weighted_f1=0.0, balanced_accuracy=0.0
    )
    for epoch in range(epochs):
        # print('epoch is {}'.format(epoch))
        model.train()
        tk0 = trainloader
        train_loss = []
        # temploss = total_loss / (1860*100)
        for view1, view2, label, name, _ in tk0:
            model.zero_grad()
            view1 = view1.to(device)
            view2 = view2.to(device)
            data_dict = model.forward(view1.to(device, non_blocking=True), view2.to(device, non_blocking=True))
            loss = data_dict['loss'].mean()  # ddp
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # fx = model(view1)
            # fx1 = model(view2)
            # loss = criterion(fx, fx1)
            train_loss.append(loss.item())
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        if epoch % 10 == 0 or epoch == epochs-1:
            logger.info('The epoch is %d, simclr train loss is %f', epoch, np.mean(train_loss))
            # print('The epoch is {}, Vic train loss is {}'.format(epoch, np.mean(train_loss)))
            # train the style classifier every 500 iterations
        if epoch % classifier_training_gap_ == 0 and epoch != 0 or epoch == epochs-1:
            classifier = Classifier(ssc_output_, class_number).cuda()
            classifier_criterion = nn.CrossEntropyLoss()
            classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr_)
            total_loss = 0.0
            style_loss = torch.zeros(1).cuda()
            model.eval()
            # logger.info('SSC classifier model is ready...')
            # model.eval()
            # correct = 0.0
            # total_number = len(trainset)
            for i in range(classifier_iteration_):
                trainstyle_loss = []
                total_correct = 0.0
                tk1 = trainloader
                tk2 = testloader
                for view1, view2, label, name, original in tk1:
                    correct = 0.0
                    view1 = view1.to(device).detach()
                    view2 = view2.to(device).detach()
                    data_dict = model.forward(view1.to(device, non_blocking=True), view2.to(device, non_blocking=True))
                    #############simclr in ssc way
                    original = original.to(device)
                    backbone_view = resnet50(original)
                    test1 = backbone_view - data_dict['z1']  # only use view 1
                    test2 = backbone_view - data_dict['z2']
                    ###########################
                    test = test1 + test2
                    prediction = classifier(test)
                    # val, idx = prediction.topk(1)
                    # idx = idx.t().squeeze()
                    # idx = idx.cpu().float()
                    # original_label = label
                    # label = label.cpu().float()-1
                    label = label - 1
                    label = Variable(label).cuda()
                    style_loss = classifier_criterion(prediction, label)
                    classifier_optimizer.zero_grad()
                    # style_loss.requires_grad_()
                    style_loss.backward()
                    classifier_optimizer.step()
                    pred = prediction.data.max(1, keepdim=True)[1]
                    correct += pred.eq(label.data.view_as(pred)).cpu().sum()
                    # correct = idx.eq(label).cpu().sum()
                    total_correct += correct
                # total_loss += style_loss
                trainstyle_loss.append(style_loss.item())
                # print('The correct/total_correct--total is {}/{}--{}'.format(correct, total_correct, len(view1)))
                if i % 10 == 9:
                    logger.info('The classifer-train round is %d, the training accuracy is %d/%d', i, total_correct,
                                len(trainset))
                    # print('The cla-train round is {}, the training ratio is {}/{}'.format(i, total_correct, len(trainset)))
                if i % 20 == 19:
                    test_correct = 0.0
                    y_true: list[int] = []
                    y_pred: list[int] = []
                    classifier.eval()
                    for view1, view2, label, name, original in tk2:
                        correct_ = 0.0
                        view1 = view1.to(device).detach()
                        view2 = view2.to(device).detach()
                        data_dict = model.forward(view1.to(device, non_blocking=True),
                                                  view2.to(device, non_blocking=True))
                        #############simclr in ssc way
                        original = original.to(device)
                        backbone_view = resnet50(original)
                        test1 = backbone_view - data_dict['z1']  # only use view 1
                        test2 = backbone_view - data_dict['z2']
                        ###########################
                        test = test1 + test2
                        prediction = classifier(test)
                        label = label - 1
                        label = Variable(label).cuda()
                        pred = prediction.data.max(1, keepdim=True)[1]
                        y_true.extend(label.cpu().tolist())
                        y_pred.extend(pred.squeeze(1).cpu().tolist())
                        correct_ += pred.eq(label.data.view_as(pred)).cpu().sum()
                        test_correct += correct_

                    test_accuracy = float(test_correct / max(len(testset), 1))
                    last_accuracy = test_accuracy
                    if test_accuracy > best_accuracy:  # the current best classifier
                        best_accuracy = test_accuracy
                        best_metrics = _compute_metrics(y_true, y_pred, class_number)
                        lt_classifier_name = model_name_ + '-SSC-resnet50-' + time_str + '-simclr-classifier-best.pth'
                        lt_base_name = model_name_ + '-SSC-resnet50-' + time_str + '-simclr-base-best.pth'
                        torch.save(model, model_path + lt_base_name)
                        torch.save(classifier, model_path + lt_classifier_name)
                        logger.info(
                            '+++THE BEST MODEL is saved+++. The best accuracy is %f, and the current accuracy is %f',
                            best_accuracy, test_accuracy)
                        logger.info(
                            'Best metrics: macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f',
                            best_metrics["macro_f1"],
                            best_metrics["weighted_f1"],
                            best_metrics["balanced_accuracy"],
                        )
                    logger.info(
                        'Test result is: The test round is %d, the test ratio is %d/%d, the test accuracy is %f', i,
                        test_correct,
                        len(testset), test_accuracy)
            total_loss += np.mean(trainstyle_loss)
            # total_loss = total_loss / 50
            if epoch == epochs - 1:
                lt_classifier_name = model_name_ + '-SSR-resnet50-' + time_str + '-SSC-classifier-last.pth'
                lt_base_name = model_name_ + '-SSR-resnet50-' + time_str + '-SSC-base-last.pth'
                torch.save(model, model_path + lt_base_name)
                torch.save(classifier, model_path + lt_classifier_name)
                logger.info('The last models are saved. The last accuracy is %f', last_accuracy)
    logger.info('The best accuracy is %f, and the last accuracy is %f', best_accuracy, last_accuracy)
    logger.info(
        'Done. acc=%.4f macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f',
        best_metrics["accuracy"],
        best_metrics["macro_f1"],
        best_metrics["weighted_f1"],
        best_metrics["balanced_accuracy"],
    )
    return best_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{MODEL_NAME} 本地训练，多轮实验 + Markdown 输出")
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    p.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS, help="重复训练次数（默认 5）")
    p.add_argument("--result_md", type=str, default=DEFAULT_RESULT_MD)
    p.add_argument("--model_path", type=str, default="./model/")
    return p.parse_args()


def _save_run_markdown(
    result_md: str,
    dataset_name: str,
    num_classes: int,
    data_root: str,
    all_runs: list[RunMetrics],
    pretrain_epochs: int,
    total_runs: int,
) -> None:
    """将当前已完成 run 的结果写入指定 md（每轮 run 结束后调用）。"""
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
        pretrain_epochs,
        total_runs,
        completed_runs=len(all_runs),
    )


def main():
    """本地 SSC 版 SimCLR：多轮训练，四项指标写入 ieee_access_paperdata/。"""
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("错误: --runs 须 >= 1")

    data_root = os.path.abspath(args.data_root.rstrip(os.sep))
    if not data_root.endswith(os.sep):
        data_root += os.sep
    dataset_name = os.path.basename(os.path.normpath(data_root))
    model_name = f"simclr_{dataset_name.lower()}"
    pretrain_epochs = parameter_load()[0]

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs("./log", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.result_md)), exist_ok=True)

    all_runs: list[RunMetrics] = []
    for r in range(1, args.runs + 1):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        log_path = os.path.join("./log", f"{model_name}-run{r}-{current_time}.log")
        logger = _make_logger(log_path)
        print(f"[{dataset_name} run{r}/{args.runs}] 开始训练…")
        try:
            metrics = simclr_train(
                logger, args.model_path, current_time, model_name, data_root, args.num_classes
            )
            all_runs.append(metrics)
            print(
                f"[{dataset_name} run{r}/{args.runs}] "
                f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}, "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )
        except Exception as e:
            logger.exception("Run %d failed", r)
            all_runs.append(
                RunMetrics(
                    accuracy=float("nan"),
                    macro_f1=float("nan"),
                    weighted_f1=float("nan"),
                    balanced_accuracy=float("nan"),
                )
            )
            print(f"[{dataset_name} run{r}/{args.runs}] FAILED: {e}")

        # 每轮 run 结束后立即更新 md（未完成轮次显示为 '-'）
        _save_run_markdown(
            args.result_md,
            dataset_name,
            args.num_classes,
            data_root,
            all_runs,
            pretrain_epochs,
            args.runs,
        )
        print(f"结果已更新: {args.result_md} ({len(all_runs)}/{args.runs} runs)")

    print(
        f"[{dataset_name}] Accuracy "
        f"{_format_mean_std([m['accuracy'] for m in all_runs])}"
    )


if __name__ == "__main__":
    main()













