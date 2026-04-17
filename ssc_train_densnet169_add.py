# SSC 共性风格增强（add 版）：主干为 DenseNet169 features + 6 通道 RGB+HSV 输入
# 与 ssc_train_transformer_add.py 流程一致，仅替换：
#   - SscReg → SscRegDensenet169（1664→1664 projector）
#   - 数据变换 → get_ssc_transforms_rgb_hsv6
# backbone 特征：不读 ./pretrainFeatures，用当前模型冻结的 DenseNet169 features+GAP 在线提取，
# 按文件名缓存到内存 dict（与原先 pkl 用法一致）。
#
# 分类器阶段耗时说明（与 transformer_add 相同结构）：
#   每次触发分类器训练前，会用 trainloader 连续跑 K 遍（默认见 parameter_load），
#   每遍对全训练集做「view1+view2」两次 SSC 前向以写入缓存 → 约 K×2×ceil(N/batch) 次 DenseNet+MLP。
#   若此处很慢：① 增大 dataloader_num_workers；② 适当减小 classifier_cache_k（如 4～6）；
#   ③ 已启用 cudnn.benchmark（固定分辨率时加速卷积）。

import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from ssc.Sscreg_densenet169 import SscRegDensenet169
from ssc.utils_add import criterion_align, get_ssc_transforms_rgb_hsv6, MultiViewDataInjector
from SscDataSet_new import SscDataset
from ssc.classifier_enhance_add import EfficientClassifier

device0 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device1 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def parameter_load():
    """DenseNet169 池化维 1664；SSC 输出与 backbone 缓存同维以便 EfficientRWPClassifier。"""
    epochs = 20
    ssc_input = 1664
    ssc_output = 1664
    batch_size_ = 128
    offset_bs = 512
    base_lr = 0.001
    image_size = 224
    classfier_iteration = 200
    classifier_lr = 0.00005
    classifier_training_gap = 5
    classifier_test_gap = 5
    backbone_cache_workers = 8  # 构建内存 backbone 缓存时的 DataLoader workers
    # SSC 训练 + 分类器缓存构建共用；>0 可并行做 numpy→PIL→6ch 与增强，显著缩短「卡很久」的等待
    dataloader_num_workers = 8
    # 预计算 K 份不同随机增强的 SSC 缓存；K 与分类器阶段最前段耗时近似成正比（可改为 4～6 换速度）
    classifier_cache_k = 12
    model_name = ""
    return (
        epochs,
        batch_size_,
        offset_bs,
        base_lr,
        image_size,
        classfier_iteration,
        classifier_lr,
        model_name,
        classifier_training_gap,
        ssc_input,
        ssc_output,
        classifier_test_gap,
        backbone_cache_workers,
        dataloader_num_workers,
        classifier_cache_k,
    )


def _setup_hub(root: str) -> None:
    hub_dir = os.path.join(root, "pretrainModels", "hub")
    os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)
    os.environ.setdefault("TORCH_HOME", os.path.join(root, "pretrainModels"))


@torch.no_grad()
def _backbone_gap_vec(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """与 SscRegDensenet169.forward 中冻结段一致：features → ReLU → GAP → (B, 1664)。"""
    feat = model.features(x)
    feat = F.relu(feat, inplace=True)
    return F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)


def build_dense_backbone_cache_in_memory(
    model: nn.Module,
    dataset_root: str,
    split: str,
    transform_eval,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    logger: logging.Logger,
) -> dict:
    """
    用当前 DenseNet169 冻结骨干对 train/test 全部图像做一次前向，按「文件名」缓存 1664 维向量到内存。
    与 SscDataset 返回的 name 字段一致（各类文件夹内 basename）。
    """
    root = os.path.join(dataset_root.rstrip("/\\"), split)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"未找到目录: {root}")

    ds = datasets.ImageFolder(root, transform=transform_eval)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    feat_dict: dict = {}
    idx = 0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = _backbone_gap_vec(model, imgs).cpu()
        b = imgs.size(0)
        for j in range(b):
            path, _ = ds.samples[idx + j]
            key = os.path.basename(path)
            feat_dict[key] = feats[j].clone()
        idx += b
    logger.info(
        "DenseNet169 backbone 内存缓存 [%s]: %d 张, dim=%d",
        split,
        len(feat_dict),
        next(iter(feat_dict.values())).numel() if feat_dict else 0,
    )
    return feat_dict


def SSCtrain(
    logger,
    model_path,
    current_time,
    opt_model_name,
    dataset,
    class_number,
    iterations,
    training_mode,
    base_model_path,
):
    logger.debug("=" * 110)
    logger.debug("SSC STYLE-ALIGNMENT (DenseNet169-6ch + utils_add + classifier_enhance_add)")
    logger.debug("=" * 110)
    logger.info("SSC parameter setting up...")

    (
        epochs_,
        batch_size_,
        offset_bs_,
        base_lr_,
        image_size_,
        classifier_iteration_,
        classifier_lr_,
        model_name_,
        classifier_training_gap_,
        ssc_input_,
        ssc_output_,
        classifier_test_gap_,
        backbone_cache_workers_,
        dataloader_num_workers_,
        classifier_cache_k_,
    ) = parameter_load()

    root = os.path.abspath(os.path.dirname(__file__))
    _setup_hub(root)

    epochs = epochs_
    batch_size = batch_size_
    offset_bs = offset_bs_
    base_lr = base_lr_
    image_size = image_size_
    model_name_ = opt_model_name

    logger.info("dataset = %s", dataset)
    logger.info("[ADD+D169] SSC loss = criterion_align (align + var + supcon)")
    logger.info("[ADD+D169] Backbone feats = DenseNet169 6ch GAP 1664-d (内存缓存，无 pkl)")
    logger.info("[ADD+D169] Classifier = EfficientClassifier (4-branch fusion)")
    logger.info("epochs = %d", epochs)
    logger.info("batch_size = %d, offset_batch_size = %d", batch_size, offset_bs)
    logger.info("SSC DenseNet169: input_dim=%d, projector_out=%d", ssc_input_, ssc_output_)
    logger.info("SSC learning rate = %f", base_lr)
    logger.info("sub patch size = (%d, %d)", image_size, image_size)
    logger.info("classifier iteration = %d", classifier_iteration_)
    logger.info("classifier learning rate = %f", classifier_lr_)
    logger.info("classifier training gap = %d", classifier_training_gap_)
    logger.info("classifier test gap = %d", classifier_test_gap_)
    logger.info("dataloader_num_workers = %d (0=主进程加载，建议 4+)", dataloader_num_workers_)
    logger.info(
        "classifier_cache_k = %d (每触发分类器前需 K 遍全训练集 SSC 双视图前向，耗时≈正比于 K)",
        classifier_cache_k_,
    )
    logger.info("model name = %s", model_name_)

    transformT, transformT1, transformEvalT = get_ssc_transforms_rgb_hsv6(image_size)

    def _make_loader(ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=dataloader_num_workers_,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=dataloader_num_workers_ > 0,
        )

    if training_mode == "original":
        model = SscRegDensenet169(
            input_size=ssc_input_,
            output_size=ssc_output_,
            depth_projector=3,
            in_channels=6,
            target_size=image_size,
        )
        model = model.to(device0)
        params = model.projector.parameters()
        lr = base_lr * batch_size / offset_bs
        optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_, eta_min=lr * 0.01
        )
        time_str = current_time
        best_accuracy = 0.0
        last_accuracy = 0.0
        logger.info("SSC original mode (DenseNet169-6ch) is ready...")
    else:
        model = torch.load(model_path + "base-best.pth")
        model = model.to(device0)
        params = model.projector.parameters()
        lr = base_lr * batch_size / offset_bs
        optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_, eta_min=lr * 0.01
        )
        time_str = current_time
        best_accuracy = 0.0
        last_accuracy = 0.0
        logger.info("SSC fine-tuning mode is ready...")

    # 一次性提取 backbone 到内存（冻结层权重在训练中不变，全迭代复用）
    train_feature_dict = build_dense_backbone_cache_in_memory(
        model,
        dataset,
        "train",
        transformEvalT,
        device0,
        batch_size,
        backbone_cache_workers_,
        logger,
    )
    test_feature_dict = build_dense_backbone_cache_in_memory(
        model,
        dataset,
        "test",
        transformEvalT,
        device0,
        batch_size,
        backbone_cache_workers_,
        logger,
    )

    for iteration in range(iterations):
        logger.info("The iteration is %d", iteration)

        trainset = SscDataset(
            dataset, "train", transform=MultiViewDataInjector([transformT, transformT1])
        )
        trainloader = _make_loader(trainset, shuffle=True)

        testset = SscDataset(
            dataset, "test", transform=MultiViewDataInjector([transformT, transformT1])
        )
        testloader = _make_loader(testset, shuffle=False)
        logger.info("SSC %s for %d iterations is ready...", dataset, iteration)

        for epoch in range(epochs):
            print(f"current epoch is {epoch}")
            model.train()
            train_loss = []

            for view1, view2, label, name, _ in trainloader:
                view1 = view1.to(device0)
                view2 = view2.to(device0)
                cls_label = (label - 1).to(device0)
                fx = model(view1)
                fx1 = model(view2)
                loss = criterion_align(fx, fx1, labels=cls_label)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            if epoch % 10 == 0 or epoch == epochs - 1:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "The epoch is %d, SSC train loss is %f, lr is %.2e",
                    epoch,
                    np.mean(train_loss),
                    current_lr,
                )

            if (epoch % classifier_training_gap_ == 0 and epoch != 0) or (
                epoch == epochs - 1 and epoch != 0
            ):
                model.eval()

                classifier = EfficientClassifier(ssc_output_, class_number).to(device0)
                classifier_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                classifier_optimizer = torch.optim.Adam(
                    classifier.parameters(), lr=classifier_lr_, weight_decay=1e-3
                )
                classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    classifier_optimizer,
                    T_max=classifier_iteration_,
                    eta_min=classifier_lr_ * 0.01,
                )

                total_loss = 0.0
                loss_count = 0
                style_loss = torch.zeros(1).to(device1)

                K = classifier_cache_k_
                t_cache0 = time.time()
                logger.info(
                    "开始构建 SSC 分类缓存: K=%d 遍 × 全训练集双视图前向（DenseNet169+MLP）…",
                    K,
                )
                train_ssc_caches = []
                for _ in range(K):
                    cache = []
                    with torch.no_grad():
                        for view1, view2, label, names, original in trainloader:
                            view1 = view1.to(device1)
                            view2 = view2.to(device1)
                            backbone_view = torch.stack(
                                [train_feature_dict[n] for n in names], dim=0
                            ).to(device1)
                            ssc_view1 = model(view1)
                            ssc_view2 = model(view2)
                            cache.append(
                                (
                                    backbone_view.cpu(),
                                    ssc_view1.cpu(),
                                    ssc_view2.cpu(),
                                    (label - 1),
                                )
                            )
                    train_ssc_caches.append(cache)

                logger.info(
                    "训练集 SSC 缓存构建完成，耗时 %.1f s；开始构建测试集缓存…",
                    time.time() - t_cache0,
                )
                t_test0 = time.time()

                test_ssc_cache = []
                with torch.no_grad():
                    for view1, view2, label, names_, original in testloader:
                        view1 = view1.to(device1)
                        view2 = view2.to(device1)
                        backbone_view = torch.stack(
                            [test_feature_dict[n] for n in names_], dim=0
                        ).to(device1)
                        ssc_view1 = model(view1)
                        ssc_view2 = model(view2)
                        test_ssc_cache.append(
                            (
                                backbone_view.cpu(),
                                ssc_view1.cpu(),
                                ssc_view2.cpu(),
                                (label - 1),
                            )
                        )

                logger.info(
                    "测试集 SSC 缓存构建完成，耗时 %.1f s；开始分类器迭代…",
                    time.time() - t_test0,
                )

                es_patience = 20
                es_no_improve = 10
                es_best_acc = 0.0

                for i in range(classifier_iteration_):
                    trainstyle_loss = []
                    total_correct = 0.0
                    train_ssc_cache = train_ssc_caches[i % K]

                    classifier.train()
                    for bb_feat, ssc_v1, ssc_v2, label in train_ssc_cache:
                        bb_feat = bb_feat.to(device1)
                        ssc_v1 = ssc_v1.to(device1)
                        ssc_v2 = ssc_v2.to(device1)
                        label = label.to(device1)

                        bb_feat = bb_feat + torch.randn_like(bb_feat) * 0.01
                        ssc_v1 = ssc_v1 + torch.randn_like(ssc_v1) * 0.01
                        ssc_v2 = ssc_v2 + torch.randn_like(ssc_v2) * 0.01

                        prediction = classifier(ssc_v1, ssc_v2, bb_feat)
                        style_loss = classifier_criterion(prediction, label)
                        classifier_optimizer.zero_grad()
                        style_loss.backward()
                        classifier_optimizer.step()

                        trainstyle_loss.append(style_loss.item())
                        with torch.no_grad():
                            pred = prediction.data.max(1, keepdim=True)[1]
                            total_correct += pred.eq(label.data.view_as(pred)).cpu().sum()

                    classifier_scheduler.step()

                    if i % classifier_training_gap_ == classifier_training_gap_ - 1:
                        logger.info(
                            "The classifer-train round is %d, the training accuracy is %d/%d",
                            i,
                            total_correct,
                            len(trainset),
                        )

                    if i % classifier_test_gap_ == classifier_test_gap_ - 1:
                        test_correct = 0.0
                        classifier.eval()
                        with torch.no_grad():
                            for bb_feat, ssc_v1, ssc_v2, label in test_ssc_cache:
                                bb_feat = bb_feat.to(device1)
                                ssc_v1 = ssc_v1.to(device1)
                                ssc_v2 = ssc_v2.to(device1)
                                label = label.to(device1)
                                prediction = classifier(ssc_v1, ssc_v2, bb_feat)
                                pred = prediction.data.max(1, keepdim=True)[1]
                                test_correct += pred.eq(label.data.view_as(pred)).cpu().sum()

                        test_accuracy = float(test_correct / len(testset))
                        last_accuracy = test_accuracy

                        if test_accuracy > best_accuracy and test_accuracy > 0.45:
                            accuracy_str = f"{test_accuracy:.4f}".split(".")[1][:4]
                            lt_cls_name = (
                                model_name_
                                + "-ADD-DENSE169-"
                                + time_str
                                + "-iteration-"
                                + str(iteration)
                                + "-accuracy-"
                                + accuracy_str
                                + "-SSC-classifier-best.pth"
                            )
                            lt_base_name = (
                                model_name_
                                + "-ADD-DENSE169-"
                                + time_str
                                + "-iteration-"
                                + str(iteration)
                                + "-accuracy-"
                                + accuracy_str
                                + "-SSC-base-best.pth"
                            )
                            torch.save(model, model_path + lt_base_name)
                            torch.save(classifier, model_path + lt_cls_name)
                            logger.info(
                                "+++THE BEST MODEL is saved+++. The iteration is %d, "
                                "the best accuracy is %f, and the current accuracy is %f",
                                iteration,
                                best_accuracy,
                                test_accuracy,
                            )
                            best_accuracy = test_accuracy

                        logger.info(
                            "Test result is: The test round is %d, the test ratio is %d/%d, "
                            "the test accuracy is %f",
                            i,
                            test_correct,
                            len(testset),
                            test_accuracy,
                        )

                        if test_accuracy > es_best_acc:
                            es_best_acc = test_accuracy
                            es_no_improve = 0
                        else:
                            es_no_improve += 1
                            if es_no_improve >= es_patience:
                                logger.info(
                                    "Early stopping at round %d, best local acc: %f",
                                    i,
                                    es_best_acc,
                                )
                                break

                total_loss += np.mean(trainstyle_loss)
                loss_count += 1
                avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
                logger.info("The average loss is %.6e", avg_loss)

                if epoch == epochs - 1 and iteration == iterations - 1:
                    lt_cls_name = (
                        model_name_
                        + "-ADD-DENSE169-"
                        + time_str
                        + "-iteration-"
                        + str(iteration)
                        + "-SSC-classifier-last.pth"
                    )
                    lt_base_name = (
                        model_name_
                        + "-ADD-DENSE169-"
                        + time_str
                        + "-iteration-"
                        + str(iteration)
                        + "-SSC-base-last.pth"
                    )
                    torch.save(model, model_path + lt_base_name)
                    torch.save(classifier, model_path + lt_cls_name)
                    logger.info(
                        "The last models are saved. The last accuracy is %f", last_accuracy
                    )

    logger.info(
        "The best accuracy is %f, and the last accuracy is %f",
        best_accuracy,
        last_accuracy,
    )
    logging.shutdown()


if __name__ == "__main__":
    model_path = "./model/"

    dataset_name = "Painting91"
    class_num_dict = {
        "Painting91": 13,
        "Pandora": 12,
        "WikiArt3": 15,
        "WikiArt3_small": 15,
        "Arch": 25,
        "FashionStyle14": 14,
        "artbench": 10,
        "webstyle/subImages": 10,
        "AVAstyle": 14,
    }

    data_root = "/mnt/codes/data/style/"
    dataSource = os.path.join(data_root, dataset_name) + "/"
    class_number = class_num_dict.get(dataset_name, 10)
    safe_name = dataset_name.replace("/", "_")
    model_name = f"ssc-add-d169-{safe_name}"
    logger = logging.getLogger("ssc_add_d169_logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_name = model_name + "-" + current_time + ".log"
    filehandler = logging.FileHandler("./log/" + log_name)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    iterations = 5
    training_mode = "original"
    base_model_path = "./model/base-best.pth"

    SSCtrain(
        logger,
        model_path,
        current_time,
        model_name,
        dataSource,
        class_number,
        iterations,
        training_mode,
        base_model_path,
    )

    logger.removeHandler(filehandler)
    logger.removeHandler(handler)
    logging.shutdown()
