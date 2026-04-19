# ssc new training code after 20250425
# Author: cuijia1247
# Date: 2026-3-28
# version: 1.1
import argparse
import statistics
from typing import List, Optional, Tuple
import logging                                          # 日志模块
import time                                             # 时间模块，用于生成时间戳
import os                                               # 操作系统接口模块
import torch                                            # PyTorch 深度学习框架
from torch import nn                                    # 神经网络模块
import torch.optim as optim                             # 优化器模块
import torchvision.models as models                     # 预训练视觉模型库
from torch.autograd import Variable                     # 自动微分变量（兼容旧版写法）
import numpy as np                                      # 数值计算库
from ssc.Sscreg import SscReg                           # SSC ResNet 编码器模型
from ssc.utils import criterion, get_ssc_transforms, MultiViewDataInjector  # 损失函数、数据增强、多视图注入器
from SscDataSet_new import SscDataset                   # 懒加载版数据集（__getitem__ 实时随机增强）
from ssc.classifier import Classifier, EfficientClassifier  # 风格分类头（四路融合，与 ssc_train_transformer 一致）
from utils.pretrainFeatureExtraction import load_dataFeatures  # 加载预提取的骨干网络特征

# 与 remote_sh/run_ssc_train_vit_bat.sh / ssc_train_transformer 中分类器设置保持一致：双卡占位，实际可同卡
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device1 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def parameter_load():
    """
    集中管理本脚本正式训练超参（remote_sh/run_ssc_resnet.sh 无参启动时即使用此处默认值）。
    与 param_optim 网格搜索无关；调参请直接改本函数或经 parse_train_args 显式覆盖。
    """
    epochs = 100
    backbone = "resnet50"
    ssc_backend = "resnet50"
    ssc_input = 2048
    ssc_output = 2048
    batch_size_ = 64
    batch_size_sample = "None"
    offset_bs = 512
    base_lr = 0.001
    image_size = 112
    classfier_iteration = 100
    classifier_lr = 0.00003
    classifier_training_gap = 5
    classifier_test_gap = 2
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
        batch_size_sample,
        classifier_training_gap,
        backbone,
        ssc_backend,
        ssc_input,
        ssc_output,
        classifier_test_gap,
    )


def parse_train_args() -> argparse.Namespace:
    """命令行非空项覆盖 parameter_load()；run_ssc_resnet.sh 可选通过环境变量传入。"""
    p = argparse.ArgumentParser(description="SSC ResNet 训练（ssc_train_resnet_copy.py）")
    p.add_argument(
        "--dataset_name",
        type=str,
        default="Painting91",
        help="数据集子目录名（位于 --data_root 下），如 Painting91、WikiArt3",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/mnt/codes/data/style/",
        help="数据根目录（各数据集为其子目录）",
    )
    p.add_argument(
        "--pre_feature_path",
        type=str,
        default="./pretrainFeatures",
        help="预提取 ResNet50 特征目录（内含 {dataset}_resnet50_train_features.pkl 等）",
    )
    p.add_argument("--model_path", type=str, default="./model/", help="模型 checkpoint 保存目录")
    p.add_argument(
        "--training_mode",
        type=str,
        default="original",
        choices=("original", "fine-tuning"),
        help="original：从头训练 SSC；fine-tuning：从 model_path 下 base-best.pth 加载",
    )
    p.add_argument(
        "--base_model_path",
        type=str,
        default="###",
        help="预留；微调时实际使用 model_path + 'base-best.pth'",
    )
    p.add_argument("--epochs", type=int, default=None, help="SSC 编码器 epoch 数，默认见 parameter_load()")
    p.add_argument("--batch_size", type=int, default=None, help="batch size，默认 64")
    p.add_argument("--base_lr", type=float, default=None, help="SSC 基础学习率（余弦初值），默认 0.001")
    p.add_argument("--image_size", type=int, default=None, help="子图边长，默认 112")
    p.add_argument(
        "--classifier_iteration",
        type=int,
        default=None,
        help="每次触发分类器训练时的迭代轮数，默认见 parameter_load()",
    )
    p.add_argument(
        "--classifier_lr",
        type=float,
        default=None,
        help="分类器 Adam lr，默认见 parameter_load()",
    )
    p.add_argument(
        "--classifier_training_gap",
        type=int,
        default=None,
        help="每隔多少 SSC epoch 触发一次分类器训练，默认见 parameter_load()",
    )
    p.add_argument(
        "--classifier_test_gap",
        type=int,
        default=None,
        help="分类器训练内每隔多少轮评估测试集，默认 2",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="外层迭代次数（每轮重扫数据集），默认 1",
    )
    p.add_argument(
        "--dataset_repeat_runs",
        type=int,
        default=1,
        help="同一数据集独立完整训练次数（每轮重置 SSC）；每轮仅记录 best，汇总 mean±std",
    )
    return p.parse_args()


def merge_params_with_args(base_tuple: tuple, args: argparse.Namespace) -> tuple:
    """将 CLI 非 None 项覆盖到 parameter_load 返回的元组；元组顺序与 SSCtrain 解包一致。"""
    (
        epochs,
        batch_size_,
        offset_bs,
        base_lr,
        image_size,
        classfier_iteration,
        classifier_lr,
        model_name,
        batch_size_sample,
        classifier_training_gap,
        backbone_,
        ssc_backend_,
        ssc_input_,
        ssc_output_,
        classifier_test_gap_,
    ) = base_tuple
    if args.epochs is not None:
        epochs = args.epochs
    if args.batch_size is not None:
        batch_size_ = args.batch_size
    if args.base_lr is not None:
        base_lr = args.base_lr
    if args.image_size is not None:
        image_size = args.image_size
    if args.classifier_iteration is not None:
        classfier_iteration = args.classifier_iteration
    if args.classifier_lr is not None:
        classifier_lr = args.classifier_lr
    if args.classifier_training_gap is not None:
        classifier_training_gap = args.classifier_training_gap
    if args.classifier_test_gap is not None:
        classifier_test_gap_ = args.classifier_test_gap
    return (
        epochs,
        batch_size_,
        offset_bs,
        base_lr,
        image_size,
        classfier_iteration,
        classifier_lr,
        model_name,
        batch_size_sample,
        classifier_training_gap,
        backbone_,
        ssc_backend_,
        ssc_input_,
        ssc_output_,
        classifier_test_gap_,
    )

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
    preFeaturePath,
    feature_name,
    train_args: Optional[argparse.Namespace] = None,
    dataset_repeat_runs: int = 1,
) -> Tuple[List[float], float, float]:
    """
    SSC ResNet 版主训练函数
    Args:
        logger:              日志记录器
        model_path:          模型保存目录
        current_time:        训练启动时间戳（用于文件命名）
        opt_model_name:      模型名称前缀
        dataset:             数据集根目录路径
        class_number:        分类类别数
        iterations:          外层迭代次数（每次重新加载数据集）
        training_mode:       'original'（从头训练）或 'fine-tuning'（微调已有模型）
        base_model_path:     fine-tuning 模式下加载的预训练模型路径
        preFeaturePath:      预提取特征的存放路径
        feature_name:        预提取特征的文件名前缀
        dataset_repeat_runs: 同一数据集独立完整训练轮数（每轮重置 SSC）；>=1。汇总 mean±std 仅记录各轮 best，不汇总 last。
    Returns:
        (各轮 best 准确率列表, 均值, 样本标准差)；仅一轮时 std 为 0.0。
    """
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.debug('THIS IS THE FORMAL TRAINING PROCESS OF SSC TRAIN')
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('SSC parameter setting up...')

    # 加载超参数；若传入 train_args 则 CLI 覆盖 parameter_load() 默认值
    _params = parameter_load()
    if train_args is not None:
        _params = merge_params_with_args(_params, train_args)
    (
        epochs_,
        batch_size_,
        offset_bs_,
        base_lr_,
        image_size_,
        classifier_iteration_,
        classifier_lr_,
        model_name_,
        batch_size_sample_,
        classifier_training_gap_,
        backbone_,
        ssc_backend_,
        ssc_input_,
        ssc_output_,
        classifier_test_gap_,
    ) = _params

    # 与 parameter_load() 中各项一致（含 train_args 覆盖后），直接打印到终端便于核对
    print(
        "[SSC 超参 | parameter_load 生效值]\n"
        f"  epochs={epochs_}\n"
        f"  backbone={backbone_!r}, ssc_backend={ssc_backend_!r}\n"
        f"  ssc_input={ssc_input_}, ssc_output={ssc_output_}\n"
        f"  batch_size_={batch_size_}, offset_bs={offset_bs_}\n"
        f"  batch_size_sample={batch_size_sample_!r}\n"
        f"  base_lr={base_lr_}\n"
        f"  image_size={image_size_}\n"
        f"  classifier_iteration={classifier_iteration_}\n"
        f"  classifier_lr={classifier_lr_}\n"
        f"  classifier_training_gap={classifier_training_gap_}\n"
        f"  classifier_test_gap={classifier_test_gap_}",
        flush=True,
    )

    # 将参数赋值给局部变量（便于后续使用）
    epochs = epochs_
    batch_size = batch_size_
    offset_bs = offset_bs_
    base_lr = base_lr_
    image_size = image_size_
    model_name_ = opt_model_name        # 使用主程序传入的模型名称覆盖默认值

    # 将所有关键参数写入日志
    logger.info('dataset = %s', dataset)
    logger.info('backbone is %s', backbone_)
    logger.info('epochs = %d', epochs)
    logger.info('batch_size = %d, offset_batch_size = %d', batch_size, offset_bs)
    logger.info('SSC backend = %s', ssc_backend_)
    logger.info('SSC input = %d', ssc_input_)
    logger.info('SSC output = %d', ssc_output_)
    logger.info('SSC learning rate = %f', base_lr)
    logger.info('sub patch size = (%d, %d)', image_size, image_size)
    logger.info('sub pathc sample is %s', batch_size_sample_)
    logger.info('classifier iteration is %d', classifier_iteration_)
    logger.info('classifier learning rate = %f', classifier_lr_)
    logger.info('classifier iteration is %d', classifier_iteration_)
    logger.info('classifier learning rate = %f', classifier_lr_)
    # logger.info('classifier structure = %s', classifier_structure_)
    logger.info('model name is %s', model_name_)
    # logger.info('SSC output is %d', ssc_output)

    # 构建数据增强变换：训练视图1、训练视图2、评估变换
    transformT, transformT1, transformEvalT = get_ssc_transforms(image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if dataset_repeat_runs < 1:
        raise ValueError("dataset_repeat_runs 须 >= 1")
    logger.info(
        "dataset_repeat_runs = %d（每轮结束记录 [RUN_BEST]；数据集结束 [DATASET_SUMMARY] 为各轮 best 的 mean±std，不汇总 last）",
        dataset_repeat_runs,
    )
    train_feature_path = os.path.join(preFeaturePath, f"{feature_name}_train_features.pkl")
    train_feature_dict = load_dataFeatures(train_feature_path)
    test_feature_path = os.path.join(preFeaturePath, f"{feature_name}_test_features.pkl")
    test_feature_dict = load_dataFeatures(test_feature_path)

    run_bests: List[float] = []
    for run_idx in range(dataset_repeat_runs):
        logger.info(
            "========== 数据集重复训练 [%d/%d]（每轮从头初始化 SSC）==========",
            run_idx + 1,
            dataset_repeat_runs,
        )
        run_suffix = f"-run{run_idx}" if dataset_repeat_runs > 1 else ""
        # 多轮训练时把时间戳后缀写入文件名，避免 best/last checkpoint 互相覆盖
        time_str = current_time + run_suffix

        if training_mode == 'original':
            # 从头初始化 SSC 编码器
            model = SscReg(input_size=ssc_input_, output_size=ssc_output_, backend=ssc_backend_)
            model = model.to(device)
            params = model.parameters()
            lr = base_lr * batch_size / offset_bs
            optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_, eta_min=lr * 0.01
            )
            best_accuracy = 0.0
            last_accuracy = 0.0
            logger.info('SSC original mode is ready...')
        else:
            model = torch.load(model_path + 'base-best.pth')
            resnet50 = models.resnet50(pretrained=True)
            resnet50.fc = nn.Linear(ssc_input_, ssc_output_)
            resnet50 = resnet50.eval()
            model = model.to(device)
            resnet50 = resnet50.to(device)
            params = model.parameters()
            lr = base_lr * batch_size / offset_bs
            optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_, eta_min=lr * 0.01
            )
            best_accuracy = 0.0
            last_accuracy = 0.0
            logger.info('SSC fine-tuning mode is ready...')

        # 外层循环：多次迭代训练，每次重新打乱数据
        for iteration in range(iterations):
            logger.info('The iteration is %d', iteration)

            # 构建训练集与 DataLoader（用于 SSC 编码器训练，需要随机增强子图）
            dataSource = dataset
            trainData = 'train'
            trainset = SscDataset(dataSource, trainData, transform=MultiViewDataInjector([transformT, transformT1]))
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

            # 构建测试集与 DataLoader（用于 SSC 编码器训练阶段）
            testData = 'test'
            testset = SscDataset(dataSource, testData, transform=MultiViewDataInjector([transformT, transformT1]))
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
            logger.info('SSC ' + dataSource + 'for ' + str(iteration) + ' iterations is ready...')

            # EfficientClassifier 需 backbone 预提取特征 + SSC 双视图；触发分类器训练时再构建缓存（同 ssc_train_transformer）。

            # 内层循环：SSC 编码器 epoch 级训练
            for epoch in range(epochs):
                model.train()                       # 恢复 SSC 训练模式（分类器段末尾为 eval）
                tk0 = trainloader                   # 训练数据迭代器
                train_loss = []                     # 记录每个 batch 的损失值

                # 遍历所有训练 batch，更新 SSC 编码器参数
                for view1, view2, label, name, _ in tk0:
                    view1 = view1.to(device)                        # 子图视图1 移至设备
                    view2 = view2.to(device)                        # 子图视图2 移至设备
                    fx = model(view1)                               # 编码器前向：视图1 特征
                    fx1 = model(view2)                              # 编码器前向：视图2 特征
                    loss = criterion(fx, fx1, device=device)        # 计算 VICReg 风格三项损失
                    train_loss.append(loss.item())                  # 记录当前 batch 损失
                    optimizer.zero_grad()                           # 清空梯度
                    loss.backward()                                 # 反向传播
                    optimizer.step()                                # 更新编码器参数

                scheduler.step()

                if epoch % 10 == 0 or epoch == epochs - 1:
                    cur_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        'The epoch is %d, SSC train loss is %f, lr is %.2e',
                        epoch,
                        float(np.mean(train_loss)) if train_loss else 0.0,
                        cur_lr,
                    )

                # 按间隔触发分类器训练（epoch 0 跳过，防止编码器未收敛）
                if (epoch % classifier_training_gap_ == 0 and epoch != 0) or (epoch == epochs - 1 and epoch != 0):

                    model.eval()                        # 构建缓存与分类器训练时 SSC 用 eval（与 transformer 一致）

                    # 每次重新初始化分类器（防止分类器记忆旧编码器的表示）
                    classifier = EfficientClassifier(ssc_output_, class_number).to(device)
                    classifier_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                    classifier_optimizer = torch.optim.Adam(
                        classifier.parameters(), lr=classifier_lr_, weight_decay=1e-3
                    )
                    total_loss = 0.0                                # 累计分类损失（用于统计）
                    loss_rounds = 0                                 # 参与平均的轮次数

                    # K 份不同随机增强的 SSC 表示缓存；分类器迭代时轮换，避免过拟合单一增强（同 transformer）
                    K = 12
                    train_ssc_caches = []
                    for _ in range(K):
                        cache = []
                        with torch.no_grad():
                            for view1, view2, label, names, _ in trainloader:
                                view1 = view1.to(device)
                                view2 = view2.to(device)
                                backbone_view = torch.stack(
                                    [train_feature_dict[n] for n in names], dim=0
                                ).to(device)
                                ssc_view1 = model(view1)
                                ssc_view2 = model(view2)
                                cache.append(
                                    (
                                        backbone_view.cpu(),
                                        ssc_view1.cpu(),
                                        ssc_view2.cpu(),
                                        (label - 1).long().cpu(),
                                    )
                                )
                        train_ssc_caches.append(cache)

                    test_ssc_cache = []
                    with torch.no_grad():
                        for view1, view2, label, names_, _ in testloader:
                            view1 = view1.to(device)
                            view2 = view2.to(device)
                            backbone_view = torch.stack(
                                [test_feature_dict[n] for n in names_], dim=0
                            ).to(device)
                            ssc_view1 = model(view1)
                            ssc_view2 = model(view2)
                            test_ssc_cache.append(
                                (
                                    backbone_view.cpu(),
                                    ssc_view1.cpu(),
                                    ssc_view2.cpu(),
                                    (label - 1).long().cpu(),
                                )
                            )

                    for i in range(classifier_iteration_):
                        trainstyle_loss = []            # 本轮每个 batch 的分类损失
                        total_correct = 0.0             # 本轮训练集累计正确预测数
                        classifier.train()
                        rot_cache = train_ssc_caches[i % K]

                        for bb_feat, ssc_v1, ssc_v2, label in rot_cache:
                            bb_feat = bb_feat.to(device)
                            ssc_v1 = ssc_v1.to(device)
                            ssc_v2 = ssc_v2.to(device)
                            label = label.to(device)
                            # 轻微特征噪声，缓解对固定缓存的过拟合（同 transformer）
                            bb_feat = bb_feat + torch.randn_like(bb_feat) * 0.01
                            ssc_v1 = ssc_v1 + torch.randn_like(ssc_v1) * 0.01
                            ssc_v2 = ssc_v2 + torch.randn_like(ssc_v2) * 0.01

                            # ssc.classifier.EfficientClassifier.forward(ssc_view1, ssc_view2, backbone_feat)
                            prediction = classifier(ssc_v1, ssc_v2, bb_feat)
                            style_loss = classifier_criterion(prediction, label)
                            classifier_optimizer.zero_grad()
                            style_loss.backward()
                            classifier_optimizer.step()

                            trainstyle_loss.append(style_loss.item())
                            with torch.no_grad():
                                pred = prediction.data.max(1, keepdim=True)[1]
                                total_correct += pred.eq(label.data.view_as(pred)).cpu().sum()

                        if i % classifier_training_gap_ == classifier_training_gap_ - 1:
                            logger.info(
                                'The classifer-train round is %d, the training accuracy is %d/%d',
                                i,
                                total_correct,
                                len(trainset),
                            )

                        if i % classifier_test_gap_ == classifier_test_gap_ - 1:
                            test_correct = 0.0
                            classifier.eval()

                            with torch.no_grad():
                                for bb_feat, ssc_v1, ssc_v2, label in test_ssc_cache:
                                    bb_feat = bb_feat.to(device)
                                    ssc_v1 = ssc_v1.to(device)
                                    ssc_v2 = ssc_v2.to(device)
                                    label = label.to(device)
                                    prediction = classifier(ssc_v1, ssc_v2, bb_feat)
                                    pred = prediction.data.max(1, keepdim=True)[1]
                                    test_correct += pred.eq(label.data.view_as(pred)).cpu().sum()

                            test_accuracy = float(test_correct / len(testset))
                            last_accuracy = test_accuracy

                            if test_accuracy > best_accuracy:
                                accuracy_str = f"{test_accuracy:.4f}".split('.')[1][:4]
                                lt_classifier_name = (
                                    model_name_
                                    + '-SSC-resnet50-'
                                    + time_str
                                    + '-iteration-'
                                    + str(iteration)
                                    + '-accuracy-'
                                    + accuracy_str
                                    + '-SSC-classifier-best.pth'
                                )
                                lt_base_name = (
                                    model_name_
                                    + '-SSC-resnet50-'
                                    + time_str
                                    + '-iteration-'
                                    + str(iteration)
                                    + '-accuracy-'
                                    + accuracy_str
                                    + '-SSC-base-best.pth'
                                )
                                torch.save(model, model_path + lt_base_name)
                                torch.save(classifier, model_path + lt_classifier_name)
                                logger.info(
                                    '+++THE BEST MODEL is saved+++. The iteration is %d, the best accuracy is %f, and the current accuracy is %f',
                                    iteration,
                                    best_accuracy,
                                    test_accuracy,
                                )
                                best_accuracy = test_accuracy

                            logger.info(
                                'Test result is: The test round is %d, the test ratio is %d/%d, the test accuracy is %f',
                                i,
                                test_correct,
                                len(testset),
                                test_accuracy,
                            )

                        total_loss += float(np.mean(trainstyle_loss)) if trainstyle_loss else 0.0
                        loss_rounds += 1

                    if loss_rounds > 0:
                        logger.info(
                            'Classifier train avg loss (this trigger): %f',
                            total_loss / loss_rounds,
                        )

                    # 全部训练结束时保存最终模型（无论精度高低）
                    if epoch == epochs - 1 and iteration == iterations - 1:
                        lt_classifier_name = model_name_ + '-SSR-resnet50-' + time_str + '-iteration-' + str(iteration) + '-SSC-classifier-last.pth'
                        lt_base_name = model_name_ + '-SSR-resnet50-' + time_str + '-iteration-' + str(iteration) + '-SSC-base-last.pth'
                        torch.save(model, model_path + lt_base_name)            # 保存最终编码器
                        torch.save(classifier, model_path + lt_classifier_name) # 保存最终分类器
                        logger.info('The last models are saved. The last accuracy is %f', last_accuracy)

        run_bests.append(best_accuracy)
        logger.info(
            '[RUN_BEST] run=%d/%d best_accuracy=%.6f',
            run_idx + 1,
            dataset_repeat_runs,
            best_accuracy,
        )

    mean_b = float(statistics.mean(run_bests))
    std_b = float(statistics.stdev(run_bests)) if len(run_bests) >= 2 else 0.0
    logger.info(
        '[DATASET_SUMMARY] dataset=%s repeat_runs=%d best_per_run=%s mean=%.6f std=%.6f',
        dataset,
        dataset_repeat_runs,
        [round(x, 6) for x in run_bests],
        mean_b,
        std_b,
    )
    return (run_bests, mean_b, std_b)


if __name__ == '__main__':
    # 数据集类别数映射表（与历史脚本一致；未知数据集默认 10 类）
    class_num_dict = {
        'Painting91': 13,
        'Pandora': 12,
        'WikiArt3': 15,
        'Arch': 25,
        'FashionStyle14': 14,
        'artbench': 10,
        'webstyle': 10,
        'webstyle/subImages': 10,  # 旧目录结构兼容
        'AVAstyle': 14,
    }

    args = parse_train_args()
    dataset_name = args.dataset_name
    data_root = args.data_root.rstrip('/') + '/'
    dataSource = os.path.join(data_root.rstrip('/'), dataset_name).rstrip('/') + '/'
    class_number = class_num_dict.get(dataset_name, 10)

    safe_dataset_name = dataset_name.replace('/', '_')
    model_name = f'ssc-{safe_dataset_name}'
    feature_name = f'{safe_dataset_name}_resnet50'
    preFeaturePath = args.pre_feature_path
    model_path = args.model_path.rstrip('/') + '/'
    iterations = args.iterations
    training_mode = args.training_mode
    base_model_path = args.base_model_path

    os.makedirs(model_path, exist_ok=True)
    os.makedirs('./log', exist_ok=True)
    os.makedirs(preFeaturePath, exist_ok=True)

    # 初始化日志记录器
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)                              # 日志级别：DEBUG（记录所有级别）
    logger.propagate = False                                    # 禁止日志向根 logger 冒泡，防止产生重复 log 文件
    handler = logging.StreamHandler()                           # 控制台输出 handler
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(message)s") # 日志格式：时间 + 消息
    handler.setFormatter(formatter)
    logger.addHandler(handler)                                  # 注册控制台 handler

    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())  # 生成训练启动时间戳
    log_name = model_name + '-' + current_time + '.log'        # 日志文件名（含时间戳防止覆盖）
    filehandler = logging.FileHandler("./log/" + log_name)     # 文件输出 handler
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)                              # 注册文件 handler

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
        preFeaturePath,
        feature_name,
        train_args=args,
        dataset_repeat_runs=args.dataset_repeat_runs,
    )

    logger.removeHandler(filehandler)                           # 移除文件 handler，释放文件锁
    logger.removeHandler(handler)                               # 移除控制台 handler
    logging.shutdown()                                          # 关闭日志系统
