# SSC 共性风格增强策略训练脚本
# 策略：SSC 编码器学习双视图公共风格特征（一致性对齐损失），
#       分类器将公共风格门控叠加到 backbone_feat 上进行增强融合分类。
# 与 ssc_train_transformer.py 的核心区别：
#   1. criterion → criterion_align（去掉 ortho_loss，改为对角对齐损失）
#   2. EfficientClassifier → 来自 classifier_enhance_add.py（StyleEnhancer 门控增强）
#   3. 其余训练流程、早停、缓存机制完全保持一致，便于公平对比

import logging
import time
import os
import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
import numpy as np

from ssc.Sscreg_transformer import SscReg
from ssc.utils_add import criterion_align, get_ssc_transforms, MultiViewDataInjector
from SscDataSet_new import SscDataset
from ssc.classifier_enhance_add import Classifier, EfficientClassifier
from utils.pretrainFeatureExtraction import load_dataFeatures

device0 = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device1 = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parameter_load():
    """超参数集中管理（与 ssc_train_transformer.py 保持一致，便于对比）"""
    epochs                  = 35
    ssc_backend             = 'swin_base_patch4_window7_224'
    ssc_input               = 1024
    ssc_output              = 1024
    batch_size_             = 128
    offset_bs               = 512
    base_lr                 = 0.001
    image_size              = 224
    classfier_iteration     = 100
    classifier_lr           = 0.00005
    classifier_training_gap = 2
    classifier_test_gap     = 2
    model_name              = ''
    return (epochs, batch_size_, offset_bs, base_lr, image_size, classfier_iteration,
            classifier_lr, model_name, classifier_training_gap,
            ssc_backend, ssc_input, ssc_output, classifier_test_gap)


def SSCtrain(logger, model_path, current_time, opt_model_name, dataset, class_number,
             iterations, training_mode, base_model_path, preFeaturePath, feature_name):
    """
    SSC 共性风格增强策略主训练函数。

    与 ssc_train_transformer.SSCtrain 的唯一逻辑差异：
      - SSC 损失：criterion_align（一致性对齐）替代 criterion（正交化）
      - 分类器：EfficientClassifier from classifier_enhance_add（StyleEnhancer 门控增强）
    """
    logger.debug('=' * 110)
    logger.debug('SSC STYLE-ALIGNMENT TRAINING (utils_add + classifier_enhance_add)')
    logger.debug('=' * 110)
    logger.info('SSC parameter setting up...')

    (epochs_, batch_size_, offset_bs_, base_lr_, image_size_, classifier_iteration_,
     classifier_lr_, model_name_, classifier_training_gap_,
     ssc_backend_, ssc_input_, ssc_output_, classifier_test_gap_) = parameter_load()

    epochs      = epochs_
    batch_size  = batch_size_
    offset_bs   = offset_bs_
    base_lr     = base_lr_
    image_size  = image_size_
    model_name_ = opt_model_name

    logger.info('dataset = %s', dataset)
    logger.info('[ADD] SSC loss = criterion_align (align + var, NO ortho)')
    logger.info('[ADD] Classifier = StyleFusionClassifier (StyleEnhancer gate + 3-branch)')
    logger.info('epochs = %d', epochs)
    logger.info('batch_size = %d, offset_batch_size = %d', batch_size, offset_bs)
    logger.info('SSC backend = %s', ssc_backend_)
    logger.info('SSC input = %d, output = %d', ssc_input_, ssc_output_)
    logger.info('SSC learning rate = %f', base_lr)
    logger.info('sub patch size = (%d, %d)', image_size, image_size)
    logger.info('classifier iteration = %d', classifier_iteration_)
    logger.info('classifier learning rate = %f', classifier_lr_)
    logger.info('classifier training gap = %d', classifier_training_gap_)
    logger.info('classifier test gap = %d', classifier_test_gap_)
    logger.info('model name = %s', model_name_)

    transformT, transformT1, transformEvalT = get_ssc_transforms(
        image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if training_mode == 'original':
        model = SscReg(input_size=ssc_input_, output_size=ssc_output_, backend=ssc_backend_)
        model = model.to(device0)
        params    = model.projector.parameters()
        lr        = base_lr * batch_size / offset_bs
        optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
        # 余弦退火：与原脚本相同，防止后期过度优化对齐目标
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_, eta_min=lr * 0.01)
        time_str      = current_time
        best_accuracy = 0.0
        last_accuracy = 0.0
        logger.info('SSC original mode is ready...')
    else:
        model = torch.load(model_path + 'base-best.pth')
        model = model.to(device0)
        params    = model.projector.parameters()
        lr        = base_lr * batch_size / offset_bs
        optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_, eta_min=lr * 0.01)
        time_str      = current_time
        best_accuracy = 0.0
        last_accuracy = 0.0
        logger.info('SSC fine-tuning mode is ready...')

    train_feature_path = os.path.join(preFeaturePath, f"{feature_name}_train_features.pkl")
    train_feature_dict = load_dataFeatures(train_feature_path)
    test_feature_path  = os.path.join(preFeaturePath, f"{feature_name}_test_features.pkl")
    test_feature_dict  = load_dataFeatures(test_feature_path)

    for iteration in range(iterations):
        logger.info('The iteration is %d', iteration)

        trainset   = SscDataset(dataset, 'train',
                                transform=MultiViewDataInjector([transformT, transformT1]))
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True)

        testset    = SscDataset(dataset, 'test',
                                transform=MultiViewDataInjector([transformT, transformT1]))
        testloader  = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False)
        logger.info('SSC %s for %d iterations is ready...', dataset, iteration)

        for epoch in range(epochs):
            print(f'current epoch is {epoch}')
            model.train()
            train_loss = []

            for view1, view2, label, name, _ in trainloader:
                view1 = view1.to(device0)
                view2 = view2.to(device0)
                # label 从 1-indexed 转为 0-indexed，传入 SupCon 监督对齐方向
                cls_label = (label - 1).to(device0)
                fx    = model(view1)
                fx1   = model(view2)
                # 对齐损失 + 有监督对比损失，保证公共风格具有类间判别性
                loss  = criterion_align(fx, fx1, labels=cls_label)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            if epoch % 10 == 0 or epoch == epochs - 1:
                current_lr = scheduler.get_last_lr()[0]
                logger.info('The epoch is %d, SSC train loss is %f, lr is %.2e',
                            epoch, np.mean(train_loss), current_lr)

            if (epoch % classifier_training_gap_ == 0 and epoch != 0) or \
               (epoch == epochs - 1 and epoch != 0):
                model.eval()

                # 每次从零初始化：classifier loss 未回传给 SSC，热启动会导致过拟合
                classifier = EfficientClassifier(ssc_output_, class_number).to(device0)
                classifier_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                classifier_optimizer = torch.optim.Adam(
                    classifier.parameters(), lr=classifier_lr_, weight_decay=1e-3)
                # 余弦退火：让 lr 在 classifier_iteration 轮内平滑衰减至 0
                classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    classifier_optimizer, T_max=classifier_iteration_,
                    eta_min=classifier_lr_ * 0.01)

                total_loss  = 0.0
                loss_count  = 0
                style_loss  = torch.zeros(1).to(device1)

                # 预计算 K 份随机增强缓存（与原脚本相同，保证公平对比）
                K = 12
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
                            cache.append((backbone_view.cpu(), ssc_view1.cpu(),
                                          ssc_view2.cpu(), (label - 1)))
                    train_ssc_caches.append(cache)

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
                        test_ssc_cache.append((backbone_view.cpu(), ssc_view1.cpu(),
                                               ssc_view2.cpu(), (label - 1)))

                es_patience   = 20
                es_no_improve = 10
                es_best_acc   = 0.0

                for i in range(classifier_iteration_):
                    trainstyle_loss = []
                    total_correct   = 0.0
                    train_ssc_cache = train_ssc_caches[i % K]

                    classifier.train()
                    for bb_feat, ssc_v1, ssc_v2, label in train_ssc_cache:
                        bb_feat = bb_feat.to(device1)
                        ssc_v1  = ssc_v1.to(device1)
                        ssc_v2  = ssc_v2.to(device1)
                        label   = label.to(device1)

                        # 特征噪声扰动（与原脚本相同）
                        bb_feat = bb_feat + torch.randn_like(bb_feat) * 0.01
                        ssc_v1  = ssc_v1  + torch.randn_like(ssc_v1)  * 0.01
                        ssc_v2  = ssc_v2  + torch.randn_like(ssc_v2)  * 0.01

                        prediction = classifier(ssc_v1, ssc_v2, bb_feat)
                        style_loss = classifier_criterion(prediction, label)
                        classifier_optimizer.zero_grad()
                        style_loss.backward()
                        classifier_optimizer.step()

                        trainstyle_loss.append(style_loss.item())
                        with torch.no_grad():
                            pred = prediction.data.max(1, keepdim=True)[1]
                            total_correct += pred.eq(
                                label.data.view_as(pred)).cpu().sum()

                    classifier_scheduler.step()

                    if i % classifier_training_gap_ == classifier_training_gap_ - 1:
                        logger.info(
                            'The classifer-train round is %d, the training accuracy is %d/%d',
                            i, total_correct, len(trainset))

                    if i % classifier_test_gap_ == classifier_test_gap_ - 1:
                        test_correct = 0.0
                        classifier.eval()
                        with torch.no_grad():
                            for bb_feat, ssc_v1, ssc_v2, label in test_ssc_cache:
                                bb_feat = bb_feat.to(device1)
                                ssc_v1  = ssc_v1.to(device1)
                                ssc_v2  = ssc_v2.to(device1)
                                label   = label.to(device1)
                                prediction = classifier(ssc_v1, ssc_v2, bb_feat)
                                pred = prediction.data.max(1, keepdim=True)[1]
                                test_correct += pred.eq(
                                    label.data.view_as(pred)).cpu().sum()

                        test_accuracy = float(test_correct / len(testset))
                        last_accuracy = test_accuracy

                        if test_accuracy > best_accuracy and test_accuracy > 0.45:
                            accuracy_str = f"{test_accuracy:.4f}".split('.')[1][:4]
                            lt_cls_name  = (model_name_ + '-ADD-SWIN-BASE-' + time_str
                                            + '-iteration-' + str(iteration)
                                            + '-accuracy-' + accuracy_str
                                            + '-SSC-classifier-best.pth')
                            lt_base_name = (model_name_ + '-ADD-SWIN-BASE-' + time_str
                                            + '-iteration-' + str(iteration)
                                            + '-accuracy-' + accuracy_str
                                            + '-SSC-base-best.pth')
                            torch.save(model, model_path + lt_base_name)
                            torch.save(classifier, model_path + lt_cls_name)
                            logger.info(
                                '+++THE BEST MODEL is saved+++. The iteration is %d, '
                                'the best accuracy is %f, and the current accuracy is %f',
                                iteration, best_accuracy, test_accuracy)
                            best_accuracy = test_accuracy

                        logger.info(
                            'Test result is: The test round is %d, the test ratio is %d/%d, '
                            'the test accuracy is %f',
                            i, test_correct, len(testset), test_accuracy)

                        if test_accuracy > es_best_acc:
                            es_best_acc   = test_accuracy
                            es_no_improve = 0
                        else:
                            es_no_improve += 1
                            if es_no_improve >= es_patience:
                                logger.info('Early stopping at round %d, best local acc: %f',
                                            i, es_best_acc)
                                break

                total_loss += np.mean(trainstyle_loss)
                loss_count += 1
                avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
                logger.info('The average loss is %.6e', avg_loss)

                if epoch == epochs - 1 and iteration == iterations - 1:
                    lt_cls_name  = (model_name_ + '-ADD-SWIN-BASE-' + time_str
                                    + '-iteration-' + str(iteration)
                                    + '-SSC-classifier-last.pth')
                    lt_base_name = (model_name_ + '-ADD-SWIN-BASE-' + time_str
                                    + '-iteration-' + str(iteration)
                                    + '-SSC-base-last.pth')
                    torch.save(model, model_path + lt_base_name)
                    torch.save(classifier, model_path + lt_cls_name)
                    logger.info('The last models are saved. The last accuracy is %f',
                                last_accuracy)

    logger.info('The best accuracy is %f, and the last accuracy is %f',
                best_accuracy, last_accuracy)
    logging.shutdown()


if __name__ == '__main__':
    model_path = './model/'

    # ── 配置：只需修改数据集名称 ────────────────────────────────────────────────
    dataset_name = 'Painting91'

    class_num_dict = {
        'Painting91': 13, 'Pandora': 12, 'WikiArt3': 15, 'WikiArt3_small': 15,
        'Arch': 25, 'FashionStyle14': 14, 'artbench': 10,
        'webstyle/subImages': 10, 'AVAstyle': 14,
    }

    data_root        = '/mnt/codes/data/style/'
    dataSource       = os.path.join(data_root, dataset_name) + '/'
    class_number     = class_num_dict.get(dataset_name, 10)
    safe_name        = dataset_name.replace('/', '_')
    model_name       = f'ssc-add-{safe_name}'      # 文件名含 'add'，与原脚本区分
    feature_name     = f'{safe_name}_vit'
    preFeaturePath   = './pretrainFeatures'

    logger = logging.getLogger('ssc_add_logger')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    log_name     = model_name + '-' + current_time + '.log'
    filehandler  = logging.FileHandler('./log/' + log_name)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    iterations    = 5
    training_mode = 'original'
    base_model_path = './model/base-best.pth'

    SSCtrain(logger, model_path, current_time, model_name, dataSource, class_number,
             iterations, training_mode, base_model_path, preFeaturePath, feature_name)

    logger.removeHandler(filehandler)
    logger.removeHandler(handler)
    logging.shutdown()
