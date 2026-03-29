# ssc new training code after 20250425
# Author: cuijia1247
# Date: 2026-3-28
# version: 1.1
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
from SscDataSet import SscDataset                       # 自定义数据集加载器
from ssc.classifier import Classifier, EfficientClassifier, AdvancedClassifier  # 三种风格分类头
from utils.pretrainFeatureExtraction import load_dataFeatures  # 加载预提取的骨干网络特征

# 优先使用 GPU cuda:1，不可用时退回 CPU
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def parameter_load():
    """集中管理所有超参数，便于统一调整"""
    epochs = 200                    # SSC 编码器训练轮数
    backbone = 'resnet50'           # 骨干网络名称（用于日志记录）
    ssc_backend = 'resnet50'        # SSC 编码器实际使用的 backend
    ssc_input = 2048                # backbone 输出特征维度（resnet50 为 2048）
    ssc_output = 2048               # SSC 投影头输出维度
    batch_size_ = 64                # 训练批大小
    batch_size_sample = 'None'      # 子图采样策略（当前未启用）
    offset_bs = 512                 # 学习率缩放基准 batch size（LARS 风格 lr scaling）
    base_lr = 0.008                 # 基础学习率
    image_size = 64                 # 随机裁剪子图的边长（像素）
    classfier_iteration = 200       # 每次触发分类器训练时的迭代轮数
    classifier_lr = 0.0005          # 分类器学习率
    # classifier_training_gap = 30  # 旧版：每隔多少 epoch 训练一次分类器
    # classifier_test_gap = 30      # 旧版：每隔多少轮测试一次分类器
    classifier_training_gap = 40    # 当前：每隔 20 个 epoch 触发一次分类器训练
    classifier_test_gap = 40        # 当前：分类器每训练 10 轮评估一次测试集
    model_name = ''                 # 模型名称前缀（由主程序传入覆盖）
    return (epochs, batch_size_, offset_bs, base_lr, image_size, classfier_iteration, classifier_lr, model_name, batch_size_sample,
            classifier_training_gap, backbone, ssc_backend, ssc_input, ssc_output, classifier_test_gap)

def SSCtrain(logger, model_path, current_time, opt_model_name, dataset, class_number, iterations, training_mode, base_model_path, preFeaturePath, feature_name):
    """
    SSC ResNet 版主训练函数
    Args:
        logger:          日志记录器
        model_path:      模型保存目录
        current_time:    训练启动时间戳（用于文件命名）
        opt_model_name:  模型名称前缀
        dataset:         数据集根目录路径
        class_number:    分类类别数
        iterations:      外层迭代次数（每次重新加载数据集）
        training_mode:   'original'（从头训练）或 'fine-tuning'（微调已有模型）
        base_model_path: fine-tuning 模式下加载的预训练模型路径
        preFeaturePath:  预提取特征的存放路径
        feature_name:    预提取特征的文件名前缀
    """
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.debug('THIS IS THE FORMAL TRAINING PROCESS OF SSC TRAIN')
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('SSC parameter setting up...')

    # 加载所有超参数
    (epochs_, batch_size_, offset_bs_, base_lr_, image_size_, classifier_iteration_, classifier_lr_, model_name_, batch_size_sample_,
     classifier_training_gap_, backbone_, ssc_backend_, ssc_input_, ssc_output_, classifier_test_gap_) = parameter_load()

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

    if training_mode == 'original':
        # 从头初始化 SSC 编码器
        model = SscReg(input_size=ssc_input_, output_size=ssc_output_, backend=ssc_backend_)
        model = model.to(device)                            # 将模型移至指定设备
        params = model.parameters()                         # 获取所有可训练参数
        lr = base_lr * batch_size / offset_bs               # LARS 风格线性 lr scaling
        optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)  # SGD 优化器，带 L2 正则
        time_str = current_time                             # 使用传入的时间戳（保证同一次训练文件名一致）
        best_accuracy = 0.0                                 # 历史最优测试精度
        last_accuracy = 0.0                                 # 最后一次测试精度
        logger.info('SSC original mode is ready...')
    else:
        # 从已保存的模型文件加载 SSC 编码器（微调模式）
        model = torch.load(model_path + 'base-best.pth')
        resnet50 = models.resnet50(pretrained=True)         # 加载 ImageNet 预训练 ResNet50
        resnet50.fc = nn.Linear(ssc_input_, ssc_output_)   # 替换全连接层以匹配输出维度
        resnet50 = resnet50.eval()                          # ResNet50 设为评估模式（仅用于参考，不参与反向传播）
        model = model.to(device)
        resnet50 = resnet50.to(device)
        params = model.parameters()
        lr = base_lr * batch_size / offset_bs
        optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
        time_str = current_time
        best_accuracy = 0.0
        last_accuracy = 0.0
        logger.info('SSC fine-tuning mode is ready...')

    # 加载预提取的骨干网络特征（避免每轮重复前向推理，显著加速训练）
    train_feature_path = os.path.join(preFeaturePath, f"{feature_name}_train_features.pkl")
    train_feature_dict = load_dataFeatures(train_feature_path)  # 训练集特征字典 {文件名: 特征张量}
    test_feature_path = os.path.join(preFeaturePath, f"{feature_name}_test_features.pkl")
    test_feature_dict = load_dataFeatures(test_feature_path)    # 测试集特征字典

    # 外层循环：多次迭代训练，每次重新打乱数据
    for iteration in range(iterations):
        logger.info('The iteration is %d', iteration)

        # 构建训练集与 DataLoader
        dataSource = dataset
        trainData = 'train'
        trainset = SscDataset(dataSource, trainData, transform=MultiViewDataInjector([transformT, transformT1]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        # 构建测试集与 DataLoader（不打乱，保证评估可复现）
        testData = 'test'
        testset = SscDataset(dataSource, testData, transform=MultiViewDataInjector([transformT, transformT1]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        logger.info('SSC ' + dataSource + 'for ' + str(iteration) + ' iterations is ready...')

        # 内层循环：SSC 编码器 epoch 级训练
        for epoch in range(epochs):
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

            # 每 10 个 epoch 或最后一个 epoch 打印一次训练损失
            if epoch % 10 == 9 or epoch == epochs - 1:
                logger.info('The epoch is %d, SSC train loss is %f', epoch, np.mean(train_loss))

            # 按间隔触发分类器训练（epoch 0 跳过，防止编码器未收敛）
            if epoch % classifier_training_gap_ == 0 and epoch != 0 or epoch == epochs - 1:

                # 每次重新初始化分类器（防止分类器记忆旧编码器的表示）
                classifier = EfficientClassifier(ssc_output_, class_number).to(device)
                classifier_criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失
                classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr_)  # Adam 优化器
                total_loss = 0.0                                # 累计分类损失（用于统计）
                style_loss = torch.zeros(1).to(device)         # 初始化风格损失占位张量

                # 分类器训练循环
                for i in range(classifier_iteration_):
                    trainstyle_loss = []            # 记录本轮每个 batch 的分类损失
                    total_correct = 0.0             # 本轮训练集累计正确预测数
                    tk1 = trainloader               # 训练数据迭代器
                    tk2 = testloader                # 测试数据迭代器

                    # 遍历训练集，更新分类器参数
                    for view1, view2, label, names, original in tk1:
                        correct = 0.0
                        view1 = view1.to(device).detach()       # 子图特征不参与编码器梯度计算
                        view2 = view2.to(device).detach()

                        # 从预提取特征字典中按文件名取出对应骨干特征（替代实时前向推理）
                        feature_list = []
                        for name in names:
                            feature_list.append(train_feature_dict[name])
                        backbone_view = torch.stack(feature_list, dim=0).to(device)  # 组装为 batch 张量

                        prediction = classifier(backbone_view)  # 分类器前向推理

                        label = label - 1                       # 标签从 1-based 转为 0-based（符合 CrossEntropyLoss 要求）
                        label = Variable(label).to(device)

                        style_loss = classifier_criterion(prediction, label)    # 计算分类损失
                        classifier_optimizer.zero_grad()                        # 清空分类器梯度
                        style_loss.backward()                                   # 反向传播
                        classifier_optimizer.step()                             # 更新分类器参数

                        pred = prediction.data.max(1, keepdim=True)[1]         # 取预测概率最大的类别索引
                        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # 统计当前 batch 正确数
                        total_correct += correct                                # 累加到本轮总正确数

                    trainstyle_loss.append(style_loss.item())   # 记录本轮最后一个 batch 的分类损失

                    # 每 10 轮打印一次训练集准确率
                    if i % 10 == 9:
                        logger.info('The classifer-train round is %d, the training accuracy is %d/%d', i, total_correct, len(trainset))

                    # 按间隔在测试集上评估分类器性能
                    if i % classifier_test_gap_ == classifier_test_gap_ - 1:
                        test_correct = 0.0
                        classifier.eval()                       # 切换为评估模式（关闭 Dropout/BN 的训练行为）

                        # 遍历测试集，统计正确预测数（不更新参数）
                        for view1, view2, label, names, original in tk2:
                            correct_ = 0.0
                            view1 = view1.to(device).detach()
                            view2 = view2.to(device).detach()

                            # 从预提取特征字典取测试集特征
                            feature_list = []
                            for name in names:
                                feature_list.append(test_feature_dict[name])
                            backbone_view = torch.stack(feature_list, dim=0).to(device)

                            prediction = classifier(backbone_view)  # 测试集分类推理

                            label = label - 1                       # 标签 0-based 转换
                            label = Variable(label).to(device)

                            pred = prediction.data.max(1, keepdim=True)[1]          # 取预测类别
                            correct_ += pred.eq(label.data.view_as(pred)).cpu().sum()  # 统计正确数
                            test_correct += correct_                                 # 累加测试正确数

                        # 计算测试集准确率
                        test_accuracy = float(test_correct / len(testset))
                        last_accuracy = test_accuracy                           # 更新最后一次精度

                        # 若当前精度超过历史最优，保存最佳模型
                        if test_accuracy > best_accuracy:
                            accuracy_str = f"{test_accuracy:.4f}".split('.')[1][:4]  # 取精度小数部分前4位用于文件命名
                            lt_classifier_name = model_name_ + '-SSC-resnet50-' + time_str + '-iteration-' + str(iteration) + '-accuracy-' + accuracy_str + '-SSC-classifier-best.pth'
                            lt_base_name = model_name_ + '-SSC-resnet50-' + time_str + '-iteration-' + str(iteration) + '-accuracy-' + accuracy_str + '-SSC-base-best.pth'
                            torch.save(model, model_path + lt_base_name)            # 保存最佳编码器
                            torch.save(classifier, model_path + lt_classifier_name) # 保存最佳分类器
                            logger.info(
                                '+++THE BEST MODEL is saved+++. The iteration is %d, the best accuracy is %f, and the current accuracy is %f',
                                iteration, best_accuracy, test_accuracy)
                            best_accuracy = test_accuracy                       # 更新历史最优精度

                        logger.info(
                            'Test result is: The test round is %d, the test ratio is %d/%d, the test accuracy is %f', i,
                            test_correct, len(testset), test_accuracy)

                total_loss += np.mean(trainstyle_loss)          # 累加本次分类器训练的平均损失
                total_loss = total_loss / 50                    # 归一化（固定分母，仅用于量级参考）

                # 全部训练结束时保存最终模型（无论精度高低）
                if epoch == epochs - 1 and iteration == iterations - 1:
                    lt_classifier_name = model_name_ + '-SSR-resnet50-' + time_str + '-iteration-' + str(iteration) + '-SSC-classifier-last.pth'
                    lt_base_name = model_name_ + '-SSR-resnet50-' + time_str + '-iteration-' + str(iteration) + '-SSC-base-last.pth'
                    torch.save(model, model_path + lt_base_name)            # 保存最终编码器
                    torch.save(classifier, model_path + lt_classifier_name) # 保存最终分类器
                    logger.info('The last models are saved. The last accuracy is %f', last_accuracy)

    # 输出训练全程最优精度和最终精度
    logger.info('The best accuracy is %f, and the last accuracy is %f', best_accuracy, last_accuracy)
    logging.shutdown()                                          # 关闭所有日志处理器


if __name__ == '__main__':
    model_path = './model/'                                     # 模型保存根目录

    # =====================================================================
    # 1. 核心配置：只需修改数据集名称
    # =====================================================================
    dataset_name = 'Painting91'                                 # 可选: 'Painting91', 'AVAstyle', 'Pandora', 'WikiArt3', 'Arch', 'FashionStyle14', 'artbench', 'webstyle/subImages'

    # 数据集类别数映射表
    class_num_dict = {
        'Painting91': 13,
        'Pandora': 12,
        'WikiArt3': 15,
        'Arch': 25,
        'FashionStyle14': 14,
        'artbench': 10,
        'webstyle/subImages': 10,
        'AVAstyle': 14
    }

    # =====================================================================
    # 2. 自动生成相关变量
    # =====================================================================
    data_root = '/mnt/codes/data/style/'                        # 数据集统一根目录
    dataSource = os.path.join(data_root, dataset_name) + '/'    # 自动生成数据源路径（末尾保留 '/'）
    class_number = class_num_dict.get(dataset_name, 10)         # 自动获取类别数，默认10
    
    # 处理带有子目录的数据集名称（如 webstyle/subImages）
    safe_dataset_name = dataset_name.replace('/', '_')
    model_name = f'ssc-{safe_dataset_name}'                     # 自动生成模型文件名前缀
    feature_name = f'{safe_dataset_name}_resnet50'                # 自动生成预提取特征文件名前缀
    preFeaturePath = './pretrainFeatures'                       # 预提取特征保存路径

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

    ##########new add 20251018####################
    iterations = 3                                              # 外层迭代次数（每次重新打乱数据集）
    training_mode = 'original'                                  # 训练模式：'original'从头训练 / 'fine-tuning'微调
    base_model_path = '###'                                     # 微调模式下的预训练模型路径（当前模式下不生效）
    ##########new add 20251018####################

    SSCtrain(logger, model_path, current_time, model_name, dataSource, class_number, iterations, training_mode, base_model_path, preFeaturePath, feature_name)

    logger.removeHandler(filehandler)                           # 移除文件 handler，释放文件锁
    logger.removeHandler(handler)                               # 移除控制台 handler
    logging.shutdown()                                          # 关闭日志系统
