# ssc transformer training code after 20250425
# Author: cuijia1247
# Date: 2025-10-19
# version: 2.0
import logging
import os
import time
import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from ssc.Sscreg_transformer import SscReg
from ssc.utils import criterion, get_ssc_transforms, MultiViewDataInjector
from SscDataSet import SscDataset
from ssc.classifier import Classifier

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not available. Please install it with: pip install timm")

#setup device for cuda:1 or cpu
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def parameter_load():
    epochs = 210 #best, perhaps6001
    backbone = 'swin_base_patch4_window7_224'
    ssc_backend = 'swin_base_patch4_window7_224'
    ssc_input = 2048
    ssc_output = 2048
    batch_size_ = 32
    batch_size_sample = 'None'
    offset_bs = 512
    # base_lr = 0.008 # best
    base_lr = 0.009 # current
    image_size = 224 # best
    # classfier_iteration = 180 # best
    classfier_iteration = 210  # current
    classifier_lr = 0.0004 #best
    # classifier_structure = '2048-1024-512-13 with dropout'
    # classifier_training_gap = 30 # best
    # classifier_test_gap = 30 # best
    classifier_training_gap = 15 # current
    classifier_test_gap = 30 # current
    model_name = ''
    return (epochs, batch_size_, offset_bs, base_lr, image_size, classfier_iteration, classifier_lr, model_name, batch_size_sample,
            classifier_training_gap, backbone, ssc_backend, ssc_input, ssc_output, classifier_test_gap)#, classifier_structure

def SSCtrain(logger, model_path, current_time, opt_model_name, dataset, class_number, iterations, training_mode, base_model_path):
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.debug('THIS IS THE FORMAL TRAINING PROCESS OF SSC TRAIN WITH TRANSFORMER')
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('SSC parameter setting up...')
    # load all the parameters
    (epochs_, batch_size_, offset_bs_, base_lr_, image_size_, classifier_iteration_, classifier_lr_, model_name_, batch_size_sample_,
     classifier_training_gap_, backbone_, ssc_backend_, ssc_input_, ssc_output_, classifier_test_gap_)= parameter_load()
    # the training parameters
    epochs = epochs_
    batch_size = batch_size_
    offset_bs = offset_bs_
    base_lr = base_lr_
    image_size = image_size_
    model_name_ = opt_model_name  ####optimal
    # display all the necessary parameters & record them in logger
    logger.info('dataset = %s', dataset)
    logger.info('backbone is %s', backbone_) # for now backbone == backend
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
    # logger.info('classifier structure = %s', classifier_structure_)  ####optimal
    logger.info('model name is %s', model_name_)
    # logger.info('SSC output is %d', ssc_output)

    #normalize and randomcrop input images
    transformT, transformT1, transformEvalT = get_ssc_transforms(image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if training_mode == 'original':
        #set up the SSC model
        # model = SscReg(input_size=2048, output_size = 2048, backend='swin_base_patch4_window7_224')
        # SscReg类会自动处理本地模型加载，无需网络下载
        model = SscReg(input_size=ssc_input_, output_size=ssc_output_, backend=ssc_backend_, pretrained_backend=False)
        model = model.to(device)
        params = model.parameters()
        lr = base_lr*batch_size/offset_bs
        optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
        # time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        time_str = current_time
        best_accuracy = 0.0
        last_accuracy = 0.0
        logger.info('SSC original mode is ready...')
    else:
        #set up the SSC model
        # model = SscReg(input_size=2048, output_size = 2048, backend='swin_base_patch4_window7_224')
        # model = SscReg(input_size=ssc_input_, output_size=ssc_output_, backend=ssc_backend_)
        model = torch.load(model_path+'base-best.pth')
        if TIMM_AVAILABLE:
            # 使用本地预训练模型，避免网络下载
            swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=ssc_output_)
            # 加载本地预训练权重
            local_model_path = 'pretrainModels/swin_base_patch4_window7_224.pth'
            if os.path.exists(local_model_path):
                print(f"加载本地预训练模型: {local_model_path}")
                state_dict = torch.load(local_model_path, map_location='cpu')
                # 移除分类器权重，只保留backbone特征提取器
                backbone_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
                swin_transformer.load_state_dict(backbone_state_dict, strict=False)
                print("本地模型加载成功")
            else:
                print(f"警告: 本地模型文件不存在 {local_model_path}，使用随机初始化权重")
        else:
            raise ImportError("timm library is required for Swin Transformer. Please install it with: pip install timm")
        swin_transformer = swin_transformer.eval()
        model = model.to(device)
        swin_transformer = swin_transformer.to(device)
        params = model.parameters()
        lr = base_lr*batch_size/offset_bs
        optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
        # time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        time_str = current_time
        best_accuracy = 0.0
        last_accuracy = 0.0
        logger.info('SSC fine-tuning mode is ready...')

    for iteration in range(iterations):
        logger.info('The iteration is %d', iteration)
        #setup dataset & dataloader
        dataSource = dataset
        trainData = 'train'
        trainset = SscDataset(dataSource, trainData, transform=MultiViewDataInjector([transformT, transformT1]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testData = 'test'
        testset = SscDataset(dataSource, testData, transform=MultiViewDataInjector([transformT, transformT1]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        logger.info('SSC ' + dataSource + 'for ' + str(iteration) + ' iterations is ready...')
    

        for epoch in range(epochs):
            # print('epoch is {}'.format(epoch))
            tk0 = trainloader
            train_loss = []
            # temploss = total_loss / (1860*100)
            for view1, view2, label, name, _ in tk0:
                view1 = view1.to(device)
                view2 = view2.to(device)
                fx = model(view1)
                fx1 = model(view2)
                loss = criterion(fx, fx1, device=device)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0 or epoch == epochs-1:
                logger.info('The epoch is %d, SSC train loss is %f', epoch, np.mean(train_loss))
                # print('The epoch is {}, Vic train loss is {}'.format(epoch, np.mean(train_loss)))
                # train the style classifier every 500 iterations
            if epoch % classifier_training_gap_ == 0 and epoch != 0 or epoch == epochs-1:
                classifier = Classifier(ssc_output_, class_number).to(device)
                classifier_criterion = nn.CrossEntropyLoss()
                classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr_)
                total_loss = 0.0
                style_loss = torch.zeros(1).to(device)
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
                        original = original.to(device)
                        backbone_view = swin_transformer(original)
                        img1 = model(view1)  # only use view 1
                        img2 = model(view2)
                        test1 = backbone_view - img1
                        test2 = backbone_view - img2
                        test = test1 + test2
                        prediction = classifier(test)
                        # val, idx = prediction.topk(1)
                        # idx = idx.t().squeeze()
                        # idx = idx.cpu().float()
                        # original_label = label
                        # label = label.cpu().float()-1
                        label = label - 1
                        label = Variable(label).to(device)
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
                    if i % 30 == 29: #################need adjust based on performance#####################
                        logger.info('The classifer-train round is %d, the training accuracy is %d/%d', i, total_correct,
                                    len(trainset))
                        # print('The cla-train round is {}, the training ratio is {}/{}'.format(i, total_correct, len(trainset)))
                    # if i % 10 == 9:
                    if i % classifier_test_gap_ == classifier_test_gap_ - 1:
                        test_correct = 0.0
                        classifier.eval()
                        for view1, view2, label, name, original in tk2:
                            correct_ = 0.0
                            view1 = view1.to(device).detach()
                            view2 = view2.to(device).detach()
                            original = original.to(device)
                            backbone_view = swin_transformer(original)
                            img1 = model(view1)  # only use view 1
                            img2 = model(view2)
                            test1 = backbone_view - img1
                            test2 = backbone_view - img2
                            test = test1 + test2
                            prediction = classifier(test)
                            # val, idx = prediction.topk(1)
                            # idx = idx.t().squeeze()
                            # idx = idx.cpu().float()
                            # original_label = label
                            # label = label.cpu().float()-1
                            label = label - 1
                            label = Variable(label).to(device)
                            # style_loss = classifier_criterion(prediction, label)
                            # classifier_optimizer.zero_grad()
                            # style_loss.requires_grad_()
                            # style_loss.backward()
                            # classifier_optimizer.step()
                            pred = prediction.data.max(1, keepdim=True)[1]
                            correct_ += pred.eq(label.data.view_as(pred)).cpu().sum()
                            # correct = idx.eq(label).cpu().sum()
                            test_correct += correct_

                        # print('TEST RESULTS: The test round is {}, the test ratio is {}/{}, the test accuracy is {}'.format(i,
                        #             test_correct, len(testset), float(test_correct/len(testset))))
                        test_accuracy = float(test_correct / len(testset))
                        last_accuracy = test_accuracy
                        if test_accuracy > best_accuracy:  # the current best classifier
                            # Format accuracy to 4 decimal places, extract first 4 digits after decimal point
                            accuracy_str = f"{test_accuracy:.4f}".split('.')[1][:4]
                            lt_classifier_name = model_name_ + '-SSC-transformer-' + time_str + '-iteration-' + str(iteration) + '-accuracy-' + accuracy_str + '-SSC-classifier-best.pth'
                            lt_base_name = model_name_ + '-SSC-transformer-' + time_str + '-iteration-' + str(iteration) + '-accuracy-' + accuracy_str + '-SSC-base-best.pth'
                            torch.save(model, model_path + lt_base_name)
                            torch.save(classifier, model_path + lt_classifier_name)
                            logger.info(
                                '+++THE BEST MODEL is saved+++. The iteration is %d, the best accuracy is %f, and the current accuracy is %f',
                                iteration, best_accuracy, test_accuracy)
                                # best_accuracy, test_accuracy)
                            best_accuracy = test_accuracy
                        logger.info(
                            'Test result is: The test round is %d, the test ratio is %d/%d, the test accuracy is %f', i,
                            test_correct,
                            len(testset), test_accuracy)
                total_loss += np.mean(trainstyle_loss)
                total_loss = total_loss / 50
                if epoch == epochs - 1 and iteration == iterations - 1:
                    lt_classifier_name = model_name_ + '-SSR-transformer-' + time_str + '-iteration-' + str(iteration) + '-SSC-classifier-last.pth'
                    lt_base_name = model_name_ + '-SSR-transformer-' + time_str + '-iteration-' + str(iteration) + '-SSC-base-last.pth'
                    torch.save(model, model_path + lt_base_name)
                    torch.save(classifier, model_path + lt_classifier_name)
                    logger.info('The last models are saved. The last accuracy is %f', last_accuracy)
    logger.info('The best accuracy is %f, and the last accuracy is %f', best_accuracy, last_accuracy)
    logging.shutdown()


if __name__ == '__main__':
    # save_iteration = 1001 #not used for now
    model_path = './model/'
    # dataSource = './data/Painting91/' #painting 91 dataset, classes = 13
    # dataSource = './data/Pandora/'  # pandora dataset, classes = 12
    # dataSource = './data/WikiArt3/'  # WikiArt3 dataset, classes = 15
    # dataSource = './data/Arch/'  # Arch dataset, classes = 25
    # dataSource = './data/FashionStyle14/'  # FashionStyle14 dataset, classes = 14
    # dataSource = './data/artbench/' #artbench dataset, classes = 10
    # dataSource = './data/webstyle/subImages/'  # artbench dataset, classes = 10
    # class_number = 10
    dataSource = '/home/cuijia1247/Codes/SubStyleClassfication/data/Painting91/'  # the '/' is necessary
    class_number = 13
    # ssc_output = 2048 #the best
    model_name = 'ssc-painting91-transformer'
    #setup logger for record the process data
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    log_name = model_name + '-' + current_time + '.log'
    filehandler = logging.FileHandler("./log/" + log_name)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    ##########new add 20251018####################
    iterations = 3
    training_mode = 'original' # 'original' or 'fine-tuning'
    base_model_path = './model/ssc-painting91-SSR-transformer-2025-10-18-10-10-10-SSC-base-best.pth' # only for the fine-tuning mode
    ##########new add 20251018####################
    SSCtrain(logger, model_path, current_time, model_name, dataSource, class_number, iterations, training_mode, base_model_path)
    logger.removeHandler(filehandler)
    logger.removeHandler(handler)
    logging.shutdown()
