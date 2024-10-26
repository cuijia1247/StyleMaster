# Author: cuijia1247
# Date: 2024-7-19
# version: 1.0
import logging
import time
import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
from ssc.Sscreg import SscReg
from ssc.utils import criterion, get_byol_transforms, MultiViewDataInjector
from tqdm import tqdm
from SscDataSet import SscDataset

#setup device for cuda or cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def SSCtrain(logger, save_iteration, model_path, current_time):
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.debug('/3-level/ classifier /(2048, 1024, 256, 13)/, /ReLU/, /100/ iteration, lr = /0.0005/')
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('SSC parameter setting up...')
    #the training parameters
    epochs = 1001
    batch_size = 64
    offset_bs = 512
    base_lr = 0.05
    image_size = 64 # 32*32
    logger.info('epochs = %d', epochs)
    logger.info('batch_size = %d, offset_batch_size = %d', batch_size, offset_bs)
    logger.info('learning rate = %f', base_lr)
    logger.info('sub patch size = (%d, %d)', image_size, image_size)

    #normalize and randomcrop input images
    transformT, transformT1, transformEvalT = get_byol_transforms(image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


    #setup dataset & dataloader
    dataSource = './data/Painting91/'
    trainData = 'train'
    trainset = SscDataset(dataSource, trainData, transform=MultiViewDataInjector([transformT, transformT1]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testData = 'test'
    testset = SscDataset(dataSource, testData, transform=MultiViewDataInjector([transformT, transformT1]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    logger.info('SSC ' + dataSource + ' is ready...')

    lr = base_lr*batch_size/offset_bs
    #set up the SSC model
    model = SscReg(input_size=2048, output_size = 2048, backend='resnet50')
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(2048, 2048)
    resnet50 = resnet50.eval()
    model = model.to(device)
    resnet50 = resnet50.to(device)
    params = model.parameters()
    optimizer = optim.SGD( params, lr=lr, weight_decay=1.5e-6)
    logger.info('SSC model is ready...')


    # time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    time_str = current_time

    for epoch in range(epochs):
        # print('epoch is {}'.format(epoch))
        tk0 = trainloader
        train_loss = []
        best_accuracy = 0.0
        last_accuracy = 0.0
        # temploss = total_loss / (1860*100)
        for view1, view2, label, name, _ in tk0:
            view1 = view1.to(device)
            view2 = view2.to(device)
            fx = model(view1)
            fx1 = model(view2)
            loss = criterion(fx, fx1)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            logger.info('The epoch is %d, Vic train loss is %f', epoch, np.mean(train_loss))
            # print('The epoch is {}, Vic train loss is {}'.format(epoch, np.mean(train_loss)))
        #train the style classifier every 500 iterations
        if epoch%100 == 0:
            # set up the classification model
            classifier = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                # nn.Linear(4096, 1024),
                # nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 13),
                # nn.ReLU(),
            ).cuda()
            classifier_criterion = nn.CrossEntropyLoss()
            classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
            total_loss = 0.0
            style_loss = torch.zeros(1).cuda()
            logger.info('SSC classifier model is ready...')
            # model.eval()
            correct = 0.0
            total_number = len(trainset)
            for i in range(100):
                trainstyle_loss = []
                total_correct = 0.0
                tk1 = trainloader
                tk2 = testloader
                for view1, view2, label, name, original in tk1:
                    correct = 0.0
                    view1 = view1.to(device).detach()
                    view2 = view2.to(device).detach()
                    original = original.to(device)
                    res_view1 = resnet50(original)
                    img1 = model(view1)  # only use view 1
                    img2 = model(view2)
                    test1 = res_view1 - img1
                    test2 = res_view1 - img2
                    test = test1 + test2
                    prediction = classifier(test)
                    # val, idx = prediction.topk(1)
                    # idx = idx.t().squeeze()
                    # idx = idx.cpu().float()
                    original_label = label
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
                if i % 20 == 19:
                    logger.info('The classifer-train round is %d, the training accuracy is %d/%d', i, total_correct, len(trainset))
                    # print('The cla-train round is {}, the training ratio is {}/{}'.format(i, total_correct, len(trainset)))
                if i % 40 == 39:
                    test_correct = 0.0
                    classifier.eval()
                    for view1, view2, label, name, original in tk2:
                        correct_ = 0.0
                        view1 = view1.to(device).detach()
                        view2 = view2.to(device).detach()
                        original = original.to(device)
                        res_view1 = resnet50(original)
                        img1 = model(view1) # only use view 1
                        img2 = model(view2)
                        test1 = res_view1 - img1
                        test2 = res_view1 - img2
                        test = test1 + test2
                        prediction = classifier(test)
                        # val, idx = prediction.topk(1)
                        # idx = idx.t().squeeze()
                        # idx = idx.cpu().float()
                        original_label = label
                        # label = label.cpu().float()-1
                        label = label - 1
                        label = Variable(label).cuda()
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
                    test_accuracy = float(test_correct/len(testset))
                    last_accuracy = test_accuracy
                    if test_accuracy > best_accuracy: # the current best classifier
                        best_accuracy = test_accuracy
                        lt_classifier_name = 'SSR-resnet50-' + time_str + '-SSC-classifier-best.pth'
                        lt_base_name = 'SSR-resnet50-' + time_str + '-SSC-base-best.pth'
                        torch.save(model, model_path + lt_base_name)
                        torch.save(classifier, model_path + lt_classifier_name)
                        logger.info('The best models are saved. The best accuracy is %f', best_accuracy)
                    logger.info('TEST RESULTS: The test round is %d, the test ratio is %d/%d, the test accuracy is %f', i, test_correct,
                                len(testset), test_accuracy)
            total_loss += np.mean(trainstyle_loss)
            total_loss = total_loss / 50
            if epoch > 0:
                if epoch == epochs-1 or epoch%save_iteration==0 :
                    lt_classifier_name = 'SSR-resnet50-' + time_str + '-SSC-classifier-last.pth'
                    lt_base_name = 'SSR-resnet50-' + time_str + '-SSC-base-last.pth'
                    torch.save(model, model_path + lt_base_name)
                    torch.save(classifier, model_path + lt_classifier_name)
                    logger.info('The last models are saved. The last accuracy is %f', last_accuracy)
            # print('the epoch is {}, style classifier training loss is {}, correct number is {}/{}'.format(epoch, total_loss, total_correct, total_number))


if __name__ == '__main__':
    #setup logger for record the process data
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    log_name = current_time + '.log'
    filehandler = logging.FileHandler("./log/" + log_name)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    #begin to train.
    save_iteration = 1001
    model_path = './model/'
    SSCtrain(logger, save_iteration, model_path, current_time)












