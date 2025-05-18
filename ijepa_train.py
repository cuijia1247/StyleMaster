# Author: cuijia1247
# Date: 2025-4-27
# version: 1.0
import logging
import time
import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from ijepa.pretrain_IJEPA import IJEPA
# from barlowtwins.barlow import BarlowTwins, barlow_loss_fun
from ssc.utils import criterion, get_byol_transforms, MultiViewDataInjector
from SscDataSet import SscDataset
from ssc.classifier import Classifier


#setup device for cuda or cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parameter_load():
    epochs = 200 #best, perhaps6001
    # backbone = 'resnet50'
    # ssc_backend = 'resnet50'
    ssc_input = 2048
    ssc_output = 2048
    batch_size_ = 196
    # batch_size_sample = 'None'
    offset_bs = 512
    base_lr = 0.008 #best
    image_size = 64 #best
    classfier_iteration = 100 #best
    # classfier_iteration = 300  # best
    classifier_lr = 0.0005 #best
    # classifier_structure = '2048-1024-512-13 with dropout'
    classifier_training_gap = 25
    classifier_test_gap = 25
    model_name = ''
    return (epochs, batch_size_, base_lr, image_size, classfier_iteration, classifier_lr, model_name,
            classifier_training_gap, ssc_input, ssc_output, classifier_test_gap, offset_bs)#, classifier_structure

def ijepa_train(logger, model_path, current_time, opt_model_name, dataset, class_number):
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.debug('THIS IS THE FORMAL TRAINING PROCESS')
    logger.debug('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('barlow parameter setting up...')
    # load all the parameters
    (epochs_, batch_size_, base_lr_, image_size_, classifier_iteration_, classifier_lr_, model_name_,
     classifier_training_gap_, ssc_input_, ssc_output_, classifier_test_gap_, offset_bs_)= parameter_load()
    # the training parameters
    epochs = epochs_
    batch_size = batch_size_
    offset_bs = offset_bs_
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
    logger.info('IJEPA learning rate = %f', base_lr)
    # logger.info('sub patch size = (%d, %d)', image_size, image_size)
    # logger.info('sub pathc sample is %s', batch_size_sample_)
    logger.info('classifier training gap = %d', classifier_training_gap_)
    logger.info('classifier test gap = %d', classifier_test_gap_)
    logger.info('classifier iteration is %d', classifier_iteration_)
    logger.info('classifier learning rate = %f', classifier_lr_)
    # logger.info('classifier structure = %s', classifier_structure_)  ####optimal
    logger.info('model name is %s', model_name_)
    # logger.info('SSC output is %d', ssc_output)

    #normalize and randomcrop input images
    transformT, transformT1, transformEvalT = get_byol_transforms(image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    #setup dataset & dataloader
    dataSource = dataset
    trainData = 'train'
    trainset = SscDataset(dataSource, trainData, transform=MultiViewDataInjector([transformT, transformT1]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testData = 'test'
    testset = SscDataset(dataSource, testData, transform=MultiViewDataInjector([transformT, transformT1]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    logger.info('IJEPA ' + dataSource + ' is ready...')

    lr = 3e-4
    # define optimizer
    pretrained_model_path = '/home/cuijia1247/Codes/SubStyleClassfication/ijepa/lightning_logs/version_7/checkpoints/epoch=9-step=70.ckpt'
    model = IJEPA.load_from_checkpoint(pretrained_model_path) #if this is work, the param load func could be unused.
    ssc_output_ = 50176
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(ssc_input_, ssc_output_)
    resnet50 = resnet50.eval()
    model = model.to(device)
    resnet50 = resnet50.to(device)
    params = model.parameters()
    optimizer = optim.SGD(params, lr=lr, weight_decay=1.5e-6)
    logger.info('IJEPA test model is ready...')
    # barlow_lambda = 5e-3
    # classifier = Classifier(ssc_output_, class_number).cuda()
    classifier = Classifier(ssc_output_, class_number).cuda()
    classifier_criterion = nn.CrossEntropyLoss()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr_)
    total_loss = 0.0
    style_loss = torch.zeros(1).cuda()
    # time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    time_str = current_time
    best_accuracy = 0.0
    last_accuracy = 0.0
    for epoch in range(epochs):
        # print('epoch is {}'.format(epoch))
        trainstyle_loss = []
        total_correct = 0.0
        tk1 = trainloader
        tk2 = testloader
        num = 0
        for view1, view2, label, name, original in tk1:
            num = num + 1
            correct = 0.0
            view1 = view1.to(device).detach()
            view2 = view2.to(device).detach()
            test1 = model.model(view2)[0]
            test1 = test1.permute(1, 0, 2, 3)
            test1 = test1.reshape(test1.shape[0], -1)
            # test2 = model.model(view2)[0]
            # test2 = test2.permute(1, 0, 2, 3)
            # test2 = test2.reshape(test2.shape[0], -1)
            ##combine the dimensionalities
            prediction1 = classifier(test1)
            # prediction2 = classifier(view2)
            label = label - 1
            label = Variable(label).cuda()
            style_loss = classifier_criterion(prediction1, label)
            classifier_optimizer.zero_grad()
            # style_loss.requires_grad_()
            style_loss.backward()
            classifier_optimizer.step()
            pred = prediction1.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            # correct = idx.eq(label).cpu().sum()
            total_correct += correct
            print('total accumulating is {}, iterations is {}'.format(total_correct, num))
            # total_loss += style_loss
        print('total correct is {} / {} This is the {}th round'.format(total_correct, len(trainset), epoch))
        trainstyle_loss.append(style_loss.item())
        total_loss += np.mean(trainstyle_loss)
            # total_loss = total_loss / 50
        if epoch == epochs - 1:
            lt_classifier_name = model_name_ + '-IJEPA-resnet50-' + time_str + '-SSC-classifier-last.pth'
            # lt_base_name = model_name_ + '-IJEPA-resnet50-' + time_str + '-SSC-base-last.pth'
            # torch.save(model, model_path + lt_base_name)
            torch.save(classifier, model_path + lt_classifier_name)
            logger.info('The last models are saved. The last accuracy is %f', last_accuracy)
    logger.info('The best accuracy is %f, and the last accuracy is %f', best_accuracy, last_accuracy)
    logging.shutdown()


if __name__ == '__main__':
    model_path = './model/'
    # dataSource = './data/Painting91/' #painting 91 dataset, classes = 13
    # dataSource = './data/Pandora/'  # pandora dataset, classes = 12
    # dataSource = './data/WikiArt3/'  # WikiArt3 dataset, classes = 15
    # dataSource = './data/Arch/'  # Arch dataset, classes = 25
    # dataSource = './data/FashionStyle14/'  # FashionStyle14 dataset, classes = 14
    # dataSource = './data/artbench/' #artbench dataset, classes = 10
    dataSource = '/home/cuijia1247/Codes/SubStyleClassfication/data/Painting91/'  # the '/' is necessary
    class_number = 13
    model_name = 'IJEPA_painting91'
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
    ijepa_train(logger, model_path, current_time, model_name, dataSource, class_number)
    logger.removeHandler(filehandler)
    logger.removeHandler(handler)













