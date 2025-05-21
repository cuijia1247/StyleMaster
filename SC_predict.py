# Author: cuijia1247
# Date: 2024-7-19
# version: 1.0
# The codes do not finished yet by 20250425
import logging
import os
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
from ssc.classifier import Classifier
import torch.nn.functional as F
import torchvision.transforms as tr
from torchvision.transforms import transforms
from ssc.utils import criterion, get_byol_transforms, MultiViewDataInjector
import cv2

#setup device for cuda or cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

####################################################predict codes#############################

def SSC_predict(model_path, dataSource, label):
    label = label - 1
    base_model = torch.load(model_path+'base-best.pth')
    classfier_model = torch.load(model_path+'classifier-best.pth')
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(2048, ssc_output)
    base_model = base_model.eval()
    classfier_model = classfier_model.eval()
    resnet50 = resnet50.eval()
    base_model = base_model.to(device)
    classfier_model = classfier_model.to(device)
    resnet50 = resnet50.to(device)
    transformT, transformT1, transformEvalT = get_byol_transforms(64, (0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transforms_original = transforms.Compose([

        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(norm_mean, norm_std),
    ])
    transform = MultiViewDataInjector([transformT, transformT1])
    correct = 0
    total = 0
    for root, dirs, files in os.walk(dataSource):
        for file in files:
            total += 1
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
            # img = cv2.resize(img, (256, 256))
            img1, img2 = transform(img)
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
            view1 = base_model(img1)
            view2 = base_model(img2)
            img = transforms_original(img).to(device)
            img = img.unsqueeze(0)
            res_view = resnet50(img)
            test1 = res_view - view1
            test2 = res_view - view2
            test = test1 + test2
            prediction = classfier_model(test)

            # prediction1 = classfier_model(view1)
            # prediction2 = classfier_model(view2)
            pred = prediction.data.max(1, keepdim=True)[1]
            pred_num = pred.cpu().numpy().sum()

            test = int(pred_num)
            print('pred_num is {}'.format(test))
            if test == label:
                correct  = correct + 1
            print('test is {}, label is {}'.format(test, label))
            # pred1 = prediction1.data.max(1, keepdim=True)[1]
            # pred2 = prediction2.data.max(1, keepdim=True)[1]
            # print('The pred is {}, {}, {}'.format(pred, pred1, pred2))
    print('Accuacy is {} / {}'.format(correct, total))

def SSC_predict_in_batch(model_path, dataSource):
    # normalize and randomcrop input images
    # transformT, transformT1, transformEvalT = get_byol_transforms(64, (0.485, 0.456, 0.406),
    #                                                               (0.229, 0.224, 0.225))
    # testData = 'test'
    # testset = SscDataset(dataSource, testData, transform=MultiViewDataInjector([transformT, transformT1]))
    testset = torch.load('my_testset.pth')
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    tk2 = testloader


    base_model = torch.load(model_path+'base-best.pth')
    classfier_model = torch.load(model_path+'classifier-best.pth')
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(2048, 2048)
    # base_model = base_model.eval()
    # classfier_model = classfier_model.eval()
    # resnet50 = resnet50.eval()
    base_model = base_model.to(device)
    classfier_model = classfier_model.to(device)
    resnet50 = resnet50.to(device)

    correct = 0
    for view1, view2, label, name, original in tk2:
        correct_ = 0.0
        view1 = view1.to(device).detach()
        view2 = view2.to(device).detach()
        original = original.to(device)
        res_view1 = resnet50(original)
        img1 = base_model(view1)  # only use view 1
        img2 = base_model(view2)
        test1 = res_view1 - img1
        test2 = res_view1 - img2
        test = test1 + test2
        if 'JACKSON_POLLOCK_16.jpg' in name:
            print(test)
        prediction = classfier_model(test)
        # val, idx = prediction.topk(1)
        # idx = idx.t().squeeze()
        # idx = idx.cpu().float()
        # original_label = label
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
        for i in range(len(name)):
            print('{} pred is {}, label is {}'.format(name[i], pred[i], label[i]))
        # correct = idx.eq(label).cpu().sum()
        correct += correct_
    print('Accuacy is {} / {}'.format(correct, len(testset)))



if __name__ == '__main__':
    # label = 1
    # dataSource = '/home/cuijia1247/Codes/SubStyleClassfication/data/Painting91/test/' + str(label)  # artbench dataset, classes = 10
    # class_number = 13
    # ssc_output = 2048 #the best
    # model_path = '/home/cuijia1247/Codes/SubStyleClassfication/model/ssc_painting91-72.69/painting91-SSR-resnet50-0.7268907427787781-SSC-'
    # output_path = '/home/cuijia1247/Codes/SubStyleClassfication/data/style_output'
    #
    # SSC_predict(model_path, dataSource, class_number, output_path, label)

    dataSource = '/home/cuijia1247/Codes/SubStyleClassfication/data/Painting91/' # artbench dataset, classes = 10
    # class_number = 13
    # ssc_output = 2048  # the best
    model_path = '/home/cuijia1247/Codes/SubStyleClassfication/model/painting91-SSR-resnet50-0.5861344337463379-SSC-'
    # output_path = '/home/cuijia1247/Codes/SubStyleClassfication/data/style_output'

    SSC_predict_in_batch(model_path, dataSource)














