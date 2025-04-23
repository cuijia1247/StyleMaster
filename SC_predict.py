# Author: cuijia1247
# Date: 2024-7-19
# version: 1.0
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

def SSC_predict(model_path, dataSource, class_number, output_path):
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
    for root, dirs, files in os.walk(dataSource):
        for file in files:
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (256, 256))
            img = transforms_original(img).to(device)
            img1, img2 = transform(img)
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
            view1 = base_model(img1)
            view2 = base_model(img2)
            res_view = resnet50(img)
            test1 = view1 - res_view
            test2 = view2 - res_view
            test = test1 + test2
            prediction = classfier_model(test)
            print(prediction)





if __name__ == '__main__':
    dataSource = '/home/cuijia1247/Codes/SubStyleClassfication/data/style_test_for_HR_20250423/0_for_test'  # artbench dataset, classes = 10
    class_number = 10
    ssc_output = 2048 #the best
    model_path = '/home/cuijia1247/Codes/SubStyleClassfication/model/SSC_20250423/webstyle-62.68/webstyle-SSR-resnet50-2025-04-22-06-36-26-SSC-'
    output_path = '/home/cuijia1247/Codes/SubStyleClassfication/data/style_output'
    SSC_predict(model_path, dataSource, class_number, output_path)














