# Author: HR
# Date: 2025-5-23
# version: 2.0
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

def make_models(target_layer=4):
    resnet = models.resnet50(pretrained=True).to(device)

    # 获取模型的特定层的输出
    layers = [
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                      resnet.layer3),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                      resnet.layer3, resnet.layer4, resnet.avgpool),
    ]

    # if target_layer < 1 or target_layer > 4:
    #     raise ValueError(f"Invalid target layer: {target_layer}. Target layer should be between 1 and 4.")

    # 截取模型到指定层
    feature_extractor = nn.Sequential(*layers[target_layer - 1]).eval()
    return feature_extractor

def SSC_predict(model_path, dataSource, class_number, output_path):
    base_model = torch.load(model_path + 'base-best.pth')
    classfier_model = torch.load(model_path + 'classifier-best.pth')
    
    resnet50 = make_models()
    resnet50 = resnet50.eval().to(device)
    # resnet50 = resnet50.eval().to(device)

    base_model = base_model.eval().to(device)
    print(base_model)
    classfier_model = classfier_model.eval().to(device)
    print(classfier_model)  # 查看最后一层的out_features


    transformT, transformT1,transformEval = get_byol_transforms(64, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transforms_original = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(norm_mean, norm_std),
    ])
    transform = MultiViewDataInjector([transformT, transformT1])

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "prediction_result.txt")

    # 结果统计
    correct = 0
    total = 0

    with open(output_file, "w") as f_out:
        for root, dirs, files in os.walk(dataSource):
            for file in files:
                if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                file_path = os.path.join(root, file)
                # 提取 true label（根目录下的文件夹名）
                true_label = os.path.basename(os.path.dirname(file_path))
                # 读取图片
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (256, 256))
                # 两种视图输入 base_model
                img1, img2 = transform(img)
                img1 = img1.unsqueeze(0).to(device)
                img2 = img2.unsqueeze(0).to(device)
                view1 = base_model(img1)
                view2 = base_model(img2)
                # 输入 resnet50
                img_res = transforms_original(img).unsqueeze(0).to(device)
                # res_view = resnet50(img_res)
                res_view = resnet50(img_res).squeeze()
                test1 =res_view-view1
                test2 =res_view-view2
                test = test1 + test2
                # print(test1)
                # print(test2)
                # print(test)

                prediction = classfier_model(test)
                print(prediction)
                pred_class = torch.argmax(prediction, dim=1).item()
                print(pred_class)
                pred_class=pred_class+1
                total += 1
                if pred_class==int(true_label):
                    correct = correct+1 # 假设true_label是数字

                # 写入预测结果
                f_out.write(f"{file},{pred_class},{true_label}\n")
                print(f"Processed: {file}, Pred: {pred_class}, True: {true_label}")

    print(f"\nFinal Accuracy: {correct/total:.4f} ({correct}/{total})")


if __name__ == '__main__':
    dataSource = '/home/huangrui/Codes/SubStyleClassfication/train_data/Painting91/test'  # painting91 13
    class_number = 13 # painting91 13
    ssc_output = 2048 # the best
    model_path = '/home/huangrui/Codes/SubStyleClassfication/model/1000/painting91-SSR-resnet50-0.7341772317886353-SSC-'
    output_path = '/home/huangrui/Codes/SubStyleClassfication/data/style_output/painting91'
    SSC_predict(model_path, dataSource, class_number, output_path)
