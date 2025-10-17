#calcute gram distance to find best threshold

import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ssc.utils import get_byol_transforms, MultiViewDataInjector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, layer='layer1'):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        if layer == 'layer1':
            self.features = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
            )
        elif layer == 'layer2':
            self.features = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
                resnet.layer1, resnet.layer2
            )
        else:
            raise ValueError("Invalid layer: choose 'layer1' or 'layer2'")
        self.features.eval().to(device)

    def forward(self, x):
        with torch.no_grad():
            return self.features(x)

def compute_gram_matrix(features):
    B, C, H, W = features.size()
    features = features.view(B, C, -1)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (C * H * W)



def compute_avg_gram_distances(model_path,root_dir, thresholds, image_limit=500, layer='layer1'):
    # model = ResNetFeatureExtractor(layer=layer)
    resnet50=make_models()
    resnet50 = resnet50.eval().to(device)
    base_model = torch.load(model_path + 'base-last.pth', map_location=device)
    base_model.eval().to(device)
    results = {}

    transformT, transformT1, _ = get_byol_transforms(64, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = MultiViewDataInjector([transformT, transformT1])
    transform_original = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

    for t in thresholds:
        folder = os.path.join(root_dir, str(t), 'cleaned')
        if not os.path.exists(folder):
            print(f"[!] 跳过不存在的文件夹: {folder}")
            results[t] = None
            continue

        images = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < 2:
            print(f"[!] 阈值 {t} 图像数不足 2，跳过")
            results[t] = None
            continue

        images = random.sample(images, min(image_limit, len(images)))
        grams = []
        for img_name in tqdm(images, desc=f"τ={t}"):
            path = os.path.join(folder, img_name)
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_res = transform_original(img).unsqueeze(0).to(device)
            img1, img2 = transform(img)
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)

            feat0 = resnet50(img_res)
            feat1= base_model(img1)
            feat2= base_model(img2)
            feat=(feat0-feat1)+(feat0-feat2)

            gram = compute_gram_matrix(feat)
            grams.append(gram.squeeze(0).cpu())

        dists = []
        for i in range(len(grams)):
            for j in range(i + 1, len(grams)):
                dist = torch.norm(grams[i] - grams[j], p='fro').item()
                dists.append(dist)
        # 平均数
        # avg_dist = np.mean(dists)
        # 中位数
        avg_dist = np.median(dists)
        results[t] = avg_dist
        print(f"✅ τ={t}: 平均 Gram 距离 = {avg_dist:.4f}")

    return results

def plot_results(results):
    keys = sorted(results.keys())
    values = [results[k] if results[k] is not None else 0 for k in keys]

    plt.figure(figsize=(8, 5))
    plt.plot(keys, values, marker='o', color='crimson')
    plt.xlabel("similarity Threshold τ")
    plt.ylabel("Average Gram Distance")
    plt.title("Gram Distance vs. Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/huangrui/Codes/SubStyleClassfication/gram_threshold.png")
    plt.show()

if __name__ == "__main__":
    root = "/home/huangrui/Codes/SubStyleClassfication/uncertain_test"
    thresholds = [-0.5,-0.4,-0.3, -0.25, -0.2, -0.15, -0.1,0,0.1,0.2,0.3,0.4,0.5]
    model_path='/home/huangrui/Codes/SubStyleClassfication/model/pandora-SSR-resnet50-2025-05-23-18-01-31-SSC-'
    results = compute_avg_gram_distances(model_path,root, thresholds, image_limit=200, layer='layer1')
    plot_results(results)
