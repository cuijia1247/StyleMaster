import os
import torch
import torch.nn.functional as F
from torch import nn
import cv2
import shutil
from torchvision import models, transforms
from ssc.utils import get_byol_transforms, MultiViewDataInjector
import heapq  # 用于获取最高和最低相似度的图像

# 设置设备
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
def count_images_in_folder(folder_path):
    """统计文件夹中的图像数量"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    count = 0
    for filename in os.listdir(folder_path):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            count += 1
    return count

def save_similarity_to_txt(similarity_dict, output_path):
    """将图像名和相似度保存到txt文件，按照相似度从低到高排序"""
    # 将字典转换为列表并按相似度排序
    sorted_items = sorted(similarity_dict.items(), key=lambda x: x[1])
    
    with open(output_path, 'w') as f:
        for filename, similarity in sorted_items:
            f.write(f"{filename}, {similarity:.4f}\n")
    print(f"已保存排序后的相似度结果到: {output_path}")

def calculate_similarities(model,resnet50, transforms_original,transform, dataSource):
    """
    计算所有图像的相似度
    返回: 
    - similarity_dict: 包含所有图像名称和对应相似度的字典
    - top_5: 相似度最高的5张图像及相似度
    - bottom_5: 相似度最低的5张图像及相似度
    """
    similarity_dict = {}
    top_5 = []
    bottom_5 = []
    
    for root, dirs, files in os.walk(dataSource):
        for file in files:
            img_path = os.path.join(root, file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"无法读取图像: {img_path}")
                    continue

                img = cv2.resize(img, (256, 256))
                img1, img2 = transform(img)
                img1 = img1.unsqueeze(0).to(device)
                img2 = img2.unsqueeze(0).to(device)
                img_res = transforms_original(img).unsqueeze(0).to(device)
                

                # 提取特征
                with torch.no_grad():
                    view1 = model(img1)
                    view2 = model(img2)
                    res_view = resnet50(img_res).squeeze()
                    test1 =res_view-view1
                    test2 =res_view-view2

                # 计算相似度
                similarity = F.cosine_similarity(test1, test2).item()
                similarity_dict[file] = similarity
                
                # 维护最高和最低相似度的列表
                heapq.heappush(top_5, (similarity, file))
                heapq.heappush(bottom_5, (-similarity, file))  # 使用负数实现最小堆
                
                if len(top_5) > 5:
                    heapq.heappop(top_5)
                if len(bottom_5) > 5:
                    heapq.heappop(bottom_5)
                    
            except Exception as e:
                print(f"处理图像失败: {img_path}, 错误: {str(e)}")
    
    # 转换堆为排序后的列表
    top_5 = sorted([(s, f) for s, f in top_5], reverse=True)
    bottom_5 = sorted([(-s, f) for s, f in bottom_5])
    
    return similarity_dict, top_5, bottom_5

def process_threshold(similarity_dict, dataSource, output_root, threshold):
    """
    处理单个阈值的数据清洗
    返回: (保留图片数, 删除图片数)
    """
    # 创建输出目录
    threshold_dir = os.path.join(output_root, str(threshold))
    cleaned_dir = os.path.join(threshold_dir, "cleaned")
    deleted_dir = os.path.join(threshold_dir, "deleted")
    
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(deleted_dir, exist_ok=True)
    
    kept_count = 0
    deleted_count = 0
    
    for root, dirs, files in os.walk(dataSource):
        for file in files:
            if file not in similarity_dict:
                continue
                
            img_path = os.path.join(root, file)
            similarity = similarity_dict[file]
            
            try:
                if similarity > threshold:
                    shutil.copy(img_path, os.path.join(cleaned_dir, file))
                    kept_count += 1
                else:
                    shutil.copy(img_path, os.path.join(deleted_dir, file))
                    deleted_count += 1
            except Exception as e:
                print(f"复制图像失败: {img_path}, 错误: {str(e)}")
    
    return kept_count, deleted_count

def clean_images_by_similarity(model_path, dataSource, output_root, thresholds):
    """主清洗函数"""
    # 加载模型
    base_model = torch.load(model_path + 'base-last.pth', map_location=device)
    base_model.eval().to(device)
    resnet50 = make_models()
    resnet50 = resnet50.eval().to(device)

    # 数据增强器
    transformT, transformT1, _ = get_byol_transforms(64, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = MultiViewDataInjector([transformT, transformT1])
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transforms_original = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 清空输出目录
    shutil.rmtree(output_root, ignore_errors=True)
    os.makedirs(output_root, exist_ok=True)

    # 计算所有图像的相似度
    similarity_dict, top_5, bottom_5 = calculate_similarities(base_model,resnet50,transforms_original, transform, dataSource)
    
    # 保存相似度结果到txt文件
    similarity_txt_path = os.path.join(output_root, "similarity_results.txt")
    save_similarity_to_txt(similarity_dict, similarity_txt_path)

    # 打印最高和最低相似度的图像
    print("\n相似度最高的5张图像:")
    for similarity, filename in top_5:
        print(f"{filename}: {similarity:.4f}")

    print("\n相似度最低的5张图像:")
    for similarity, filename in bottom_5:
        print(f"{filename}: {similarity:.4f}")

    # 处理每个阈值
    results = {}
    for threshold in thresholds:
        print(f"\n正在处理阈值: {threshold}")
        kept, deleted = process_threshold(similarity_dict, dataSource, output_root, threshold)
        results[threshold] = (kept, deleted)
        
        # 保存每个阈值的统计信息
        threshold_dir = os.path.join(output_root, str(threshold))
        with open(os.path.join(threshold_dir, "stats.txt"), 'w') as f:
            f.write(f"阈值: {threshold}\n")
            f.write(f"保留图片数: {kept}\n")
            f.write(f"删除图片数: {deleted}\n")
            f.write(f"保留比例: {kept/(kept+deleted):.2%}\n")
    
    # 打印总体结果
    print("\n所有阈值处理结果:")
    for threshold, (kept, deleted) in results.items():
        print(f"阈值 {threshold}: 保留 {kept} 张, 删除 {deleted} 张, 保留比例 {kept/(kept+deleted):.2%}")

if __name__ == '__main__':
    dataSource = '/home/huangrui/Codes/SubStyleClassfication/merged_data/2'
    model_path = '/home/huangrui/Codes/SubStyleClassfication/model/pandora-SSR-resnet50-2025-05-23-18-01-31-SSC-'
    output_root = '/home/huangrui/Codes/SubStyleClassfication/uncertain_test'
    
    # 自定义阈值集合
    thresholds = [-0.5,-0.4,-0.3, -0.25, -0.2, -0.15, -0.1,0,0.1,0.2,0.3,0.4,0.5]  # 可以修改这个列表
    
    clean_images_by_similarity(model_path, dataSource, output_root, thresholds)
   
    # 统计原始图片数量
    image_count_before = count_images_in_folder(dataSource)
    print(f"\n原来文件夹中共有 {image_count_before} 张图像")
    
    # # 打印每个阈值的结果
