import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTModel
import torch.nn as nn

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# 1. 特征提取阶段
# -----------------------

# 自定义数据集类
class SingleLabelDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        """
        Args:
            image_dir (str): 图像文件夹路径
            label_file (str): 标注文件路径
            transform (callable, optional): 图像预处理操作
        """
        self.image_dir = image_dir
        self.transform = transform
        self.data = self._load_data(label_file)

    def _load_data(self, label_file):
        """
        从标注文件加载图像路径和对应标签
        """
        data = []
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # 确保行不为空
                    parts = line.split(",")
                    if len(parts) < 2:  # 确保至少包含图像名和一个标签
                        print(f"跳过无效行: {line}")
                        continue
                    image_name = parts[0].strip("'")  # 去掉单引号
                    try:
                        label_values = [int(x.strip()) for x in parts[1:]]
                        label = label_values.index(1)  # 找到值为1的索引
                    except ValueError:
                        print(f"标签解析失败，跳过: {line}")
                        continue
                    data.append((image_name, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        """
        image_name, label = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")  # 打开图像

        if self.transform:
            image = self.transform(image)

        return image, label


# 定义特征提取和保存函数
def extract_and_save_features(data_loader, vit_model, save_path):
    """
    提取特征并保存到磁盘
    Args:
        data_loader (DataLoader): 数据加载器
        vit_model (nn.Module): 预训练的ViT模型
        save_path (str): 保存路径
    """
    vit_model.eval()  # 设置ViT为评估模式
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            lbls = lbls.to(device)

            # 提取 [CLS] 特征
            outputs = vit_model(pixel_values=images).last_hidden_state
            cls_features = outputs[:, 0, :]  # 取 [CLS] token 特征

            features.append(cls_features.cpu())  # 转到CPU，便于后续存储
            labels.append(lbls.cpu())

    # 合并所有特征和标签
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # 保存到磁盘
    torch.save((features, labels), save_path)
    print(f"Features saved to {save_path}")


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集路径和标注文件路径
image_dir = "/home/huangrui/Codes/SubStyleClassfication/data/Painting91/Images"  # 替换为图像文件夹路径
train_label_file = "/home/huangrui/Codes/SubStyleClassfication/data/Painting91/train.txt"  # 替换为训练集标注文件路径
test_label_file = "/home/huangrui/Codes/SubStyleClassfication/data/Painting91/test.txt"  # 替换为测试集标注文件路径

# 创建数据加载器
train_dataset = SingleLabelDataset(image_dir=image_dir, label_file=train_label_file, transform=transform)
test_dataset = SingleLabelDataset(image_dir=image_dir, label_file=test_label_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的ViT模型
vit_model = ViTModel.from_pretrained("Models/vit1k").to(device)

# 提取并保存训练集和测试集特征
extract_and_save_features(train_loader, vit_model, "train_features.pth")
extract_and_save_features(test_loader, vit_model, "test_features.pth")

# -----------------------
# 2. MLP 训练阶段
# -----------------------

# 加载提取好的特征
train_features, train_labels = torch.load("train_features.pth")
test_features, test_labels = torch.load("test_features.pth")

# 自定义数据集
class PreExtractedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 创建数据加载器
train_dataset = PreExtractedDataset(train_features, train_labels)
test_dataset = PreExtractedDataset(test_features, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义 MLP 分类器
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 初始化 MLP 分类器
num_classes = len(set(train_labels.numpy()))  # 获取类别数量
input_dim = train_features.shape[1]  # 输入特征维度
mlp_classifier = MLPClassifier(input_dim=input_dim, output_dim=num_classes).to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(mlp_classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练和测试循环
num_epochs = 100
for epoch in range(num_epochs):
    # 训练阶段
    mlp_classifier.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # 前向传播
        outputs = mlp_classifier(features)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失和准确率
        train_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, dim=1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    avg_train_accuracy = train_correct / train_total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")

    # 测试阶段
    mlp_classifier.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            # 前向传播
            outputs = mlp_classifier(features)
            loss = criterion(outputs, labels)

            # 累计损失和准确率
            test_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    avg_test_loss = test_loss / test_total
    avg_test_accuracy = test_correct / test_total
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
