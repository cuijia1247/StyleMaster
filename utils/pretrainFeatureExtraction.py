"""
Pretrained Feature Extraction Script
Author: cuijia1247
Date: 2025-10-20
Version: 1.0

This script extracts features from images using a pretrained ViT model.
"""

import os
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from pathlib import Path
import pickle


def get_image_transform(image_size=224):
    """
    Get the image transformation pipeline for feature extraction.
    
    Args:
        image_size: Size of the input image (default: 224)
    
    Returns:
        transforms.Compose: Image transformation pipeline
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return transform


def load_vit_model(model_name='vit_large_patch16_224', pretrained_path='pretrainModels/vit_large_patch16_224.pth', device='cuda'):
    """
    Load the pretrained ViT model.
    
    Args:
        model_name: Name of the ViT model
        pretrained_path: Path to the pretrained model weights
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded ViT model in eval mode
    """
    print(f"Loading model: {model_name}")
    
    # Create model with num_classes=0 to extract features
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    
    # Load pretrained weights
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        print("Using model without pretrained weights.")
    
    model = model.eval()
    model = model.to(device)
    
    return model


def load_resnet_model(model_name='resnet50', pretrained_path=None, device='cuda'):
    """
    Load the pretrained ResNet model.
    
    Args:
        model_name: Name of the ResNet model ('resnet50', 'resnet101', etc.)
        pretrained_path: Path to the pretrained model weights (optional, if None uses torchvision pretrained)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded ResNet model in eval mode (without final FC layer)
    """
    print(f"Loading model: {model_name}")
    
    # Create ResNet model
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=False)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=False)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=False)
    else:
        raise ValueError(f"Unsupported ResNet model: {model_name}")
    
    # Load pretrained weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        # Use torchvision pretrained weights
        print(f"Loading torchvision pretrained weights for {model_name}")
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True)
    
    # Remove the final fully connected layer to get features
    # ResNet features are extracted before avgpool + fc
    model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    
    model = model.eval()
    model = model.to(device)
    
    return model


def collect_image_paths(data_dir, extensions=['.jpg', '.jpeg', '.png']):
    """
    Collect all image paths from the data directory.
    
    Args:
        data_dir: Root directory containing images
        extensions: List of valid image extensions
    
    Returns:
        image_paths: List of image file paths
        relative_paths: List of filenames only (without subfolder paths)
    """
    image_paths = []
    relative_paths = []
    
    data_dir = Path(data_dir)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in extensions:
                full_path = os.path.join(root, file)
                # 只使用文件名和后缀，不包含子文件夹路径
                filename_only = file
                image_paths.append(full_path)
                relative_paths.append(filename_only)
    
    return image_paths, relative_paths


def extract_features(model, image_paths, relative_paths, transform, device='cuda', batch_size=32, flatten=True):
    """
    Extract features from images using the model.
    
    Args:
        model: Pretrained model for feature extraction
        image_paths: List of image file paths
        relative_paths: List of filenames only (without subfolder paths)
        transform: Image transformation pipeline
        device: Device to run inference on
        batch_size: Batch size for processing
        flatten: Whether to flatten the output features (needed for ResNet)
    
    Returns:
        feature_dict: Dictionary mapping filename to feature vector
    """
    feature_dict = {}
    failed_images = []
    
    model.eval()
    
    with torch.no_grad():
        batch_images = []
        batch_names = []
        
        for img_path, rel_path in tqdm(zip(image_paths, relative_paths), total=len(image_paths), desc="Extracting features"):
            try:
                # Load and transform image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                
                batch_images.append(image_tensor)
                batch_names.append(rel_path)
                
                # Process batch when it reaches batch_size
                if len(batch_images) == batch_size:
                    batch_tensor = torch.stack(batch_images).to(device)
                    batch_features = model(batch_tensor)
                    
                    # Flatten if needed (for ResNet output which is [B, C, 1, 1])
                    if flatten and len(batch_features.shape) > 2:
                        batch_features = batch_features.view(batch_features.size(0), -1)
                    
                    # Store features in dictionary with filename as key
                    for filename, feature_vector in zip(batch_names, batch_features.cpu()):
                        feature_dict[filename] = feature_vector
                    
                    batch_images = []
                    batch_names = []
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                failed_images.append(img_path)
        
        # Process remaining images
        if len(batch_images) > 0:
            batch_tensor = torch.stack(batch_images).to(device)
            batch_features = model(batch_tensor)
            
            # Flatten if needed (for ResNet output which is [B, C, 1, 1])
            if flatten and len(batch_features.shape) > 2:
                batch_features = batch_features.view(batch_features.size(0), -1)
            
            # Store features in dictionary with filename as key
            for filename, feature_vector in zip(batch_names, batch_features.cpu()):
                feature_dict[filename] = feature_vector
    
    if failed_images:
        print(f"\nFailed to process {len(failed_images)} images:")
        for img in failed_images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    return feature_dict


def save_features(feature_dict, output_dir, dataset_name):
    """
    Save extracted features to a .pkl file.
    
    Args:
        feature_dict: Dictionary mapping filename to feature vector
        output_dir: Directory to save the features
        dataset_name: Name of the dataset (for filename)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature dimension from first feature vector
    first_feature = next(iter(feature_dict.values()))
    feature_dim = first_feature.shape[0]
    num_samples = len(feature_dict)
    
    # Prepare data to save
    data = {
        'feature_dict': feature_dict,
        'feature_dim': feature_dim,
        'num_samples': num_samples
    }
    
    # Generate output filename
    output_path = os.path.join(output_dir, f'{dataset_name}_features.pkl')
    
    # Save to file using pickle
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nFeatures saved to: {output_path}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of samples: {num_samples}")


def load_dataFeatures(feature_path):
    """
    Load extracted features from a .pkl file.
    
    Args:
        feature_path: Path to the .pkl file containing extracted features
    
    Returns:
        feature_dict: Dictionary mapping filename to feature vector
    
    Example:
        >>> feature_dict = load_dataFeatures('pretrainFeatures/Painting91_features.pkl')
        >>> feature_vector = feature_dict['image.jpg']
        >>> print(f"Feature shape: {feature_vector.shape}")
    """
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    
    print(f"Loading features from: {feature_path}")
    
    # Load the .pkl file
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract feature dictionary
    feature_dict = data['feature_dict']
    # feature_dim = data['feature_dim']
    # num_samples = data['num_samples']
    
    # print(f"Successfully loaded features:")
    # print(f"  - Number of samples: {num_samples}")
    # print(f"  - Feature dimension: {feature_dim}")
    # print(f"  - Sample filenames (first 5): {list(feature_dict.keys())[:5]}")
    
    return feature_dict


def get_feature_by_filename(feature_dict, filename):
    """
    通过文件名获取特征向量。
    
    Args:
        feature_dict: 特征字典
        filename: 文件名（不包含路径）
    
    Returns:
        feature_vector: 对应的特征向量
    
    Example:
        >>> feature_dict = load_dataFeatures('features.pkl')
        >>> feature = get_feature_by_filename(feature_dict, 'image.jpg')
    """
    if filename not in feature_dict:
        raise KeyError(f"Filename '{filename}' not found in feature dictionary")
    return feature_dict[filename]


def get_all_filenames(feature_dict):
    """
    获取所有文件名列表。
    
    Args:
        feature_dict: 特征字典
    
    Returns:
        filenames: 所有文件名的列表
    """
    return list(feature_dict.keys())


def get_feature_matrix(feature_dict, filenames=None):
    """
    将特征字典转换为特征矩阵。
    
    Args:
        feature_dict: 特征字典
        filenames: 指定的文件名列表，如果为None则使用所有文件名
    
    Returns:
        feature_matrix: 特征矩阵 (n_samples, feature_dim)
        filename_list: 对应的文件名列表
    """
    if filenames is None:
        filenames = list(feature_dict.keys())
    
    features = []
    valid_filenames = []
    
    for filename in filenames:
        if filename in feature_dict:
            features.append(feature_dict[filename])
            valid_filenames.append(filename)
    
    if not features:
        raise ValueError("No valid features found for the given filenames")
    
    feature_matrix = torch.stack(features)
    return feature_matrix, valid_filenames


def main():
    """
    Main function for feature extraction.
    直接在此函数内配置所有参数，无需命令行传参。
    """
    # ==================== 参数配置区域 ====================
    model_type = 'resnet'  # 'vit' 或 'resnet'
    data_dir = '/home/cuijia1247/Codes/SubStyleClassfication/data/Painting91/train/'  # 数据目录路径
    output_dir = 'pretrainFeatures'  # 输出特征保存目录
    batch_size = 64  # 批处理大小
    image_size = 224  # 输入图像尺寸
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 计算设备
    
    # ViT 配置
    vit_model_name = 'vit_large_patch16_224'  # ViT模型名称
    vit_pretrained_path = 'pretrainModels/vit_large_patch16_224.pth'  # 预训练权重路径
    vit_feature_dim = 1024  # ViT特征维度
    
    # ResNet 配置
    resnet_model_name = 'resnet50'  # ResNet模型名称 ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
    resnet_pretrained_path = None  # ResNet预训练权重路径 (None则使用torchvision预训练权重)
    resnet_feature_dim = 2048  # ResNet50特征维度
    # =====================================================
    
    print("="*80)
    print("Pretrained Feature Extraction")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Model type: {model_type.upper()}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("="*80)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Collect image paths
    print("\nCollecting image paths...")
    image_paths, relative_paths = collect_image_paths(data_dir)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    # Load model based on model_type
    if model_type.lower() == 'vit':
        model_name = vit_model_name
        pretrained_path = vit_pretrained_path
        feature_dim = vit_feature_dim
        dataset_name = f'Painting91_vit_test'
        
        print(f"\nLoading ViT model: {model_name}")
        model = load_vit_model(model_name, pretrained_path, device)
        flatten = False  # ViT already outputs flattened features
        
    elif model_type.lower() == 'resnet':
        model_name = resnet_model_name
        pretrained_path = resnet_pretrained_path
        feature_dim = resnet_feature_dim
        dataset_name = f'Painting91_resnet50_test'
        
        print(f"\nLoading ResNet model: {model_name}")
        model = load_resnet_model(model_name, pretrained_path, device)
        flatten = True  # ResNet outputs [B, C, 1, 1], needs flattening
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'vit' or 'resnet'")
    
    print(f"Expected feature dimension: {feature_dim}")
    
    # Get image transform
    transform = get_image_transform(image_size)
    
    # Extract features
    print("\nExtracting features...")
    feature_dict = extract_features(
        model, image_paths, relative_paths, transform, 
        device=device, batch_size=batch_size, flatten=flatten
    )
    
    # Verify feature dimension
    first_feature = next(iter(feature_dict.values()))
    actual_feature_dim = first_feature.shape[0]
    print(f"\nActual feature dimension: {actual_feature_dim}")
    if actual_feature_dim != feature_dim:
        print(f"Warning: Extracted feature dimension ({actual_feature_dim}) "
              f"does not match expected dimension ({feature_dim})")
    
    # Save features
    save_features(feature_dict, output_dir, dataset_name)
    
    print("\nFeature extraction completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
    
    # ==================== 使用 load_dataFeatures 的示例 ====================
    # 如果需要加载已保存的特征，可以取消下面代码的注释：
    # 
    # feature_path = '/home/cuijia1247/Codes/SubStyleClassfication/pretrainFeatures/Painting91_vit_train_features.pkl'
    # feature_dict = load_dataFeatures(feature_path)
    
    # print('\n' + '='*80)
    # print('feature loading DONE.')
    # print(f"Total features: {len(feature_dict)}")
    
    # # 示例：通过文件名获取特征
    # print('\n示例：通过文件名访问特征向量（新格式：只包含文件名.后缀）')
    # print('-'*80)
    # sample_filenames = list(feature_dict.keys())[:5]
    # for filename in sample_filenames:
    #     feature = feature_dict[filename]
    #     print(f"  文件名: {filename}")
    #     print(f"  特征形状: {feature.shape}")
    #     print(f"  特征前5个值: {feature[:5]}")
    #     print()
    # print('='*80)
    # ======================================================================

