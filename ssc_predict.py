# ssc prediction script for FashionStyle14 dataset
# Author: cuijia1247
# Date: 2025-01-XX
# version: 1.0
import logging
import time
import torch
import torch.nn.functional as F
import numpy as np
from SscDataSet import SscDataset
from ssc.utils import get_ssc_transforms, MultiViewDataInjector
import os

# Setup device for cuda or cpu
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def compute_cosine_similarity(features1, features2):
    """
    计算两个特征向量的余弦相似度
    
    Args:
        features1: 第一个特征向量 (batch_size, feature_dim)
        features2: 第二个特征向量 (batch_size, feature_dim)
    
    Returns:
        cosine_similarity: 余弦相似度值 (batch_size,)
    """
    # 归一化特征向量
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)
    
    # 计算余弦相似度
    cosine_sim = (features1_norm * features2_norm).sum(dim=1)
    
    return cosine_sim

def ssc_predict(model_path, dataSource, dataset_name='FashionStyle14'):
    """
    使用SSC模型对数据集进行子图计算，计算view1和view2的余弦相似度
    遍历整个数据库（train + test + val），只统计cosine_sim的统计信息
    
    Args:
        model_path: 模型文件路径
        dataSource: 数据集路径
        dataset_name: 数据集名称（默认：FashionStyle14）
    """
    logger = logging.getLogger("predict_logger")
    logger.info('=' * 100)
    logger.info('SSC PREDICTION PROCESS - Computing view1 and view2 cosine similarity')
    logger.info(f'TRAVERSING ENTIRE {dataset_name.upper()} DATABASE (train + test + val)')
    logger.info('=' * 100)
    
    # Load model
    logger.info(f'Loading model from: {model_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    logger.info('Model loaded successfully')
    
    # Setup transforms and dataset
    image_size = 64
    transformT, transformT1, transformEvalT = get_ssc_transforms(
        image_size, 
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225)
    )
    
    # Process all datasets: train, test, val
    data_splits = ['train', 'test', 'val']
    all_cosine_sims = []  # 只存储cosine_sim值
    total_samples = 0
    
    for split_name in data_splits:
        split_path = os.path.join(dataSource, split_name)
        if not os.path.exists(split_path):
            logger.warning(f'Split {split_name} not found at {split_path}, skipping...')
            continue
        
        logger.info('=' * 100)
        logger.info(f'Processing {split_name.upper()} dataset...')
        logger.info('=' * 100)
        
        dataset = SscDataset(
            dataSource, 
            split_name, 
            transform=MultiViewDataInjector([transformT, transformT1])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=False
        )
        
        logger.info(f'{split_name.upper()} dataset loaded: {len(dataset)} samples')
        
        # Process each batch
        split_samples = 0
        with torch.no_grad():
            for batch_idx, (view1, view2, label, names, original) in enumerate(dataloader):
                # Move to device
                view1 = view1.to(device)
                view2 = view2.to(device)
                
                # Compute features
                features1 = model(view1)  # (batch_size, feature_dim)
                features2 = model(view2)   # (batch_size, feature_dim)
                
                # Compute cosine similarity
                cosine_sim = compute_cosine_similarity(features1, features2)
                
                # Convert to numpy and store only cosine_sim values
                cosine_sim_np = cosine_sim.cpu().numpy()
                all_cosine_sims.extend(cosine_sim_np.tolist())
                
                batch_size = cosine_sim_np.shape[0]
                split_samples += batch_size
                total_samples += batch_size
                
                # Log progress
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    logger.info(f'{split_name.upper()}: Processed {split_samples}/{len(dataset)} samples (batch {batch_idx + 1})')
        
        logger.info(f'{split_name.upper()} completed: {split_samples} samples processed')
    
    logger.info('=' * 100)
    logger.info(f'Prediction completed. Total samples across all splits: {total_samples}')
    logger.info('=' * 100)
    
    # Compute statistics for entire database
    cosine_sims_array = np.array(all_cosine_sims)
    mean_sim = np.mean(cosine_sims_array)
    std_sim = np.std(cosine_sims_array)  # 标准差（方差的开方）
    variance_sim = np.var(cosine_sims_array)  # 方差
    min_sim = np.min(cosine_sims_array)
    max_sim = np.max(cosine_sims_array)
    
    # 在屏幕上显示统计结果
    print('\n' + '=' * 100)
    print('COSINE SIMILARITY STATISTICS - ENTIRE DATABASE:')
    print('=' * 100)
    print(f'Total Samples: {total_samples}')
    print(f'Mean (均值):     {mean_sim:.6f}')
    print(f'Variance (方差): {variance_sim:.6f}')
    print(f'Std (标准差):    {std_sim:.6f}')
    print(f'Min (最小值):    {min_sim:.6f}')
    print(f'Max (最大值):    {max_sim:.6f}')
    print('=' * 100 + '\n')
    
    logger.info('=' * 100)
    logger.info('COSINE SIMILARITY STATISTICS - ENTIRE DATABASE:')
    logger.info(f'Total Samples: {total_samples}')
    logger.info(f'Mean (均值):     {mean_sim:.6f}')
    logger.info(f'Variance (方差): {variance_sim:.6f}')
    logger.info(f'Std (标准差):    {std_sim:.6f}')
    logger.info(f'Min (最小值):    {min_sim:.6f}')
    logger.info(f'Max (最大值):    {max_sim:.6f}')
    logger.info('=' * 100)
    
    return {
        'mean': mean_sim,
        'variance': variance_sim,
        'std': std_sim,
        'min': min_sim,
        'max': max_sim,
        'total_samples': total_samples
    }

if __name__ == '__main__':
    # Setup logger
    logger = logging.getLogger("predict_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    log_name = f'ssc_predict_avastyle_{current_time}.log'
    os.makedirs('./log', exist_ok=True)
    filehandler = logging.FileHandler("./log/" + log_name)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    
    # Configuration for AVAstyle dataset
    model_path = './model/ssc_AVAstyle_resnet_51.30/ssc-AVAstyle-SSC-resnet50-2025-10-26-13-28-51-iteration-0-accuracy-5130-SSC-base-best.pth'
    dataSource = '/home/cuijia1247/Codes/SubStyleClassfication/data/AVAstyle/'  # the '/' is necessary
    dataset_name = 'AVAstyle'
    
    logger.info('Configuration:')
    logger.info(f'Model path: {model_path}')
    logger.info(f'Data source: {dataSource}')
    logger.info(f'Device: {device}')
    
    try:
        # Run prediction
        stats = ssc_predict(model_path, dataSource, dataset_name)
        
        logger.info('\n' + '=' * 100)
        logger.info('PREDICTION COMPLETED SUCCESSFULLY')
        logger.info('=' * 100)
        
    except Exception as e:
        logger.error(f'Error during prediction: {str(e)}', exc_info=True)
        raise
    
    finally:
        logger.removeHandler(filehandler)
        logger.removeHandler(handler)
        logging.shutdown()

