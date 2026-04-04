# CLIP 风格分类脚本
# 使用预训练 CLIP 视觉编码器提取特征，配合 Classifier 线性分类头进行风格分类
# 支持两种骨干：
#   - EVA-CLIP-5B  (EVA02-E-14-plus, 5.0B 参数, feat_dim=1024)
#   - OpenCLIP ViT-L-14 (openai, 0.3B 参数, feat_dim=768)
# 权重自动下载到 pretrainModels/ 目录
# 数据集目录结构：<data_root>/<train_split>/<class_id>/img.jpg（class_id 从 1 开始）
#
# 用法示例：
#   python clip_train.py --data_root /data/Painting91 --train train --test test --class_num 13
#   python clip_train.py --data_root /data/WikiArt3   --train train --test test --class_num 15
#
# Author: cuijia1247  Date: 2026-04-04

import os
import sys
import time
import argparse
import urllib.request
import logging

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 添加项目根目录到 sys.path（支持从任意目录运行）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from ssc.classifier import Classifier

# ─── 固定配置 ─────────────────────────────────────────────────────────────────
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAIN_DIR      = os.path.join(PROJECT_ROOT, 'pretrainModels')
MODEL_DIR         = os.path.join(PROJECT_ROOT, 'model')
LOG_DIR           = os.path.join(PROJECT_ROOT, 'log')
BATCH_SIZE        = 32
CLASSIFIER_LR     = 1e-3
CLASSIFIER_EPOCHS = 50   # 分类头训练轮数
IMAGE_SIZE        = 224   # CLIP 标准输入尺寸

os.makedirs(PRETRAIN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(LOG_DIR,      exist_ok=True)

# ─── 日志配置（屏幕实时打印 + 文件持久保存）─────────────────────────────────
class _FlushStreamHandler(logging.StreamHandler):
    """每条日志写入后立即 flush，保证终端实时可见。"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(log_filename: str) -> logging.Logger:
    """
    同时向终端和日志文件输出。
    终端格式：[HH:MM:SS] message（左对齐，无多余缩进）
    文件格式：与终端相同，方便直接阅读。
    """
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    lg  = logging.getLogger('clip_train')
    lg.setLevel(logging.INFO)
    lg.propagate = False
    lg.handlers.clear()   # 避免重复添加 handler

    sh = _FlushStreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    fh = logging.FileHandler(os.path.join(LOG_DIR, log_filename),
                             encoding='utf-8', delay=False)
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg

logger = logging.getLogger('clip_train')   # 占位，main() 中正式初始化


# ─── 权重下载工具 ──────────────────────────────────────────────────────────────
def _download_with_progress(url: str, dst: str):
    """带进度显示的文件下载（支持断点续传）"""
    tmp = dst + '.tmp'
    downloaded = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {'Range': f'bytes={downloaded}-'} if downloaded else {}

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get('Content-Length', 0)) + downloaded
        mode = 'ab' if downloaded else 'wb'
        with open(tmp, mode) as f:
            chunk, t0 = 8192 * 16, time.time()
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded / total * 100
                    speed = downloaded / (time.time() - t0 + 1e-6) / 1024 / 1024
                    print(f'\r  {pct:.1f}%  {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB  {speed:.1f} MB/s',
                          end='', flush=True)
    print()
    os.rename(tmp, dst)


def find_weight_in_dir(keywords: list) -> 'str | None':
    """
    在 pretrainModels/ 中按关键词列表查找已有权重文件。
    文件名（不含扩展名）须包含所有关键词（大小写不敏感）。
    返回匹配文件的完整路径，未找到则返回 None。
    """
    for fname in os.listdir(PRETRAIN_DIR):
        lower = fname.lower()
        if all(kw.lower() in lower for kw in keywords):
            path = os.path.join(PRETRAIN_DIR, fname)
            logger.info('在 pretrainModels/ 中找到已有权重: %s', path)
            return path
    return None


def ensure_weight(keywords: list, fallback_filename: str, url: str) -> str:
    """
    优先在 pretrainModels/ 中按关键词匹配已有权重，找到则直接使用；
    否则下载到 pretrainModels/<fallback_filename>。
    Args:
        keywords:         用于匹配文件名的关键词列表（如 ['vit-l-14', 'openai']）
        fallback_filename: 找不到时保存的文件名
        url:              下载地址
    """
    found = find_weight_in_dir(keywords)
    if found:
        return found
    dst = os.path.join(PRETRAIN_DIR, fallback_filename)
    logger.info('未找到匹配权重，开始下载: %s', url)
    _download_with_progress(url, dst)
    logger.info('下载完成: %s', dst)
    return dst


# ─── 权重路径配置 ──────────────────────────────────────────────────────────────
# ViT-L-14 (openai)：从 OpenAI CDN 下载
VIT_L14_URL      = ('https://openaipublic.azureedge.net/clip/models/'
                    'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt')
VIT_L14_FILE     = 'ViT-L-14-openai.pt'
VIT_L14_KEYWORDS = ['vit-l-14', 'openai']   # 匹配 pretrainModels/ 中已有文件的关键词

# EVA02-E-14-plus (5B)：从 hf-mirror 下载
EVA_URL      = ('https://hf-mirror.com/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k'
                '/resolve/main/open_clip_pytorch_model.bin')
EVA_FILE     = 'EVA02-E-14-plus-laion2b_s9b_b144k.bin'
EVA_KEYWORDS = ['eva02-e-14-plus', 'laion2b']   # 匹配关键词


# ─── 数据集 ────────────────────────────────────────────────────────────────────
class StyleDataset(Dataset):
    """
    通用风格分类数据集加载器。
    目录结构：<data_root>/<split>/<class_id>/<image>
    class_id 为从 1 开始的数字文件夹，返回 0-based label。
    """
    CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, data_root: str, split: str, image_size: int = 224):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(self.CLIP_MEAN, self.CLIP_STD),
        ])
        self.samples = []   # [(img_path, label_0based), ...]
        skipped = 0
        split_dir = os.path.join(data_root, split)
        class_ids = sorted(int(d) for d in os.listdir(split_dir) if d.isdigit())
        for cid in class_ids:
            cls_dir = os.path.join(split_dir, str(cid))
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                # 初始化时预检：跳过损坏/无法解码的图片
                if cv2.imread(path, cv2.IMREAD_COLOR) is None:
                    logger.warning('跳过损坏图片: %s', path)
                    skipped += 1
                    continue
                self.samples.append((path, cid - 1))
        if skipped:
            logger.warning('共跳过 %d 张损坏图片', skipped)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), label


# ─── 特征预提取 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader) -> tuple:
    """用 CLIP 视觉编码器批量提取特征，返回 (features: Tensor, labels: Tensor)"""
    encoder.eval()
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        feats = encoder.encode_image(imgs)          # (B, feat_dim)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 归一化
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


# ─── 分类器训练 & 评估 ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(classifier: Classifier, test_feats: torch.Tensor,
             test_labels: torch.Tensor) -> float:
    """在测试集特征上评估分类准确率"""
    classifier.eval()
    dataset = torch.utils.data.TensorDataset(test_feats, test_labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)
    correct, total = 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        pred = classifier(feats).argmax(1)
        correct += pred.eq(labels).sum().item()
        total   += len(labels)
    return correct / total


def train_classifier(train_feats: torch.Tensor, train_labels: torch.Tensor,
                     test_feats: torch.Tensor,  test_labels: torch.Tensor,
                     feat_dim: int, class_num: int) -> tuple:
    """
    在预提取特征上训练 Classifier 分类头，每 epoch 打印训练损失、训练准确率和测试准确率。
    返回 (最佳分类器, 最佳测试准确率)。
    """
    classifier  = Classifier(feat_dim, class_num).to(DEVICE)
    optimizer   = optim.Adam(classifier.parameters(), lr=CLASSIFIER_LR)
    criterion   = nn.CrossEntropyLoss()
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, CLASSIFIER_EPOCHS)

    train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_acc        = 0.0
    best_state_dict = None

    # 表头：列宽与数据行严格对齐
    # [HH:MM:SS] 前缀固定 11 字符，之后正文左对齐
    ROW = '%-7s  %-10s  %-10s  %-10s'
    SEP = '-' * 43
    logger.info(SEP)
    logger.info(ROW, 'Epoch', 'Loss', 'TrainAcc', 'TestAcc')
    logger.info(SEP)

    for epoch in range(CLASSIFIER_EPOCHS):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = classifier(feats)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += len(labels)
        scheduler.step()

        avg_loss  = total_loss / total
        train_acc = correct / total
        test_acc  = evaluate(classifier, test_feats, test_labels)

        logger.info(ROW, epoch + 1, f'{avg_loss:.4f}',
                    f'{train_acc:.4f}', f'{test_acc:.4f}')

        if test_acc > best_acc:
            best_acc        = test_acc
            best_state_dict = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    # 恢复最佳权重
    classifier.load_state_dict(best_state_dict)
    logger.info(SEP)
    logger.info('Best test acc: %.4f  (%.2f%%)', best_acc, best_acc * 100)
    return classifier, best_acc


# ─── 单次实验流程 ──────────────────────────────────────────────────────────────
def run_experiment(model_name: str, pretrained_tag: str, weight_path: str,
                   feat_dim: int, data_root: str, train_split: str,
                   test_split: str, class_num: int, timestamp: str) -> float:
    """
    完整实验：加载 CLIP 编码器 → 提取特征 → 训练分类头 → 保存最佳模型
    Args:
        model_name:     open_clip 模型名
        pretrained_tag: open_clip pretrained 标签
        weight_path:    本地权重文件路径
        feat_dim:       视觉特征维度
        data_root:      数据集根目录
        train_split:    训练集子目录名
        test_split:     测试集子目录名
        class_num:      分类类别数
        timestamp:      本次运行时间戳（用于模型文件命名）
    """
    logger.info('=' * 60)
    logger.info('Model   : %s (%s)', model_name, pretrained_tag)
    logger.info('Dataset : %s  train=%s  test=%s  classes=%d',
                data_root, train_split, test_split, class_num)

    # ── 1. 构建 CLIP 编码器并加载本地权重 ────────────────────────────────────
    logger.info('加载 CLIP 编码器...')
    encoder, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=weight_path,
        device=DEVICE,
    )
    encoder.eval()
    logger.info('编码器加载完成，参数量: %.2f B',
                sum(p.numel() for p in encoder.parameters()) / 1e9)

    # ── 2. 构建数据集 & DataLoader ────────────────────────────────────────────
    train_set = StyleDataset(data_root, train_split, IMAGE_SIZE)
    test_set  = StyleDataset(data_root, test_split,  IMAGE_SIZE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    logger.info('训练集: %d 样本  测试集: %d 样本', len(train_set), len(test_set))

    # ── 3. 预提取特征（冻结编码器，只需跑一次）────────────────────────────────
    t0 = time.time()
    logger.info('提取训练集特征...')
    train_feats, train_labels = extract_features(encoder, train_loader)
    logger.info('提取测试集特征...')
    test_feats, test_labels = extract_features(encoder, test_loader)
    logger.info('特征提取完成，耗时 %.1f s', time.time() - t0)

    # ── 4. 训练分类头（每 epoch 显示训练/测试准确率）────────────────────────
    logger.info('开始训练分类头  lr=%.4f  epochs=%d', CLASSIFIER_LR, CLASSIFIER_EPOCHS)
    t0 = time.time()
    classifier, best_acc = train_classifier(
        train_feats, train_labels, test_feats, test_labels, feat_dim, class_num)
    logger.info('分类头训练完成，耗时 %.1f s', time.time() - t0)

    # ── 5. 保存最佳分类器 ────────────────────────────────────────────────────
    dataset_name  = os.path.basename(data_root.rstrip('/'))
    # 模型名缩写：EVA02-E-14-plus → EVA5B，ViT-L-14 → ViTL14
    model_tag = model_name.replace('EVA02-E-14-plus', 'EVA5B').replace('ViT-L-14', 'ViTL14')
    acc_str   = f'{best_acc * 100:.2f}'.replace('.', '_')
    save_name = f'clip-{dataset_name}-{model_tag}-acc{acc_str}-{timestamp}.pth'
    save_path = os.path.join(MODEL_DIR, save_name)
    torch.save(classifier.state_dict(), save_path)
    logger.info('Model saved : %s', save_path)
    logger.info('Best acc    : %.4f  (%.2f%%)', best_acc, best_acc * 100)
    logger.info('=' * 60)
    return best_acc


# ─── 命令行参数解析 ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='CLIP 风格分类（EVA-CLIP-5B & OpenCLIP ViT-L-14）')
    parser.add_argument('--data_root',  type=str, default='/mnt/codes/data/style/Painting91',
                        help='数据集根目录')
    parser.add_argument('--train',      type=str, default='train',
                        help='训练集子目录名（默认 train）')
    parser.add_argument('--test',       type=str, default='test',
                        help='测试集子目录名（默认 test）')
    parser.add_argument('--class_num',  type=int, default=13,
                        help='分类类别数（默认 13，对应 Painting91）')
    return parser.parse_args()


# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 生成时间戳，用于日志文件名和模型文件命名
    timestamp    = time.strftime('%Y%m%d-%H%M%S')
    dataset_name = os.path.basename(args.data_root.rstrip('/'))
    log_filename = f'clip-{dataset_name}-{timestamp}.log'

    # 初始化 logger（屏幕 + 文件双输出）
    global logger
    logger = setup_logger(log_filename)

    logger.info('Device  : %s', DEVICE)
    logger.info('Dataset : %s', args.data_root)
    logger.info('Train   : %s  Test: %s  Classes: %d',
                args.train, args.test, args.class_num)
    logger.info('Log     : %s', os.path.join(LOG_DIR, log_filename))

    results = {}

    # ════════════════ 实验 1：OpenCLIP ViT-L-14 (openai) ════════════════
    vit_path = ensure_weight(VIT_L14_KEYWORDS, VIT_L14_FILE, VIT_L14_URL)
    results['OpenCLIP ViT-L-14'] = run_experiment(
        model_name='ViT-L-14',
        pretrained_tag='openai',
        weight_path=vit_path,
        feat_dim=768,
        data_root=args.data_root,
        train_split=args.train,
        test_split=args.test,
        class_num=args.class_num,
        timestamp=timestamp,
    )

    # ════════════════ 实验 2：EVA-CLIP-5B (EVA02-E-14-plus) ═════════════
    eva_path = ensure_weight(EVA_KEYWORDS, EVA_FILE, EVA_URL)
    results['EVA-CLIP-5B (EVA02-E-14-plus)'] = run_experiment(
        model_name='EVA02-E-14-plus',
        pretrained_tag='laion2b_s9b_b144k',
        weight_path=eva_path,
        feat_dim=1024,
        data_root=args.data_root,
        train_split=args.train,
        test_split=args.test,
        class_num=args.class_num,
        timestamp=timestamp,
    )

    # ════════════════ 汇总 ═══════════════════════════════════════════════
    logger.info('')
    logger.info('=' * 55)
    logger.info('Result: %s style classification', dataset_name)
    logger.info('-' * 55)
    for name, acc in results.items():
        logger.info('%-40s: %.2f%%', name, acc * 100)
    logger.info('=' * 55)


if __name__ == '__main__':
    main()
