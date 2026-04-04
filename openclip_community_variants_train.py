# OpenCLIP Community Variants 风格分类脚本
# 依次使用以下三个社区大模型提取视觉特征，配合 Classifier 线性分类头进行风格分类：
#   - ViT-H-14  / laion2b_s32b_b79k  (632M 参数, feat_dim=1024)
#   - ViT-g-14  / laion2b_s34b_b88k  (1.01B 参数, feat_dim=1024)
#   - ViT-bigG-14 / laion2b_s39b_b160k (1.84B 参数, feat_dim=1280)
# 权重优先从 pretrainModels/ 目录按关键词匹配，找不到则通过 open_clip 内置机制下载
# 数据集目录结构：<data_root>/<split>/<class_id>/<image>（class_id 从 1 开始）
#
# 用法示例：
#   python openclip_community_variants_train.py --data_root /data/Painting91 --class_num 13
#
# Author: cuijia1247  Date: 2026-04-04

import os
import sys
import time
import argparse
import logging

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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
CLASSIFIER_EPOCHS = 50
IMAGE_SIZE        = 224

os.makedirs(PRETRAIN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(LOG_DIR,      exist_ok=True)

# ─── 实验配置表 ───────────────────────────────────────────────────────────────
# (实验名称, open_clip model_name, pretrained_tag, feat_dim, 匹配权重的关键词列表)
# pretrained_tag 使用 open_clip 内置标签，权重由 open_clip 自动从 HuggingFace 下载
# 若 pretrainModels/ 中已有对应权重文件（文件名含所有关键词），则直接使用本地路径
EXPERIMENTS = [
    ('ViT-H-14 (laion2b_s32b_b79k)',
     'ViT-H-14', 'laion2b_s32b_b79k', 1024,
     ['vit-h-14', 'laion2b']),
    ('ViT-g-14 (laion2b_s34b_b88k)',
     'ViT-g-14', 'laion2b_s34b_b88k', 1024,
     ['vit-g-14', 'laion2b']),
    ('ViT-bigG-14 (laion2b_s39b_b160k)',
     'ViT-bigG-14', 'laion2b_s39b_b160k', 1280,
     ['vit-bigg-14', 'laion2b']),
]


# ─── 日志配置 ──────────────────────────────────────────────────────────────────
class _FlushStreamHandler(logging.StreamHandler):
    """每条日志写入后立即 flush，保证终端实时可见。"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(log_filename: str) -> logging.Logger:
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    lg  = logging.getLogger('oc_community_train')
    lg.setLevel(logging.INFO)
    lg.propagate = False
    lg.handlers.clear()

    sh = _FlushStreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    fh = logging.FileHandler(os.path.join(LOG_DIR, log_filename),
                             encoding='utf-8', delay=False)
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg


logger = logging.getLogger('oc_community_train')


# ─── 本地权重查找 ──────────────────────────────────────────────────────────────
def find_local_weight(keywords: list) -> 'str | None':
    """
    在 pretrainModels/ 中按关键词列表（大小写不敏感）查找已有权重文件。
    文件名须包含所有关键词，返回完整路径；未找到返回 None。
    """
    for fname in os.listdir(PRETRAIN_DIR):
        lower = fname.lower()
        if all(kw.lower() in lower for kw in keywords):
            path = os.path.join(PRETRAIN_DIR, fname)
            logger.info('在 pretrainModels/ 中找到已有权重: %s', path)
            return path
    return None


# ─── 数据集 ────────────────────────────────────────────────────────────────────
class StyleDataset(Dataset):
    """
    通用风格分类数据集加载器。
    目录结构：<data_root>/<split>/<class_id>/<image>
    class_id 为从 1 开始的数字文件夹，返回 0-based label。
    """
    CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, data_root: str, split: str, image_size: int = IMAGE_SIZE):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(self.CLIP_MEAN, self.CLIP_STD),
        ])
        self.samples = []
        skipped  = 0
        split_dir = os.path.join(data_root, split)
        class_ids = sorted(int(d) for d in os.listdir(split_dir) if d.isdigit())
        for cid in class_ids:
            cls_dir = os.path.join(split_dir, str(cid))
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
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


# ─── 特征提取 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader) -> tuple:
    """用 CLIP 视觉编码器批量提取 L2 归一化特征，返回 (features, labels)。"""
    encoder.eval()
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs  = imgs.to(DEVICE)
        feats = encoder.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


# ─── 分类器训练 & 评估 ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(classifier: Classifier, test_feats: torch.Tensor,
             test_labels: torch.Tensor) -> float:
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
    """在预提取特征上训练 Classifier，返回 (最佳分类器, 最佳测试准确率)。"""
    classifier = Classifier(feat_dim, class_num).to(DEVICE)
    optimizer  = optim.Adam(classifier.parameters(), lr=CLASSIFIER_LR)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, CLASSIFIER_EPOCHS)

    train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_acc, best_state_dict = 0.0, None
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

    classifier.load_state_dict(best_state_dict)
    logger.info(SEP)
    logger.info('Best test acc: %.4f  (%.2f%%)', best_acc, best_acc * 100)
    return classifier, best_acc


# ─── 单次实验流程 ──────────────────────────────────────────────────────────────
def run_experiment(exp_name: str, model_name: str, pretrained_tag: str,
                   feat_dim: int, local_keywords: list,
                   data_root: str, train_split: str, test_split: str,
                   class_num: int, dataset_name: str, timestamp: str) -> float:
    """
    完整实验：加载编码器 → 提取特征 → 训练分类头 → 保存最佳模型。
    若 pretrainModels/ 中存在匹配的本地权重文件，则直接加载；
    否则 open_clip 自动从 HuggingFace 下载到默认缓存目录。
    """
    logger.info('=' * 60)
    logger.info('Model   : %s', exp_name)
    logger.info('Dataset : %s  train=%s  test=%s  classes=%d',
                data_root, train_split, test_split, class_num)

    # ── 1. 加载编码器 ─────────────────────────────────────────────────────────
    logger.info('加载 CLIP 编码器...')
    local_path = find_local_weight(local_keywords)
    # 有本地权重时直接传路径，否则传 pretrained_tag 让 open_clip 自动下载
    pretrained = local_path if local_path else pretrained_tag
    encoder, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=DEVICE,
    )
    encoder.eval()
    logger.info('编码器加载完成，参数量: %.2f B',
                sum(p.numel() for p in encoder.parameters()) / 1e9)

    # ── 2. 构建数据集 & DataLoader ────────────────────────────────────────────
    train_set    = StyleDataset(data_root, train_split)
    test_set     = StyleDataset(data_root, test_split)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    logger.info('训练集: %d 样本  测试集: %d 样本', len(train_set), len(test_set))

    # ── 3. 预提取特征 ─────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info('提取训练集特征...')
    train_feats, train_labels = extract_features(encoder, train_loader)
    logger.info('提取测试集特征...')
    test_feats, test_labels   = extract_features(encoder, test_loader)
    logger.info('特征提取完成，耗时 %.1f s', time.time() - t0)

    # ── 4. 训练分类头 ─────────────────────────────────────────────────────────
    logger.info('开始训练分类头  feat_dim=%d  lr=%.4f  epochs=%d',
                feat_dim, CLASSIFIER_LR, CLASSIFIER_EPOCHS)
    t0 = time.time()
    classifier, best_acc = train_classifier(
        train_feats, train_labels, test_feats, test_labels, feat_dim, class_num)
    logger.info('分类头训练完成，耗时 %.1f s', time.time() - t0)

    # ── 5. 保存最佳分类器 ────────────────────────────────────────────────────
    # 模型标签：去掉特殊字符，便于文件命名
    model_tag = model_name.replace('-', '').replace('_', '')
    acc_str   = f'{best_acc * 100:.2f}'.replace('.', '_')
    save_name = f'oc-{dataset_name}-{model_tag}-acc{acc_str}-{timestamp}.pth'
    save_path = os.path.join(MODEL_DIR, save_name)
    torch.save(classifier.state_dict(), save_path)
    logger.info('Model saved : %s', save_path)
    logger.info('Best acc    : %.4f  (%.2f%%)', best_acc, best_acc * 100)
    logger.info('=' * 60)
    return best_acc


# ─── 命令行参数 ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='OpenCLIP Community Variants 风格分类（ViT-H-14 / ViT-g-14 / ViT-bigG-14）')
    parser.add_argument('--data_root',  type=str,
                        default='/mnt/codes/data/style/Painting91')
    parser.add_argument('--train',      type=str, default='train')
    parser.add_argument('--test',       type=str, default='test')
    parser.add_argument('--class_num',  type=int, default=13)
    return parser.parse_args()


# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    timestamp    = time.strftime('%Y%m%d-%H%M%S')
    dataset_name = os.path.basename(args.data_root.rstrip('/'))
    log_filename = f'oc-community-{dataset_name}-{timestamp}.log'

    global logger
    logger = setup_logger(log_filename)

    logger.info('Device  : %s', DEVICE)
    logger.info('Dataset : %s', args.data_root)
    logger.info('Train   : %s  Test: %s  Classes: %d',
                args.train, args.test, args.class_num)
    logger.info('Log     : %s', os.path.join(LOG_DIR, log_filename))

    results = {}
    for exp_name, model_name, pretrained_tag, feat_dim, keywords in EXPERIMENTS:
        results[exp_name] = run_experiment(
            exp_name      = exp_name,
            model_name    = model_name,
            pretrained_tag= pretrained_tag,
            feat_dim      = feat_dim,
            local_keywords= keywords,
            data_root     = args.data_root,
            train_split   = args.train,
            test_split    = args.test,
            class_num     = args.class_num,
            dataset_name  = dataset_name,
            timestamp     = timestamp,
        )

    # ════ 汇总 ════
    logger.info('')
    logger.info('=' * 60)
    logger.info('Result: %s style classification', dataset_name)
    logger.info('-' * 60)
    for name, acc in results.items():
        logger.info('%-45s: %.2f%%', name, acc * 100)
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
