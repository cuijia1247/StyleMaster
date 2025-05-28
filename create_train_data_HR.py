import os
import shutil
import random

def split_dataset(
    src_root='painting91',
    dst_root='output_dataset',
    train_ratio=0.8,
    test_ratio=0.2,
    seed=42
):
    assert abs(train_ratio + test_ratio - 1.0) < 1e-6, "划分比例之和必须为1"

    random.seed(seed)

    for class_name in os.listdir(src_root):
        class_src_path = os.path.join(src_root, class_name)
        if not os.path.isdir(class_src_path):
            continue

        # 获取所有图片
        images = [f for f in os.listdir(class_src_path)
                  if os.path.isfile(os.path.join(class_src_path, f))]
        random.shuffle(images)

        # 计算数量
        total = len(images)
        train_end = int(total * train_ratio)

        subsets = {
            'train': images[:train_end],
            'test': images[train_end:]
        }

        # 拷贝文件
        for subset_name, subset_images in subsets.items():
            dst_class_dir = os.path.join(dst_root, subset_name, class_name)
            os.makedirs(dst_class_dir, exist_ok=True)

            for img in subset_images:
                src_img_path = os.path.join(class_src_path, img)
                dst_img_path = os.path.join(dst_class_dir, img)
                shutil.copy2(src_img_path, dst_img_path)

    print("✅ 数据集已成功划分为 train/test 结构。")

# 使用
split_dataset(
    src_root='/home/huangrui/Codes/SubStyleClassfication/split_data/WikiArt3',
    dst_root='/home/huangrui/Codes/SubStyleClassfication/train_data/WikiArt3',
    train_ratio=0.8,
    test_ratio=0.2
)