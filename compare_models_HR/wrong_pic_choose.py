import numpy as np
import os
import shutil
from collections import defaultdict

# 初始化存储变量
names = []       # 存储图片文件名
true_labels = [] # 存储真实标签
pred_labels = [] # 存储预测标签

# 文件路径配置
results_file = "/home/huangrui/Codes/SubStyleClassfication/data/style_output/pandora/prediction_result.txt"
images_folder = "/home/huangrui/Codes/SubStyleClassfication/data/Pandora/Images"
output_base_folder = "/home/huangrui/Codes/SubStyleClassfication/compare_models_HR/wrong_pic_all_ssc"

# 感兴趣的真实标签
target_true_labels = [10,11]

# 创建顶层输出目录
for label in target_true_labels:
    os.makedirs(os.path.join(output_base_folder, f"misclassified_{label}"), exist_ok=True)

def extract_filename(full_str):
    if full_str.startswith("Image: "):
        return full_str[7:]
    return full_str

def read_ssc_results(file_path):
    true_labels = []
    pred_labels = []
    skipped_lines = 0

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                skipped_lines += 1
                continue
            try:
                pred_label = int(parts[1])
                true_label = int(parts[2])
                true_labels.append(true_label)
                pred_labels.append(pred_label)

                raw_name = parts[0]
                clean_name = extract_filename(raw_name)
                names.append(clean_name)

            except (IndexError, ValueError):
                skipped_lines += 1
            

    print(f"Skipped {skipped_lines} malformed lines.")
    return np.array(true_labels), np.array(pred_labels)

def read_results_vit(file_path):
    skipped_lines = 0
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) < 2:
                skipped_lines += 1
                continue
            try:
                true_label = int(parts[1].split(": ")[1]) + 1
                if len(parts) >= 3:
                    pred_label = int(parts[2].split(": ")[1]) + 1
                else:
                    pred_label = true_label

                raw_name = parts[0]
                clean_name = extract_filename(raw_name)
                true_labels.append(true_label)
                names.append(clean_name)
                pred_labels.append(pred_label)
            except (IndexError, ValueError):
                skipped_lines += 1

    print(f"跳过了 {skipped_lines} 行格式错误的数据")
    return np.array(true_labels), np.array(pred_labels)

# true_labels, pred_labels = read_results_vit(results_file)
true_labels, pred_labels = read_ssc_results(results_file)


# 准确率
correct = sum(t == p for t, p in zip(true_labels, pred_labels))
acc = correct / len(true_labels)
print(f"标签文件中的准确率：{acc:.4f}")

def process_misclassified(true_label):
    """将误分类图片根据预测标签分别存放，并记录数量"""
    # matches = [(name, t, p) for name, t, p in zip(names, true_labels, pred_labels)
    #            if t == true_label and p != true_label]
    matches = [(name, t, p) for name, t, p in zip(names, true_labels, pred_labels)
               if t == true_label ]

    misclass_count = defaultdict(int)  # 统计误分类为哪个预测标签

    print(f"\n处理真实标签为 {true_label} 的误分类图片，共 {len(matches)} 张")

    for name, t, p in matches:
        # 构建目标路径
        subfolder = os.path.join(output_base_folder, f"misclassified_{true_label}", f"to_{p}")
        os.makedirs(subfolder, exist_ok=True)

        src_path = os.path.join(images_folder, name)
        dst_path = os.path.join(subfolder, name)

        try:
            shutil.copy(src_path, dst_path)
            misclass_count[p] += 1
        except FileNotFoundError:
            print(f"警告：未找到文件 - {name}")
        except Exception as e:
            print(f"复制文件 {name} 时出错：{str(e)}")

    # 打印每种误分类的数量
    for pred, count in sorted(misclass_count.items()):
        print(f"  错误预测为 {pred} 的数量：{count}")

# 执行处理
for label in target_true_labels:
    process_misclassified(label)

print("\n所有处理完成！")
