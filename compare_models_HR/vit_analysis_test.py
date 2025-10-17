import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import collections
import pandas as pd
import os
import plotly.graph_objects as go
import pandas as pd

# 读取txt文件
def read_results(file_path):
    true_labels = []
    pred_labels = []
    skipped_lines = 0

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) < 2:
                skipped_lines += 1
                continue
            try:
                true_label = int(parts[1].split(": ")[1])+ 1
                if len(parts) >= 3:
                    pred_label = int(parts[2].split(": ")[1])+ 1
                else:
                    pred_label = true_label

                true_labels.append(true_label)
                pred_labels.append(pred_label)
            except (IndexError, ValueError):
                skipped_lines += 1

    print(f"Skipped {skipped_lines} malformed lines.")
    return np.array(true_labels), np.array(pred_labels)

# 加载数据
true_labels, pred_labels = read_results("/home/huangrui/Codes/SubStyleClassfication/compare_models/classify_result/vit-wiki3-classification.txt")

# 准确率
accuracy = accuracy_score(true_labels, pred_labels)
print('====================vit-wiki3======================')
print(f"Overall Accuracy: {accuracy:.4f}")

# 混淆矩阵
labels = np.unique(true_labels)
conf_matrix = confusion_matrix(true_labels, pred_labels, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("vit-wiki3-Confusion Matrix")
plt.savefig("/home/huangrui/Codes/SubStyleClassfication/compare_models/analysis_matrix_pic/vit-wiki3-confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# 分类报告
print(classification_report(true_labels, pred_labels))

# ========= 每类被误识别成什么类别的分析 + 导出CSV + 可视化 ========= #
output_rows = []
save_dir = "misclassification_plots"
os.makedirs(save_dir, exist_ok=True)

print("\n====== Per-Class Misclassification Details (Sorted by Percentage) ======")
for i, label in enumerate(labels):
    row = conf_matrix[i]
    total = np.sum(row)
    correct = conf_matrix[i][i]
    misclassified_total = total - correct

    if misclassified_total == 0:
        continue

    mis_list = []
    print(f"\nTrue Label {label} was misclassified {misclassified_total} times:")

    for j, count in enumerate(row):
        if i != j and count > 0:
            percent = (count / total) * 100
            mis_list.append((labels[j], count, percent))

    # 排序输出
    mis_list.sort(key=lambda x: x[2], reverse=True)
    for pred_label, count, percent in mis_list:
        print(f"  → Predicted as {pred_label}: {count} times ({percent:.2f}%)")
        output_rows.append({
            "True Label": label,
            "Predicted Label": pred_label,
            "Misclassified Count": count,
            "Total True Label Count": total,
            "Percentage": f"{percent:.2f}%"
        })

    # # 可视化误分类条形图
    # plt.figure(figsize=(6, 4))
    # sns.barplot(
    #     x=[str(p[0]) for p in mis_list],
    #     y=[p[2] for p in mis_list],
    #     palette="Reds_r"
    # )
    # plt.title(f"True Label {label} - Misclassification")
    # plt.xlabel("Predicted Label")
    # plt.ylabel("Misclassification %")
    # plt.ylim(0, 100)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f"misclass_label_{label}.png"), dpi=300)
    # plt.close()

# 导出CSV文件
df = pd.DataFrame(output_rows)
df.to_csv("/home/huangrui/Codes/SubStyleClassfication/compare_models/analysis_csv/wiki3-vit-per_class_misclassification.csv", index=False)
print("\nMisclassification report saved to: per_class_misclassification.csv")
print(f"Per-label misclassification bar plots saved to folder: {save_dir}/")


# 统计误分类情况
misclassified = [(t, p) for t, p in zip(true_labels, pred_labels) if t != p]
misclass_count = collections.Counter(misclassified)
# 打印最容易混淆的前5个类别
print("Top 5 Most Confused Pairs:")
for (true_class, pred_class), count in misclass_count.most_common(5):
    print(f"True Label {true_class} → Predicted Label {pred_class}: {count} times")


# 统计误分类流向：每个 true_label 被误预测为哪些 pred_label
confused_pairs = []
for t, p in zip(true_labels, pred_labels):
    if t != p:
        confused_pairs.append((t, p))

# 统计频数
pair_counts = collections.Counter(confused_pairs)

# label不加1，构造 Sankey 所需的数据
label_set = sorted(set(true_labels) | set(pred_labels))  # 所有类的合集，确保顺序一致
label_to_index = {label: idx for idx, label in enumerate(label_set)}



source = []
target = []
value = []

for (t, p), count in pair_counts.items():
    source.append(label_to_index[t])   # 真实标签位置
    target.append(label_to_index[p])   # 被预测成的位置
    value.append(count)                # 数量

# 构建 Sankey 图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=[f"Class {i}" for i in label_set],
        color="blue"
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color="rgba(255,0,0,0.4)"  # 淡红色表示错误流向
    )
)])


fig.update_layout(title_text="vit-wiki3-Misclassification Flow - Sankey Diagram", font_size=12)
fig.show()

