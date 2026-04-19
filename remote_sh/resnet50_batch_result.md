# ResNet50 批量训练结果

每数据集独立重复完整训练 **5** 次；每轮仅记录该轮 **best** 测试准确率，**不**汇总 last。 末列 **mean±std** 为该库各轮 best 的均值 ± 样本标准差。

超参来源: `ssc_train_resnet_copy.parameter_load()`

- epochs=100, base_lr=0.001, image_size=112
- cls_iteration=100, cls_lr=3e-05, cls_train_gap=5, cls_test_gap=2
- 每数据集重复 DATASET_REPEAT_RUNS=5，批量外层 ITERATIONS=1

| 数据集 | R1 | R2 | R3 | R4 | R5 | mean±std | 训练时长(min) | 开始时间 | 状态 |
|--------|-----|-----|-----|-----|-----|----------|--------------|----------|------|
| Painting91 | 0.7605 | 0.7521 | 0.7563 | 0.7563 | 0.7668 | 0.7584±0.0056 | 90.8 | 2026-04-19 07:37:01 | OK |
