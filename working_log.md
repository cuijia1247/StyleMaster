20260328清理项目文件
脚本清单与功能
SSC 核心训练脚本（当前有效）
脚本	功能
ssc_train.py	SSC ResNet50 版本主训练脚本，含分类器联合训练
ssc_train_transformer.py	SSC Transformer（Swin/ViT）版本主训练脚本
ssc_predict.py	对指定数据集进行推理，统计 view1/view2 余弦相似度分布
SscDataSet.py	SSC 数据集加载类（Dataset 定义，被训练脚本导入使用）
visualize_sscreg_transformer.py	可视化 SSCReg Transformer 模型输出
基线对比模型训练脚本（均有效，用于横向对比）
脚本	方法
barlowtwins_train.py	Barlow Twins 对比学习
byol_train.py	BYOL 自监督学习
simclr_train.py	SimCLR 对比学习
simsiam_train.py	SimSiam 自监督学习
ijepa_train.py	I-JEPA 联合嵌入预测架构
HR 作者贡献脚本
脚本	功能
SC_predict_HR.py	HR 版本预测脚本（v2.0，2025-05-23）
SC_clean_HR.py	基于 ResNet50 特征余弦相似度过滤/清洗图像数据
gram_threshold_HR.py	计算 Gram 矩阵距离，用于确定风格相似度阈值
create_train_data_HR.py	将原始数据集按 train/test 比例自动划分
Shell 管理脚本
脚本	功能
run_ssc_resnet_background.sh	后台启动 ResNet 训练（nohup，写入 PID 文件）
run_ssc_vit_background.sh	后台启动 Transformer 训练（nohup，写入 PID 文件）
manage_ssc_resnet.sh	ResNet 训练进程管理：start/stop/restart/status/tail/logs
manage_ssc_vit.sh	Transformer 训练进程管理：status/start/stop/restart/logs/tail/list/clean
问题发现
1. 无效/废弃脚本（建议清理或归档）
脚本	原因
SC_train.py	文件头明确注明："DO NOT USE IT NOW. THIS IS THE OLD VERSION BEFORE 202504"
SC_train_optimization.py	同上，旧版参数搜索训练脚本
SC_predict.py	文件头注明："The codes do not finished yet by 20250425"，未完成脚本
Logging_test.py	仅 18 行的临时日志测试文件，无任何实际功能
2. 功能重复脚本
重复组	说明
ssc_train.py / SC_train.py / SC_train_optimization.py	三者都是 ResNet50 SSC 训练，后两者是旧版
ssc_predict.py / SC_predict.py / SC_predict_HR.py	三者都是预测推理，SC_predict.py 未完成
manage_ssc_resnet.sh 与 run_ssc_resnet_background.sh	前者的 start 命令直接调用后者，功能层叠
manage_ssc_vit.sh 与 run_ssc_vit_background.sh	同上
3. Shell 脚本 Bug（引用文件不存在）
manage_ssc_vit.sh 的 start_training 函数调用的是：

./run_ssc_background.sh   # ← 此文件不存在！
实际文件名为 run_ssc_vit_background.sh，导致 manage_ssc_vit.sh start 命令无法正常工作。