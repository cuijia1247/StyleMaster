# 启动训练
./remote_sh/run_ssc_resnet.sh
# 或使用管理脚本
./remote_sh/manage_ssc_resnet.sh start    # 启动
./remote_sh/manage_ssc_resnet.sh status   # 查看状态
./remote_sh/manage_ssc_resnet.sh tail     # 实时日志
./remote_sh/manage_ssc_resnet.sh stop     # 停止训练


# 启动训练
./remote_sh/run_resnet50_param_optim.sh
# 或使用管理脚本
./remote_sh/manage_resnet50_param_optim.sh status        # 最常用，一览全局
./remote_sh/manage_resnet50_param_optim.sh tail          # 看整体进度
./remote_sh/manage_resnet50_param_optim.sh tail-current  # 看当前这组的详细 loss
./remote_sh/manage_resnet50_param_optim.sh result        # 看已完成的精度汇总

