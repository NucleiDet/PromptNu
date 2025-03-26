#!/bin/bash

# 定义基本的命令和路径
base_command="python tools/test.py"
config_path="configs/hovernet/hovernet_adam-lr1e-4_bs8_256x256_300e_monuseg_ruby.py"
base_model_path="work_dirs/hovernet/hovernet_adam-lr1e-4_bs8_256x256_300e_monuseg_ruby/epoch_"

# 循环，从 20 到 300，每次增加 20
for epoch in {20..300..20}
do
    # 构建完整的命令
    full_command="$base_command $config_path ${base_model_path}${epoch}.pth"
    
    # 执行命令
    echo "Executing: $full_command"
    $full_command

    # 如果需要在命令之间暂停，可以使用 sleep 命令
    # 例如，sleep 10 会暂停 10 秒
    # sleep 10
done

echo "All tests completed."
