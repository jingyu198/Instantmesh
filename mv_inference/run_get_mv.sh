#!/bin/bash

# 密码设置
SUDO_PASS="gjy@5zt/CrMgXTta/1Jhbata"
ID=8
# 创建并启动50个screen会话
for i in $(seq 1 $ID); do
    # 使用 `expect` 工具来自动输入 sudo 密码
    screen -dmS f$i bash -c "echo $SUDO_PASS | sudo -S /home/gjy/anaconda3/envs/im/bin/python get_mv.py configs/instant-nerf-large.yaml $i $ID"
done
