#!/bin/bash

N=8  # 设置屏幕数量为 8
for i in $(seq 1 $N); do
    screen -dmS m$i bash -c "cd /home/gjy/jingyu/InstantMesh/mesh2npz; conda activate mesh; python mesh2wt2npy.py --screen_id $i --total_screens $N; sleep 5"
done

cd /home/gjy/jingyu/InstantMesh/mesh2npz; conda activate mesh; python mesh2wt2npy.py --screen_id 8 --total_screens 8