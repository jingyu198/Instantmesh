#!/bin/bash

N=8  # 设置屏幕数量为 8
for i in $(seq 1 $N); do
    screen -dmS m$i bash -c "conda activate im; python /home/gjy/jingyu/InstantMesh/mesh2npz/grad/get_grad.py --screen_id $i --total_screens $N; sleep 5"
done

conda activate mesh; python /home/gjy/jingyu/InstantMesh/mesh2npz/grad/get_grad_surface.py --screen_id 8 --total_screens 8
