import os
import numpy as np
import torch
import json,shutil
import rembg, random, math
from PIL import Image
from tqdm import tqdm



json_path = f"/home/gjy/jingyu/InstantMesh/filtered_obj_name.json"
root_dir = "/data/model"


# Process input files
with open(json_path) as f:
    filtered_dict = json.load(f)
paths = filtered_dict['all_objs'][:]

###############################################################################
# Stage 1: Multiview generation.
###############################################################################

skipped_count = 0  # 初始化跳过计数器
skipped_list=[]
for idx, image_file in enumerate(paths):
    try:
        rendering_path = os.path.join(root_dir, image_file, "camera_info_cam7.txt")
        image_file+="_Gold"
        cam7_path = os.path.join("/data/model/process_images/jewelry7w/", image_file, "camera_info.txt").replace("_cam32", "_cam7")

        shutil.copy2(cam7_path, rendering_path)
    
    except Exception as e:
        print(e)
        skipped_count += 1  # 增加跳过计数器
        skipped_list.append(idx+1)
        continue  # 跳过当前图像，继续处理下一个图像
    if idx % 100 ==0:
        print(idx)

# 输出总共跳过的路径数量
print(f"Total skipped paths: {skipped_count}")
print(f"Skipped paths: {skipped_list}")

