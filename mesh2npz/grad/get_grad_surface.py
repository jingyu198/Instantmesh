import argparse
import json
import os
from tqdm import tqdm
from grad_util_surface import get_normal
import numpy as np 

def main(screen_id, total_screens):
    json_path = "/home/gjy/jingyu/InstantMesh/json_files/obj_filtered_obj_name_cam32_all.json"
    with open(json_path, 'r') as f:
        data = json.load(f)["all_objs"]

    # 计算每个屏幕处理的数量
    n = len(data)
    chunk_size = n // total_screens
    # 将 screen_id - 1 作为索引
    start_index = (screen_id - 1) * chunk_size
    end_index = start_index + chunk_size if screen_id < total_screens else n
    # 处理分配给当前屏幕的部分
    dir_path = "/data/model/objaverse_npys_100w_surface_w_grad/"
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    for path in tqdm(data[start_index:end_index], desc=f"Processing meshes in screen {screen_id}", unit="mesh"): 
        obj_id = path.split('/')[-1]
        occ_path = os.path.join("/data/model/objaverse_npys_100w_surface/", obj_id + '.npy')
        output_dir = os.path.join(dir_path, obj_id + '.npy')
        
        try:
            if os.path.exists(output_dir):
                # 检查文件的精度
                # arr = np.load(output_dir)
                # if arr.dtype != np.float32:
                #     print(f"文件 {obj_id}.npy 的精度不是 float32，正在转换...")
                #     arr = arr.astype(np.float32)
                #     np.save(output_dir, arr)
                #     continue

                # if list(arr.shape) != [2000000,8]:
                #     grid_gradient(occ_path, output_dir, screen_id-1)
                #     continue
                
                continue
        except Exception as e:
            print(e)
        get_normal(occ_path, output_dir, screen_id-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process meshes in parallel using screens.")
    parser.add_argument("--screen_id", type=int, required=True, help="ID of the screen (1 to N).")
    parser.add_argument("--total_screens", type=int, required=True, help="Total number of screens to use.")
    
    args = parser.parse_args()
    main(args.screen_id, args.total_screens)
