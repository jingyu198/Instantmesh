import argparse
import json
import os
from tqdm import tqdm
from mesh2watertight import get_watertight_mesh
from MeshSample.sample import generate_volume_dataset_new, generate_surface_dataset

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
    dir_path = "/data/model/objaverse_npys_100w_surface/"
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    for path in tqdm(data[start_index:end_index], desc=f"Processing meshes in screen {screen_id}", unit="mesh"): 
        obj_id = path.split('/')[-1]
        output_dir = os.path.join(dir_path, obj_id + '.npy')
        if os.path.exists(output_dir):
            continue
        mesh_dir = os.path.join("/data/model/objaverse_glbs/", obj_id + '.glb')
        wt_mesh = get_watertight_mesh(mesh_dir, '')  # 获取水密网格
        # generate_volume_dataset_new(wt_mesh, output_dir, 1000000, 0.1)  # 生成体积数据集
        generate_surface_dataset(wt_mesh, output_dir, 1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process meshes in parallel using screens.")
    parser.add_argument("--screen_id", type=int, required=True, help="ID of the screen (1 to N).")
    parser.add_argument("--total_screens", type=int, required=True, help="Total number of screens to use.")
    
    args = parser.parse_args()
    main(args.screen_id, args.total_screens)
