import json
import os
import shutil
import random

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def transform_path(original_path):
    parts = original_path.split('/')
    
    # Extract the parts we need
    part1 = parts[0]
    part2 = parts[1:]

    parts = part1.split('_')
    part11,part12=parts[2],parts[3]
    # Construct the new path
    new_path = f"obj_models_{part11}_{part12}/" + "/".join(part2) + ".obj"
    
    return new_path


def copy_images(json_path, root_dir, dest_dir, obj_dir, num_samples=None, seed=None):
    # 读取JSON文件
    data = read_json(json_path)
    all_objs = data["all_objs"]
    
    # 设置随机数种子
    if seed is not None:
        random.seed(seed)
    
    # 随机选择num_samples个地址
    if num_samples is not None:
        all_objs = random.sample(all_objs, num_samples)
        #print(all_objs)

    # 定义列表
    numbers = [0, 1, 3, 6, 19]

    # 构建完整路径并复制图片
    for i, obj in enumerate(all_objs):
        idx_pick = random.choice(numbers)
        src_img_path = os.path.join(root_dir, obj, '%03d.png' % idx_pick)
        dest_img_path = os.path.join(dest_dir, f"{i+1}.png")
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dest_img_path)
        else:
            print(f"Warning: {src_img_path} does not exist")

        src_obj_path = os.path.join(root_dir, transform_path(obj))
        dest_obj_path = os.path.join(obj_dir, f"{i+1}.obj")
        print(src_img_path)
        #print()
        if os.path.exists(src_obj_path):
            shutil.copy(src_obj_path, dest_obj_path)
        else:
            print(f"Warning: {src_obj_path} does not exist")

if __name__ == "__main__":
    num_samples = 100  # 设置为None表示选择所有地址，或者设置为一个整数以选择特定数量的地址
    json_path = "/home/gjy/jingyu/InstantMesh/val_data.json"
    root_dir = '/data/model/'
    dest_dir = f'/home/gjy/jingyu/InstantMesh/examples_{num_samples}/img'
    obj_dir = f'/home/gjy/jingyu/InstantMesh/examples_{num_samples}/obj'    
    seed = 43  # 设置随机数种子 4

    exp_dir=os.path.join(*dest_dir.split('/')[:-1])
    #print(exp_dir)

    if os.path.exists(exp_dir): shutil.rmtree(exp_dir)
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)

    copy_images(json_path, root_dir, dest_dir, obj_dir, num_samples, seed)

