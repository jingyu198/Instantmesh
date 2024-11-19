import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import numpy as np
import torch
import json
import rembg, random, math
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

###############################################################################
# Arguments.
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('ID', type=int, help='ID for selecting dataset and paths')
parser.add_argument('NUM', type=int, help='NUM for total num of paths')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config
IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

ID = args.ID
NUM = args.NUM
json_path = "/home/gjy/jingyu/InstantMesh/filtered_obj_name.json"
root_dir = "/data/model"
model_name = "isnet-general-use"
remove_files = True

# Select GPU
gpu_id = ID % 8
torch.cuda.set_device(gpu_id)
device = torch.device(f'cuda:{gpu_id}')
print(f'Running on GPU {gpu_id}')


# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[10:]: v for k, v in state_dict.items() if k.startswith('unet.unet.')}
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

# Process input files
with open(json_path) as f:
    filtered_dict = json.load(f)
# Get range of index for processing json paths
filtered_dict['all_objs'] = filtered_dict['all_objs'][:]
total_paths = len(filtered_dict['all_objs'])
chunk_size = math.ceil(total_paths / NUM)
id_to_range = {}
for i in range(NUM):
    start_index = i * chunk_size
    end_index = min((i + 1) * chunk_size, total_paths)
    id_to_range[i + 1] = (start_index, end_index)
start, end = id_to_range.get(ID, (0, 0))

paths = filtered_dict['all_objs'][start:end]
print(f'Range: ({start}, {end}), Total number of dataset: {len(paths)}/{total_paths}')


###############################################################################
# Stage 1: Multiview generation.
###############################################################################

skipped_count = 0  # 初始化跳过计数器
skipped_list=[]
for idx, image_file in enumerate(paths):
    try:
        rendering_path = os.path.join(root_dir, image_file)

        # 检查路径是否存在
        if not os.path.exists(rendering_path):
            print(f"[{idx+1}/{len(paths)}] Skipping {rendering_path}, path does not exist.")
            skipped_count += 1  # 增加跳过计数器
            skipped_list.append(idx+1)
            continue

        if remove_files:
            for filename in os.listdir(rendering_path):
                file_path = os.path.join(rendering_path, filename)
                if filename.startswith('032') or filename.startswith('033') or filename.startswith('034')\
                 or filename.startswith('035') or filename.startswith('036') or filename.startswith('037')\
                 or (filename == "camera_info_cam7.txt") or filename.startswith('mv'):
                    os.remove(file_path)
                    #print(f"Deleted: {file_path}")

        # 复制相机参宿
        # target_path = os.path.join(root_dir, image_file, "camera_info_cam7.txt")
        # image_file+="_Gold"
        # cam7_path = os.path.join("/data/model/process_images/jewelry7w/", image_file, "camera_info.txt").replace("_cam32", "_cam7")

        # shutil.copy2(cam7_path, rendering_path)

        idx_pick = random.randint(0, 31)
        input_image = Image.open(os.path.join(rendering_path, '%03d.png' % idx_pick))
        
        print(f'[{idx+1}/{len(paths)}] Imagining  ..., Randomly choose view {idx_pick}')
        
        # sampling
        output_image = pipeline(
            input_image, 
            num_inference_steps=args.diffusion_steps, 
        ).images[0]
        output_image.save(os.path.join(rendering_path, 'mv_rand.png'))

        # remove background
        session = rembg.new_session(model_name)
        output_image = rembg.remove(output_image, session=session)  #RGBA
        output_image.save(os.path.join(rendering_path, 'mv_rembg_rand.png'))

        # 切分图像为六个部分
        width, height = output_image.size
        part_width = width // 2  # 320 pixels
        part_height = height // 3  # 320 pixels

        # 定义切分的位置
        boxes = [
            (0, 0, part_width, part_height),              # mv_001.png 左上
            (part_width, 0, width, part_height),          # mv_002.png 右上
            (0, part_height, part_width, 2 * part_height), # mv_003.png 左中
            (part_width, part_height, width, 2 * part_height), # mv_004.png 右中
            (0, 2 * part_height, part_width, height),     # mv_005.png 左下
            (part_width, 2 * part_height, width, height)  # mv_006.png 右下
        ]

        # 保存每个部分
        for i, box in enumerate(boxes):
            part_image = output_image.crop(box)
            #part_image = resize_foreground(part_image, 0.5)  #错误使用resize，会使得中心不一致
            part_image.save(os.path.join(rendering_path, '%03d_rand_%d.png' % (i + 32, idx_pick)))
            
            # if os.path.exists(os.path.join(rendering_path, 'mv_%03d_rand.png' % (i+1))): 
            #     os.remove(os.path.join(rendering_path, 'mv_%03d_rand.png' % (i+1))) # 删除原本错误的mv_001.png
    
    except Exception as e:
        skipped_count += 1  # 增加跳过计数器
        skipped_list.append(idx+1)
        print(e)
        continue  # 跳过当前图像，继续处理下一个图像

# 输出总共跳过的路径数量
print(f"Total skipped paths: {skipped_count}")
print(f"Skipped paths: {skipped_list}")



# delete pipeline to save memory
del pipeline
