import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
import mcubes
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import json
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
parser.add_argument('--input_path', type=str, default='/home/gjy/jingyu/InstantMesh/json_files/obj_filtered_obj_name_val_cam32.json', help='Path to input image or directory.')
parser.add_argument('--config', type=str, default='configs/instant-nerf-large.yaml', help='Path to config file.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--root_dir', type=str, default='/data/model', help='root dir.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', type=int, default=False, help='Do not remove input background.')
parser.add_argument('--export_texmap', type=int, default=False, help='Export a mesh with texture map.')
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

device = torch.device('cuda')

# load diffusion model
# print('Loading diffusion model ...')
# pipeline = DiffusionPipeline.from_pretrained(
#     "sudo-ai/zero123plus-v1.2", 
#     custom_pipeline="zero123plus",
#     torch_dtype=torch.float16,
# )
# pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
#     pipeline.scheduler.config, timestep_spacing='trailing'
# )

# # load custom white-background UNet
# print('Loading custom white-background unet ...')
# if os.path.exists(infer_config.unet_path):
#     unet_ckpt_path = infer_config.unet_path
# else:
#     unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
# state_dict = torch.load(unet_ckpt_path, map_location='cpu')['state_dict']
# state_dict = {k[10:]: v for k, v in state_dict.items() if k.startswith('unet.unet.')}
# pipeline.unet.load_state_dict(state_dict, strict=True)

# pipeline = pipeline.to(device)

# load reconstruction model
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)

# process input files
input_path = args.input_path
with open(input_path) as f:
    filtered_dict = json.load(f)
paths = filtered_dict['all_objs'][8:]
print(f'Total number of input images: {len(paths)}')


###############################################################################
# Stage 1: Multiview generation.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()

outputs = []

def read_images(input_path):
    import os
    import torch
    from PIL import Image
    from torchvision import transforms


    # 创建一个转换，调整图像大小并转为Tensor
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    # 创建一个空列表来存储图像张量
    image_tensors = []

    # 读取每张图像
    for i in range(1, 7):  # 从1到6
        image_name = f"{i:03d}.png"  # 格式化文件名
        image_path_mv = os.path.join(input_path, image_name)
        
        # 打开图像
        image = Image.open(image_path_mv).convert('RGBA')  # 确保是RGBA格式
        
        # 创建一个白色背景
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        
        # 将图像合成到白色背景上
        image_with_background = Image.alpha_composite(background, image)
        
        # 转换为RGB
        image_with_background = image_with_background.convert('RGB')

        # 应用转换
        image_tensor = transform(image_with_background)

        # 添加到列表中
        image_tensors.append(image_tensor)

    # 将列表转换为张量并堆叠
    images_tensor = torch.stack(image_tensors)

    #print(images_tensor[0, :, 0, 0])  # 应该输出 torch.Size([6, 3, 320, 320])

    return images_tensor


for idx, image_file in enumerate(paths):
    name = os.path.basename(image_file).split('/')[-1]
    print(f'[{idx+1}/{len(paths)}] Imagining {name} ...')

    sampling_mv = False
    if sampling_mv:
        # remove background optionally
        input_image = Image.open(os.path.join(args.root_dir,image_file,"000.png"))
        if not args.no_rembg:
            input_image = remove_background(input_image, rembg_session)
            input_image = resize_foreground(input_image, 0.85)
        
        # sampling
        output_image = pipeline(
            input_image, 
            num_inference_steps=args.diffusion_steps, 
        ).images[0]

        output_image.save(os.path.join(image_path, f'{name}.png'))
        print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
        print(images[0,:,0,0])
    
    
    #replace generated mv to ground truth mv
    gt_mv_path = os.path.join(args.root_dir,image_file)
    images = read_images(gt_mv_path)

    outputs.append({'name': name, 'images': images})

# delete pipeline to save memory
#del pipeline

###############################################################################
# Stage 2: Reconstruction.
###############################################################################

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=2.5).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )

        vertices, faces = mesh_out
        mcubes.export_obj(vertices, faces, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")
