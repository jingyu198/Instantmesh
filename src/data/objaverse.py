import os, sys
import math
import json
import importlib
from pathlib import Path
import glob
import cv2
import random
import numpy as np
from PIL import Image
import webdataset as wds
import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from src.utils.train_util import instantiate_from_config
from src.utils.data_util import process_paths, get_pick_index
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    center_looking_at_camera_pose, 
    get_circular_camera_poses,
    load_c2ws_mv,load_c2ws_gt,
    get_zero123plus_c2ws
)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=1, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        input_image_dir='rendering_random_32views',
        target_image_dir='rendering_random_32views',
        input_view_num=6,
        target_view_num=4,
        total_view_n=32,
        fov=30,
        camera_rotation=True,
        validation=False,
        data_mode = 'gt',     # 'gt' or 'mv'
        n_obj = -1,     # -1 means using all 3D data. 
        input_image_size=320,
    ):
        self.root_dir = Path(root_dir)
        self.meta_fname = Path(meta_fname)
        self.input_image_dir = input_image_dir
        self.target_image_dir = target_image_dir

        self.input_view_num = input_view_num
        self.target_view_num = target_view_num
        self.total_view_n = total_view_n
        self.fov = fov
        self.camera_rotation = camera_rotation
        self.input_image_size=input_image_size
        self.depth_scale = 6.0

        self.data_mode = data_mode
        self.n_obj = n_obj


        with open(meta_fname) as f:
            filtered_dict = json.load(f)
        paths_all = filtered_dict['all_objs'][:]

        # filter npy != [200000,8]
        paths = []
        for path in paths_all:
            obj_id = path.split('/')[-1]
            grad_path = os.path.join("/data/model/objaverse_npys_100w_w_grad/", obj_id + '.npy')
            if os.path.exists(grad_path):
                paths.append(path)

        if self.n_obj == -1: 
            self.n_obj = len(paths)
        self.paths = process_paths(paths, self.data_mode, self.n_obj)
            
        total_objects = len(self.paths)
        print('============= length of train dataset %d =============' % len(self.paths))
        #print(self.paths[:2])

    def __len__(self):
        return len(self.paths)
    
    def load_im(self, path, color,normal=False):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)
        pil_img = pil_img.resize((self.input_image_size, self.input_image_size), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)
        
            
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_xyz_occ_sdf(self, path, surf_path, sample = 300000):
        pointcloud = np.load(path)   # [n, 5]
        surface = np.load(surf_path)  # [m, 3]
        
        # 原点云数据
        xyz = torch.from_numpy(pointcloud[:, :3]).contiguous().float()  
        sdf = torch.from_numpy(pointcloud[:, 4]).contiguous().float()
        grad = torch.from_numpy(pointcloud[:, 5:]).contiguous().float()

        # Surface 数据
        surf_xyz = torch.from_numpy(surface[:, :3]).contiguous().float()
        surf_sdf = torch.zeros(surf_xyz.size(0))  # Surface的SDF设置为0
        surf_grad = torch.from_numpy(surface[:, 3:]).contiguous().float()

        # 将surface的xyz, occupancy, sdf, grad加到原来的数据中
        xyz = torch.cat([xyz, surf_xyz], dim=0)
        sdf = torch.cat([sdf, surf_sdf], dim=0)
        grad = torch.cat([grad, surf_grad], dim=0)

        # 创建新的 sdf_new
        sdf_new = torch.where(sdf < -0.02, torch.tensor(-1.0, device=sdf.device),
                        torch.where(sdf > 0.02, torch.tensor(1.0, device=sdf.device), sdf))

        # 将 SDF 从 [-1, 1] 转换为 [1, 0]
        sdf_new = 1 - (sdf_new + 1) / 2  # 转换公式

        # 创建 mask
        mask = torch.where(
            sdf == 0,  # SDF 等于 0 的区域
            torch.tensor(0, device=sdf.device),  # mask 为 0
            torch.where(
                (sdf >= -0.02) & (sdf <= 0.02),  # SDF 在 (-0.02, 0.02) 之间
                torch.tensor(1, device=sdf.device),  # mask 为 1
                torch.tensor(2, device=sdf.device)  # 其他区域 mask 为 2
            )
        )

        total_points = xyz.size(0)
        indices = torch.randperm(total_points)[:sample]
        xyz = xyz[indices]
        sdf_new = sdf_new[indices]
        grad = grad[indices]
        mask = mask[indices]
        return xyz, sdf_new, grad, mask


    def __getitem__(self, index):
        index_mode = self.paths[index][:2]
        index_dim = self.paths[index][3] # 2d or 3d
        while True:
            # set loss_mask
            if index_dim == '3':  # for 3D data
                loss_mask = torch.ones(self.input_view_num + self.target_view_num)
            elif index_dim == '2':  # for 2D data
                loss_mask = torch.cat([torch.ones(self.input_view_num), torch.zeros(self.target_view_num)])  

            input_image_path = os.path.join(self.root_dir, self.input_image_dir, self.paths[index][5:])

            indices = np.random.choice(range(self.total_view_n), self.input_view_num + self.target_view_num, replace=False)
            input_indices_gt, input_indices_mv = [],[]
            
            if index_mode == "gt":
                input_indices_gt = indices[:self.input_view_num]
            elif index_mode == "mv":
                input_indices_mv = list(range(self.total_view_n, self.input_view_num + self.total_view_n))
            

            '''background color, default: white'''
            bg_white = [1., 1., 1.]
            bg_black = [0., 0., 0.]

            image_list = []
            pose_list = []

            #check number of input
            if len(input_indices_gt)+len(input_indices_mv) != self.input_view_num:
                print("WRONG NUM: ", len(input_indices_gt)+len(input_indices_mv))
            try:
                cameras_32 = load_c2ws_gt(os.path.join(input_image_path, 'camera_info.txt'))
                for idx in input_indices_gt:
                    image, alpha = self.load_im(os.path.join(input_image_path, '%03d.png' % idx), bg_white)
                    pose = cameras_32[idx]

                    image_list.append(image)
                    pose_list.append(pose)

                pick_index = get_pick_index(input_image_path)
                cameras_7 = load_c2ws_mv(os.path.join(input_image_path, 'camera_info.txt'), pick_index)
                for idx in input_indices_mv:
                    print("注意输入有generated MV!!!")
                    image, alpha = self.load_im(os.path.join(input_image_path, '%03d_rand_%d.png' % (idx, pick_index)), bg_white)
                    pose = cameras_7[idx-self.total_view_n]

                    image_list.append(image)
                    alpha_list.append(alpha)
                    depth_list.append(depth)
                    normal_list.append(normal)
                    pose_list.append(pose)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
        
        images = torch.stack(image_list, dim=0).float()                 # (6+V, 3, H, W)
        c2ws = torch.from_numpy(np.stack(pose_list, axis=0)).float()    # (6+V, 4, 4)



        # random rotation along z axis
        if self.camera_rotation:
            degree = np.random.uniform(0, math.pi * 2)
            rot = torch.tensor([
                [np.cos(degree), -np.sin(degree), 0, 0],
                [np.sin(degree), np.cos(degree), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]).unsqueeze(0).float()
            c2ws = torch.matmul(rot, c2ws)

        # random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.7, 1.1)
            c2ws[:, :3, 3] *= scale

        # instrinsics of perspective cameras
        K = FOV_to_intrinsics(self.fov)
        Ks = K.unsqueeze(0).repeat(self.input_view_num + self.target_view_num, 1, 1).float()

        # load point cloud and occupancies
        obj_id = self.paths[index].split('/')[-1]
        occ_path = os.path.join("/data/model/objaverse_npys_100w_w_grad/", obj_id + '.npy')
        surf_path = os.path.join("/data/model/objaverse_npys_100w_surface_w_grad/", obj_id + '.npy')
        xyz, sdf, grad, mask = self.load_xyz_occ_sdf(occ_path, surf_path)
        
        data = {
            'input_images': images[:self.input_view_num],           # (6, 3, H, W)
            'input_c2ws': c2ws[:self.input_view_num],               # (6, 4, 4)
            'input_Ks': Ks[:self.input_view_num],                   # (6, 3, 3)

            "xyz": xyz,
            "sdf": sdf,
            "mask": mask,
            "grad": grad
        }
        return data


class ValidationData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        input_view_num=6,
        input_image_size=320,
        fov=30,
        total_view_n=32
    ):
        self.root_dir = Path(root_dir)
        self.meta_fname = Path(meta_fname)

        self.input_view_num = input_view_num
        self.input_image_size = input_image_size
        self.fov = fov
        self.total_view_n = total_view_n
        
        #self.paths = sorted(os.listdir(self.root_dir))
        with open(meta_fname) as f:
            filtered_dict = json.load(f)
        paths = filtered_dict['all_objs'][8:]
        self.paths = paths

        print('============= length of val dataset %d =============' % len(self.paths))

        # cam_distance = 4.0
        # azimuths = np.array([30, 90, 150, 210, 270, 330])
        # elevations = np.array([20, -10, 20, -10, 20, -10])
        # azimuths = np.deg2rad(azimuths)
        # elevations = np.deg2rad(elevations)

        # x = cam_distance * np.cos(elevations) * np.cos(azimuths)
        # y = cam_distance * np.cos(elevations) * np.sin(azimuths)
        # z = cam_distance * np.sin(elevations)

        # cam_locations = np.stack([x, y, z], axis=-1)  # (6, 3)
        # cam_locations = torch.from_numpy(cam_locations).float()
        # c2ws = center_looking_at_camera_pose(cam_locations)
        # self.c2ws = c2ws.float()
        # self.Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(6, 1, 1).float()

        # render_c2ws = get_circular_camera_poses(M=8, radius=cam_distance, elevation=20.0)
        # render_Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(render_c2ws.shape[0], 1, 1)
        # self.render_c2ws = render_c2ws.float()
        # self.render_Ks = render_Ks.float()

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)
        pil_img = pil_img.resize((self.input_image_size, self.input_image_size), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        if image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        else:
            alpha = np.ones_like(image[:, :, :1])

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_xyz_occ_sdf(self, path, surf_path):
        pointcloud = np.load(path)   # [n, 5]
        surface = np.load(surf_path)  # [m, 3]
        
        # 原点云数据
        xyz = torch.from_numpy(pointcloud[:, :3]).contiguous().float()  
        sdf = torch.from_numpy(pointcloud[:, 4]).contiguous().float()
        grad = torch.from_numpy(pointcloud[:, 5:]).contiguous().float()

        # Surface 数据
        surf_xyz = torch.from_numpy(surface[:, :3]).contiguous().float()
        surf_sdf = torch.zeros(surf_xyz.size(0))  # Surface的SDF设置为0
        surf_grad = torch.from_numpy(surface[:, 3:]).contiguous().float()

        # 将surface的xyz, occupancy, sdf, grad加到原来的数据中
        xyz = torch.cat([xyz, surf_xyz], dim=0)
        sdf = torch.cat([sdf, surf_sdf], dim=0)
        grad = torch.cat([grad, surf_grad], dim=0)

        # 创建新的 sdf_new
        sdf_new = torch.where(sdf < -0.02, torch.tensor(-1.0, device=sdf.device),
                        torch.where(sdf > 0.02, torch.tensor(1.0, device=sdf.device), sdf))

        # 将 SDF 从 [-1, 1] 转换为 [1, 0]
        sdf_new = 1 - (sdf_new + 1) / 2  # 转换公式

        # 创建 mask
        mask = torch.where(
            sdf == 0,  # SDF 等于 0 的区域
            torch.tensor(0, device=sdf.device),  # mask 为 0
            torch.where(
                (sdf >= -0.02) & (sdf <= 0.02),  # SDF 在 (-0.02, 0.02) 之间
                torch.tensor(1, device=sdf.device),  # mask 为 1
                torch.tensor(2, device=sdf.device)  # 其他区域 mask 为 2
            )
        )

        return xyz, sdf_new, grad, mask

    def __getitem__(self, index):
        # load data
        input_image_path = os.path.join(self.root_dir, self.paths[index])
        
        input_indices = list(range(1,7))
        
        '''background color, default: white'''
        bkg_color = [1.0, 1.0, 1.0]

        image_list = []
        alpha_list = []
        pose_list = []


        input_cameras = load_c2ws_mv(os.path.join(input_image_path, 'camera_info.txt'), 0)
        # input_cameras = get_zero123plus_c2ws(cam_distance=2.5)
        for idx in input_indices:
            image, alpha = self.load_im(os.path.join(input_image_path, '%03d.png' % (idx)), bkg_color)
            pose = input_cameras[idx-1]

            image_list.append(image)
            pose_list.append(pose)
        
        images = torch.stack(image_list, dim=0).float()
        self.c2ws = torch.from_numpy(np.stack(pose_list, axis=0)).float()    # (6, 4, 4)
        K = FOV_to_intrinsics(self.fov)
        self.Ks = K.unsqueeze(0).repeat(self.input_view_num, 1, 1).float()
        
        # load point cloud and occupancies
        obj_id = self.paths[index].split('/')[-1]
        occ_path = os.path.join("/data/model/objaverse_npys_100w_w_grad/", obj_id + '.npy')
        surf_path = os.path.join("/data/model/objaverse_npys_100w_surface_w_grad/", obj_id + '.npy')
        xyz, sdf, grad, mask = self.load_xyz_occ_sdf(occ_path, surf_path)
        
        
        data = {
            'input_images': images,
            'input_c2ws': self.c2ws,
            'input_Ks': self.Ks,

            "xyz": xyz,
            "sdf": sdf,
            "mask": mask,
            "grad": grad,
        }
        return data