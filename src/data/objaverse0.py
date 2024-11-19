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
        paths = filtered_dict['all_objs'][:]
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
    
    def load_xyz_occ_sdf(self, path):
        pointcloud = np.load(path)

        xyz = torch.from_numpy(pointcloud[:, :3]).contiguous().float()  
        occupancy = torch.from_numpy(pointcloud[:, 3]).contiguous().float() 
        sdf = torch.from_numpy(pointcloud[:, 4]).contiguous().float()  

        return xyz, occupancy, sdf
    
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
            range_mv = list(range(self.total_view_n, self.input_view_num + self.total_view_n))
            if index_mode == "gt":
                input_indices_gt = indices[:self.input_view_num]
            elif index_mode == "mv":
                input_indices_mv = range_mv
            
            target_indices = indices[self.input_view_num:]

            '''background color, default: white'''
            bg_white = [1., 1., 1.]
            bg_black = [0., 0., 0.]

            image_list = []
            alpha_list = []
            depth_list = []
            normal_list = []
            pose_list = []

            #check number of input
            if len(input_indices_gt)+len(input_indices_mv) != self.input_view_num:
                print("WRONG NUM: ", len(input_indices_gt)+len(input_indices_mv))

            try:
                cameras_32 = load_c2ws_gt(os.path.join(input_image_path, 'camera_info.txt'))
                for idx in input_indices_gt:
                    image, alpha = self.load_im(os.path.join(input_image_path, '%03d.png' % idx), bg_white)
                    normal, _ = self.load_im(os.path.join(input_image_path, '%03d_normal.png' % idx), bg_black)
                    depth = cv2.imread(os.path.join(input_image_path, '%03d_depth.png' % idx), cv2.IMREAD_UNCHANGED) / 255.0 * self.depth_scale
                    depth = torch.from_numpy(depth).unsqueeze(0)
                    pose = cameras_32[idx]

                    image_list.append(image)
                    alpha_list.append(alpha)
                    depth_list.append(depth)
                    normal_list.append(normal)
                    pose_list.append(pose)

                pick_index = get_pick_index(input_image_path)
                cameras_7 = load_c2ws_mv(os.path.join(input_image_path, 'camera_info.txt'), pick_index)
                for idx in input_indices_mv:
                    image, alpha = self.load_im(os.path.join(input_image_path, '%03d_rand_%d.png' % (idx, pick_index)), bg_white)
                    # 因为mv是没有normal和depth的，所以用backup
                    normal, _ = self.load_im(os.path.join(self.root_dir, '/home/gjy/jingyu/InstantMesh/backup/000_normal.png'), bg_black)
                    depth = cv2.imread(os.path.join(self.root_dir, '/home/gjy/jingyu/InstantMesh/backup/000_depth.png'), cv2.IMREAD_UNCHANGED) / 255.0 * self.depth_scale
                    depth = torch.from_numpy(depth).unsqueeze(0)
                    pose = cameras_7[idx-self.total_view_n]

                    image_list.append(image)
                    alpha_list.append(alpha)
                    depth_list.append(depth)
                    normal_list.append(normal)
                    pose_list.append(pose)

                for idx in target_indices:
                    image, alpha = self.load_im(os.path.join(input_image_path, '%03d.png' % idx), bg_white)
                    normal, _ = self.load_im(os.path.join(input_image_path, '%03d_normal.png' % idx), bg_black)
                    depth = cv2.imread(os.path.join(input_image_path, '%03d_depth.png' % idx), cv2.IMREAD_UNCHANGED) / 255.0 * self.depth_scale
                    depth = torch.from_numpy(depth).unsqueeze(0)
                    pose = cameras_32[idx]

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
        alphas = torch.stack(alpha_list, dim=0).float()                 # (6+V, 1, H, W)
        depths = torch.stack(depth_list, dim=0).float()                 # (6+V, 1, H, W)
        normals = torch.stack(normal_list, dim=0).float()               # (6+V, 3, H, W)
        c2ws = torch.from_numpy(np.stack(pose_list, axis=0)).float()    # (6+V, 4, 4)
        #c2ws = torch.linalg.inv(w2cs).float()

        def invert_channel(normals, channels):
            inverted_normals = normals.clone()
            if 'x' in channels:
                inverted_normals[:, 0, :, :] = 1.0 - inverted_normals[:, 0, :, :]  # 翻转第0通道 (R)
            if 'y' in channels:
                inverted_normals[:, 1, :, :] = 1.0 - inverted_normals[:, 1, :, :]  # 翻转第1通道 (G)
            if 'z' in channels:
                inverted_normals[:, 2, :, :] = 1.0 - inverted_normals[:, 2, :, :]  # 翻转第2通道 (B)
            return inverted_normals

        # 翻转法线坐标轴（例如，翻转x和y轴）
        channels_to_invert = ['x', 'y']  # 根据需要翻转的轴进行调整
        normals = invert_channel(normals, channels_to_invert)

        normals = normals * 2.0 - 1.0
        normals = F.normalize(normals, dim=1)
        normals = (normals + 1.0) / 2.0
        normals = torch.lerp(torch.zeros_like(normals), normals, alphas)

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

            # rotate normals
            N, _, H, W = normals.shape
            normals = normals * 2.0 - 1.0
            normals = torch.matmul(rot[:, :3, :3], normals.view(N, 3, -1)).view(N, 3, H, W)
            normals = F.normalize(normals, dim=1)
            normals = (normals + 1.0) / 2.0
            normals = torch.lerp(torch.zeros_like(normals), normals, alphas)

        # random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.7, 1.1)
            c2ws[:, :3, 3] *= scale
            depths *= scale

        # instrinsics of perspective cameras
        K = FOV_to_intrinsics(self.fov)
        Ks = K.unsqueeze(0).repeat(self.input_view_num + self.target_view_num, 1, 1).float()

        # load point cloud and occupancies
        obj_id = self.paths[index].split('/')[-1]
        occ_path = os.path.join("/data/model/objaverse_npys/", obj_id + '.npy')
        xyz, occupancy, sdf = self.load_xyz_occ_sdf(occ_path)

        data = {
            'input_images': images[:self.input_view_num],           # (6, 3, H, W)
            'input_alphas': alphas[:self.input_view_num],           # (6, 1, H, W) 
            'input_depths': depths[:self.input_view_num],           # (6, 1, H, W)
            'input_normals': normals[:self.input_view_num],         # (6, 3, H, W)
            'input_c2ws': c2ws[:self.input_view_num],               # (6, 4, 4)
            'input_Ks': Ks[:self.input_view_num],                   # (6, 3, 3)

            # lrm generator input and supervision
            'target_images': images[self.input_view_num:],          # (V, 3, H, W)
            'target_alphas': alphas[self.input_view_num:],          # (V, 1, H, W)
            'target_depths': depths[self.input_view_num:],          # (V, 1, H, W)
            'target_normals': normals[self.input_view_num:],        # (V, 3, H, W)
            'target_c2ws': c2ws[self.input_view_num:],              # (V, 4, 4)
            'target_Ks': Ks[self.input_view_num:],                  # (V, 3, 3)

            # a mask indicates which views to be considered for loss computing
            "mask": loss_mask,

            "xyz": xyz,
            "occupancy": occupancy,
            "sdf": sdf,
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
        paths = filtered_dict['all_objs'][:8]
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
    
    def load_xyz_occ_sdf(self, path):
        pointcloud = np.load(path)

        xyz = torch.from_numpy(pointcloud[:, :3]).contiguous().float()  
        occupancy = torch.from_numpy(pointcloud[:, 3]).contiguous().float() 
        sdf = torch.from_numpy(pointcloud[:, 4]).contiguous().float()  

        return xyz, occupancy, sdf
    
    def __getitem__(self, index):
        # load data
        input_image_path = os.path.join(self.root_dir, self.paths[index])
        
        input_indices = list(range(self.total_view_n, self.input_view_num + self.total_view_n))
        
        '''background color, default: white'''
        bkg_color = [1.0, 1.0, 1.0]

        image_list = []
        alpha_list = []
        pose_list = []

        pick_index = get_pick_index(input_image_path)
        cameras_7 = load_c2ws_mv(os.path.join(input_image_path, 'camera_info.txt'), pick_index)
        for idx in input_indices:
            image, alpha = self.load_im(os.path.join(input_image_path, '%03d_rand_%d.png' % (idx, pick_index)), bkg_color)
            pose = cameras_7[idx-self.total_view_n]

            image_list.append(image)
            alpha_list.append(alpha)
            pose_list.append(pose)
        
        images = torch.stack(image_list, dim=0).float()
        alphas = torch.stack(alpha_list, dim=0).float()
        self.c2ws = torch.from_numpy(np.stack(pose_list, axis=0)).float()    # (6, 4, 4)
        K = FOV_to_intrinsics(self.fov)
        self.Ks = K.unsqueeze(0).repeat(self.input_view_num, 1, 1).float()
        
        render_c2ws = get_circular_camera_poses(M=8, radius=4, elevation=20.0)
        render_Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(render_c2ws.shape[0], 1, 1)
        self.render_c2ws = render_c2ws.float()
        self.render_Ks = render_Ks.float()

        # load point cloud and occupancies
        obj_id = self.paths[index].split('/')[-1]
        occ_path = os.path.join("/data/model/objaverse_npys/", obj_id + '.npy')
        xyz, occupancy, sdf = self.load_xyz_occ_sdf(occ_path)

        data = {
            'input_images': images,
            'input_alphas': alphas,
            'input_c2ws': self.c2ws,
            'input_Ks': self.Ks,

            'render_c2ws': self.render_c2ws,
            'render_Ks': self.render_Ks,

            "xyz": xyz,
            "occupancy": occupancy,
            "sdf": sdf,
        }
        return data