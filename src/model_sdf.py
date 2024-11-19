import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pytorch_lightning as pl
from einops import rearrange, repeat
from collections import OrderedDict
from src.utils.train_util import instantiate_from_config
import mcubes
import shutil

class MVRecon(pl.LightningModule):
    def __init__(
        self,
        lrm_generator_config,
        lrm_path=None,
        input_size=256,
        render_size=192,
        train_check_interval=200,
    ):
        super(MVRecon, self).__init__()

        self.input_size = input_size
        self.render_size = render_size
        self.res = 1000
        self.train_check_interval = train_check_interval
        # init modules
        self.lrm_generator = instantiate_from_config(lrm_generator_config)

        if lrm_path is not None:
            if os.path.basename(lrm_path) == 'instant_nerf_large.ckpt':  # 复用原始lrm权重
                sd = torch.load(lrm_path, map_location='cpu')['state_dict']
                sd = {k: v for k, v in sd.items() if k.startswith('lrm_generator')}
                sd_fc = {}
                # for k, v in sd.items():
                #     if k.startswith('lrm_generator.synthesizer.decoder.net.6.'):    # last layer
                #         # Here we assume the density filed's isosurface threshold is t, 
                #         # we reverse the sign of density filed to initialize SDF field.  
                #         # -(w*x + b - t) = (-w)*x + (t - b)
                #         if 'weight' in k:
                #             sd_fc[k] = -v[0:1]
                #         else:
                #             sd_fc[k] = 10.0 - v[0:1]
                #     else:
                #         sd_fc[k] = v
                for k, v in sd.items():
                    if k.startswith('lrm_generator.synthesizer.decoder.net.6.'):    # last layer
                        # Here we assume the density filed's isosurface threshold is t, 
                        # we reverse the sign of density filed to initialize SDF field.  
                        # -(w*x + b - t) = (-w)*x + (t - b)
                        sd_fc[k] = v[0:1]
                    else:
                        sd_fc[k] = v

                sd_fc = {k.replace('lrm_generator.', ''): v for k, v in sd_fc.items()}
                self.lrm_generator.load_state_dict(sd_fc, strict=False)
            elif os.path.basename(lrm_path) == 'instant_mesh_large.ckpt':  # 复用原始lrm权重
                sd = torch.load(lrm_path, map_location='cpu')['state_dict']
                sd = {k: v for k, v in sd.items() if k.startswith('lrm_generator')}
                sd_fc = {}
                for k, v in sd.items():
                    if k.startswith('lrm_generator.synthesizer.decoder.net_sdf.'):    # last layer
                        # Here we assume the density filed's isosurface threshold is t, 
                        # we reverse the sign of density filed to initialize SDF field.  
                        # -(w*x + b - t) = (-w)*x + (t - b)
                        sd_fc[k.replace('net_sdf.', 'net.')] = v

                    elif not (k.startswith('lrm_generator.synthesizer.decoder.net_rgb.') or \
                                k.startswith('lrm_generator.synthesizer.decoder.net_deformation.') or\
                                k.startswith('lrm_generator.synthesizer.decoder.net_weight.')):
                        sd_fc[k] = v

                sd_fc = {k.replace('lrm_generator.', ''): v for k, v in sd_fc.items()}
                self.lrm_generator.load_state_dict(sd_fc, strict=False)
            else:   # 接着上次权重训练
                sd = torch.load(lrm_path, map_location='cpu')['state_dict']
                sd = {k.replace('lrm_generator.', ''): v for k, v in sd.items() if k.startswith('lrm_generator')}
                self.lrm_generator.load_state_dict(sd, strict=False)

            # #冻结 lrm_generator 中除了 synthesizer 层以外的所有参数
            for name, param in self.lrm_generator.named_parameters():
                #print(name)
                if name.startswith("synthesizer.decoder.net."):
                    param.requires_grad = True  # 需要训练
                else:
                    param.requires_grad = True   # 冻结


        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        
        self.current_round= 0
    
    def on_fit_start(self):
        if self.global_rank == 0:
            code_folder = os.path.join(self.logdir, 'code')
            os.makedirs(code_folder, exist_ok=True)
            src_folder = '/home/gjy/jingyu/InstantMesh/src'
            dest_folder = os.path.join(code_folder, 'src')
            shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)

            os.makedirs(os.path.join(self.logdir, 'sdf'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'sdf_val'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'mc_val'), exist_ok=True)
    
    def random_crop(self, query_xyz, sdf_gt, grad_gt, mask, min_points=400000):
        query_xyz_list = []
        sdf_gt_list = []
        mask_list = []
        grad_gt_list = []

        for i in range(query_xyz.size(0)):  # 遍历每个 batch 中的样本
            # 提取单个样本
            query_xyz_sample_0 = query_xyz[i].to(self.device)  # torch.Size([num_points, 3])
            sdf_gt_sample_0 = sdf_gt[i].to(self.device)  # torch.Size([num_points])
            mask_sample_0 = mask[i].to(self.device)  # torch.Size([num_points])
            grad_gt_sample_0 = grad_gt[i].to(self.device)

            # 生成随机立方体中心，范围在 [-0.5, 0.5]
            cube_center = (torch.rand(3) - 0.5).to(self.device)
            lower_bound = cube_center - 0.5
            upper_bound = cube_center + 0.5
            mask_ = ((query_xyz_sample_0 >= lower_bound) & (query_xyz_sample_0 <= upper_bound)).all(dim=-1)

            query_xyz_sample = query_xyz_sample_0[mask_]
            sdf_gt_sample = sdf_gt_sample_0[mask_]
            mask_sample = mask_sample_0[mask_]
            grad_gt_sample = grad_gt_sample_0[mask_]

            filtered_num = query_xyz_sample.size(0)
            # 确保每个样本的点数达到 min_points
            if filtered_num < min_points:
                # 如果点数少于 min_points，进行填充
                padding_num = min_points - filtered_num
                extra_indices = torch.randperm(query_xyz_sample_0.size(0))[:padding_num]
                query_xyz_sample = torch.cat([query_xyz_sample, query_xyz_sample_0[extra_indices]], dim=0)
                sdf_gt_sample = torch.cat([sdf_gt_sample, sdf_gt_sample_0[extra_indices]], dim=0)
                mask_sample = torch.cat([mask_sample, mask_sample_0[extra_indices]], dim=0)
                grad_gt_sample = torch.cat([grad_gt_sample, grad_gt_sample_0[extra_indices]], dim=0)
            elif filtered_num > min_points:
                # 如果点数超过 min_points，进行随机采样
                indices = torch.randperm(filtered_num)[:min_points]
                query_xyz_sample = query_xyz_sample[indices]
                sdf_gt_sample = sdf_gt_sample[indices]
                mask_sample = mask_sample[indices]
                grad_gt_sample = grad_gt_sample[indices]

            query_xyz_list.append(query_xyz_sample)
            sdf_gt_list.append(sdf_gt_sample)
            mask_list.append(mask_sample)
            grad_gt_list.append(grad_gt_sample)

        query_xyz = torch.stack(query_xyz_list, dim=0)
        sdf_gt = torch.stack(sdf_gt_list, dim=0)
        mask = torch.stack(mask_list, dim=0)
        grad_gt = torch.stack(grad_gt_list, dim=0)

        return query_xyz, sdf_gt, grad_gt, mask


    def prepare_batch_data(self, batch):
        lrm_generator_input = {}
        sdf_gt = {}   # for supervision

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)

        # input cameras and render cameras
        input_c2ws = batch['input_c2ws'].flatten(-2)
        input_Ks = batch['input_Ks'].flatten(-2)
        input_extrinsics = input_c2ws[:, :, :12]
        input_intrinsics = torch.stack([
            input_Ks[:, :, 0], input_Ks[:, :, 4], 
            input_Ks[:, :, 2], input_Ks[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        # add noise to input cameras
        cameras = cameras + torch.rand_like(cameras) * 0.04 - 0.02

        lrm_generator_input['cameras'] = cameras.to(self.device)

        # xyz of query points
        query_xyz = batch['xyz'].to(self.device)

        # ground truth sdf
        sdf_gt = batch['sdf'].to(self.device)

        mask = batch['mask'].to(self.device)
        grad_gt = batch['grad'].to(self.device)

        #query_xyz, sdf_gt, grad_gt, mask = self.random_crop(query_xyz, sdf_gt, grad_gt, mask)

        return lrm_generator_input, query_xyz, sdf_gt, grad_gt, mask

    
    def prepare_validation_batch_data(self, batch):
        lrm_generator_input = {}

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)

        input_c2ws = batch['input_c2ws'].flatten(-2)
        input_Ks = batch['input_Ks'].flatten(-2)

        input_extrinsics = input_c2ws[:, :, :12]
        input_intrinsics = torch.stack([
            input_Ks[:, :, 0], input_Ks[:, :, 4], 
            input_Ks[:, :, 2], input_Ks[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        lrm_generator_input['cameras'] = cameras.to(self.device)

        # xyz of query points
        query_xyz = batch['xyz'].to(self.device)

        # ground truth sdf
        sdf_gt = batch['sdf'].to(self.device)

        return lrm_generator_input, query_xyz, sdf_gt
    
    
    def forward_lrm_generator(
        self, 
        images, 
        cameras, 
        query_xyz,
        chunk_size=1,
    ):
        planes = torch.utils.checkpoint.checkpoint(
            self.lrm_generator.forward_planes, 
            images, 
            cameras, 
            use_reentrant=False,
        )
        
        out = torch.utils.checkpoint.checkpoint(
            self.lrm_generator.synthesizer,
            planes,
            query_xyz,
            use_reentrant=False
        )

        return out

    
    def forward(self, lrm_generator_input, query_xyz):
        images = lrm_generator_input['images']
        cameras = lrm_generator_input['cameras']

        out = self.forward_lrm_generator(
            images, 
            cameras, 
            query_xyz
        )

        return out

    def gradient(self, x, y):
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def diff_gradient(self, lrm_generator_input, transformed_xyz, epsilon=5e-4):
        """
        使用差分法计算每个点的 SDF 对 x, y, z 的梯度。

        Args:
            self: 包含 forward 方法的类实例。
            lrm_generator_input: 生成器的输入，传递给 forward 方法。
            transformed_xyz: torch.Tensor, shape [batch, 200000, 3]，表示点云坐标。
            epsilon: float，偏移距离，用于差分计算。

        Returns:
            gradients: torch.Tensor, shape [batch, 200000, 3]，每个点的 SDF 对 x, y, z 的梯度。
        """
        # 初始化梯度张量
        batch_size, num_points, _ = transformed_xyz.shape
        gradients = torch.zeros((batch_size, num_points, 3), device=transformed_xyz.device)
        
        # 创建偏移量
        for i, axis in enumerate([0, 1, 2]):  # 分别处理 x, y, z 轴
            offset = torch.zeros_like(transformed_xyz)
            offset[..., axis] = epsilon

            sdf_pos, _ = self.forward(lrm_generator_input, transformed_xyz + offset)
            sdf_neg, _ = self.forward(lrm_generator_input, transformed_xyz - offset)

            # factor comes from "sdf_new = 1 - (sdf + 1) / 2"
            factor_ = 2
            gradients[..., i] = ((sdf_pos - sdf_neg) * factor_ / (2 * epsilon)).squeeze(-1)
        
        return gradients

    def transform_xyz(self, query_xyz):
        x_f = query_xyz[:, :, 2]  # GT的z轴 -> f的x轴
        y_f = query_xyz[:, :, 0]  # GT的x轴 -> f的y轴
        z_f = query_xyz[:, :, 1]  # GT的y轴 -> f的z轴
        return torch.stack([x_f, y_f, z_f], dim=-1)

    def training_step(self, batch, batch_idx):
        lrm_generator_input, query_xyz, sdf_gt, grad_gt, mask = self.prepare_batch_data(batch)   # torch.Size([b, 200000, 3]), # torch.Size([b, 200000]) 
        transformed_xyz, grad_gt = self.transform_xyz(query_xyz), self.transform_xyz(grad_gt)# 转换坐标和梯度

        transformed_xyz.requires_grad_(True)
        sdf_out = self.forward(lrm_generator_input, transformed_xyz) # torch.Size([b, 200000, 1])
        gradients = self.gradient(transformed_xyz, sdf_out)
        #diff_gradients = self.diff_gradient(lrm_generator_input, transformed_xyz)
        
        grad_magnitude = torch.norm(gradients, dim=-1)  # 计算每个点的梯度模长，结果形状是 [batch, points]
        batch_grad_mean = torch.mean(grad_magnitude, dim=1)  # 计算每个批次的平均梯度模长，形状是 [batch]
        if self.global_rank == 0:
           print(batch_grad_mean)

        loss, loss_dict = self.compute_loss(sdf_out, sdf_gt.unsqueeze(-1), gradients, grad_gt, mask, 'train')
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        

        if self.global_step % self.train_check_interval == 0:     #1000
            print("train_sdf_out min:", sdf_out.min().item(), "max:", sdf_out.max().item(), "GT min:", sdf_gt.unsqueeze(-1).min().item(), "max:", sdf_gt.unsqueeze(-1).max().item())   
            if self.global_rank == 0:
                batch_size = sdf_out.size(0) 
                for bs in range(batch_size):
                    filtered_points = query_xyz[bs][sdf_out[bs].squeeze() >= 0.5]  # torch.Size([n, 3])
                    out_path = os.path.join(self.logdir, 'sdf', f'train_{self.global_step:07d}_{self.global_rank * batch_size + bs}.obj')
                    with open(out_path, 'w') as f:
                        for point in filtered_points:
                            f.write(f'v {point[0]} {point[1]} {point[2]}\n')
                            f.write(f'v {point[0]} {point[1]} {point[2]}\n')
                    
                    out_path = os.path.join(self.logdir, 'sdf', f'train_{self.global_step:07d}_{self.global_rank * batch_size + bs}_gt.obj')
                    filtered_gt = query_xyz[bs][sdf_gt[bs] >= 0.5]    # torch.Size([n, 3])
                    with open(out_path, 'w') as f:
                        for point in filtered_gt:
                            f.write(f'v {point[0]} {point[1]} {point[2]}\n')

            input_images = lrm_generator_input['images']
            input_images = rearrange(
                input_images, 'b n c h w -> b c h (n w)')

            save_image(input_images, os.path.join(self.logdir, 'sdf', f'train_{self.global_step:07d}.png'))
        
        #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        torch.cuda.empty_cache()
        return loss

    def compute_loss(self, sdf_out, sdf_gt, gradients, grad_gt, mask, prefix):
        # Mask counts: {'mask == 0': 1000000, 'mask == 1': 218098, 'mask == 2': 1781902}
        mask_surface = (mask == 0)
        mask_vol = (mask == 2)
        mask_near = (mask == 1)
        loss_dict = {}
        mask_surface_count = mask_surface.sum().item()
        mask_near_count = mask_near.sum().item()
        mask_vol_count = mask_vol.sum().item()

        #print(f"Mask counts: {{'mask == 0': {mask_surface_count}, 'mask == 1': {mask_near_count}, 'mask == 2': {mask_vol_count}}}")
        
        # surface loss
        # 注意：grad_gt是根据[-1, 1]区间的sdf值计算，而转化后的sdf_new = 1 - (sdf_new + 1) / 2
        # 因此梯度值也应当做相应变换
        grad_loss_surf = (gradients[mask_surface] - grad_gt[mask_surface] * (-0.5)).norm(2, dim=-1).mean()
        sdf_loss_surf = (sdf_out[mask_surface] - sdf_gt[mask_surface]).abs().mean()
        criteria = torch.nn.BCEWithLogitsLoss()
        sdf_loss_surf = criteria(sdf_out[mask_surface], sdf_gt[mask_surface]).mean() 
        # print(gradients[mask_surface][:2])
        # print(grad_gt[mask_surface][:2])
        
        # near-surface loss
        sdf_loss_near = (sdf_out[mask_near] - sdf_gt[mask_near]).abs().mean()

        # volume loss
        occ_loss = criteria(sdf_out[~mask_surface], sdf_gt[~mask_surface]).mean() 
        eikonal_loss = ((gradients[~mask_vol].norm(2, dim=-1) - 0.5) ** 2).mean()

        loss_dict.update({f'{prefix}_surf_sdf': sdf_loss_surf, f'{prefix}_surf_grad': grad_loss_surf, \
                        f'{prefix}_near_sdf': sdf_loss_near, \
                        f'{prefix}_vol_occ': occ_loss, f'{prefix}_eik': eikonal_loss, \
                        })


        loss =  occ_loss + sdf_loss_surf
                # + 5 * sdf_loss_surf + 1 * sdf_loss_near \
                #+ 0.1 * grad_loss_surf \
                #+ 0.1 * eikonal_loss

        return loss, loss_dict


    @torch.no_grad()
    
    def validation_step(self, batch, batch_idx):

        lrm_generator_input, query_xyz, sdf_gt = self.prepare_validation_batch_data(batch)
        
        x_f = query_xyz[:, :, 2]  # GT的z轴 -> f的x轴
        y_f = query_xyz[:, :, 0]  # GT的x轴 -> f的y轴
        z_f = query_xyz[:, :, 1]  # GT的y轴 -> f的z轴
        transformed_xyz = torch.stack([x_f, y_f, z_f], dim=-1)
        
        sdf_out = self.forward(lrm_generator_input, transformed_xyz) # torch.Size([b, 200000, 1])

        print("val_sdf_out min:", sdf_out.min().item(), "max:", sdf_out.max().item())
        #loss, loss_dict = self.compute_loss(sdf_out, sdf_gt.unsqueeze(-1), gradients, grad_gt, 'val')
        #self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        filtered_points = query_xyz[sdf_out.squeeze(-1) >= 0.5]


        out_path = os.path.join(self.logdir, 'sdf_val', f'val_{self.global_step:07d}_rank{self.global_rank}.obj')     
        with open(out_path, 'w') as f:
            for point in filtered_points:
                f.write(f'v {point[0]} {point[1]} {point[2]}\n')
        
        # marching cube extract mesh
        mesh_path = os.path.join(self.logdir, 'mc_val', f'val_{self.global_step:07d}_rank{self.global_rank}.obj')
        x = torch.linspace(-1, 1, self.res)
        y = torch.linspace(-1, 1, self.res)
        z = torch.linspace(-1, 1, self.res)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        batch_size = 1
        grid = grid.reshape(1, -1, 3).repeat(batch_size, 1, 1)
        grid = grid.to(self.device)
        grid_out = self.forward(lrm_generator_input, grid).view(self.res, self.res, self.res)
        grid_out = grid_out.cpu().detach().numpy()
        #grid_out = (grid_out > 0).astype(np.uint8)
        vertices, triangles = mcubes.marching_cubes(grid_out, 0.5)
        mcubes.export_obj(vertices, triangles, mesh_path)

        # smoothed_grid = mcubes.smooth(grid_out)
        # vertices, triangles = mcubes.marching_cubes(smoothed_grid, 0)
        # mcubes.export_obj(vertices, triangles, os.path.join(self.logdir, 'mc_val', f'val_{self.global_step:07d}_rank{self.global_rank}_smooth.obj'))


        # save ground truth images
        out_path = os.path.join(self.logdir, 'sdf_val', f'val_images_rank{self.global_rank}.png')
        if not os.path.exists(out_path):
            input_images = lrm_generator_input['images']
            input_images = rearrange(
                input_images, 'b n c h w -> b c h (n w)')
            save_image(input_images, out_path)

    def configure_optimizers(self):
        lr = self.learning_rate

        params = []

        params.append({"params": [p for p in self.lrm_generator.parameters() if p.requires_grad], "lr": lr, "weight_decay": 0.01})

        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.90, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=lr/10)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}