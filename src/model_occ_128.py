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
        self.res = 512
        # init modules
        self.lrm_generator = instantiate_from_config(lrm_generator_config)
        self.train_check_interval = train_check_interval

        if lrm_path is not None:
            #if os.path.basename(lrm_path) == 'instant_nerf_large.ckpt':  # 复用原始lrm权重
                
            # print("加载原文权重.......")
            # sd = torch.load(lrm_path, map_location='cpu')['state_dict']
            # sd = {k: v for k, v in sd.items() if k.startswith('lrm_generator')}
            # sd_fc = {}
            # for k, v in sd.items():

            #     if k.startswith('lrm_generator.transformer.deconv.'):
            #         continue
            #     # elif k.startswith('lrm_generator.synthesizer.decoder.net.6.'): 
            #     #     sd_fc[k] = v[0:1]
            #     else:
            #         sd_fc[k] = v

            # sd_fc = {k.replace('lrm_generator.', ''): v for k, v in sd_fc.items()}
            # self.lrm_generator.load_state_dict(sd_fc, strict=False)

            # #冻结 lrm_generator 中除了 synthesizer 层以外的所有参数
            # for name, param in self.lrm_generator.named_parameters():
            #     print(name)
            #     if name.startswith("transformer.deconv."): # or name.startswith("synthesizer.decoder.net."):
            #         param.requires_grad = True  # 需要训练
            #     else:
            #         param.requires_grad = False   # 冻结

            
            
            # else:   # 接着上次权重训练
            sd = torch.load(lrm_path, map_location='cpu')['state_dict']
            sd = {k.replace('lrm_generator.', ''): v for k, v in sd.items() if k.startswith('lrm_generator')}
            self.lrm_generator.load_state_dict(sd, strict=False)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        
        self.current_round= 0
    
    def on_fit_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'occ'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'occ_val'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'mc_val'), exist_ok=True)

    def prepare_batch_data(self, batch):
        lrm_generator_input = {}
        occ_gt = {}   # for supervision

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

        # ground truth occupancies
        occ_gt = batch['occupancy'].to(self.device)

        return lrm_generator_input, query_xyz, occ_gt

    
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

        # ground truth occupancies
        occ_gt = batch['occupancy'].to(self.device)

        return lrm_generator_input, query_xyz, occ_gt
    
    
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

    def training_step(self, batch, batch_idx):
        lrm_generator_input, query_xyz, occ_gt = self.prepare_batch_data(batch)   # torch.Size([b, 200000, 3]), # torch.Size([b, 200000]) 
        
        x_f = query_xyz[:, :, 2]  # GT的z轴 -> f的x轴
        y_f = query_xyz[:, :, 0]  # GT的x轴 -> f的y轴
        z_f = query_xyz[:, :, 1]  # GT的y轴 -> f的z轴
        transformed_xyz = torch.stack([x_f, y_f, z_f], dim=-1)
        
        occ_out = self.forward(lrm_generator_input, transformed_xyz) # torch.Size([b, 200000, 1])
        
        loss, loss_dict = self.compute_loss(occ_out, occ_gt.unsqueeze(-1), 'train')

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.global_step % self.train_check_interval == 0:     #1000
            print("train_occ_out min:", occ_out.min().item(), "max:", occ_out.max().item())   
            if self.global_rank == 0:
                batch_size = occ_out.size(0) 
                for bs in range(batch_size):
                    filtered_points = query_xyz[bs][occ_out[bs].squeeze() >= 0.5]  # torch.Size([n, 3])
                    out_path = os.path.join(self.logdir, 'occ', f'train_{self.global_step:07d}_{self.global_rank * batch_size + bs}.obj')
                    with open(out_path, 'w') as f:
                        for point in filtered_points:
                            f.write(f'v {point[0]} {point[1]} {point[2]}\n')
                    
                    out_path = os.path.join(self.logdir, 'occ', f'train_{self.global_step:07d}_{self.global_rank * batch_size + bs}_gt.obj')
                    filtered_gt = query_xyz[bs][occ_gt[bs] >= 0.5]    # torch.Size([n, 3])
                    with open(out_path, 'w') as f:
                        for point in filtered_gt:
                            f.write(f'v {point[0]} {point[1]} {point[2]}\n')

            input_images = lrm_generator_input['images']
            input_images = rearrange(
                input_images, 'b n c h w -> b c h (n w)')

            save_image(input_images, os.path.join(self.logdir, 'occ', f'train_{self.global_step:07d}.png'))

        return loss

    def compute_loss(self, occ_out, occ_gt, prefix):
        criteria = torch.nn.BCEWithLogitsLoss()
        loss_logits = criteria(occ_out, occ_gt).mean()
        loss_dict = {}
        loss_dict.update({f'{prefix}_loss': loss_logits})
        return loss_logits, loss_dict

    @torch.no_grad()
    
    def validation_step(self, batch, batch_idx):

        lrm_generator_input, query_xyz, occ_gt = self.prepare_validation_batch_data(batch)
        
        x_f = query_xyz[:, :, 2]  # GT的z轴 -> f的x轴
        y_f = query_xyz[:, :, 0]  # GT的x轴 -> f的y轴
        z_f = query_xyz[:, :, 1]  # GT的y轴 -> f的z轴
        transformed_xyz = torch.stack([x_f, y_f, z_f], dim=-1)
        
        occ_out = self.forward(lrm_generator_input, transformed_xyz) # torch.Size([b, 200000, 1])
        print("val_occ_out min:", occ_out.min().item(), "max:", occ_out.max().item())

        filtered_points = query_xyz[occ_out.squeeze(-1) >= 0.5]
        
        loss, loss_dict = self.compute_loss(occ_out, occ_gt.unsqueeze(-1), 'val')
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        out_path = os.path.join(self.logdir, 'occ_val', f'val_{self.global_step:07d}_rank{self.global_rank}.obj')
        
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
        grid_out = (grid_out > 0.5).astype(np.uint8)
        vertices, triangles = mcubes.marching_cubes(grid_out, 0.5)
        mcubes.export_obj(vertices, triangles, mesh_path)

        # smoothed_grid = mcubes.smooth(grid_out)
        # vertices, triangles = mcubes.marching_cubes(smoothed_grid, 0)
        # mcubes.export_obj(vertices, triangles, os.path.join(self.logdir, 'mc_val', f'val_{self.global_step:07d}_rank{self.global_rank}_smooth.obj'))


        # save ground truth images
        out_path = os.path.join(self.logdir, 'occ_val', f'val_images_rank{self.global_rank}.png')
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
